"""Standalone scientific analysis; runnable with the bundled Python runtime."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def physical_hash(plan):
    return identity({k: plan[k] for k in ("period", "assignments", "routes", "repositions", "expired")})


def interpolate(times, values, when):
    """Independent linear interpolator, including the frozen endpoint clamp."""
    if when <= times[0]:
        return values[0]
    if when >= times[-1]:
        return values[-1]
    for i in range(1, len(times)):
        if when <= times[i]:
            weight = (when-times[i-1])/(times[i]-times[i-1])
            return values[i-1]*(1-weight)+values[i]*weight
    raise AssertionError("unreachable interpolation")


def coefficient_checks(state, signal, probe, config, table):
    hubs, scales = config["action"]["hubs"], config["action"]["scales"]
    t = state["period"]
    T = 24  # protocol is fixed to 24 and validated against physical signatures
    maximum = 0.
    for row in probe["coefficients"]:
        when, hub, q = row["time"], row["hub"], row["supply_slot"]
        index = hubs.index(hub)
        if when >= T:
            base, residual = 0., 0.
        else:
            group = table["values"][hub][str(q)]
            base = interpolate(group["times"], group["values"], when)
            fraction = (when-t)/(T-t)
            residual = scales[index]*((1-fraction)*signal[2*index]+fraction*signal[2*index+1])
        for key, expected in (("base", base), ("residual", residual), ("value", base+residual)):
            if not math.isclose(row[key], expected, abs_tol=1e-6, rel_tol=1e-10):
                raise RuntimeError("independent frozen-table/residual mismatch")
        maximum = max(maximum, abs(residual))
    return maximum


def collect(target):
    manifest = read(target/"manifest.json")
    if manifest["status"] != "NUMERICAL_COMPLETE_QA_PENDING":
        raise RuntimeError("analysis needs a completed numerical attempt")
    protocol = read(target/"protocol/preregistration.json")
    states, probes, signals, state_map = [], [], [], {}
    raw_count, unique_count = 0, 0
    for relative, digest in manifest["raw_hashes"].items():
        path = target/relative
        if sha(path) != digest:
            raise RuntimeError("raw file changed")
        raw = read(path)
        raw_count += 1
        item = read(target/"data"/(raw["condition"]+".json"))
        table = read(item["config"]["baseline"])
        for state in raw["states"]:
            sid = f"{state['condition']}/s{state['scenario_seed']}/t{state['period']}"
            if sid in state_map or identity(state["observation"]) != state["observation_hash"]:
                raise RuntimeError("duplicate state or observation identity changed")
            state_map[sid] = state
            reference = state["probes"][state["reference_key"]]
            changed_axis = sum(r["changed"] for r in state["logical"] if r["method"] == "axis")
            active = state["pending_order_count"] >= 1 and state["idle_vehicle_count"] >= 1
            states.append(dict(state_id=sid, condition=state["condition"], scenario_seed=state["scenario_seed"],
                period=state["period"], pending_order_count=state["pending_order_count"],
                idle_vehicle_count=state["idle_vehicle_count"], resource_active=active,
                axis_changed_count=changed_axis, axis_responsive=changed_axis>0,
                reference_candidate_route_count=reference["plan"]["solver"]["candidate_route_count"],
                MC_action_hash=reference["action_hash"], unique_probe_solves=len(state["probes"])))
            unique_count += len(state["probes"])
            # Every unique record must have a logical signal which independently reproduces its key.
            referenced = set()
            for row in state["logical"]:
                key = row["probe_key"]
                probe = state["probes"][key]
                if key != identity(dict(observation=state["observation_hash"], revision=state["observation"]["revision"], signal=row["signal"])):
                    raise RuntimeError("signal/probe cache identity mismatch")
                if physical_hash(probe["plan"]) != probe["action_hash"]:
                    raise RuntimeError("physical action identity mismatch")
                changed = probe["action_hash"] != reference["action_hash"]
                if changed != row["changed"] or probe["plan"]["solver"]["status"] != "OPTIMAL":
                    raise RuntimeError("changed flag/optimal status mismatch")
                maximum = coefficient_checks(state, row["signal"], probe, item["config"], table)
                if row["method"] == "PPO_scaled":
                    actor = next(a["signal"] for a in state["actors"] if a["training_seed"] == row["training_seed"])
                    raw_signal = [row["multiplier"]*a for a in actor]
                    if row["signal"] != [max(-1., min(1., a)) for a in raw_signal]:
                        raise RuntimeError("independent multiplier/clip mismatch")
                    if row["clipped_components"] != sum(abs(a)>1 for a in raw_signal):
                        raise RuntimeError("clipped component count mismatch")
                probes.append(dict(state_id=sid, condition=state["condition"], scenario_seed=state["scenario_seed"],
                    period=state["period"], resource_active=active, **{k: row[k] for k in
                        ("method", "training_seed", "multiplier", "axis", "sign", "cached")},
                    changed=changed, clipped_components=row.get("clipped_components", 0),
                    max_abs_residual_USD=maximum, action_hash=probe["action_hash"],
                    selected_assignment_count=len(probe["plan"]["assignments"]),
                    current_operating_profit_USD=probe["plan"]["operating_profit"],
                    candidate_route_count=probe["plan"]["solver"]["candidate_route_count"],
                    solver_runtime_seconds=probe["plan"]["solver"]["runtime"], probe_key=key))
                referenced.add(key)
            if referenced != set(state["probes"]):
                raise RuntimeError("orphan raw probe record")
            for actor in state["actors"]:
                for axis, value in enumerate(actor["signal"]):
                    signals.append(dict(state_id=sid, condition=state["condition"], scenario_seed=state["scenario_seed"],
                        period=state["period"], training_seed=actor["training_seed"], axis=axis,
                        hub=item["config"]["action"]["hubs"][axis//2], band="near" if axis%2==0 else "far",
                        signal=value, action_scale_USD=item["config"]["action"]["scales"][axis//2]))
    qa = read(target/"qa/numerical.json")
    if (raw_count != 4 or len(states) != 24 or len(probes) != 888 or len(signals) != 720 or
            unique_count+96 != qa["actual_solver_calls"]):
        raise RuntimeError("raw accounting/coverage differs from registration")
    effects, adaptation = [], []
    for condition in protocol["conditions"]:
        for seed in protocol["training_seeds"]:
            subset = [p for p in probes if p["condition"] == condition and p["training_seed"] == seed]
            native = {p["state_id"]:p["changed"] for p in subset if p["multiplier"] == 1}
            for c in protocol["multipliers"]:
                rows = [p for p in subset if p["multiplier"] == c]
                active = [p for p in rows if p["resource_active"]]
                effects.append(dict(condition=condition, training_seed=seed, multiplier=c,
                    states=len(rows), changed_states=sum(p["changed"] for p in rows),
                    changed_fraction=sum(p["changed"] for p in rows)/len(rows),
                    active_states=len(active), active_changed=sum(p["changed"] for p in active),
                    extra_vs_native=sum(p["changed"] and not native[p["state_id"]] for p in rows),
                    clipped_components=sum(p["clipped_components"] for p in rows),
                    max_abs_residual_USD=max(p["max_abs_residual_USD"] for p in rows)))
            by_period = defaultdict(list)
            by_scene = defaultdict(list)
            for sid, state in state_map.items():
                if state["condition"] != condition:
                    continue
                a = next(a["signal"] for a in state["actors"] if a["training_seed"] == seed)
                by_period[state["period"]].append(a)
                by_scene[state["scenario_seed"]].append(a)
            between = [max(abs(a-b) for a,b in zip(*group)) for group in by_period.values()]
            temporal = [max(max(values)-min(values) for values in zip(*group)) for group in by_scene.values()]
            adaptation.append(dict(condition=condition, training_seed=seed,
                max_between_scene_same_period_Linf=max(between),
                mean_between_scene_same_period_Linf=sum(between)/len(between),
                max_within_scene_temporal_Linf=max(temporal)))
    return dict(states=states, probes=probes, signals=signals, effects=effects,
                adaptation=adaptation, protocol=protocol, qa=qa, unique_probes=unique_count)


def csv_rows(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse(target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = collect(target)
    if not (target/"qa/csv-marker.json").exists():
        raise RuntimeError("scientific CSV create marker must execute once before authoring")
    if (target/"qa/analysis.json").exists():
        raise RuntimeError("analysis already generated; do not silently overwrite")
    for name in ("states", "probes", "signals"):
        csv_rows(target/"analysis"/(name+".csv"), result[name])
    write(target/"analysis/summary.json", {k:v for k,v in result.items() if k not in ("probes", "signals", "states")})
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.spines.top":False,
        "axes.spines.right":False, "svg.fonttype":"none", "savefig.bbox":"tight"})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), sharey=True)
    for ax, condition, title in zip(axes, result["protocol"]["conditions"], ("CA80/3", "W80/3")):
        for seed, color, marker in zip((30001,30002,30003), ("#0072B2", "#D55E00", "#009E73"), ("o","s","^")):
            e = [r for r in result["effects"] if r["condition"]==condition and r["training_seed"]==seed]
            ax.plot([r["multiplier"] for r in e], [100*r["changed_fraction"] for r in e],
                marker=marker, color=color, label=str(seed), linewidth=1.2)
        axis_rows = [s for s in result["states"] if s["condition"]==condition]
        ax.axhline(100*sum(s["axis_responsive"] for s in axis_rows)/len(axis_rows), color="#333333",
            linestyle="--", label="Any +/- axis response")
        ax.set_title(title)
        ax.set_xlabel("Actor signal multiplier (clipped to [-1, 1])")
        ax.set_ylim(-2, 102)
        ax.set_xticks((0,.5,1,2,4))
    axes[0].set_ylabel("States with a different integer action (%)")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(target/"figures"/("01-signal-to-action."+ext), dpi=600)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    for ax, condition, title in zip(axes, result["protocol"]["conditions"], ("CA80/3", "W80/3")):
        rows = [a for a in result["adaptation"] if a["condition"]==condition]
        xs = list(range(len(rows)))
        ax.bar([x-.17 for x in xs], [r["max_between_scene_same_period_Linf"] for r in rows],
            width=.34, color="#0072B2", label="Same-period scene difference")
        ax.bar([x+.17 for x in xs], [r["max_within_scene_temporal_Linf"] for r in rows],
            width=.34, color="#D55E00", hatch="//", label="Within-scene time range")
        ax.set_xticks(xs, [str(r["training_seed"]) for r in rows])
        ax.set_title(title)
        ax.set_xlabel("Training seed")
        ax.set_ylabel("Maximum absolute signal difference")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(target/"figures"/("02-state-adaptation."+ext), dpi=600)
    plt.close(fig)
    write(target/"figures/plot-config.json", dict(dpi=600, sizes_inches=[8,3.1],
        source="analysis/{states,probes,signals}.csv and independently recalculated summary.json",
        plots=["integer action response", "public-state signal variation"], profit_claim=False))
    lines = ["# PPO 信号—MILP 动作传递诊断报告", "", "## 研究问题与范围", "",
        "本轮检查既有短训练 PPO 信号的幅度和状态变化能否传递到实际整数动作。使用两个工况、三个冻结模型、新开发场景 23002/23003，共 24 个 MC 同状态快照。仅作机制探索，不训练、不选择利润赢家。",
        "", "## 主要结果", "", "| 工况 | training seed | 原信号改变状态 | 2倍 | 4倍 | 4倍新增改变状态 |", "|---|---:|---:|---:|---:|---:|"]
    for condition in result["protocol"]["conditions"]:
        for seed in result["protocol"]["training_seeds"]:
            e = {r["multiplier"]:r for r in result["effects"] if r["condition"]==condition and r["training_seed"]==seed}
            lines.append(f"| {'CA80/3' if condition.startswith('california') else 'W80/3'} | {seed} | {e[1]['changed_states']}/12 | {e[2]['changed_states']}/12 | {e[4]['changed_states']}/12 | {e[4]['extra_vs_native']} |")
    lines += ["", "分母是同状态快照数，快照嵌套于两个场景，不能当成 12 个独立场景或做总体显著性结论。", "", "## 边界控制与资源活动状态", ""]
    for condition in result["protocol"]["conditions"]:
        states = [s for s in result["states"] if s["condition"]==condition]
        active = [s for s in states if s["resource_active"]]
        cal = [r for r in result["probes"] if r["condition"]==condition and r["method"]=="CAL"]
        lines.append(f"- {condition}：各轴边界至少一种改变动作的状态 {sum(s['axis_responsive'] for s in states)}/12；预定义资源活动状态 {len(active)}/12，其中边界可响应 {sum(s['axis_responsive'] for s in active)}/{len(active)}；CAL 改变 {sum(r['changed'] for r in cal)}/12。")
    lines += ["", "资源活动仅指订单与空闲车辆同时存在，不代表已证明其他动作合法；边界未响应也不证明所有联合方向均不可控。", "", "## 状态适应性", "", "| 工况 | training seed | 同时刻两场景最大信号差 | 同场景跨时刻最大信号差 |", "|---|---:|---:|---:|"]
    for a in result["adaptation"]:
        lines.append(f"| {a['condition']} | {a['training_seed']} | {a['max_between_scene_same_period_Linf']:.6g} | {a['max_within_scene_temporal_Linf']:.6g} |")
    lines += ["", "这些是公开真实状态的局部输出变化。时间变化和不同场景的历史差异混合在观察中，不是对单一状态特征的因果敏感度；非零且变化的信号不等于有经济收益。", "", "## 下一步判断", ""]
    extra = sum(e["extra_vs_native"] for e in result["effects"] if e["multiplier"]==4)
    native = sum(e["changed_states"] for e in result["effects"] if e["multiplier"]==1)
    axes = sum(s["axis_responsive"] for s in result["states"])
    if extra:
        lines += [f"预先指定的 4 倍探针出现 {extra} 个新增模型—状态动作分歧，原信号共 {native} 个分歧。这支持“幅度会限制部分状态的控制作用”这一局部解释。",
            "建议下一批只比较原幅度和一个预先固定的幅度变体，在一个工况、相同小训练预算下从头训练，再用新的独立开发场景检验是否提高 realized profit。不能把本轮动作变多直接写成利润变好，也不能直接部署放大策略。"]
    elif axes:
        lines += [f"边界控制可改变 {axes}/24 个快照，但沿已学方向放大未产生额外动作分歧。优先检查已学方向和短训练的状态适应，再做有限训练预算扩展；单纯放大尺度缺少本轮支持。"]
    else:
        lines += ["本轮有限边界方向和已学方向均没有提供动作响应。先检查候选支持、资源活动和价格接入；不能把有限探针当作整个连续动作域的穷举。"]
    lines += ["", "本轮不自动运行后续训练或参数搜索；下一批须重新登记问题、数据角色、预算和停止规则。", "", "## 成本、核验与局限", "",
        f"实际数值耗时 {result['qa']['wall_seconds']/60:.2f} 分钟；真实 solver 调用 {result['qa']['actual_solver_calls']} 次，逻辑探针 888 条。缓存仅合并相同状态/信号；实际 unique probe {result['unique_probes']} 次，另有 96 个 MC 轨迹决策求解。",
        "全部独立探针检查 OPTIMAL、整数动作再编译、现金核算、价格系数重算和 live 状态不变；四条 MC 完整日做 full cooldown 及独立逐期重放。分析另用独立线性插值、clip、缓存身份和计数实现复核。",
        "冻结检查点及原始场景身份严格匹配。场景由 CFS 样本加模型生成时序构造，不是新真实日期观测。支持计数未知，不能用查表模式代替支持量。不同残差的 solver window objective 未用于跨策略利润比较。",
        "", "真实最终测试、图形视觉核验与所有来源/输出 hash 完成后才冻结 COMPLETED；证据见 qa。", "", "## 产物", "",
        "- `analysis/states.csv`：24 个公开状态及活动分类。",
        "- `analysis/probes.csv`：888 条逻辑探针、响应、缓存、系数幅度和当期现金。",
        "- `analysis/signals.csv`：720 个原始 actor 信号分量。",
        "- `raw/`：全部探针解、公开快照与完整 MC 轨迹。",
        "- `figures/`：两组 600 dpi PNG 与 SVG，配置和来源保存。",
        "", "当期 operating profit 列只用于核算和机制描述，没有未来回报，不能当作 full-cooldown rollout advantage。"]
    (target/"signal_action_diagnostic_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    outputs = {p.relative_to(target).as_posix():sha(p) for folder in ("analysis", "figures")
               for p in sorted((target/folder).glob("*")) if p.is_file()}
    outputs["signal_action_diagnostic_report.md"] = sha(target/"signal_action_diagnostic_report.md")
    write(target/"qa/analysis.json", dict(status="PASSED", raw_scenes=4, states=24, logical_probes=888,
        signals=720, unique_probe_solves=result["unique_probes"], coefficient_implementation="independent linear interpolation",
        all_clip_cache_action_counts_independently_recalculated=True, output_hashes=outputs))
    print(json.dumps(dict(effects=result["effects"], adaptation=result["adaptation"]), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    analyse(Path(args.attempt).resolve())
