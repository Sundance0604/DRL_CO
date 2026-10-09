"""Independent descriptive recomputation; data with bundled Python, plots with Matplotlib."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from experiments.cfs_ppo_signal_action_diagnostic.analyse import (
    read, write, sha, identity, physical_hash, coefficient_checks)


def canonical_hash(plan):
    data = {"period":plan["period"]}
    for name in ("assignments", "routes", "repositions", "expired"):
        encoded = sorted(json.dumps(x, sort_keys=True) for x in plan[name])
        data[name] = [json.loads(x) for x in encoded]
    return identity(data)


def summarize(states, old_times):
    if len({(s["scenario_seed"], s["period"]) for s in states}) != len(states):
        raise RuntimeError("duplicate scenario-period observation")
    result = []
    for seed in sorted({s["scenario_seed"] for s in states})+[None]:
        source = [s for s in states if seed is None or s["scenario_seed"] == seed]
        for name, subset in (("all", source),
                ("fixed_six", [s for s in source if s["period"] in old_times]),
                ("omitted_eighteen", [s for s in source if s["period"] not in old_times])):
            active = [s for s in subset if s["resource_active"]]
            result.append(dict(scenario_seed=seed, subset=name, states=len(subset),
                responsive=sum(s["responsive"] for s in subset),
                coefficient_responsive=sum(s["coefficient_responsive"] for s in subset),
                active_states=len(active), active_responsive=sum(s["responsive"] for s in active),
                responsive_periods=[s["period"] for s in subset if s["responsive"]]))
    return result


def collect(target):
    manifest = read(target/"manifest.json")
    if manifest["status"] != "NUMERICAL_COMPLETE_QA_PENDING":
        raise RuntimeError("completed numerical attempt required")
    protocol = read(target/"protocol/preregistration.json")
    item = read(target/"data"/(protocol["condition"]+".json"))
    table = read(item["config"]["baseline"])
    states, probes, vehicles = [], [], []
    for rel, digest in manifest["raw_hashes"].items():
        if sha(target/rel) != digest:
            raise RuntimeError("raw hash changed")
        raw = read(target/rel)
        if [s["period"] for s in raw["states"]] != protocol["periods"]:
            raise RuntimeError("raw coverage differs from preregistration")
        committed = {t["period"]:t for t in raw["trajectory"]["transitions"]}
        for s in raw["states"]:
            if (s["horizon"] != 24 or identity(s["observation"]) != s["observation_hash"] or
                identity(s["before_state"]) != committed[s["period"]]["before_state_hash"]):
                raise RuntimeError("public snapshot/MC identity changed")
            sid = f"s{s['scenario_seed']}/t{s['period']}"
            ref = s["probes"][s["reference_key"]]
            refhash = canonical_hash(ref["plan"])
            if refhash != s["reference_canonical_hash"] or committed[s["period"]]["action_hash"] != ref["action_hash"]:
                raise RuntimeError("MC probe/commit mismatch")
            if len(s["logical"]) != 21 or len(s["probes"]) != 21:
                raise RuntimeError("missing probe")
            controls = {(r["axis"], r["sign"]) for r in s["logical"] if r["method"] == "axis"}
            if controls != {(i,v) for i in range(10) for v in (-1.,1.)}:
                raise RuntimeError("incomplete axis/sign coverage")
            active = s["pending_order_count"] >= 1 and s["idle_vehicle_count"] >= 1
            changed = coefficient_changed = order_only = 0
            referenced = set()
            for r in s["logical"]:
                p = s["probes"][r["probe_key"]]
                expected = [r["sign"] if i==r["axis"] else 0. for i in range(10)] if r["method"]=="axis" else [0.]*10
                if expected != r["signal"]:
                    raise RuntimeError("signal axis changed")
                if r["probe_key"] != identity(dict(observation=s["observation_hash"], revision=s["observation"]["revision"], signal=r["signal"])):
                    raise RuntimeError("cache identity changed")
                if physical_hash(p["plan"]) != p["action_hash"] or p["plan"]["solver"]["status"] != "OPTIMAL":
                    raise RuntimeError("raw physical hash/solve mismatch")
                responsive = canonical_hash(p["plan"]) != refhash
                raw_changed = p["action_hash"] != ref["action_hash"]
                if responsive != r["changed"] or raw_changed != r["raw_hash_changed"]:
                    raise RuntimeError("action response flag mismatch")
                maxres = coefficient_checks(s, r["signal"], p, item["config"], table)
                terminal_rows = sum(q["time"]>=24 for q in p["coefficients"])
                if terminal_rows:
                    raise RuntimeError("terminal gate leaked provider queries")
                coeffchange = any(abs(q["residual"])>1e-9 for q in p["coefficients"])
                changed += responsive
                coefficient_changed += coeffchange
                order_only += raw_changed and not responsive
                probes.append(dict(state_id=sid, scenario_seed=s["scenario_seed"], period=s["period"],
                    method=r["method"], axis=r["axis"], sign=r["sign"], changed=responsive,
                    raw_hash_changed=raw_changed, ordering_only_changed=raw_changed and not responsive,
                    coefficient_changed=coeffchange, coefficient_count=len(p["coefficients"]),
                    max_abs_residual_USD=maxres, candidate_route_count=p["plan"]["solver"]["candidate_route_count"],
                    current_operating_profit_USD=p["plan"]["operating_profit"],
                    assignments=len(p["plan"]["assignments"]), action_hash=p["action_hash"],
                    canonical_action_hash=canonical_hash(p["plan"]), probe_key=r["probe_key"]))
                referenced.add(r["probe_key"])
            if referenced != set(s["probes"]):
                raise RuntimeError("orphan probe")
            if len(s["vehicles"]) != 3:
                raise RuntimeError("vehicle coverage differs")
            for v in s["vehicles"]:
                if v["terminal_release"] != (v["committed_free_time"] >= 24):
                    raise RuntimeError("committed terminal classification differs")
                vehicles.append(dict(state_id=sid, scenario_seed=s["scenario_seed"], period=s["period"], **v))
            states.append(dict(state_id=sid, scenario_seed=s["scenario_seed"], period=s["period"],
                fixed_six=s["period"] in protocol["previous_periods"], pending_order_count=s["pending_order_count"],
                idle_vehicle_count=s["idle_vehicle_count"], resource_active=active, responsive=changed>0,
                changed_axis_count=changed, ordering_only_count=order_only,
                coefficient_responsive=coefficient_changed>0, changed_coefficient_axis_count=coefficient_changed,
                nonterminal_coefficient_count=len(ref["coefficients"]),
                last_decision_terminal_gate=s["period"]==23,
                committed_terminal_vehicle_count=sum(v["terminal_release"] for v in s["vehicles"]),
                candidate_route_count=ref["plan"]["solver"]["candidate_route_count"],
                MC_assignments=len(ref["plan"]["assignments"])))
    qa = read(target/"qa/numerical.json")
    if (len(states), len(probes), len(vehicles), qa["actual_solver_calls"]) != (48,1008,144,1056):
        raise RuntimeError("coverage/cost mismatch")
    if {s["scenario_seed"] for s in states} != set(protocol["scenario_seeds"]):
        raise RuntimeError("scenario identity differs")
    return states, probes, vehicles, summarize(states, protocol["previous_periods"]), qa


def data(target):
    if (target/"qa/analysis.json").exists():
        raise RuntimeError("do not rerun analysis")
    if not (target/"qa/csv-marker.json").exists():
        raise RuntimeError("first CSV authoring requires successful marker")
    states, probes, vehicles, summary, qa = collect(target)
    for name, rows in (("states", states), ("probes",probes), ("vehicles",vehicles)):
        with (target/"analysis"/(name+".csv")).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(target/"analysis/summary.json", dict(summary=summary, numerical=qa,
        responsive_states=[s for s in states if s["responsive"]],
        ordering_only_count=sum(s["ordering_only_count"] for s in states)))
    totals = {r["subset"]:r for r in summary if r["scenario_seed"] is None}
    allrow, fixed, omitted = (totals[k] for k in ("all", "fixed_six", "omitted_eighteen"))
    lines = ["# W80/3 全时刻控制覆盖诊断", "", "## 问题与实验边界", "",
        "上一批发现放大既有 PPO 信号没有新增动作分歧，而轴向边界控制只有少量响应。本批只检查固定六时刻是否遗漏可响应状态，不训练、不改变物理/价值表/尺度/奖励，不选择利润赢家。",
        "", "使用两个新的开发模拟场景 23004/23005，在 MC 自然轨迹 t=0…23 全覆盖。每状态运行零信号与相同十轴 ±1 控制，共 48 状态、1,008 探针。", "",
        "主动作身份忽略顶层列表排列但保留路线内部次序，避免仅列举顺序变化冒充决策变化。原 decoder hash、完整计划和现金同时保存。", "",
        "## 主要结果", "", "| 场景 | 时刻子组 | 状态数 | 有轴向动作响应 | 有系数变化 | 资源活动状态 | 活动且响应 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    labels = dict(all="全部时刻", fixed_six="原六时刻", omitted_eighteen="其余十八时刻")
    for r in summary:
        lines.append(f"| {r['scenario_seed'] if r['scenario_seed'] is not None else '合计'} | {labels[r['subset']]} | {r['states']} | {r['responsive']} | {r['coefficient_responsive']} | {r['active_states']} | {r['active_responsive']} |")
    lines += ["", f"全时刻响应 {allrow['responsive']}/48；原六时刻响应 {fixed['responsive']}/12；其余时刻响应 {omitted['responsive']}/36。比较发生于同一批完整轨迹，不用新旧场景差异推断纯取样效应。", "",
        "逐时刻响应（t 为零起始索引）："]
    for r in summary:
        if r["scenario_seed"] is not None and r["subset"] == "all":
            lines.append(f"- seed {r['scenario_seed']}：{r['responsive_periods']}。")
    lines += ["", "## 接入、承诺与日终", "",
        f"至少一个控制改变非终端价值系数的状态 {allrow['coefficient_responsive']}/48；列表排列独有差异 {sum(s['ordering_only_count'] for s in states)} 条。",
        f"最后决策 t=23 的两个状态，非终端系数计数为 {[s['nonterminal_coefficient_count'] for s in states if s['period']==23]}。所有存储 provider 查询均早于 T=24，日终门核验通过。",
        "车辆既定自由释放的 hub/time 按公开承诺路线保存于 vehicles.csv；这不是每个备选动作执行后的释放时间。候选 route 数为宏路线数，并不证明该状态存在多个合法联合动作。没有 provider 查询可说明该求解中无非终端价值系数，不等于没有路线或物理决策。",
        "", "## 判断及下一批", ""]
    if omitted["responsive"]:
        lines += [f"本批存在 {omitted['responsive']} 个位于原取样方案之外的响应状态，提供了固定时刻遗漏的局部证据。但全时刻响应仍为 {allrow['responsive']}/48，不能把遗漏现象直接解释为 PPO 不提升的全部原因。",
            "建议下一批回答一个问题：在已证实能够响应的公开状态上，既有 actor 的方向是否能跨过决策边界？只读加载三个冻结 actor，在新开发场景中保存全时刻输出及相同控制对照，区分固定偏移与状态适应，不直接启动长训练或扩大搜索。响应子组仅作机制分析，保留所有自然状态和分母，不按利润筛选。"]
    else:
        lines += ["本批未在原取样方案之外检出轴向响应，固定六时刻遗漏不是这两个新场景的已建立解释。",
            "建议下一批检查候选动作、承诺与终端限制：在公开同状态上列出完整合法的替代决策，并比较它们的非终端释放特征是否不同。不要因轴向探针少变就直接增加 PPO 训练预算，也不要认为连续动作域已被穷举。"]
    lines += ["", "本批只提出下一假设，不自动运行后续批次。", "", "## 投入与 QA", "",
        f"数值求解 {qa['actual_solver_calls']} 次，运行 {qa['wall_seconds']:.2f} 秒，峰值进程工作集 {qa['peak_working_set_bytes']/2**20:.1f} MiB。预算预登记为 1,200 次、30 分钟、2 GiB。",
        "所有探针 OPTIMAL，独立整数编译、现金和系数核验，probe 不改变 live 状态；两条完整 MC 轨迹包含 full cooldown，48 个提交边界独立回放。原始数据 SHA、冻结父实验/前批 SHA 与数值/分析源码身份均核验。最终测试 XML 与真实图形审阅通过后才冻结，证据保存于 qa/。",
        "", "## 可以与不可以推出的结论", "",
        "两个独立场景、48 个嵌套状态，只有描述性机制证据，不作总体显著性或利润优势检验。新 seed 是 CFS 衍生模拟场景，不是新增真实日期观测。零信号完整利润仅为核算核验，不是探针策略的未来回报。轴向 ±1 不能穷举联合作用；资源活动条件不是替代动作可行性证明。既有表缺少 support counts，保持 unknown，不填零。",
        "", "## 产物", "", "- analysis/states.csv：逐 t 状态、系数接入、动作响应与日终分类。",
        "- analysis/probes.csv：每个轴向/零信号实际整数动作、当期现金及系数幅度。",
        "- analysis/vehicles.csv：每个 t、每辆车的公开承诺释放状态。",
        "- raw/：两条完整 MC 轨迹及所有同状态探针。",
        "- figures/coverage.png 与 .svg：逐时刻动作与系数响应图（600 dpi PNG）。"]
    (target/"decision_coverage_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    outputs = {p.relative_to(target).as_posix():sha(p) for p in sorted((target/"analysis").glob("*"))}
    outputs["decision_coverage_report.md"] = sha(target/"decision_coverage_report.md")
    write(target/"qa/analysis.json", dict(status="DATA_PASSED_FIGURES_PENDING", states=48,
        probes=1008, vehicles=144, independent_identity_coefficient_counts=True, output_hashes=outputs))
    print(json.dumps(summary, ensure_ascii=False))


def figures(target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    qa = read(target/"qa/analysis.json")
    if qa["status"] != "DATA_PASSED_FIGURES_PENDING":
        raise RuntimeError("do not overwrite completed figures")
    for rel,digest in qa["output_hashes"].items():
        if sha(target/rel) != digest:
            raise RuntimeError("data/report changed before figures")
    with (target/"analysis/states.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(2,1, figsize=(8,4.6), sharex=True)
    old = read(target/"protocol/preregistration.json")["previous_periods"]
    plt.rcParams.update({"font.family":"DejaVu Sans", "svg.fonttype":"none"})
    for ax, seed in zip(axes, (23004,23005)):
        group = [r for r in rows if int(r["scenario_seed"]) == seed]
        xs = [int(r["period"]) for r in group]
        ax.bar(xs, [int(r["changed_coefficient_axis_count"]) for r in group],
            color="#DDDDDD", edgecolor="#888888", label="Axes changing coefficients (of 20)")
        ax.bar(xs, [int(r["changed_axis_count"]) for r in group], width=.5,
            color="#0072B2", hatch="//", label="Axes changing integer action (of 20)")
        ax.scatter(old, [-1.2]*len(old), marker="v", color="#D55E00", label="Previous six sampled periods")
        ax.set_title(f"W80/3, scenario {seed}", fontsize=10)
        ax.set_ylim(-2,22)
        ax.set_ylabel("Probe count")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Decision period t (zero-based)")
    axes[1].set_xticks(range(24))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5,0), ncol=1, fontsize=8)
    fig.tight_layout(rect=(0,.15,1,1))
    for ext in ("png", "svg"):
        fig.savefig(target/"figures"/("coverage."+ext), dpi=600, bbox_inches="tight")
    plt.close(fig)
    write(target/"figures/plot-config.json", dict(source="analysis/states.csv", size_inches=[8,4.6], dpi=600,
        primary_metric="axes changing canonical integer action", denominator=20, fixed_times=old,
        colors={"coefficients":"gray", "integer_action":"blue+hatched", "old_times":"orange+triangle"}))
    qa["output_hashes"].update({p.relative_to(target).as_posix():sha(p) for p in sorted((target/"figures").glob("*"))})
    qa["status"] = "PASSED"
    qa["data_analysis_environment"] = "bundled Python; standard library"
    qa["figure_environment"] = "repository existing Matplotlib; bundled Python lacks Matplotlib as recorded in prior frozen QA"
    write(target/"qa/analysis.json", qa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("data", "figures"))
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    globals()[args.stage](Path(args.attempt).resolve())
