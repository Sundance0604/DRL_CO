"""Preregistered crossed bootstrap, six figure families and transparent reporting."""
import csv
import json
from pathlib import Path
import numpy as np
from model_families.single.strategic.artifacts import read_json, write_json
from .study_common import CONDITIONS, key, load, sha, now


def bootstrap_crossed(difference, replications=20000, seed=31000):
    x = np.asarray(difference, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("bootstrap requires finite crossed seed/scenario matrix")
    rng = np.random.default_rng(seed)
    samples = []
    for offset in range(0, replications, 1000):
        count = min(1000, replications-offset)
        models = rng.integers(0, x.shape[0], (count, x.shape[0]))
        scenarios = rng.integers(0, x.shape[1], (count, x.shape[1]))
        samples.extend(x[models[:, :, None], scenarios[:, None, :]].mean(axis=(1, 2)))
    samples = np.asarray(samples)
    alpha = .05/6
    return dict(mean_USD=float(x.mean()), seed_mean_USD=x.mean(axis=1).tolist(),
                scenario_mean_USD=x.mean(axis=0).tolist(), paired_win_rate=float((x>0).mean()),
                marginal_95=np.quantile(samples, [.025, .975]).tolist(),
                bonferroni_99_1667=np.quantile(samples, [alpha/2, 1-alpha/2]).tolist(),
                one_sided_95_lower=float(np.quantile(samples, .05)),
                seed_positive_count=int((x.mean(axis=1)>0).sum()), replications=replications,
                bootstrap_seed=seed, scheme="independent resampling of model and scenario axes; crossed paired means",
                coverage="nominal percentile intervals; five models limit finite-sample tail precision")


def csv_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def compute(attempt):
    target, _ = load(attempt)
    output, effects = [], []
    for condition in CONDITIONS:
        name = key(condition)
        rows = read_json(target/"evaluation"/name/"summary.json")["rows"]
        by_method = {}
        for row in rows:
            by_method.setdefault(row["method"], {})[row["scenario_seed"]] = row["profit_USD"]
            output.append(dict(condition=name, **row))
        scenarios = sorted(by_method["MC"])
        if len(scenarios) != 80 or any(set(r) != set(scenarios) for r in by_method.values()):
            raise RuntimeError("incomplete paired formal evaluation")
        ppo = np.array([[by_method[f"PPO-{seed}"][s] for s in scenarios] for seed in range(30001, 30006)])
        for comparator in ("MC", "CAL"):
            difference = ppo-np.array([by_method[comparator][s] for s in scenarios])[None, :]
            result = bootstrap_crossed(difference)
            effects.append(dict(condition=name, comparator=comparator, **result))
    csv_rows(target/"analysis/formal-results.csv", output)
    write_json(target/"analysis/main-effects.json", effects)
    csv_rows(target/"analysis/main-effects.csv", effects)
    protection = next(e for e in effects if e["condition"] == "western_multicluster-n80-k3" and e["comparator"] == "MC")
    delta = read_json(target/"protocol/selected-models.json")["W80_noninferiority_delta_USD"]
    write_json(target/"analysis/protection.json", dict(delta_USD=delta,
        lower_USD=protection["one_sided_95_lower"], passed=protection["one_sided_95_lower"] > -delta))
    # Independently recompute the point estimand from the long CSV representation.
    checks = []
    for effect in effects:
        rows = [r for r in output if r["condition"] == effect["condition"]]
        baseline = {r["scenario_seed"]: r["profit_USD"] for r in rows if r["method"] == effect["comparator"]}
        differences = [r["profit_USD"]-baseline[r["scenario_seed"]] for r in rows if r["method"].startswith("PPO-")]
        recomputed = sum(differences)/len(differences)
        if not np.isclose(recomputed, effect["mean_USD"], atol=1e-8, rtol=1e-12):
            raise RuntimeError("independent statistical point-estimate recalculation mismatch")
        checks.append(dict(condition=effect["condition"], comparator=effect["comparator"], mean_USD=recomputed))
    write_json(target/"qa/statistical-recalculation.json", dict(status="PASSED", point_estimates=checks,
               interval_checks="crossed bootstrap unit test and deterministic independent rerun; not exact coverage"))
    return output, effects


def figures(attempt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target, _ = load(attempt)
    rows, effects = compute(attempt)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                        "axes.spines.right": False, "savefig.bbox": "tight"})
    def save(fig, stem, source):
        for extension in ("pdf", "svg", "png"):
            fig.savefig(target/"figures"/(stem+"."+extension), dpi=600)
        write_json(target/"figures"/(stem+".source.json"), source)
        write_json(target/"figures"/(stem+".config.json"), dict(dpi=600, sources=source,
            code_sha256=sha(Path(__file__)), selection="three conditions; all five training seeds; diagnostic seed 23000 and model 30001"))
        plt.close(fig)
    names = [key(c) for c in CONDITIONS]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    curves = []
    for ax, name in zip(axes, names):
        for seed in range(30001, 30006):
            updates = [read_json(p) for p in sorted((target/"training"/name/str(seed)).glob("update-*.json"))]
            ax.plot([u["episodes"] for u in updates], [u["training_mean_USD"] for u in updates], alpha=.4, lw=.8)
            selected = [u for u in updates if u["validation_mean_USD"] is not None]
            ax.plot([u["episodes"] for u in selected], [u["validation_mean_USD"] for u in selected], marker=".", label=str(seed))
            curves.append(dict(condition=name, seed=seed, updates=updates))
        reference = read_json(target/"validation"/name/"MC-summary.json")["mean_profit_USD"]
        ax.axhline(reference, color="black", linestyle="--", label="MC validation")
        ax.set(title=name, xlabel="Training episodes", ylabel="Realized profit (USD)")
    axes[-1].legend(fontsize=6)
    save(fig, "P01_training_validation", curves)
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = []
    for i, result in enumerate(effects):
        lo, hi = result["bonferroni_99_1667"]
        ax.plot([lo, hi], [i, i], color="tab:blue", lw=2)
        ax.scatter([result["mean_USD"]], [i], color="black", s=25)
        ax.scatter(result["seed_mean_USD"], [i+.12]*5, color="tab:orange", s=12)
        labels.append(result["condition"]+" / PPO - "+result["comparator"])
    ax.axvline(0, color="gray", lw=1)
    ax.set(yticks=range(6), yticklabels=labels, xlabel="Profit difference (USD); nominal family-adjusted intervals")
    save(fig, "P02_main_effects", effects)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, name in zip(axes, names):
        selected = [r for r in rows if r["condition"] == name]
        for method in ("MC", "CAL", "M0", "RH4"):
            values = sorted(r["profit_USD"] for r in selected if r["method"] == method)
            ax.plot(values, np.arange(1, 81)/80, label=method)
        for seed in range(30001, 30006):
            values = sorted(r["profit_USD"] for r in selected if r["method"] == f"PPO-{seed}")
            ax.plot(values, np.arange(1, 81)/80, alpha=.45, color="tab:purple")
        ax.set(title=name, xlabel="Realized USD", ylabel="Scenario empirical CDF")
    axes[-1].legend(fontsize=7)
    save(fig, "P03_formal_distributions", rows)
    diagnostic = [read_json(target/"diagnostics"/name/"scenario-23000.json") for name in names]
    fig, axes = plt.subplots(4, 3, figsize=(12, 10))
    for j, record in enumerate(diagnostic):
        periods = [r["period"] for r in record["same_state"]]
        values = np.array([r["PPO_signal"] for r in record["same_state"]])
        axes[0, j].imshow(values.T, aspect="auto", origin="lower", extent=[0, 24, 0, 10], vmin=-1, vmax=1, cmap="coolwarm")
        axes[0, j].set(title=names[j], ylabel="Signal dimension")
        hubs = sorted({c["hub"] for r in record["same_state"] for c in r["PPO_coefficients"]})
        for hub in hubs:
            usd = [np.mean([c["residual"] for c in r["PPO_coefficients"] if c["hub"] == hub])
                   if any(c["hub"] == hub for c in r["PPO_coefficients"]) else 0. for r in record["same_state"]]
            axes[1, j].plot(periods, usd, label=hub, lw=.8)
        axes[1, j].set(ylabel="Mean queried residual (USD)")
        for field, label in (("base", "MC base"), ("value", "PPO score")):
            axes[2, j].plot(periods, [np.mean([c[field] for c in r["PPO_coefficients"]])
                                    if r["PPO_coefficients"] else 0. for r in record["same_state"]], label=label)
        axes[2, j].set(ylabel="Mean queried score (USD)")
        axes[3, j].step(periods, [r["PPO_MC_disagreement"] for r in record["same_state"]], label="PPO vs MC")
        axes[3, j].step(periods, [r["PPO_CAL_disagreement"] for r in record["same_state"]], label="PPO vs CAL", alpha=.7)
        axes[3, j].set(xlabel="Period on MC source trajectory", ylabel="Physical action differs")
    axes[2, -1].legend(fontsize=7)
    axes[3, -1].legend(fontsize=7)
    save(fig, "P04_signals_actions", diagnostic)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for j, record in enumerate(diagnostic):
        traces = record["trajectory"]["transitions"]
        for label, method in (("Public pool", lambda r: len(r["observation"]["orders"])),
                              ("Idle vehicles", lambda r: sum(v["features"][5] for v in r["observation"]["vehicles"])),
                              ("Busy vehicles", lambda r: sum(1-v["features"][5] for v in r["observation"]["vehicles"])),
                              ("Committed releases within 4", lambda r: sum(v["features"][5] == 0 and v["features"][-1]*24 <= 4
                                                                           for v in r["observation"]["vehicles"]))):
            axes[0, j].plot([r["period"] for r in traces], [method(r) for r in traces], label=label)
        axes[0, j].set(title=names[j], ylabel="Count on MC source trajectory")
        axes[1, j].plot([r["period"] for r in traces], [r["cumulative_profit_USD"] for r in traces])
        axes[1, j].set(xlabel="Period", ylabel="Cumulative realized USD")
    axes[0, -1].legend(fontsize=7)
    save(fig, "P05_service_resource", diagnostic)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    costs = read_json(target/"analysis/computation.json")
    for ax, name in zip(axes, names):
        by_method = {}
        for row in costs["evaluation"]:
            if row["condition"] == name:
                by_method.setdefault("PPO" if row["method"].startswith("PPO") else row["method"], []).extend(row["solver_times"])
        ax.boxplot([by_method[m] for m in by_method], tick_labels=list(by_method), showfliers=False)
        ax.set(title=name, ylabel="Solver seconds per decision", yscale="log")
    save(fig, "P06_computation_stability", costs)


def report(attempt):
    target, manifest = load(attempt)
    effects = read_json(target/"analysis/main-effects.json")
    protection = read_json(target/"analysis/protection.json")
    selected = read_json(target/"protocol/selected-models.json")
    lines = ["# 第八轮 PPO 首次 CFS 数值实验报告", "", "本轮条件于第七轮 bank1 非 bucket MC 底座，采用三个工况、各五次独立训练，正式评估为每工况共同的 80 个新模拟场景。完整训练预算为每 seed 1,024 个 episode，训练场景有放回抽取；不将其当成独立现实交通日。", "",
             "## 主要结果", "", "| 工况 | 比较 | 平均差 USD | 边际 95% 区间 | 六项 Bonferroni 名义区间 | 正增益训练 seed |", "|---|---|---:|---|---|---:|"]
    for e in effects:
        lines.append(f"| {e['condition']} | PPO−{e['comparator']} | {e['mean_USD']:.2f} | {e['marginal_95'][0]:.2f}, {e['marginal_95'][1]:.2f} | {e['bonferroni_99_1667'][0]:.2f}, {e['bonferroni_99_1667'][1]:.2f} | {e['seed_positive_count']}/5 |")
    lines.extend(["", "上述区间由训练 seed 与 scenario 两维独立有放回抽样的交叉矩阵均值产生，20,000 次、RNG seed=31000。只有五次训练，尾部区间是名义 bootstrap 方案，不声称有限样本精确覆盖。", ""])
    for condition in CONDITIONS:
        name = key(condition)
        pair = [e for e in effects if e["condition"] == name]
        succeeded = all(e["bonferroni_99_1667"][0] > 0 for e in pair) and pair[0]["seed_positive_count"] >= 4
        lines.append(f"- {name}：{'达到预设较强初步成功标准' if succeeded else '未达到预设较强初步成功标准'}；各训练 seed 的完整差分见 analysis/main-effects.json，未筛选最好 seed。")
    lines.extend(["", f"W80 非劣保护：预设 δ={protection['delta_USD']:.2f} USD；PPO−MC 单侧 95% 下界={protection['lower_USD']:.2f} USD，{'通过' if protection['passed'] else '未通过'}保护门。该终点不替代六项优越性比较。", "",
                  "## 选模、控制与成本", "", "PPO 按固定验证集确定性平均美元利润选择最佳已训练 update，严格平局保留较早者；未训练零均值网络没有参与候选。CAL 仅从九个固定共用 near/far 偏移中选择，两者选模计算预算不同。",
                  "", "同状态诊断以 MC 闭环轨迹的公开快照为来源，在全部 16 个诊断场景上比较 PPO（预定 seed=30001）、MC 和 CAL；图中固定展示编号最小的 23000，未按收益挑选。全部诊断汇总见 diagnostics/summary.json。训练与正式评价耗时、求解时间、输出字节及 KL/饱和诊断见 analysis/computation.json。",
                  "", "## 审计与证据", "", "完整决策与冷却按冻结输入重放；逐步公开观察、整数动作、状态 hash、实际系数与权威现金均核对。验证逐场景数据及选模复算、数据隔离、工程零残差回归、最终测试与历史来源核验分别保存在 qa。图表输出 PDF/SVG/600 dpi PNG，并保存源数据和绘图配置。", "",
                  "## 解释边界", "", "结论只针对这一银行、当前可分释放价格接口、公开观测集合编码和固定训练预算。利润 loss/critic 误差、信号熵和物理动作多样性是不同对象；不能由 loss 下降宣布控制改善。MC 支持计数未知，网格、插值和端点夹持分别登记；资源注入 q 与在线 prefix q 的语义限制并未因此消失。没有跨 construction bank 或跨现实交通日外推证据。", "",
                  f"源码身份：`{manifest['source_identity']}`。选模冻结时间：{selected['selected_at']}。"])
    (target/"ppo_first_cfs_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
