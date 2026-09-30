"""Compare supply-counterfactual and ordinary curriculum SAC on held-out scenarios."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scenario_level(frame: pd.DataFrame, sac_name: str) -> pd.DataFrame:
    learned = frame[frame.model_seed.notna()].groupby(
        ["pressure", "scenario_seed", "policy"], as_index=False
    ).agg(
        objective=("objective", "mean"),
        match_rate=("match_rate", "mean"),
        cancelled=("cancelled", "mean"),
        elapsed_seconds=("elapsed_seconds", "mean"),
    )
    learned = learned[learned.policy == "sac"].copy()
    learned["policy"] = sac_name
    return learned


def main() -> None:
    root = Path("runs/stress")
    counterfactual_raw = pd.read_csv(root / "counterfactual-heldout.csv")
    curriculum_raw = pd.read_csv(root / "curriculum-counterfactual-control.csv")
    counterfactual = scenario_level(counterfactual_raw, "counterfactual_sac")
    curriculum = scenario_level(curriculum_raw, "curriculum_sac")
    baselines = counterfactual_raw[counterfactual_raw.model_seed.isna()][
        [
            "pressure", "scenario_seed", "policy", "objective",
            "match_rate", "cancelled", "elapsed_seconds",
        ]
    ]
    combined = pd.concat([baselines, curriculum, counterfactual], ignore_index=True)
    summary = combined.groupby(["pressure", "policy"], as_index=False).agg(
        scenarios=("objective", "size"),
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        match_rate_mean=("match_rate", "mean"),
        cancelled_mean=("cancelled", "mean"),
        elapsed_seconds_mean=("elapsed_seconds", "mean"),
    )

    comparisons = []
    for pressure in ("high", "extreme"):
        wide = combined[combined.pressure == pressure].pivot(
            index="scenario_seed", columns="policy", values="objective"
        )
        for reference in ("curriculum_sac", "myopic_milp", "supply"):
            difference = (wide.counterfactual_sac - wide[reference]).dropna()
            comparisons.append({
                "pressure": pressure,
                "reference": reference,
                "scenarios": len(difference),
                "wins": int((difference > 0).sum()),
                "ties": int((difference == 0).sum()),
                "mean_difference": difference.mean(),
                "median_difference": difference.median(),
                "difference_std": difference.std(ddof=1),
            })
    comparison_frame = pd.DataFrame(comparisons)
    summary.to_csv(root / "counterfactual-heldout-summary.csv", index=False)
    comparison_frame.to_csv(root / "counterfactual-paired-comparison.csv", index=False)

    policies = ["myopic_milp", "curriculum_sac", "counterfactual_sac", "supply"]
    labels = {
        "myopic_milp": "Myopic MILP",
        "curriculum_sac": "SAC\noriginal reward",
        "counterfactual_sac": "SAC\ncounterfactual",
        "supply": "Supply",
    }
    colors = {
        "myopic_milp": "#66a61e",
        "curriculum_sac": "#80b1d3",
        "counterfactual_sac": "#1b9e77",
        "supply": "#e7298a",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for axis, pressure in zip(axes, ("high", "extreme")):
        selected = summary[summary.pressure == pressure].set_index("policy").loc[policies]
        positions = np.arange(len(policies))
        bars = axis.bar(
            positions,
            selected.objective_mean,
            yerr=selected.objective_std,
            capsize=4,
            color=[colors[policy] for policy in policies],
        )
        axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
        axis.set_xticks(positions, [labels[policy] for policy in policies])
        axis.set(title=f"{pressure.capitalize()} pressure — seeds 801–807", ylabel="Objective")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(root / "counterfactual-heldout.png", dpi=180)

    print(summary.to_string(index=False))
    print("\nPaired comparisons")
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
