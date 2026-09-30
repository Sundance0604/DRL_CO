"""Analyze independently held-out learned-Q MILP experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scenario_level(frame):
    learned = frame[frame.model_seed.notna()].groupby(
        ["pressure", "scenario_seed", "policy"], as_index=False
    ).agg(
        objective=("objective", "mean"),
        match_rate=("match_rate", "mean"),
        cancelled=("cancelled", "mean"),
        elapsed_seconds=("elapsed_seconds", "mean"),
    )
    baselines = frame[frame.model_seed.isna()][
        ["pressure", "scenario_seed", "policy", "objective", "match_rate",
         "cancelled", "elapsed_seconds"]
    ]
    return pd.concat([baselines, learned], ignore_index=True)


def main():
    root = Path("runs/stress")
    heldout = pd.read_csv(root / "hybrid-heldout.csv")
    shuffled = pd.read_csv(root / "hybrid-shuffled-control.csv")
    paired = scenario_level(heldout)
    shuffled_level = scenario_level(shuffled)
    paired = pd.concat([
        paired,
        shuffled_level[shuffled_level.policy == "hybrid_shuffled_3.0"],
    ], ignore_index=True)
    summary = paired.groupby(["pressure", "policy"]).agg(
        scenarios=("objective", "size"),
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        match_rate_mean=("match_rate", "mean"),
        cancelled_mean=("cancelled", "mean"),
        elapsed_seconds_mean=("elapsed_seconds", "mean"),
    ).reset_index()
    summary.to_csv(root / "hybrid-heldout-summary.csv", index=False)

    configurations = {
        "high": ["myopic_milp", "sac", "hybrid_q_0.3", "supply"],
        "extreme": ["myopic_milp", "sac", "hybrid_shuffled_3.0", "hybrid_q_3.0", "supply"],
    }
    labels = {
        "myopic_milp": "Myopic MILP", "sac": "SAC",
        "hybrid_q_0.3": "Q-MILP β=.3", "hybrid_q_3.0": "Q-MILP β=3",
        "hybrid_shuffled_3.0": "Shuffled-Q", "supply": "Supply",
    }
    colors = {
        "myopic_milp": "#66a61e", "sac": "#1b9e77", "hybrid_q_0.3": "#4c78a8",
        "hybrid_q_3.0": "#4c78a8", "hybrid_shuffled_3.0": "#9e9e9e", "supply": "#e7298a",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, (pressure, policies) in zip(axes, configurations.items()):
        selected = summary[summary.pressure == pressure].set_index("policy").loc[policies]
        positions = np.arange(len(policies))
        axis.bar(
            positions, selected.objective_mean,
            yerr=selected.objective_std, capsize=4,
            color=[colors[policy] for policy in policies],
        )
        axis.set_xticks(positions, [labels[policy] for policy in policies], rotation=20, ha="right")
        axis.set(title=f"{pressure.capitalize()} pressure — unseen scenarios", ylabel="Objective")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(root / "hybrid-heldout.png", dpi=180)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
