"""Combine stress runs into paired, scenario-level summaries and a figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path("runs/stress")
    frame = pd.concat([
        pd.read_csv(root / "stress-rollouts.csv"),
        pd.read_csv(root / "stress-rollouts-extra.csv"),
    ], ignore_index=True)
    # Average model seeds first, so every scenario has equal statistical weight.
    sac = frame[frame.policy == "sac"].groupby(
        ["pressure", "scenario_seed"], as_index=False
    ).agg(
        objective=("objective", "mean"),
        objective_vs_myopic=("objective_vs_myopic", "mean"),
        match_rate=("match_rate", "mean"),
        cancelled=("cancelled", "mean"),
        elapsed_seconds=("elapsed_seconds", "mean"),
    )
    sac["policy"] = "sac"
    baselines = frame[frame.policy != "sac"]
    paired = pd.concat([baselines, sac], ignore_index=True, sort=False)
    order = ["moderate", "high", "extreme"]
    paired["pressure"] = pd.Categorical(paired.pressure, order, ordered=True)

    summary = paired.groupby(["pressure", "policy"], observed=True).agg(
        scenarios=("objective", "size"),
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        objective_vs_myopic_mean=("objective_vs_myopic", "mean"),
        match_rate_mean=("match_rate", "mean"),
        cancelled_mean=("cancelled", "mean"),
        elapsed_seconds_mean=("elapsed_seconds", "mean"),
    ).reset_index()
    summary.to_csv(root / "stress-summary-7scenarios.csv", index=False)

    comparisons = []
    for pressure in order:
        pivot = paired[paired.pressure == pressure].pivot(
            index="scenario_seed", columns="policy", values="objective"
        )
        if "sac" not in pivot:
            continue
        for baseline in ("supply", "myopic_milp"):
            difference = pivot.sac - pivot[baseline]
            comparisons.append({
                "pressure": pressure,
                "baseline": baseline,
                "scenarios": len(difference),
                "sac_wins": int((difference > 0).sum()),
                "mean_difference": float(difference.mean()),
                "median_difference": float(difference.median()),
            })
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(root / "stress-paired-comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"noop": "#9e9e9e", "supply": "#e7298a", "myopic_milp": "#66a61e", "sac": "#1b9e77"}
    labels = {"noop": "No-op", "supply": "Supply", "myopic_milp": "Myopic MILP", "sac": "Candidate SAC"}
    for policy in ("noop", "supply", "myopic_milp", "sac"):
        selected = summary[summary.policy == policy].set_index("pressure").reindex(order)
        axes[0].plot(order, selected.objective_vs_myopic_mean, marker="o",
                     label=labels[policy], color=colors[policy])
        axes[1].plot(order, selected.match_rate_mean, marker="o",
                     label=labels[policy], color=colors[policy])
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Objective relative to myopic MILP", ylabel="Ratio")
    axes[1].set(title="Match rate under fleet scarcity", ylabel="Match rate")
    for axis in axes:
        axis.set_xlabel("Pressure")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(root / "stress-summary.png", dpi=180)
    print(summary.to_string(index=False))
    print("\nPaired comparison:\n" + comparison.to_string(index=False))


if __name__ == "__main__":
    main()
