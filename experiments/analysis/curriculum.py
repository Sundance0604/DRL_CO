"""Compare pressure-curriculum and normal-trained policies on held-out scenarios."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scenario_level(frame: pd.DataFrame) -> pd.DataFrame:
    """Average model seeds before treating a scenario as an independent sample."""
    learned = frame[frame.model_seed.notna()].groupby(
        ["pressure", "scenario_seed", "policy"], as_index=False
    ).agg(
        objective=("objective", "mean"),
        match_rate=("match_rate", "mean"),
        cancelled=("cancelled", "mean"),
        elapsed_seconds=("elapsed_seconds", "mean"),
    )
    baselines = frame[frame.model_seed.isna()][
        [
            "pressure",
            "scenario_seed",
            "policy",
            "objective",
            "match_rate",
            "cancelled",
            "elapsed_seconds",
        ]
    ]
    return pd.concat([baselines, learned], ignore_index=True)


def select_policies(frame: pd.DataFrame, training: str) -> pd.DataFrame:
    selected = []
    hybrid = {
        ("normal", "high"): "hybrid_q_0.3",
        ("normal", "extreme"): "hybrid_q_3.0",
        ("curriculum", "high"): "hybrid_q_0.1",
        ("curriculum", "extreme"): "hybrid_q_1.0",
    }
    for pressure in ("high", "extreme"):
        pressure_frame = frame[frame.pressure == pressure].copy()
        pressure_frame = pressure_frame[
            pressure_frame.policy.isin(["sac", hybrid[(training, pressure)]])
        ]
        pressure_frame["policy"] = pressure_frame.policy.map(
            {
                "sac": f"{training}_sac",
                hybrid[(training, pressure)]: f"{training}_hybrid",
            }
        )
        selected.append(pressure_frame)
    return pd.concat(selected, ignore_index=True)


def paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("curriculum_sac", "normal_sac"),
        ("curriculum_sac", "myopic_milp"),
        ("curriculum_sac", "supply"),
        ("curriculum_hybrid", "myopic_milp"),
        ("curriculum_hybrid", "supply"),
    ]
    rows = []
    for pressure in ("high", "extreme"):
        wide = frame[frame.pressure == pressure].pivot(
            index="scenario_seed", columns="policy", values="objective"
        )
        for left, right in pairs:
            diff = (wide[left] - wide[right]).dropna()
            rows.append(
                {
                    "pressure": pressure,
                    "left": left,
                    "right": right,
                    "scenarios": len(diff),
                    "left_wins": int((diff > 0).sum()),
                    "ties": int((diff == 0).sum()),
                    "mean_difference": diff.mean(),
                    "median_difference": diff.median(),
                    "difference_std": diff.std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    root = Path("runs/stress")
    curriculum = scenario_level(pd.read_csv(root / "curriculum-hybrid-heldout.csv"))
    normal = scenario_level(pd.read_csv(root / "normaltrained-comparison-heldout.csv"))

    learned = pd.concat(
        [
            select_policies(normal, "normal"),
            select_policies(curriculum, "curriculum"),
        ],
        ignore_index=True,
    )
    baselines = curriculum[curriculum.policy.isin(["myopic_milp", "supply"])]
    combined = pd.concat([baselines, learned], ignore_index=True)

    summary = combined.groupby(["pressure", "policy"]).agg(
        scenarios=("objective", "size"),
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        match_rate_mean=("match_rate", "mean"),
        cancelled_mean=("cancelled", "mean"),
        elapsed_seconds_mean=("elapsed_seconds", "mean"),
    ).reset_index()
    comparisons = paired_comparisons(combined)
    summary.to_csv(root / "curriculum-heldout-summary.csv", index=False)
    comparisons.to_csv(root / "curriculum-paired-comparison.csv", index=False)

    policies = [
        "myopic_milp",
        "normal_sac",
        "curriculum_sac",
        "curriculum_hybrid",
        "supply",
    ]
    labels = {
        "myopic_milp": "Myopic MILP",
        "normal_sac": "SAC\nnormal",
        "curriculum_sac": "SAC\ncurriculum",
        "curriculum_hybrid": "Q-MILP\ncurriculum",
        "supply": "Supply",
    }
    colors = {
        "myopic_milp": "#66a61e",
        "normal_sac": "#80b1d3",
        "curriculum_sac": "#1b9e77",
        "curriculum_hybrid": "#4c78a8",
        "supply": "#e7298a",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
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
        axis.set(
            title=f"{pressure.capitalize()} pressure — seeds 401–407",
            ylabel="Objective",
        )
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(root / "curriculum-heldout.png", dpi=180)

    print("Summary")
    print(summary.to_string(index=False))
    print("\nPaired scenario comparisons")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
