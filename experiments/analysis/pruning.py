"""Aggregate actor-pruning validation at the independent scenario level."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def scenario_level(frame: pd.DataFrame) -> pd.DataFrame:
    learned = frame[frame.model_seed.notna()].groupby(
        ["pressure", "scenario_seed", "policy"], as_index=False
    ).agg(
        objective=("objective", "mean"),
        match_rate=("match_rate", "mean"),
        candidate_count_mean=("candidate_count_mean", "mean"),
        candidate_fraction=("candidate_fraction", "mean"),
        uncertain_fraction=("uncertain_fraction", "mean"),
        oracle_recall=("oracle_recall", "mean"),
        feasible_pair_reduction=("feasible_pair_reduction", "mean"),
    )
    baselines = frame[frame.model_seed.isna()].copy()
    baselines = baselines.assign(
        candidate_count_mean=float("nan"),
        candidate_fraction=float("nan"),
        uncertain_fraction=float("nan"),
        oracle_recall=float("nan"),
        feasible_pair_reduction=float("nan"),
    )[
        [
            "pressure",
            "scenario_seed",
            "policy",
            "objective",
            "match_rate",
            "candidate_count_mean",
            "candidate_fraction",
            "uncertain_fraction",
            "oracle_recall",
            "feasible_pair_reduction",
        ]
    ]
    return pd.concat([baselines, learned], ignore_index=True)


def main() -> None:
    root = Path("runs/stress")
    paired = scenario_level(pd.read_csv(root / "pruning-validation.csv"))
    summary = paired.groupby(["pressure", "policy"], as_index=False).agg(
        scenarios=("objective", "size"),
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        match_rate_mean=("match_rate", "mean"),
        candidate_count_mean=("candidate_count_mean", "mean"),
        candidate_fraction=("candidate_fraction", "mean"),
        uncertain_fraction=("uncertain_fraction", "mean"),
        oracle_recall=("oracle_recall", "mean"),
        feasible_pair_reduction=("feasible_pair_reduction", "mean"),
    )

    comparisons = []
    for pressure in ("moderate", "high", "extreme"):
        wide = paired[paired.pressure == pressure].pivot(
            index="scenario_seed", columns="policy", values="objective"
        )
        for policy in ("actor_pruned_1", "actor_pruned_2"):
            for reference in ("myopic_milp", "sac", "supply"):
                difference = (wide[policy] - wide[reference]).dropna()
                comparisons.append({
                    "pressure": pressure,
                    "policy": policy,
                    "reference": reference,
                    "scenarios": len(difference),
                    "wins": int((difference > 0).sum()),
                    "mean_difference": difference.mean(),
                })
    comparison_frame = pd.DataFrame(comparisons)
    summary.to_csv(root / "pruning-validation-summary.csv", index=False)
    comparison_frame.to_csv(root / "pruning-validation-comparison.csv", index=False)
    print(summary.to_string(index=False))
    print("\nPaired objective comparisons")
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
