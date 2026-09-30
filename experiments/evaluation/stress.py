"""Evaluate policies under progressively scarcer fleet capacity."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from drl_co.paths import DEFAULT_SCENARIO
from drl_co.simulation.scenarios import generate_scenario
from experiments.evaluation.evaluate import evaluate_rollout, load_agent


PRESSURE_LEVELS = {
    "moderate": (8, 6),
    "high": (5, 8),
    "extreme": (3, 10),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("runs/candidate-ablation-no-bc"))
    parser.add_argument("--model-seeds", default="11,22,33")
    parser.add_argument("--scenario-seeds", default="201,202,203")
    parser.add_argument("--levels", default="moderate,high,extreme")
    parser.add_argument("--hybrid-betas", default="")
    parser.add_argument("--adaptive-thresholds", default="")
    parser.add_argument("--shuffled-betas", default="")
    parser.add_argument("--prune-topks", default="")
    parser.add_argument(
        "--pruning-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a same-state full MILP for pruning recall diagnostics.",
    )
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--training-scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--output", type=Path, default=Path("runs/stress/stress-rollouts.csv"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_seeds = [int(value) for value in args.model_seeds.split(",")]
    scenario_seeds = [int(value) for value in args.scenario_seeds.split(",")]
    selected_levels = {
        name: PRESSURE_LEVELS[name] for name in args.levels.split(",")
    }
    hybrid_policies = [
        f"hybrid_q_{value}" for value in args.hybrid_betas.split(",") if value
    ]
    adaptive_policies = [
        f"hybrid_adaptive_{value}"
        for value in args.adaptive_thresholds.split(",") if value
    ]
    shuffled_policies = [
        f"hybrid_shuffled_{value}"
        for value in args.shuffled_betas.split(",") if value
    ]
    pruned_policies = [
        f"actor_pruned_{value}" for value in args.prune_topks.split(",") if value
    ]
    learned_policies = [
        *hybrid_policies, *adaptive_policies, *shuffled_policies, *pruned_policies
    ]
    agents = {}
    for model_seed in model_seeds:
        checkpoint = args.checkpoint_root / f"seed-{model_seed}" / "sac_checkpoint.pt"
        agents[model_seed], _, _, _ = load_agent(
            checkpoint, args.training_scenario, args.horizon, 7, 64, args.device
        )

    rows = []
    for pressure, (num_vehicles, orders_per_step) in selected_levels.items():
        for scenario_seed in scenario_seeds:
            vehicles, orders, graph = generate_scenario(
                scenario_seed,
                horizon=args.horizon,
                num_cities=8,
                num_vehicles=num_vehicles,
                orders_per_step=orders_per_step,
                capacity=7,
            )
            scenario_rows = []
            for policy in ("noop", "supply", "myopic_milp"):
                started = time.perf_counter()
                row = evaluate_rollout(
                    policy, vehicles, orders, graph, args.horizon,
                    capacity=7, seed=scenario_seed, agent=None,
                )
                row["elapsed_seconds"] = time.perf_counter() - started
                row["model_seed"] = ""
                scenario_rows.append(row)
            for model_seed, agent in agents.items():
                for policy in ["sac", *learned_policies]:
                    started = time.perf_counter()
                    row = evaluate_rollout(
                        policy, vehicles, orders, graph, args.horizon,
                        capacity=7, seed=scenario_seed, agent=agent,
                        pruning_diagnostics=args.pruning_diagnostics,
                    )
                    row["elapsed_seconds"] = time.perf_counter() - started
                    row["model_seed"] = model_seed
                    scenario_rows.append(row)

            myopic_objective = next(
                row["objective"] for row in scenario_rows if row["policy"] == "myopic_milp"
            )
            for row in scenario_rows:
                row.update({
                    "pressure": pressure,
                    "num_vehicles": num_vehicles,
                    "orders_per_step": orders_per_step,
                    "scenario_seed": scenario_seed,
                    "objective_vs_myopic": row["objective"] / max(1.0, myopic_objective),
                })
                rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for pressure in selected_levels:
        for policy in ("noop", "supply", "myopic_milp", "sac", *learned_policies):
            selected = [row for row in rows if row["pressure"] == pressure and row["policy"] == policy]
            summary_row = {
                "pressure": pressure,
                "policy": policy,
                "rollouts": len(selected),
                "objective_mean": float(np.mean([row["objective"] for row in selected])),
                "objective_vs_myopic_mean": float(np.mean([
                    row["objective_vs_myopic"] for row in selected
                ])),
                "match_rate_mean": float(np.mean([row["match_rate"] for row in selected])),
                "cancelled_mean": float(np.mean([row["cancelled"] for row in selected])),
                "elapsed_seconds_mean": float(np.mean([
                    row["elapsed_seconds"] for row in selected
                ])),
            }
            for diagnostic in (
                "candidate_count_mean",
                "candidate_fraction",
                "uncertain_fraction",
                "oracle_recall",
                "feasible_pair_reduction",
            ):
                values = [row[diagnostic] for row in selected if diagnostic in row]
                summary_row[diagnostic] = float(np.mean(values)) if values else ""
            summary.append(summary_row)
    summary_path = args.output.with_name(args.output.stem + "-summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
