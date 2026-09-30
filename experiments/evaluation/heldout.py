"""Evaluate checkpoints on deterministic scenarios never used for training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from drl_co.paths import DEFAULT_SCENARIO
from drl_co.simulation.scenarios import generate_scenario
from experiments.evaluation.evaluate import evaluate_rollout, load_agent, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--training-scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--scenario-seeds", default="101,102,103")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("runs/heldout.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    agent, _, _, _ = load_agent(
        args.checkpoint, args.training_scenario, args.horizon, 7,
        args.batch_size, args.device,
    )
    rows = []
    for scenario_seed in [int(value) for value in args.scenario_seeds.split(",")]:
        vehicles, orders, graph = generate_scenario(scenario_seed, horizon=args.horizon)
        for policy in ("noop", "random", "supply", "myopic_milp", "sac", "sac_sample"):
            row = evaluate_rollout(
                policy, vehicles, orders, graph, args.horizon,
                capacity=7, seed=scenario_seed, agent=agent,
            )
            row["scenario_seed"] = scenario_seed
            rows.append(row)

    summary = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        columns = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output.with_name(args.output.stem + "-summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
