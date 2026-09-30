"""Multi-seed benchmark for the candidate-scoring SAC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluate_refactored import evaluate_checkpoint, summarize
from train_candidate_sac import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bc-episodes", type=int, default=0)
    parser.add_argument("--bc-epochs", type=int, default=15)
    parser.add_argument("--random-rollouts", type=int, default=5)
    parser.add_argument("--pressure-curriculum", action="store_true")
    parser.add_argument(
        "--counterfactual-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--scenario", type=Path, default=Path("sample_data.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/candidate-benchmark"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = parser.parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for seed in [int(value) for value in cli.seeds.split(",")]:
        run_dir = cli.output_dir / f"seed-{seed}"
        train(SimpleNamespace(
            scenario=cli.scenario, output_dir=run_dir, episodes=cli.episodes,
            horizon=cli.horizon, capacity=7, num_cities=8, num_vehicles=11,
            orders_per_step=5, batch_size=cli.batch_size, hidden_dim=128,
            updates_per_step=1, eval_interval=5, bc_episodes=cli.bc_episodes,
            bc_epochs=cli.bc_epochs, bc_batch_size=256,
            target_entropy_ratio=0.2, initial_alpha=0.1,
            reward_scale=1000.0, cancel_penalty=300.0,
            team_reward_weight=0.25, random_scenarios=True,
            pressure_curriculum=cli.pressure_curriculum,
            counterfactual_reward=cli.counterfactual_reward,
            counterfactual_individual_weight=1.0,
            seed=seed, device=cli.device,
        ))
        rows = evaluate_checkpoint(
            run_dir / "sac_checkpoint.pt", cli.scenario, cli.horizon, 7,
            cli.batch_size, cli.random_rollouts, cli.device,
        )
        for row in rows:
            row["training_seed"] = seed
        all_rows.extend(rows)
    summary = summarize(all_rows)
    with (cli.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    with (cli.output_dir / "rollouts.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = sorted({key for row in all_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
