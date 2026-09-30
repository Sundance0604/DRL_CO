"""Train several seeds and compare final greedy policies with fixed baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from drl_co.paths import DEFAULT_SCENARIO
from experiments.evaluation.evaluate import evaluate_checkpoint, summarize
from experiments.training.train_fixed_id import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--target-entropy-ratio", type=float, default=0.2)
    parser.add_argument("--initial-alpha", type=float, default=0.1)
    parser.add_argument("--random-rollouts", type=int, default=5)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate-only", action="store_true")
    cli = parser.parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for seed in [int(value) for value in cli.seeds.split(",")]:
        run_dir = cli.output_dir / f"seed-{seed}"
        args = SimpleNamespace(
            scenario=cli.scenario,
            output_dir=run_dir,
            episodes=cli.episodes,
            horizon=cli.horizon,
            capacity=7,
            batch_size=cli.batch_size,
            updates_per_step=cli.updates_per_step,
            eval_interval=cli.eval_interval,
            target_entropy_ratio=cli.target_entropy_ratio,
            initial_alpha=cli.initial_alpha,
            reward_scale=1000.0,
            cancel_penalty=300.0,
            seed=seed,
            device=cli.device,
        )
        if not cli.evaluate_only:
            train(args)
        elif not (run_dir / "sac_checkpoint.pt").exists():
            raise FileNotFoundError(run_dir / "sac_checkpoint.pt")
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
