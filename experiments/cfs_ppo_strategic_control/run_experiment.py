"""Run from repository root with python -m experiments.cfs_ppo_strategic_control.run_experiment."""
import argparse
from pathlib import Path
from model_families.single.strategic.artifacts import read_json


def main():
    parser = argparse.ArgumentParser(description="Single H=1 residual strategic PPO")
    parser.add_argument("mode", choices=("preflight", "train", "evaluate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new attempt directory")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    from model_families.single.learning.ppo.runner import train_run, evaluate_run, preflight_run
    data = read_json(args.dataset)
    if args.mode == "evaluate":
        if not args.checkpoint or args.resume:
            parser.error("evaluate requires --checkpoint and does not accept --resume")
        result = evaluate_run(args.checkpoint, data, args.output, args.policy_seed, not args.stochastic)
    else:
        if not args.config or args.checkpoint or (args.resume and args.mode != "train"):
            parser.error("train/preflight require --config; --resume is only for training")
        configuration = read_json(args.config)
        if isinstance(configuration.get("baseline"), str):
            configuration["baseline"] = str((args.config.parent/configuration["baseline"]).resolve())
        if args.mode == "train":
            result = train_run(configuration, data, args.output, args.resume)
        else:
            result = preflight_run(configuration, data, args.output)
    print(f"Completed {args.mode}; outputs: {args.output.resolve()}")
    return result


if __name__ == "__main__":
    main()
