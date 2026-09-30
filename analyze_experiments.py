"""Aggregate the reproducibility experiments and render compact diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_learning_curves(root: Path):
    frames = []
    for path in sorted(root.glob("seed-*/metrics.csv")):
        frame = pd.read_csv(path)
        frame["training_seed"] = int(path.parent.name.split("-")[-1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_heldout(root: Path):
    frames = []
    for path in sorted(root.glob("heldout-seed[0-9]*.csv")):
        if path.stem.endswith("-summary"):
            continue
        frame = pd.read_csv(path)
        frame["training_seed"] = int(path.stem.split("seed")[-1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_nested_heldout(root: Path):
    frames = []
    for path in sorted(root.glob("seed-*/heldout.csv")):
        frame = pd.read_csv(path)
        frame["training_seed"] = int(path.parent.name.split("-")[-1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def heldout_summary(frame):
    return frame.groupby("policy").agg(
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        matched_mean=("matched", "mean"),
        match_rate_mean=("match_rate", "mean"),
        cancelled_mean=("cancelled", "mean"),
        rollouts=("objective", "size"),
    ).reset_index()


def curve_summary(frame):
    selected = frame.dropna(subset=["eval_objective"])
    return selected.groupby("episode")["eval_objective"].agg(["mean", "std", "count"]).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-entropy", type=Path, default=Path("runs/benchmark-3seed"))
    parser.add_argument("--low-entropy", type=Path, default=Path("runs/benchmark-low-entropy"))
    parser.add_argument("--candidate-fixed", type=Path, default=Path("runs/candidate-ablation-fixed"))
    parser.add_argument("--candidate-random", type=Path, default=Path("runs/candidate-ablation-no-bc"))
    parser.add_argument("--candidate-bc", type=Path, default=Path("runs/candidate-benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/analysis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    high = load_learning_curves(args.high_entropy)
    low = load_learning_curves(args.low_entropy)
    candidate_fixed = load_learning_curves(args.candidate_fixed)
    candidate_random = load_learning_curves(args.candidate_random)
    heldout = load_heldout(args.low_entropy)
    fixed_heldout = load_nested_heldout(args.candidate_fixed)
    random_heldout = load_nested_heldout(args.candidate_random)
    bc_heldout = load_nested_heldout(args.candidate_bc)
    fixed_summary = pd.read_csv(args.low_entropy / "summary.csv")
    legacy_heldout_summary = heldout_summary(heldout)
    legacy_heldout_summary.to_csv(args.output_dir / "heldout-summary.csv", index=False)

    ablation_frames = []
    for label, frame in (
        ("legacy SAC", heldout),
        ("candidate / fixed graph", fixed_heldout),
        ("candidate / random graphs", random_heldout),
        ("candidate / random + BC", bc_heldout),
    ):
        selected = frame[frame.policy == "sac"].copy()
        selected["setting"] = label
        ablation_frames.append(selected)
    heldout_ablation = pd.concat(ablation_frames, ignore_index=True)
    heldout_ablation_summary = heldout_ablation.groupby("setting").agg(
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        match_rate_mean=("match_rate", "mean"),
        cancelled_mean=("cancelled", "mean"),
        rollouts=("objective", "size"),
    ).reset_index()
    heldout_ablation_summary.to_csv(
        args.output_dir / "candidate-ablation-summary.csv", index=False
    )

    high_curve = curve_summary(high)
    high_curve["setting"] = "high entropy (0.9)"
    low_curve = curve_summary(low)
    low_curve["setting"] = "low entropy (0.2)"
    candidate_fixed_curve = curve_summary(candidate_fixed)
    candidate_fixed_curve["setting"] = "candidate / fixed graph"
    candidate_random_curve = curve_summary(candidate_random)
    candidate_random_curve["setting"] = "candidate / random graphs"
    pd.concat([high_curve, low_curve, candidate_fixed_curve, candidate_random_curve]).to_csv(
        args.output_dir / "learning-curve-summary.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax = axes[0]
    for frame, label, color in (
        (low, "Legacy SAC", "#d95f02"),
        (candidate_fixed, "Candidate SAC / fixed graph", "#7570b3"),
        (candidate_random, "Candidate SAC / random graphs", "#1b9e77"),
    ):
        selected = frame.dropna(subset=["eval_objective"])
        for _, seed_frame in selected.groupby("training_seed"):
            ax.plot(seed_frame["episode"], seed_frame["eval_objective"], color=color, alpha=0.22)
        summary = curve_summary(frame)
        ax.plot(summary["episode"], summary["mean"], marker="o", color=color, label=label)
        std = summary["std"].fillna(0)
        ax.fill_between(summary["episode"], summary["mean"] - std, summary["mean"] + std, color=color, alpha=0.12)
    baseline_colors = {"random": "#7570b3", "supply": "#e7298a", "myopic_milp": "#66a61e"}
    for policy, color in baseline_colors.items():
        value = fixed_summary.loc[fixed_summary.policy == policy, "objective_mean"]
        if not value.empty:
            ax.axhline(value.iloc[0], linestyle="--", color=color, label=policy)
    ax.set(title="Greedy evaluation on training scenario", xlabel="Episode", ylabel="Objective")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    baseline = legacy_heldout_summary.set_index("policy")
    bar_rows = [
        {"label": "noop", **baseline.loc["noop", ["objective_mean", "objective_std"]].to_dict()},
        {"label": "random", **baseline.loc["random", ["objective_mean", "objective_std"]].to_dict()},
    ]
    for label in ("legacy SAC", "candidate / fixed graph", "candidate / random graphs", "candidate / random + BC"):
        row = heldout_ablation_summary[heldout_ablation_summary.setting == label].iloc[0]
        bar_rows.append({"label": label, "objective_mean": row.objective_mean,
                         "objective_std": row.objective_std})
    for policy in ("supply", "myopic_milp"):
        bar_rows.append({"label": policy, **baseline.loc[policy, ["objective_mean", "objective_std"]].to_dict()})
    bars = pd.DataFrame(bar_rows)
    positions = np.arange(len(bars))
    colors = ["#9e9e9e", "#9e9e9e", "#d95f02", "#7570b3", "#1b9e77", "#66c2a5", "#e7298a", "#66a61e"]
    ax.bar(positions, bars["objective_mean"], yerr=bars["objective_std"], capsize=4, color=colors)
    ax.set_xticks(positions, bars["label"], rotation=28, ha="right")
    ax.set(title="Held-out random graphs (3 model x 3 scenario seeds)", ylabel="Objective")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(args.output_dir / "experiment-summary.png", dpi=180)
    print(fixed_summary.to_string(index=False))
    print("\nHeld-out candidate ablation:\n" + heldout_ablation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
