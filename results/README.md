# Curated experiment results

This directory contains only the compact artifacts needed to audit the conclusions in the root README and [`docs/EXPERIMENT_REPORT.md`](../docs/EXPERIMENT_REPORT.md). Raw checkpoints, per-episode logs and smoke-test outputs are intentionally excluded; reruns write to the ignored `runs/` directory.

| Files | Experiment |
|---|---|
| `learning-curve-summary.csv`, `heldout-summary.csv`, `experiment-summary.png` | repaired fixed-ID SAC and baseline comparison |
| `candidate-ablation-summary.csv` | fixed-ID vs candidate sharing, random graphs and behavior cloning |
| `stress-*` | zero-shot moderate/high/extreme evaluation |
| `hybrid-*` | learned-Q MILP and shuffled-Q control |
| `curriculum-*` | pressure-curriculum SAC |
| `pruning-*` | actor top-k candidate-pruning validation |
| `counterfactual-*` | supply-paired counterfactual reward on held-out seeds 801–807 |

The summary tables are the source of the rounded figures reported in the documentation. Negative results are retained because they determine which research directions should not be treated as established improvements.
