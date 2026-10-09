# 笔记本轻量实验

依据用户“改成一个轻量化的实验”的要求，使用单进程 CPU、Torch/BLAS/MILP 各一线程，Windows 执行进程采用低于正常优先级。保留三个原工况；每个工况训练三个种子，每模型 8 次更新、每次 4 个完整日，合计 288 个训练 episode。训练池 32、验证 8、测试 12、诊断 2 场景／工况。

固定数据子集在运行前登记，复用已通过工程预检的源码、bank1 非 bucket MC 表与冻结尺度；使用未参加 pilot 的 train/validation/test/diagnostic 种子。按验证成绩冻结 PPO 和九候选 CAL 后再测试。比较三个 PPO seed、MC、CAL、M0、RH4，保存完整冷却结算和独立轨迹回放。M0 是相同 H1 解码器的零价值版本。

共 978 次 episode 执行（其中训练 288、训练验证 144、CAL 验证 216、选定模型验证重算 72、测试 252、诊断 MC 轨迹 6），外加同状态诊断探针和无需 MILP 的回放。每个恢复包留在训练目录，推理包无 critic 或 optimizer。

这是探索实验，三个 seed 和十二测试场景无法提供原完整任务书的大样本证明。不能挑最好测试 seed，不能因短训练没提升就判定整个 RL 方向失败。原完整实验目录保留，完整预算没有启动。

## 复现

在 `_review_repo` 根目录运行，输出目录必须全新；本命令依赖已有 `runs/cfs_first_ppo_v1` 冻结数据与已通过的预检证据。

```powershell
$env:PYTHONHASHSEED = '0'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$attempt = 'experiments/cfs_ppo_strategic_control/runs/my-laptop-experiment'
& .venv/Scripts/python.exe -m experiments.cfs_ppo_strategic_control.laptop.run prepare --attempt $attempt
& .venv/Scripts/python.exe -m experiments.cfs_ppo_strategic_control.laptop.run all --attempt $attempt
```

`all` 完成数值工作、统计复算、六类图形和报告，停在 `NUMERICAL_COMPLETE_QA_PENDING`。完成最终测试和逐图检查并将真实结果写入 `qa/final-tests.json`、`qa/visual-review.json` 后，`deliver` 校验并生成最终 hash 清单；不自动虚报验收通过。失败立即保留原目录，不能覆盖或自动重试以隐藏失败。

本次运行目录为 `runs/cfs_laptop_ppo_v1`，结果见其中 `laptop_experiment_report.md`、`analysis/test_results.csv` 和 `analysis/profit-comparison.json`。实际资源用量记录于 `analysis/execution.json`；预算记录于 `protocol/budget.json`。环境变量和优先级只影响当前执行进程。
