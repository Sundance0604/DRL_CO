# 动态 PPO 与每枢纽固定信号

这是上一轮笔记本轻量实验的机制验证，复用九个已冻结的短训练模型，不新增训练。每个模型的固定信号来自其在八个验证场景的 MC 轨迹上、192 个公开状态中的确定性输出均值，near/far 按枢纽分别计算。固定残差后，MC 底座与物理解码器仍按原规则运行。

固定向量仍经原 near/far 时间插值生成价格修正，因此不代表全时段价格相同。被消除的是 Actor 输出随公开状态与时间的变化。

新测试使用原已生成但未评价的种子 22012–22019。每个工况比较三组配对的动态／固定模型及 MC、原验证选择的 CAL，共 192 个测试日。三个预先指定的 MC 诊断日用 432 次探针检查同一状态下动态／固定产生的整数动作。全部记录完整冷却结算并回放。

主比较仅有三个工况的动态减固定，20,000 次交叉 bootstrap，Bonferroni 98.333% 区间，由独立计数加权实现复算。固定向量和模型 hash 在新测试利润产生前冻结。本实验不能代表充分训练的 RL，也不能代表最优静态策略。

## 复现

在 `_review_repo` 根目录执行，输出目录必须全新。依赖已完成的 `runs/cfs_laptop_ppo_v1` 及原 `runs/cfs_first_ppo_v1` 未评价数据池，先验证旧实验全文件 hash。

```powershell
$env:PYTHONHASHSEED = '0'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$attempt = 'experiments/cfs_ppo_strategic_control/runs/my-dynamic-vs-constant'
& .venv/Scripts/python.exe -m experiments.cfs_ppo_strategic_control.mechanism.run prepare --attempt $attempt
& .venv/Scripts/python.exe -m experiments.cfs_ppo_strategic_control.mechanism.run all --attempt $attempt
```

`all` 完成计算、统计复算、图形和报告，停在最终 QA 待验收状态。通过实际文件数量、测试、逐图检查后保存 `qa/final-tests.json`、`qa/visual-review.json`，运行 `deliver` 才生成最终 hash 清单。任何数值故障或回放异常均停止并保留目录，不自动重试。

本次报告见 `runs/cfs_dynamic_vs_constant_v2/mechanism_experiment_report.md`。v1 完成全部求解与回放后，在统计结果写入 JSON 时因 NumPy 整数类型停止，保留原始记录。v2 只修复计数的整数导出，复用 v1 原始结果，重做分析与审计，不新增训练或求解。源码比较确认校准、评价、诊断函数未改变。

固定向量与逐验证状态输出见 v2 的 `protocol/frozen-policies.json`、`calibration/`；新场景配对利润见 `analysis/results.json`、`analysis/paired-test-results.csv`。原始测试日完整轨迹位于 v1 的 `evaluation/`，v2 的 manifest 同时保存其 hash。

仅当原失败属于上述导出错误、全部回放已完成且冻结输入通过检查时，可以使用新目录恢复分析：

```powershell
& .venv/Scripts/python.exe -m experiments.cfs_ppo_strategic_control.mechanism.run recover-analysis --attempt experiments/cfs_ppo_strategic_control/runs/my-recovered-analysis --failed experiments/cfs_ppo_strategic_control/runs/cfs_dynamic_vs_constant_v1
```
