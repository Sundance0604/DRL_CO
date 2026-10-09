# 实现验收记录

日期：2026-10-09。以下均为实现与工程验证，不是正式 CFS 数值研究。

## 实现范围

新增 `model_families/single/strategic`、`model_families/single/learning/ppo`、独立实验入口、工程配置和测试。未修改原物理引擎、MILP、动作编译、权威账本、平台注册、公共调度、旧研究脚本或历史实验产物。未升级依赖。

支持公开状态集合编码、每 hub 两维 tanh Gaussian 修正、冻结 MC 底座、H=1 值感知 MILP、live state 编译复核、完整冷却 reward、完整 episode PPO、验证模型选择、只读正式评价、更新边界恢复、数据划分与 checkpoint 兼容性检查。

## 测试结果

最终执行 `tests/test_single_strategic_ppo.py`、`tests/test_classical_value.py` 和 `tests/test_single_rolling.py`：**46 passed，9.01 秒**。记录保存在 `workspace/ppo_strategic_control/qa/pytest-regression-03.xml`。

新增测试验证零信号与原 MC provider／planner 的目标、完整动作、逐期现金及完整冷却利润相同；恢复训练与连续训练的 Actor／编码器权重逐张量相同；评价不修改 checkpoint；隐藏未来和场景身份不改变当前输入与解码；价格扰动能够改变实际运输动作；异常、取消、过期动作、篡改和跨目录文件拒绝接受。

首次单独运行时，10 项通过、3 项因 QA 临时目录的父目录尚不存在而产生 fixture setup 错误；随后父目录创建后，13 项全部通过。之后补充动作可控性及 hub 顺序检查，运行上述 46 项最终回归。首次错误记录保留在 `pytest-attempt-01.xml`，第二次记录保留在 `pytest-attempt-02.xml`，未覆写为全程无失败。

## 可运行入口验证

- `engineering-preflight-01`：1 个合成训练场景，沿零信号轨迹共 36 次轴向正负扰动，其中 3 次改变动作 hash。
- `engineering-training-01`：2 次完整 PPO update、4 个完整训练 episode，产出恢复包、最佳推理包和末次推理包，状态为 `COMPLETED`。
- `engineering-evaluation-01`：加载最佳冻结推理包，在独立合成 test 场景评价，状态为 `COMPLETED`。

上述三个目录位于 `workspace/ppo_strategic_control/`。训练更新、冷却执行和评价均使用实际 Gurobi 规划路径；示例表是手工合成工程夹具，训练预算很小，不能据此作收益优势或统计结论。

## 仍在实施范围之外

网页平台的 PPO framework／组件注册、平台任务恢复与复现导出、正式 CFS 数据划分、MC 表选择、校准与滚动基准矩阵、多训练 seed 和分层统计尚未实施。首版限定同一网络、车队规模与时域，采用 CPU 串行采样。旧路径的验证范围是上述价值与滚动控制回归，并非全平台、legacy 或 BHH 的完整验收。
