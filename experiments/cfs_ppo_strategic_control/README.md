# Single 残差战略 PPO

实现依据 `reports/第八轮方向判断与强化学习建模建议.md` 和 `reports/PPO模块化架构设计_隔离边界与接入方案.md`。本入口完成架构方案的领域适配、PPO 训练、冻结评价与更新边界恢复阶段；平台组件注册、网页训练按钮、平台复现导出与正式第八轮科学试验尚未接入。旧 single、legacy、BHH 运行入口和历史实验文件无需修改。

## 快速运行

在 `_review_repo` 根目录的 PowerShell 中运行。每个 `--output` 必须是尚不存在的新目录，避免覆盖实验产物。

```powershell
$python = '.venv/Scripts/python.exe'
$entry = 'experiments.cfs_ppo_strategic_control.run_experiment'
$config = 'experiments/cfs_ppo_strategic_control/config/smoke.json'
$dataset = 'experiments/cfs_ppo_strategic_control/config/smoke_dataset.json'
& $python -m $entry preflight --config $config --dataset $dataset --output workspace/ppo_strategic_control/my-preflight
& $python -m $entry train --config $config --dataset $dataset --output workspace/ppo_strategic_control/my-training
& $python -m $entry evaluate --checkpoint workspace/ppo_strategic_control/my-training/best-inference --dataset $dataset --output workspace/ppo_strategic_control/my-evaluation
```

示例的场景、价格表和尺度均为手工构造的工程夹具，不是 CFS 实验，也不说明 PPO 优于 MC。真实实验需换成新的、互不重叠的训练／验证／测试场景，以及来源明确且已冻结的非 bucket MC 表；五枢纽时动作维数自动成为 10。动作按 `hubs` 顺序交错排列：`[near_A, far_A, near_B, far_B, ...]`。

价格表沿用 `method=table`、`values[hub][q].times/values` 格式。每个 hub 必须覆盖 `q=1..fleet_size`，缺少槽位会拒绝运行。插值与原 `MCTableValueProvider` 使用同样的数值实现，查询日志标记精确网格、插值或边界钳制；旧表没有逐格支持计数时明确记录 `null`。显式 `{"method":"zero"}` 可用于无底座对照。

## 控制与训练协议

- `strategic/` 只依赖 single 物理、规划与账本；不依赖 Torch。网络只能拿到不可变的公开观测，场景 ID、种子、隐藏订单和 `sim.J` 不进入输入。
- 观测保留全部公开订单，以及每辆车的承诺订单、在车状态、按序剩余路线、各段载荷与剩余容量。订单与车辆以共享编码器和集合均值／数量汇总；变长集合没有裁剪，空集合显式屏蔽。这是部分可观测的近似控制，不宣称是充分状态。
- 公开 OD 累计到达历史、公共网络距离／时间矩阵、hub 供需与底座价格进入观测。标准化仅用配置中预先冻结的正尺度，训练与测试期间均不更新。
- 一次策略采样对应一份冻结价格信号，经 H=1 原价值 MILP 产生完整动作，在复制状态和 live state 上分别编译核对。保留负系数、释放槽位前缀约束、业务可行域和 tie-break。仅接受 `OPTIMAL`，超时／取消／账本异常不伪装成正常终止，不回退到另一策略。
- 中间 reward 是权威实现账本增量；末次决策自动完成原承诺冷却，奖励总和必须等于最终实现利润。战略 bonus、预记收入和 tie-break 不作为现金奖励。
- PPO 使用 tanh Gaussian、稳定变换 Jacobian、完整 episode GAE、`gamma=1`、固定 reward scale、优势标准化、clip、梯度范数与近似 KL 早停。熵诊断／正则使用变换后的分布。
- 首版使用 CPU、单进程串行采样；每次求解独立创建和释放 solver。初始化、动作采样、minibatch 和场景抽样采用分开的私有随机流，不修改整机线程环境或其他 run 的随机数。
- 每批完整 on-policy episode 收齐后才更新。验证选择采用确定性策略的平均美元利润，首次达到的并列最佳保留；测试不参与选择或更新。

## 产物与恢复

训练产物包括 `training-episode-*.json` 的独立 transition／物理决策记录、`update-*.json` 的数值诊断、`update-*/` 的恢复包，以及 `best-inference/`、`last-inference/` 的只读推理包。每步保存 observation hash、signal、实际系数及底座查询方式、action hash、原始美元 reward、累计账本、编译金额和 solver 状态。训练曲线与正式评价产物分开，不进入现有平台的正式样本池。

checkpoint 包内保存 Actor／编码器、动作与特征配置、物理／solver 参数、MC 底座、数据划分身份、协议与网络版本、依赖版本、源码身份和文件 hash。恢复包额外保存 Critic、optimizer、三个私有随机流和历史最佳 Actor；仅支持完整 rollout/update 边界恢复。权重以 `weights_only=True` 加载，相对文件路径必须留在包内。源码、依赖、hub 顺序、尺度、底座、数据或配置不匹配会拒绝加载；需要保留同版本源码来复现历史包。

恢复时将 `ppo.updates` 改成新的**总 update 数**，其他配置和完整数据划分保持相同，再使用新输出目录：

```powershell
& $python -m $entry train --config my-extended-config.json --dataset $dataset --resume workspace/ppo_strategic_control/my-training/update-0002 --output workspace/ppo_strategic_control/my-resumed-training
```

正式评价默认确定性；`--stochastic --policy-seed 123` 启用有记录的随机策略。正式场景必须标记 `test`，与训练／验证的 ID、种子和内容均不重叠，并与 checkpoint 的网络、车队规模和决策时域兼容。推理包没有 Critic 或 optimizer，评价不修改包内任何文件。

预检在训练工程状态上沿零信号轨迹测试每个动作轴的正负边界，报告物理动作是否改变。有限扰动未找到动作变化不能证明接口不可表达，但应在大规模训练前检查尺度与覆盖。本入口没有自动安排正式预算、多训练 seed、校准策略或显著性比较；需要预注册独立研究配置再运行。

## 笔记本轻量数值实验

按用户要求新增 [轻量研究入口](laptop/README.md)：三工况 × 三训练种子，每模型仅 32 个完整训练日；每工况验证 8、测试 12 个场景，CPU 单进程、单线程执行。保留 MC、CAL、M0、RH4 对照，模型仅由验证选择；完整轨迹回放、独立统计复算和六类图形随结果保存。这是减量探索，不能替代原任务书的大样本验证。

本次结果保存于 `runs/cfs_laptop_ppo_v1/laptop_experiment_report.md`。原 `runs/cfs_first_ppo_v1` 的完整方案准备和 pilot 另行保留，原完整预算未启动。

后续 [动态与固定信号机制实验](mechanism/README.md) 复用九个冻结模型，在验证状态上计算每枢纽固定 near/far 向量，再用此前未测试的场景比较动态输出与固定输出。报告保存于 `runs/cfs_dynamic_vs_constant_v2/mechanism_experiment_report.md`；v1 原始求解结果及导出失败记录保留，v2 仅恢复分析，没有重复求解。

## 验证

```powershell
& $python -m pytest tests/test_single_strategic_ppo.py tests/test_classical_value.py tests/test_single_rolling.py -q
```

新增检查覆盖信息边界、对象隔离、零信号整轨迹回归、末步冷却、负系数、过期动作／求解故障、集合编码、概率变换与极值梯度、GAE、只读推理、数据重叠、checkpoint 篡改／路径越界，以及连续训练与恢复训练的权重一致性。
