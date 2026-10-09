# PPO 信号与整数动作的轻量机制诊断

依据 `reports/笔记本数值实验规范_假设驱动与逐级验证.md` 第七节。研究问题是：既有 PPO 的小信号是否跨不过 MILP 决策边界，还是采样状态缺少可改变决策。使用 CA80/3、W80/3，三个既有训练 seed，两个未用于既有轻量评估的开发诊断场景 23002、23003，固定 MC 状态来源，时刻 4、8、12、16、20、23。

场景由原完整实验准备阶段生成但未参与训练、验证、pilot 或轻量测试。本轮只读取它们，不重新读取未来进行控制，不修改 CFS 抽样器。候选信号为 `clip(c * deterministic_actor_signal, -1, 1)`，c 为 0、0.5、1、2、4；另有 MC、冻结 CAL、每轴 ±1 正向控制。三模型全部保存，不选赢家。最多 984 个真实求解（重复零信号按完整状态/信号身份缓存），时间上限 30 分钟、磁盘上限 2 GiB、CPU 单进程，各线程 1。

统计单位分开记录：4 场景、24 同状态快照、72 模型—快照、逻辑探针与实际求解数。资源活动子组在结果前定义为“公开待分配订单至少一个且空闲车辆至少一辆”，它不是合法替代动作存在的证明。全部预定快照保留。只作机制描述，不作利润优越性检验；改变残差后的 window objective 不能跨探针当作收益比较。

本包位于 PPO 包之外，避免修改其已冻结 source inventory；既有策略、物理、MILP、奖励和检查点都只读。严格检查推理包身份，不绕过源码和依赖门。

## 执行

在仓库根目录使用原数值 `.venv/Scripts/python.exe`，设置 PYTHONHASHSEED=0、PYTHONIOENCODING=utf-8、PYTHONDONTWRITEBYTECODE=1、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。

```powershell
& .venv/Scripts/python.exe -m experiments.cfs_ppo_signal_action_diagnostic.run prepare --attempt experiments/cfs_ppo_signal_action_diagnostic/runs/signal_action_v1
& .venv/Scripts/python.exe -m experiments.cfs_ppo_signal_action_diagnostic.run run --attempt experiments/cfs_ppo_signal_action_diagnostic/runs/signal_action_v1
```

prepare 冻结来源、数据隔离、全部模型和实验源码；run 执行有计数和截止时间的同状态探针，并独立编译各动作、重算现金和系数、检查 probe 不改变 live 状态，MC 完整轨迹完成冷却后独立重放。预算用完保存 BUDGET_LIMITED；真实故障保存 INVALID，不删样本或自动换算法。

分析脚本使用 bundled Python，生成三张科学 CSV、两组 PNG/SVG 机制图及中文报告，独立复核全部原始 probe/信号数、查表修正、clip 和变化统计。第一次 CSV 创建前按科学输出技能执行一次 marker。单独运行相关测试并记录真实 XML，实际查看图，再执行 freeze。冻结必须核验源代码、历史产物、原始文件、分析及测试证据；COMPLETED 后不再写 attempt。

## 解释与下一步

放大信号改变动作只说明控制通道可作用，不代表 profit 提高。即使各轴未改变动作，也不能证明整个连续动作域没有其他联合方向。状态适应性报告同一时间两个真实场景的信号差，不构造不物理状态、不宣称部分可观测编码充分。支持计数未知继续记录 unknown，不能当作零。

本轮结束后仅推荐一个后续问题；不会自动启动尺度搜索、PPO 长训练或独立利润确认。
