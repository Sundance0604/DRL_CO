# 第八轮：PPO 战略价格控制的首次 CFS 数值实验

> 版本：研究协议草案 v1.0，2026-10-09。根据当前已实现的代码设计，尚未执行本轮训练或正式评估。本文中的样本量、预算、阈值和超参数是拟预注册的设计，不是已有实验结果。
>
> 研究对象是当前 single-level 模型中的新 PPO 控制器，不是恢复旧 DRL/SAC 模型。本文只设计实验；不授权修改历史实验、启动其他算法、提交或推送代码。

## 1. 研究问题：先确认 PPO 学到了什么，再讨论它能否胜出

前七轮已经建立了有竞争力的 MC Table 基准，同时发现：预测误差降低不必然提高闭环利润；资源价格的反复更新可能造成不稳定；加入状态信息也不自动带来更好的控制。第七轮的联合动作诊断提供了改进方向，但尚未证明强化学习必要，也没有证明任何 PPO 实现都会优于 MC Table。

因此，首轮 PPO 实验不应同时搜索网络结构、奖励函数、动作定义和大量超参数。应先固定现有实现，回答三个具体问题：

1. **有效性：**在相同物理系统、场景和 MILP 解码器下，PPO 学到的状态依赖价格修正，能否提高相对于冻结 MC Table 的 full-cooldown realized profit？
2. **学习内容：**如果利润提高，是否超过简单固定价格偏移也能获得的改进？否则不能把收益全部归因于状态依赖控制。
3. **可重复性与代价：**收益能否跨训练随机种子复现，在困难工况下是否退化，训练和在线决策的计算代价是多少？

本轮是一个**固定 MC 底座条件下的、有限预算的 PPO 概念验证**。它不同时解决 construction bank 方差、跨网络泛化或可分价格的表达上限，也不以“必须得到正结果”为目标。

## 2. 代码告诉我们：哪些已经实现，哪些还不是正式实验

### 2.1 已核对的实现基础

当前实现分为三层：

| 层次 | 当前位置 | 本轮用途 |
|---|---|---|
| 物理、观察、战略动作与核算 | `model_families/single/strategic/` | 公开状态、价格修正、H=1 MILP、整数动作执行、完整现金回报 |
| PPO 学习与检查点 | `model_families/single/learning/ppo/` | Actor–Critic、完整 episode 采样、PPO 更新、验证选模、恢复与推理 |
| 独立研究入口 | `experiments/cfs_ppo_strategic_control/` | `preflight`、`train`、`evaluate` 三种入口 |

当前已实现 CPU 串行采样、tanh-Gaussian 策略、完整 episode GAE、梯度裁剪、有限 KL 提前结束更新，以及严格的检查点和数据身份校验。Actor 输出低维信号，MILP 负责 matching、routing、capacity 与 commitment；PPO 不直接输出完整组合动作。

已有 QA 文件和实际 XML 记录显示，相关回归测试为 **46 passed、0 failed、0 errors、0 skipped**，用时 9.010 秒。现有训练、评估和扰动测试是小型合成场景的工程验证，不能当作 CFS 学习效果。该数字也不是整个实验平台、BHH 或旧 DRL 的全量测试结果。

### 2.2 正式执行前需要补齐的实验层工作

下列工作应由新的实验准备、调度或审计工具完成，不应把它们伪装成现有 CLI 已经支持的功能：

- 正式 CFS 数据生成、分割、初始车队部署与完整来源清单。
- 冻结 MC Table 的完整性、资源数量覆盖及 USD 动作尺度提取。
- 多训练种子调度、M0/MC/固定偏移/RH4 的同口径评估。
- 验证选模的逐场景收益记录或独立复算。当前训练器保存验证均值，但不足以单独构成逐场景选模审计。
- 配对统计、训练种子不确定性、图表、逐步重放与最终交付冻结。

另外，正式训练前应审查并补齐一个数值安全细节：当前 PPO 更新逻辑将“非有限 KL”与“有限 KL 超阈值”放在同一提前结束分支；某些情况下前面已有有效 minibatch 时，非有限 KL 可能没有作为整次运行故障抛出。**有限 KL 过大是正常保护，NaN/Inf 则必须单独停止并记录，不能混同。**这属于训练前的安全门，不是根据实验结果调整算法。

上述准备及必要安全修正完成后，重新运行相关测试并冻结源码，才开始正式训练。本轮文档本身不实施这些修改。

## 3. 控制问题：沿用现有动作、状态和奖励，不在首轮同时重设计

### 3.1 Actor 输出什么

每个 hub 输出 near、far 两个归一化残差信号。五个 hub 时，动作为 10 维：

\[
a_t=(a_{1,n},a_{1,f},\ldots,a_{H,n},a_{H,f})\in[-1,1]^{2H}.
\]

hub 顺序必须写入 ActionSchema 并冻结，不能在训练和推理之间重新排序。

对当前决策时刻 \(t\)、未来释放时刻 \(\tau\)、hub \(u\) 和资源序号 \(q\)，现有解码器使用：

\[
\widetilde V_t(\tau,u,q)=V_{\mathrm{MC}}(\tau,u,q)
+s_u\left[(1-\ell)a_{u,n}+\ell a_{u,f}\right],\quad
\ell=\frac{\tau-t}{T-t},\quad\tau<T.
\]

当 \(\tau\ge T\) 时，基础值和修正都为零。保留当前 terminal gate，不借本轮暗中改成跨日价值。

这里的 \(q\) 仍由现有 release-group/prefix 接入决定；Actor 没有为每个 \(q\) 输出独立修正，也没有直接输出 joint-action advantage。因此，即使本轮失败，也不能据此排除其他 RL 动作表示。

### 3.2 USD 尺度如何固定

首轮不对动作幅度进行验证集收益搜索。建议从所选冻结表中，按 hub 使用唯一非终端网格点、\(q=1,\ldots,K\) 的绝对值中位数：

\[
s_u=\max\{100\ \mathrm{USD},\operatorname{median}_{\tau<T,\,1\le q\le K}|V_{\mathrm{MC}}(\tau,u,q)|\}.
\]

保存参与计算的网格、结果和表 SHA；不同 hub 使用自己的尺度。100 USD 是本轮预设的工程下限，不是前七轮已经证明的最优尺度。不得用正式测试收益调整它。

先用工程场景检查该尺度是否能改变真实整数动作。若所有扰动均无法改变动作，应停止进入大规模训练，查明尺度、候选限制或场景原因，而不是让 PPO 在实际上不可控的接口上消耗预算。若需要调整尺度，必须在正式训练前更新协议版本，并重新通过工程门。

### 3.3 状态与 Critic

保持当前公共观察编码：时间、hub 信息、已观察需求历史、当前订单集合、车辆状态、承诺和路径相关信息，以及可查询的冻结 MC 信息。保留完整集合，不为了提速随意截断订单或车辆。

不得输入隐藏未来订单、未来到达总量、场景 ID、scenario seed 或用于识别具体场景的标签。训练和推理使用相同特征构造。

首轮保持当前集合编码网络及 hidden size=64，不引入 RNN、GNN 或额外预测器。固定特征尺度建议为 money=1000 USD、distance=100 miles、time=24、load=40；须与 CFS 场景实际单位核对，不能直接沿用合成示例的载重尺度。

Critic 学习的是当前策略下的未来累计回报，不是再次监督拟合资源注入标签 \(V(t,u,q)\)。它与作为动作底座的 MC Table 是两个不同对象。

### 3.4 奖励与终止

沿用现有 realized ledger 的增量奖励，最后一个决策步包含所有未完成承诺的 full cooldown 核算：

\[
r_t=J^{\mathrm{realized}}_{t+1}-J^{\mathrm{realized}}_t,\qquad
\sum_t r_t=J^{\mathrm{full\ cooldown}}-J_0.
\]

训练使用 \(r_t/1000\)，\(\gamma=1\)；报告恢复为原始 USD。不增加服务率奖励、库存惩罚、MC shaping 或 ACTION target。服务率和资源利用仅作为诊断指标。

自然完成与超时必须区分：solver 未达到要求、episode 超时、数值异常都不能当作正常终止并写入 PPO buffer。每个 episode 必须核对奖励和完整现金的一致性。

## 4. 对照设计：三个工况、一个冻结底座、五次独立训练

### 4.1 工况

| 工况 | 网络/每日订单/车辆数 | 研究角色 |
|---|---|---|
| CA80/3 | California corridor / 80 / 3 | 前期反馈与 construction 敏感性较明显的困难工况 |
| W40/8 | Western multicluster / 40 / 8 | 检查正常资源配置下是否还有增益 |
| W80/3 | Western multicluster / 80 / 3 | 高压力工况及退化保护 |

**每个工况单独训练模型。**现有入口要求单次运行内网络、车队数量和 horizon 的物理签名一致，不能把三个工况混进一个训练数据集。第一轮不纳入 W20，不做跨工况迁移；这不是宣称它们不重要，而是控制首轮问题规模。

固定沿用历史 CFS 物理和成本口径：T=24、capacity=40、speed=50、loaded cost=2、empty cost=2.2、waiting cost=0、expiry cost=100、max delay=2、reposition=false，网络 kNN+MST 与 stretch=1.2，以及原订单收入、等待窗口和 backlog 规则。准备工具须从已冻结配置和实际生成逻辑核对，不能只凭本段文字重造一个相似环境。

初始车队采用历史确定性部署规则：按 eligible CFS outbound 权重排序，K=3 放在前三 hub；K=8 每 hub 一辆后再给前三 hub 各一辆；同权按 hub ID 排序。保存最终部署，不增加未声明的 fleet 随机性。

### 4.2 冻结 MC 底座

建议三工况统一采用第七轮 **bank1** 对应的非 bucket 表，条件是通过现有 FrozenBaseline 的严格完整性校验：

`experiments/cfs_joint_action_value_diagnostic/tables/bank1/{condition}.json`

其 construction 来源是该银行的既有 60 个 seed，不能复用本轮训练、验证或正式样本。选择 bank1 是为使用统一编号和可追溯底座，**不是逐工况挑历史最佳银行**。由于前七轮结果已经看过，也不能把这说成对历史结果完全盲选。

本轮结论须明确“条件于这一冻结底座”；不据此声称消除了 construction variance。四银行重新训练 PPO 留待后续独立研究。

每个 hub 必须提供所有整数 \(q=1,\ldots,K\) 所需序列。缺失则准备阶段失败，不能静默补零、换表或加入旧 bucket 回退。当前接口记录的 `support_count=None` 是支持计数未知，不是支持充分或支持为零；网格、插值与端点夹持应分别统计。

### 4.3 对照策略

| 策略 | 定义 | 用途 |
|---|---|---|
| M0 | `method=zero` 底座、零信号、相同 H=1 解码器 | 说明相对 myopic 水平；先验证与历史 M0 一致 |
| MC | 冻结 bank1 表、恒为零的战略信号 | **主要基准**，精确隔离 PPO 残差贡献 |
| CAL | 同一 MC 表，固定的全 hub 共用 near/far 信号 | 排除简单价格校准就能解释全部收益 |
| PPO | 同一 MC 表，状态依赖战略信号 | 候选学习控制器，5 个独立训练 seed |
| RH4 | 历史滚动窗口 H=4 基准、相同物理和核算 | 计算与性能的辅助参照，不进入主要六项检验 |

CAL 只在验证集上，从 \(a_n,a_f\in\{-0.5,0,0.5\}\) 的 9 个组合中选均值最高者，所有 hub 共享该对信号，但各自仍使用冻结 \(s_u\)。严格平局时依次选零信号、较小平方范数、near/far 数值字典序。每个工况选一次并冻结。它是固定策略，释放时间插值随规则变化，但不依赖订单和车队状态。

CAL 与 PPO 的验证候选数和计算成本并非完全相等，应披露，不能声称严格等预算算法竞赛。CAL 的价值在于提供一个低复杂度解释性对照。

RH4 使用实际公开信息版本，不允许读取未来订单。若历史实现涉及其他终端值或规则，须逐项登记，并将其保留为辅助基准；不得冒充与 PPO 只有窗口长度不同的干净消融。

## 5. 数据、预算与选模：先隔离正式样本，再开始学习

### 5.1 拟采用的数据划分

以下范围是本轮建议的新 seed 方案。执行前必须与历史清单做 seed/内容身份审计；本文不宣称这些范围已经完成无冲突核验。

| 角色 | 每工况 scenario seeds | 数量 | 可用于什么 |
|---|---|---:|---|
| 工程接入 | 18000–18003 | 4 | 物理一致性、奖励、解码控制性和耗时 |
| Pilot 训练 | 18100–18107 | 8 | 只验证小规模训练流程和资源预算 |
| Pilot 验证 | 18108–18115 | 8 | 验证选模流程，不比较多组超参数 |
| 正式训练池 | 20000–20255 | 256 | 每次训练从中有放回抽取完整 episode |
| 固定验证集 | 21000–21031 | 32 | PPO 检查点选择和 CAL 选择 |
| 封存正式评估集 | 22000–22079 | 80 | 所有策略、所有训练 seed 的共同配对评估 |
| 机制诊断 | 23000–23015 | 16 | 正式模型冻结后检查信号、动作分歧和接入行为 |

各工况使用相同编号便于管理，但不把跨工况的同号样本当作同一物理场景，也不合并成一个总体。

正式 training seeds 为 **30001–30005**，pilot 使用 29999。scenario seed 与 training seed 是不同随机源；默认正式推理为 deterministic，不再给策略采样随机种子。若未来增加 stochastic evaluation，应单独预注册，不能替代主结果。

同一工况五次训练共享训练/验证场景，分别使用独立训练随机性。正式 80 个场景对所有模型完全相同。检查 ID、seed、去掉标签后的内容 hash，避免同一场景改名后跨集合使用。

训练入口可以只收到 train/validation 数据。正式集应另行封存，不传给训练器；选好所有模型和 CAL 后，再交给 `evaluate`。原始 CFS 抽样仍是模拟场景：新 seed 并不意味着获得了新的独立现实交通日，报告必须区分场景泛化与真实世界外推。

### 5.2 单一首选训练配置

正式首轮不做超参数网格搜索。以下是当前 `PPOConfig` 已支持的字段；它只是完整 run config 中的 `ppo` 部分，不能单独作为完整配置使用。

```json
{
  "updates": 128,
  "episodes_per_update": 8,
  "epochs": 4,
  "minibatch_size": 64,
  "hidden_size": 64,
  "learning_rate": 0.0003,
  "clip_ratio": 0.2,
  "gae_lambda": 0.95,
  "gamma": 1.0,
  "value_coefficient": 0.5,
  "entropy_coefficient": 0.0,
  "reward_scale": 1000.0,
  "max_grad_norm": 0.5,
  "target_kl": 0.03,
  "training_seed": 30001,
  "episode_timeout_seconds": 600,
  "validation_every": 16
}
```

每次正式训练为 1,024 个完整训练 episode，最多 8 个验证检查点，每次验证 32 个场景。256 个训练实例是有放回重复采样，**1,024 个训练 episode 不等于 1,024 个独立训练场景**。

学习率、预算和网络规模只是首轮起点，不是已经针对 CFS 调好的最优值。初始 Actor 均值为零，但随机策略方差非零，训练首个 episode 不等于纯 MC。当前代码也没有把未训练的零策略自动纳入最佳检查点候选；因此 MC 应单列，不能声称选模机制保证不劣于 MC。

选模保持现有规则：验证集 deterministic 平均 USD 最高的已训练检查点，严格平局保留较早者。不在正式集上选择训练 seed，不只报告五次训练中最好的一次。

### 5.3 运行量与资源门

按上述配置，主要运行量为：

| 内容 | 数量 |
|---|---:|
| 正式训练 | 3 工况 × 5 seeds × 1,024 = 15,360 episodes |
| 训练期间验证 | 3 × 5 × 8 × 32 = 3,840 episodes |
| CAL 验证选择 | 3 × 9 × 32 = 864 episodes |
| 正式评估 | 3 × 80 ×（5 PPO + MC + CAL + M0 + RH4）= 2,160 episodes |
| 上述合计 | **22,224 episodes** |

这个数字不包括 pilot、工程扰动、多次编译核验和机制诊断。不能据此直接承诺几小时完成：MILP 耗时、状态规模、冷却和写盘开销需要实测。H=4 与 H=1 的一个 episode 也不能当作相同算力。

先执行每工况一次 4 updates、每 update 4 episodes、每次验证 8 scenes 的小 pilot；不从多套 pilot 中挑高收益配置。记录每类运行的均值、p95、最大耗时、求解次数和输出字节，估算完整预算与磁盘空间。

**预算门：**完整预算、最长预计运行时间和磁盘需求经确认后，才启动正式训练。若资源不足，应在读取正式结果之前缩减并重新登记预算；不能悄悄少跑某个亏损 seed，或因为收益不好不断追加 updates。

默认一次只运行一个训练作业，solver 单线程；外层并行只能在 pilot 证明内存和许可证允许、且资源预算明确后启用，最多 3 个数值 worker。保留 `PYTHONHASHSEED=0`、`PYTHONIOENCODING=utf-8`、`PYTHONDONTWRITEBYTECODE=1`、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。

## 6. 执行顺序：科学门与工程门分别验收

### 阶段 A：准备与冻结

1. 新建本轮独立 attempt，不覆盖合成 smoke、旧训练、前七轮结果或原 CFS 数据。
2. 补齐第 2.2 节实验层工具和必要安全检查，完成相关测试。
3. 从只读历史来源导出新 CFS 场景，登记所有物理参数、初始车队、分割和来源 hash。
4. 校验冻结表完整性、动作尺度、特征单位以及三工况分别运行的物理签名。
5. 记录协议、代码、依赖、配置、baseline、数据、统计程序和输出清单的 hash。

检查点内建 source identity 覆盖 single、experiment_core 及依赖文件，但并不等同于覆盖所有外层调度和统计脚本；后者必须另外冻结。正式训练期间不要改动其所校验的代码，否则恢复与评估可能被正确拒绝。

### 阶段 B：真实 CFS 工程门

对每工况 4 个工程场景检查：

- **零残差同一性：**新 MC 控制路径与同表的既有 value-MILP 路径得到相同整数动作、状态转移和 realized ledger；M0 另做零表回归。
- **信息隔离：**修改未到达的隐藏订单，不改变同一公开状态的 observation 和当前 deterministic 动作。
- **执行一致性：**复制状态与 live 状态编译相同；无容量、承诺、时间或资源守恒违例。
- **奖励一致性：**逐步奖励求和与 full-cooldown realized profit 差额在冻结核算容差内；不比较浮点 solver objective 来替代实际现金核验。
- **控制性：**沿零信号轨迹，对各轴正负扰动，记录信号改变、系数改变和实际动作改变，三个层面分别统计。
- **终止性：**所有承诺完成且冷却不越界；timeout 和 solver 异常不会写成自然完成。

`preflight` 目前沿其数据的 train split 执行；不要把正式训练池全部送进去当成廉价检查。五 hub 下每期包括基准 probe、20 个轴扰动和零信号实际执行，成本明显高于一次普通 episode。应准备专用工程数据。

没有动作改变不等于代码一定错误，也可能是候选退化；但在三个工况均无可控证据前，不应启动完整 PPO 训练。

### 阶段 C：小 pilot 与预算确认

只检查训练、验证、检查点、恢复、日志和预算。核验 deterministic 推理可复现，恢复后的 actor 与不中断运行一致。不用 pilot 利润决定“只保留有希望的工况”。

当前评估器使用 `StrategicEpisode` 默认 600 秒 episode timeout，而训练器读取配置。本文统一设为 600 秒，避免首轮不一致；如果 pilot 说明不足，必须同时明确修改训练与评估限制，不能只改训练参数。

### 阶段 D：15 次正式训练

按预登记顺序完成 3×5 作业；每个作业拥有独立输出目录。固定训练预算，不凭训练曲线看起来变差临时早停，也不延长表现较好的 seed。

保存每个 update 的训练指标和所有 episode 原始轨迹，记录完整验证序列及最终选模依据。允许使用现有 update 边界恢复功能，但须登记恢复位置；不能把恢复当作换学习率或换数据的手段。

### 阶段 E：封存模型后的共同正式评估

先冻结 15 个 best-inference 检查点、3 个 CAL 信号和全部对照配置，再开放正式集。所有 80 场景逐一配对执行，默认 deterministic PPO。训练 seed 不以正式利润排名后筛选。

随后使用单独诊断集，在**同一公开快照**上比较 PPO、零残差和 CAL 的信号、候选系数与整数动作。不同策略各自闭环轨迹上的动作差异不能直接当作同一状态下的决策差异；诊断必须注明状态来源。

### 阶段 F：审计、统计与交付

完成全部原始轨迹重放、统计重算、测试和图表审阅后，才能将整个科学 attempt 标记为 COMPLETED。单个 `train` 输出的 COMPLETED 仅代表该训练作业完成，不代表研究结论成立或整个实验通过验收。

当前 CLI 只有 `preflight/train/evaluate`，没有已实现的科学 `--stage all` 或统一 `delivery`。实施者可以另建调度器，但不能按不存在的入口启动任务，也不能复用历史实验的写入口。

## 7. 统计与成功标准：训练 seed 不是可挑选的结果

### 7.1 样本单位

设 \(Y_{rj}\) 为第 \(r\) 个训练 seed 在正式场景 \(j\) 的利润，\(B_j\) 为同场景对照利润。主要效应是：

\[
\widehat\Delta=\frac{1}{5}\sum_{r=1}^{5}\frac{1}{80}\sum_{j=1}^{80}(Y_{rj}-B_j).
\]

每工况的主要数据是一个 **5×80 的交叉设计**，不是 400 个相互独立样本。五个模型共享测试场景，80 个场景又共享每个模型；不得把所有 episode 平铺后做独立样本 t 检验。24 个时刻更不是 24 个独立利润样本。

### 7.2 主要比较与不确定性

主要 family 固定为六项：三个工况各做 PPO−MC 和 PPO−CAL。M0、RH4、服务率和机制图为辅助结果，不事后加入或移出主要 family。

建议预注册以下直接、透明的统计方案：

- 每项报告 USD 平均差、五个 seed 各自的场景平均差、逐场景差分分布和配对胜率。
- 对 training-seed 维和 scenario 维分别有放回重抽样，取重抽样交叉矩阵的均值；20,000 次，统计 RNG seed 固定为 31000，各对照保持相同场景索引以维持配对。
- 报告边际 95% percentile 区间；主要六项同时报告 Bonferroni 名义校正区间，单项置信水平为 \(1-0.05/6=99.1667\%\)。不把边际区间为正写成已通过 family 控制。
- 只有 5 次训练，尾部不确定性估计有限；上述为**名义上的 bootstrap 多重比较方案**，不声称有限样本精确覆盖。另列五个训练 seed 的完整结果，让读者看到结论是否由单次幸运训练驱动。

不必为了生成 p 值而临时换检验。若最终采用其他统计方法，应在正式评估前替换协议并说明理由，不能根据显著性择法。

### 7.3 怎样解释结果

**某工况取得较强初步成功**，要求同时满足：

1. PPO−MC 和 PPO−CAL 的主要校正区间下界均大于 0；
2. 至少 4/5 个训练 seed 相对 MC 的平均利润差为正；
3. 无物理、信息、核算或检查点审计失败，成本在确认预算内。

第二条是预设的稳健性描述门槛，不是独立显著性检验。若只有均值正但区间跨零，应写“方向正、精度不足”，不能写“PPO 已胜出”；若相对 MC 改善而相对 CAL 没有证据，则不能确认收益来自复杂的状态依赖学习。

W80 另设非劣保护：允许退化幅度预先设为
\(\delta=0.01\max(|\overline J_{\mathrm{MC,validation}}|,1000\ \mathrm{USD})\)，
在正式评估前记录其具体 USD。用 PPO−MC 的单侧 95% bootstrap 下界大于 \(-\delta\) 检查。它是单独的保护终点，不代替六项优越性比较，也不等于“没有显著下降所以安全”。

只有三工况分别满足成功标准，才谈本轮范围内的跨工况一致改进。否则按工况陈述，不把赢家平均后掩盖 CA 或 W80 的损失。

### 7.4 不以“显著与否”代替诊断

若失败，按证据区分：

| 观察 | 可以支持的解释 | 不能直接推出 |
|---|---|---|
| 信号变化但整数动作几乎不变 | 当前尺度/动作接口/候选集合的控制性不足 | PPO 学习算法必然无效 |
| 训练提高、验证不提高 | 过拟合或分布敏感，需结合诊断 | 必须更大网络 |
| CAL 与 PPO 类似 | 固定校准可能已解释主要收益 | 一切状态依赖控制无价值 |
| seed 间差异大 | 优化或样本方差较大 | 挑最好 seed 就是可靠策略 |
| critic 误差下降但利润不升 | 再现预测与控制不一致 | 更低 value loss 等于控制进步 |
| 全部审计通过且明显亏损 | 当前表示、底座与预算下的负向控制证据 | 已经证明所有 RL 不适用 |

## 8. 记录、图表与审计：保证第一轮可解释，而不仅是能跑

### 8.1 必须保存的证据

- **每次运行：**condition、scenario/训练 seed、split、数据/源码/配置/baseline/checkpoint hash、开始结束时间、退出状态、恢复记录。
- **每个 update：**训练和验证均值、policy/value loss、entropy、KL、clip fraction、gradient norm、KL early-stop、signal saturation、实际 minibatch 数、实际 episode 和决策步数。
- **每个 episode：**完整 realized profit、收入/各项成本、服务/失约、full cooldown 长度、求解与总耗时，以及失败原因。
- **每个决策时刻：**公开观察或可重建它的状态、原始信号、实际 USD 修正、基础值/修正值/最终值、查询时刻/hub/q、lookup 模式、整数动作、承诺变化、现金增量、累计现金。
- **选模：**所有候选检查点验证结果、选中 update、严格平局规则、独立复算证据；保存 last-inference，但不事后用它替换主模型。

现有 transition 记录包含 observation hash，并不自动意味着完整观察像素般原样保存；若观察未单独序列化，必须用冻结数据、状态和代码实际证明可重建，不能把一个 hash 当作充分可复现实验记录。

策略分布的 entropy 与经 MILP 解码后的动作多样性不是同一件事。需同时统计近边界信号比例、非零修正比例、真实整数动作改变率，以及同状态诊断中的 PPO−MC、PPO−CAL 分歧。

不沿用旧 B 的 bucket fallback 术语混称本轮行为。本轮 FrozenBaseline 无静默缺 q 回退；端点夹持、插值、终端归零和未知支持计数须分别报告。训练注入 q 与在线 release-prefix q 的语义问题仍存在，不能因为新控制器运行成功就说它已解决。

### 8.2 第一版图表只做六组

1. 训练与验证曲线：横轴累计环境决策步/训练 episode，五个 seed 不隐藏；MC 水平作为参照。
2. 主要利润差：三工况 PPO−MC、PPO−CAL 的效应和区间，同时显示训练 seed 分散程度。
3. 正式利润分布：PPO、MC、CAL、M0、RH4 的配对场景表现；不只画均值柱形图。
4. 战略信号与真实动作：选定诊断场景的 near/far、USD 修正、系数和动作变化联图；选图规则预先固定。
5. 服务与资源过程：固定诊断场景下订单池、忙闲/释放与累计 realized profit，不以最好场景代表全部。
6. 计算与稳定性：训练总成本、在线求解耗时分布、KL early-stop、饱和度和失败计数。

输出论文用 PDF/SVG、600 dpi PNG，并保存每张图的 source data 和 plotting config。选择诊断图时建议按场景编号取预定样本，另补全样本汇总；不能先看结果再只挑最漂亮轨迹。

### 8.3 停止条件与最终验收

出现下面任一情况，立即停止相关 attempt、保留证据，不静默改协议继续混跑：

- 隐藏未来泄漏、数据集合重叠或来源 hash 不一致。
- 非有限 observation、signal、log probability、loss、KL、gradient 或 reward。
- 动作不合法、容量/承诺/资源守恒错误、整数现金核算不一致。
- solver 未达到解码器要求、episode 超时、冷却无法完成。
- 检查点身份不一致、无有效更新却被记录为成功训练。

普通有限 KL 提前结束一个更新、负利润、统计不显著，不属于代码故障。不能删除亏损样本，不能把失败运行直接当缺失值后只统计幸存结果。

最终科学交付至少要求：数据与信息隔离通过、CFS 零信号回归通过、所有完成 episode 的核算与状态重放通过、选模可复算、主要统计独立重算、相关最终测试真实通过、图表真实视觉审阅、历史来源只读核验，以及完整产物 hash 清单。预检 XML 不能冒充本轮最终 QA。

完成后冻结本轮目录；修复真实数值错误必须创建新 attempt，不覆盖旧结果。前七轮已冻结目录始终只读。

## 9. 交付方式与下一步判断

建议在现有实验包下新建独立科学运行目录，例如：

```text
experiments/cfs_ppo_strategic_control/
  README.md                     # 已有工程入口说明
  ...                           # 已有源码和合成 smoke，保留
  runs/cfs_first_ppo_v1/         # 本文建议的新目录，尚未创建
    protocol/                   # 预注册、配置、统计规则、版本与 hash
    data/                       # 新 CFS splits、来源和身份清单
    baselines/                  # 只读底座引用/副本及尺度
    preflight/                  # CFS 工程门与 pilot
    training/<condition>/<seed>/
    validation/
    evaluation/<condition>/
    diagnostics/
    analysis/
    figures/
    qa/
    ppo_first_cfs_report.md
    final_manifest.json
```

其中新目录、文件名和科学调度层均为建议结构，不是声称当前已生成。每次调用现有 CLI 的 `--output` 必须指向不存在的新子目录；不能提前建空输出目录或复用另一次运行结果。

最终中文报告应集中回答：PPO 是否优于 MC；是否优于 CAL；收益依赖哪些工况和训练 seed；策略是否真正改变整数动作；训练成本是否值得；仍有哪些表示和数据局限。

若出现可信收益，下一步优先验证跨 construction bank 和新数据分布的复现，而不是马上扩大网络。若收益有限但控制性和训练都正常，应对照 CAL、动作支持和 q 语义诊断，再决定调整表示或训练预算。若存在明确负向证据，也应保留它：本轮的任务是建立可检验的学习控制基准，而不是为 PPO 制造一个赢家叙事。

**我的建议是：先完成这一个协议清楚、预算有限、对照充分的 PPO 实验。它比同时试很多“更强”的网络，更能告诉我们接下来该改哪里。**

## 附录：本设计依据与阅读入口

以下为本机阅读入口；引用它们不意味着本轮已执行。移动仓库后需相应更新本地链接。

- [当前 PPO 研究入口与限制](D:/mycodelife/workshop/AI/DRL_CO_new_model/DRL_CO_new_model_20261002/_review_repo/experiments/cfs_ppo_strategic_control/README.md)
- [当前工程 QA 记录](D:/mycodelife/workshop/AI/DRL_CO_new_model/DRL_CO_new_model_20261002/_review_repo/experiments/cfs_ppo_strategic_control/QA.md)
- [相关测试实际 XML](D:/mycodelife/workshop/AI/DRL_CO_new_model/DRL_CO_new_model_20261002/_review_repo/workspace/ppo_strategic_control/qa/pytest-regression-03.xml)
- [PPO 配置约束](D:/mycodelife/workshop/AI/DRL_CO_new_model/DRL_CO_new_model_20261002/_review_repo/model_families/single/learning/ppo/config.py)
- [战略动作与版本契约](D:/mycodelife/workshop/AI/DRL_CO_new_model/DRL_CO_new_model_20261002/_review_repo/model_families/single/strategic/contracts.py)
- [释放价格修正与 H=1 解码器](D:/mycodelife/workshop/AI/DRL_CO_new_model/DRL_CO_new_model_20261002/_review_repo/model_families/single/strategic/decoder.py)
- [episode、执行与终止](D:/mycodelife/workshop/AI/DRL_CO_new_model/DRL_CO_new_model_20261002/_review_repo/model_families/single/strategic/episode.py)
- [训练器与验证选模](D:/mycodelife/workshop/AI/DRL_CO_new_model/DRL_CO_new_model_20261002/_review_repo/model_families/single/learning/ppo/trainer.py)
- [PPO 更新与数值检查](D:/mycodelife/workshop/AI/DRL_CO_new_model/DRL_CO_new_model_20261002/_review_repo/model_families/single/learning/ppo/backend.py)
- [第七轮最终报告](D:/mycodelife/workshop/AI/DRL_CO_new_model/DRL_CO_new_model_20261002/_review_repo/experiments/cfs_joint_action_value_diagnostic/joint_action_value_diagnostic_report.md)
- [前七轮研究汇总](D:/mycodelife/workshop/AI/DRL_CO_new_model/DRL_CO_new_model_20261002/_review_repo/reports/七轮数值实验研究汇总_导师交流稿.md)
- [先前 PPO 隔离架构设计](D:/mycodelife/workshop/AI/DRL_CO_new_model/DRL_CO_new_model_20261002/_review_repo/reports/PPO模块化架构设计_隔离边界与接入方案.md)

本文保留了前七轮“不能以更低预测误差替代更高闭环利润”的判断，但实验方案是基于当前代码重新制定，并未把之前的任何模型建议当作已被证明的事实。
