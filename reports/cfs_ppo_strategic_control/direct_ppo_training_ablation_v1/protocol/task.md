# Codex 任务：Direct PPO 训练机制消融与独立验证实验

## 一、任务背景与研究目标

当前 single-level 货运调度模型已完成：

1. 七轮 MC 资源价值与优化实验；
2. MC + PPO 残差控制实验；
3. 动态 PPO 与固定残差信号机制实验；
4. 无 MC 的 Direct PPO 正式训练；
5. Direct PPO 失败机制诊断。

请先阅读以下报告及其引用的实际源码、配置和日志：

- `PPO强化学习训练架构与现状说明_20261010.md`
- `direct_ppo_vs_mc_report.md`
- `direct_ppo_failure_diagnosis_report.md`
- `scheme_b_architecture_proposal.md`（若文件存在）

最新诊断得到以下结果：

- Direct PPO 在 CA、W40、W80 的测试利润分别比 Pure MC 低 11.97%、14.87%、19.88%。
- Direct PPO 在 CA 相对 M0 有一定正向收益，但在 W40、W80 没有建立稳定改善。
- 九个 Actor 的确定性输出对验证状态变化非常不敏感。
- 九个 Critic 的历史 pooled Explained Variance 均为负，约为 −0.246 至 −0.020。
- Critic 对实际剩余利润存在明显的预测偏差。
- 独立事后梯度诊断中，加权 Critic 梯度范数约为 Actor 的 41–101 倍。
- PPO 随机动作可以改变部分 MILP 决策，尤其 W40 的随机动作切换率达到 34.7%。
- 部分反事实实验显示，当期现金收益和长期利润的变化方向可能相反。

这些结果说明学习质量值得优先调查，但**尚未证明 Critic 梯度压制 Actor、GAE 是主要失败原因，或现有动作空间已经充分**。

### 本轮核心问题

在完全不改变 MILP 和 10 维战略价格动作接口的条件下：

> 改善价值学习的数值尺度、移除不准确的 Critic 优势估计，或者解耦 Actor–Critic 训练，能否提高 PPO 的状态适应能力与实际运营利润？

本轮重点研究训练机制，而不是继续扩大动作空间。

**请实际完成代码实现、模型训练、配对测试、统计分析和报告。不要仅提交实验设计。**

---

## 二、保持不变的系统结构

继续使用无 MC 的 Direct PPO：

$$
o_t\xrightarrow{\mathrm{Actor}}a_t
\xrightarrow{\mathrm{Decoder}}V_\theta
\xrightarrow{\mathrm{MILP}}x_t
\xrightarrow{\mathrm{Environment}}r_t.
$$

当前动作：

$$
a_t\in[-1,1]^{10}.
$$

价格解码：

$$
V_\theta(h,\tau,q\mid o_t)
=
S_h[(1-\ell)a_{h,n}+\ell a_{h,f}],
$$

其中：

$$
\ell=\frac{\tau-t}{T-t},\qquad \tau<T.
$$

$\tau\ge T$ 时沿用现有终端规则。

本轮保持：

- 5 个枢纽；
- near/far 共 10 维连续动作；
- 原价格尺度 $S_h$；
- 原公开观测内容；
- 原 H=1 MILP；
- 原订单、车辆、路线、容量和承诺约束；
- 原场景生成、收入成本与现金账本；
- 原 24 时段与完整冷却结算；
- 完全不使用 MC Table 作为 Direct PPO 的输入、标签、价格底座或训练奖励。

**不要在本轮引入 $q$ 维动作、重新设计解码器、修改价格尺度或更换强化学习算法。**

MC Table 仅作为独立的外部比较基准。

---

# 三、四组 PPO 消融实验

本轮比较四个训练方案。

| 方案 | 关键改动 | 研究目的 |
|---|---|---|
| A：Original PPO | 完全保留原实现 | 对照基准 |
| B：Reward-Scaled PPO | 奖励及 Critic 目标缩小 10 倍 | 检验训练数值尺度 |
| C：Episode-Return PPO | 使用完整回报与时间基线，不使用 Critic 估计优势 | 检验 Critic/GAE 依赖 |
| D：Decoupled Actor–Critic | Actor/Critic 独立网络及梯度优化 | 检验共享训练结构的影响 |

除各组明确规定的变化外，其余条件保持一致。

## 3.1 A：原始 Direct PPO

复用已冻结的 `direct_ppo_no_mc_v3`。

原配置：

| 参数 | 数值 |
|---|---:|
| PPO updates | 32 |
| Episodes/update | 4 |
| PPO epochs | 4 |
| Minibatch size | 64 |
| Learning rate | 0.0003 |
| PPO clip | 0.2 |
| $\gamma$ | 1.0 |
| GAE $\lambda$ | 0.95 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0 |
| Max gradient norm | 0.5 |
| Target KL | 0.03 |
| Reward divisor | 1,000 |

现有 Actor 与 Critic 共享集合编码器和 64 维主干。

保留原始 checkpoint、日志、训练轨迹和验证选择规则。

**原则上不重新训练 A。**

如果为了记录真实训练期共享梯度，必须进行一次带日志插桩的基准复现，可在独立目录运行。应验证插桩没有改变随机数状态、优化行为和结果；这类诊断复现不自动增加正式比较中的独立训练 seed 数量。

## 3.2 B：Reward-Scaled PPO

本方案只调整强化学习奖励的数值单位。

原奖励：

$$
r_t^{A}=\frac{\Delta P_t}{1000}.
$$

新奖励：

$$
\boxed{
r_t^{B}=\frac{\Delta P_t}{10000}
}
$$

其余保持：

$$
\gamma=1,\qquad \lambda=0.95.
$$

Actor–Critic 网络、共享主干、优化器、PPO 损失形式及 value coefficient 均保持不变。

Critic 预测与 GAE 均使用新的奖励单位，不能只缩小奖励而保留原单位的价值预测。

每次更新仍然执行原有的优势标准化。

本质上，B 没有改变美元意义下的运营目标：

$$
\sum_t r_t^{B}=\frac{P_T}{10000}.
$$

实际测试仍用 USD 现金利润，不能将缩放回报直接与 A 的训练 loss 数值比较。

### 需要检验

1. Critic 是否更容易学习到合理的回报量级？
2. Value loss 与 Actor loss 的梯度尺度是否更平衡？
3. Explained Variance 是否提高？
4. Actor 的状态依赖是否增强？
5. 最终利润是否改善？

注意：奖励缩放也改变 Critic 数值目标、相对损失尺度及训练动力学，因此即使 B 改善，也不能直接认定唯一原因是“Critic 梯度过大”。

本轮不额外同时修改 value coefficient，避免产生更多混杂因素。

## 3.3 C：Episode-Return PPO

这是本次最重要的干预之一。

当前 Actor 的优势估计依赖 Critic：

$$
\delta_t=r_t+\gamma V(o_{t+1})-V(o_t),
$$

$$
A_t^{GAE}
=
\delta_t+\gamma\lambda A_{t+1}^{GAE}.
$$

由于现有 Critic 拟合较弱，希望测试：**如果 Actor 不再依赖 Critic 的 TD bootstrap，是否能够更稳定地学习长期运营价值？**

### 新优势估计

一个完整 episode 有 24 个主动决策时段，最后一个奖励包含全部冷却结算。

直接计算：

$$
G_t=\sum_{k=t}^{T-1}r_k.
$$

其中本方案仍使用：

$$
r_t=\frac{\Delta P_t}{1000}.
$$

采用只依赖当前时段的基线：

$$
\boxed{
A_t=G_t-b(t)
}
$$

其中 $b(t)$ 是时间基线。

建议：

- $b(t)$ 使用此前已完成训练 episode 的对应时段回报均值；
- 当前 rollout 优化时，只允许使用此前历史形成的基线；
- 当前 rollout 完成 PPO 优化后，再更新基线；
- 初始无历史时取零；
- 基线数据只来自该模型自己的训练场景；
- 不使用验证或测试回报拟合基线。

这样避免当前采样动作直接参与其自身的基线估计。

随后与原 PPO 一样，在 rollout batch 内对优势进行标准化：

$$
\widetilde A_t=
\frac{A_t-\bar A}{\sigma_A+\epsilon}.
$$

继续使用原 PPO-Clip 策略目标：

$$
L_{\mathrm{policy}}
=
-\mathbb E\left[
\min\left(
\rho_t\widetilde A_t,
\operatorname{clip}(\rho_t,0.8,1.2)\widetilde A_t
\right)
\right].
$$

本方案：

- 不训练 Critic；
- 不使用 Value loss；
- 不使用 GAE bootstrap；
- 保留现有 Actor 编码器、主干、动作分布与 PPO-Clip；
- 保留原 log probability、KL、梯度裁剪和 Adam 机制。

可以将该方案命名为 `PPO-Clip with Episodic Returns`。

这里的完整 episode 回报是策略梯度估计所用的蒙特卡洛回报，**不是原研究的 MC Table 资源价值模型**，两者不可混淆。

### 需要检验

- 不依赖 Critic 后，Actor 是否出现更明显的状态响应？
- PPO 的策略改进是否更加稳定？
- 完整回报优势是否带来更高的估计方差？
- 随机训练与确定性验证的差距是否缩小？
- 最终利润是否提高？

本方案同时改变了优势估计和 Critic 训练路径，因此如果 C 改善，只能先判断这一组合干预有效，不能马上归因于 Critic 的某个具体误差。

## 3.4 D：Decoupled Actor–Critic

当前 Actor 与 Critic 共用：

- 订单编码器；
- 车辆与路线弧编码器；
- 301→281 维调整后的观测编码输入；
- 共享 64 维主干。

因此两者的梯度可能作用于相同的表征参数。

本方案将 Actor 和 Critic 分开。

### 网络结构

Actor：

$$
o_t\xrightarrow{f_\theta}
h_\theta(o_t)
\xrightarrow{\mathrm{ActorHead}}
\pi_\theta(a\mid o_t).
$$

Critic：

$$
o_t\xrightarrow{g_\phi}
z_\phi(o_t)
\xrightarrow{\mathrm{ValueHead}}
V_\phi(o_t).
$$

两者不共享可学习参数，包括集合编码器和主干。

要求：

- Actor 保持原有 10 维动作；
- 每条独立网络的隐藏宽度仍为 64；
- Critic 仍使用 GAE $\lambda=0.95$；
- 奖励仍为 $\Delta P/1000$；
- 价值损失系数仍为 0.5；
- 两个网络使用相同 Adam 学习率和超参数；
- Actor/Critic 采用独立优化器；
- 梯度范数分别裁剪至 0.5；
- 不让 Critic loss 反向传播进入 Actor 网络。

这是一项**共享表征与梯度优化同时解耦的组合干预**。

它可以验证完全解耦是否实用，但不能仅凭该方案的利润改善，就断言“共享主干必然有害”。

请明确报告新模型参数量增加了多少，以及额外训练和推理成本。

---

# 四、必须新增的训练诊断

与前一轮事后诊断不同，本次应尽可能在**实际训练更新期间**记录信息。

## 4.1 Actor/Critic 梯度

对于 A、B 等共享网络方案，在选定的训练 minibatch 上分别计算：

$$
g_A=\nabla_{\theta_{\mathrm{shared}}}
L_{\mathrm{policy}},
$$

$$
g_C=\nabla_{\theta_{\mathrm{shared}}}
(0.5L_{\mathrm{value}}).
$$

记录：

$$
\|g_A\|,\qquad \|g_C\|,
$$

以及：

$$
\cos(g_A,g_C)
=
\frac{\langle g_A,g_C\rangle}
{\|g_A\|\|g_C\|}.
$$

这里必须针对**共享编码器与共享主干参数**，不能将 Actor/Critic 各自独立输出头拼入同一个梯度向量后解释梯度冲突。

同时记录：

- 梯度裁剪前的总范数；
- 实际裁剪系数；
- 梯度是否出现 NaN/Inf；
- KL early stopping；
- Actor/Critic 有效优化步数。

为了控制计算量，可以在每次 PPO update 中抽取预先固定的 minibatch 做详细诊断。

若旧 A 无法恢复原训练时刻的分项梯度，必须标记为缺失，不得用事后 checkpoint 梯度冒充历史训练梯度。

## 4.2 Critic 学习质量

对拥有 Critic 的 A、B、D，统一以真实剩余现金回报为参照：

$$
G_t^{USD}
=
\sum_{k=t}^{T-1}\Delta P_k.
$$

记录：

- Critic 预测均值；
- 实际 return-to-go 均值；
- RMSE；
- MAE；
- Explained Variance；
- 分时段误差；
- 不同 PPO update 的变化。

各方案预测必须先转换回 USD 后比较。

对于 C，不存在 Critic，这些指标记为不适用，不得填零。

另外，应分别报告 Critic 对真实 Monte Carlo return-to-go 与 GAE lambda-return 的拟合情况，不能混用两个目标。

## 4.3 Actor 状态敏感性

采用相同的一组固定公开参考状态，对全部方案计算：

$$
D_\pi=
\frac{1}{10N}
\sum_i
\|\pi_\theta(o_i)-\bar a\|_1.
$$

参考状态应来自训练或验证侧，并在比较之前冻结。

同时报告：

- 归一化动作均值；
- 归一化动作标准差；
- 实际战略价格 USD；
- near/far 信号差异；
- 早、中、晚时段的输出变化；
- 不同负载状态的输出变化；
- `log_std` 变化。

使用相同固定状态是为了降低不同策略访问状态分布带来的比较偏差。

但这些状态上的响应变化，不自动意味着策略获得了更高利润。

## 4.4 随机策略与确定性策略差异

训练时：

$$
a_t\sim\pi_\theta(\cdot\mid o_t).
$$

测试时：

$$
a_t=\tanh(\mu_\theta(o_t)).
$$

因此请分别记录：

- 随机训练 episode 的平均利润；
- 确定性验证 episode 的平均利润；
- 冻结 checkpoint 的随机验证利润（固定采样规则，作为辅助诊断）；
- Actor 输出均值与标准差。

特别关注是否存在：

**随机策略逐渐改善，但确定性均值策略反而恶化。**

随机验证仅作为诊断，正式 checkpoint 仍按预登记的确定性验证规则选择。

---

# 五、训练数据、预算与模型选择

## 5.1 数据协议

为实现与原 Direct PPO 的可比性，优先复用：

- 训练场景池：20032–20159；
- 验证场景池：21008–21023；
- 原三个工况；
- 原三个训练 seed：30001、30002、30003。

各方案使用相同训练场景池、验证场景池和预先固定的场景抽样规则。

应尽可能统一对应训练 seed 的 Actor 初始权重；D 可以独立初始化 Critic 分支，但 Actor 初始化仍应与 A 对齐。

训练是 on-policy，策略改变后访问的状态自然可以不同，不要求不同方案实际获得完全相同的轨迹。

正式训练：

| 设置 | 数值 |
|---|---:|
| 工况 | CA、W40、W80 |
| 每工况训练 seed | 3 |
| 新训练方案 | B、C、D |
| 每模型更新次数 | 32 |
| 每次更新 episode | 4 |
| 每模型训练 episode | 128 |
| 验证 checkpoint | 更新 8、16、24、32 |

总计新训练：

$$
3\times3\times3=27\text{ 个模型}.
$$

原 A 的九个冻结模型作为正式对照。

## 5.2 Checkpoint 选择

对每个模型，在更新 8、16、24、32 时执行原有确定性验证。

按验证场景的平均完整冷却利润选择最佳 checkpoint。

并列时选择较早更新。

不得根据正式测试结果选择训练种子、checkpoint 或超参数。

不得因为某方案前期表现较差而临时延长其训练预算。

## 5.3 独立确认性测试集

此前测试种子 22020–22043 已经被使用。

建议本轮新测试种子：

`22044–22067`

每工况 24 个完整运营场景。

实际使用前应检查场景 seed、内容哈希及此前实验档案，保证这些场景没有进入训练、验证、模型选择或历史正式测试。

新测试集在模型、checkpoint、统计规则全部冻结后才能用于正式评价。

同时在新测试场景上重新评价：

- A：原始 Direct PPO；
- B：Reward-Scaled PPO；
- C：Episode-Return PPO；
- D：Decoupled Actor–Critic；
- Pure MC：冻结 MC Table；
- M0：短视基准。

MC 和 M0 不需要重复训练。

所有方法使用相同场景与物理规则。

## 5.4 资源限制

请先执行工程 smoke test：

1. 确认四种模型均能正常加载、推理和训练；
2. 运行少量 episode；
3. 验证 GAE／Monte Carlo advantage 计算正确；
4. 确认 checkpoint 保存和恢复正确；
5. 对训练、验证和推理进行实际计时；
6. 估计完整实验的 MILP 调用量与磁盘占用。

建议设定本轮总数值预算上限为：

- 180,000 次 MILP 调用；
- 240 分钟数值运行时间；
- 15 GiB 新实验产物。

这些是资源保护上限，不是必须用完的预算。

如 smoke test 预测正式实验无法在上限内完成，先优化复用和日志策略，不能擅自改变样本量、训练次数或只完成表现好的方案。

确需修改预算时，在正式数值冻结之前写明原因并重新登记；若运行中触发上限，保留真实完成部分并明确标记 `BUDGET_LIMITED`，不得将不完整实验冒称正式完成。

---

# 六、正式利润比较与统计

## 6.1 主要利润指标

所有方案使用：

$$
P_T
=
\text{完整冷却后的实际现金利润}.
$$

不以 policy loss、Critic MSE 或 MILP 代理 objective 作为主指标。

首先比较新训练方案相对原始 PPO：

$$
\Delta_B=P_B-P_A,
$$

$$
\Delta_C=P_C-P_A,
$$

$$
\Delta_D=P_D-P_A.
$$

其次比较各方案与 Pure MC：

$$
\Delta_{X,MC}=P_X-P_{MC}.
$$

主比较为：

**B、C、D 各自在三个工况中相对 A 的利润差，共九项预登记比较。**

使用相同测试场景进行配对，并考虑三个训练种子与测试场景的交叉依赖。

建议使用：

- 20,000 次交叉 bootstrap；
- 固定 bootstrap seed；
- 三训练 seed 与测试场景分别重抽；
- 九项主比较使用 Bonferroni 调整后的区间；
- 同时报告普通 95% 配对区间。

与 Pure MC 的比较作为单独报告的次要比较族，须明确其多重比较处理方式。

不要将 3 个训练 seed × 24 个场景视为 72 个独立模型或独立运营日。

## 6.2 最终比较表

| 工况 | M0 | Pure MC | A | B | C | D |
|---|---:|---:|---:|---:|---:|---:|
| CA | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| W40 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| W80 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |

全部单位为 USD/完整运营场景。

另提供：

| 工况 | B−A | C−A | D−A | 最优新方案−MC |
|---|---:|---:|---:|---:|
| CA | 实测 | 实测 | 实测 | 描述性 |
| W40 | 实测 | 实测 | 实测 | 描述性 |
| W80 | 实测 | 实测 | 实测 | 描述性 |

若报告表现最好的方案，其身份必须由验证集预先选出；不能根据正式测试成绩事后选择“最优方案”并给予确认性结论。

## 6.3 辅助业务指标

记录：

- 完成订单数；
- 订单收入；
- 总成本；
- 空驶成本；
- 取消及积压罚金；
- 车辆利用率；
- MILP 求解时间；
- PPO 推理时间；
- 训练总时间；
- 失败及异常终止情况。

这些指标用于解释利润差异。

---

# 七、少量动作机制复核

本轮不重复上一轮的大规模动作扰动实验。

只需使用预先冻结的少量验证状态，比较各方案：

1. 确定性动作与零动作是否产生不同 MILP 决策；
2. 随机采样是否跨越整数决策切换区域；
3. 训练后 Actor 是否比 A 更依赖状态；
4. 产生不同整数决策的动作是否具有非零长期经济影响。

固定同状态探针、相同动作采样次数与业务等价判定规则。

如有必要，对少量动作切换案例执行共同隐藏未来、共同后续策略的完整反事实结算。

不要让动作机制实验扩大到挤占正式训练与利润比较的计算预算。

---

# 八、结果解释要求

本次实验应该区分以下几种情况。

### 情况一：B 明显改善

如果 Reward-Scaled PPO 的 Critic 预测、Actor 学习及实现利润同时改善，则表明训练数值尺度是有价值的干预方向。

但不能仅凭此结果判断根因一定是 Critic 梯度过大。

### 情况二：C 明显改善

如果 Episode-Return PPO 的利润超过 A，尤其状态依赖也增强，则说明当前 Critic/GAE 学习链条值得进一步改造。

可在下一轮考虑 $\lambda=1$ 的 GAE、不同 baseline 以及新的 Critic 目标。

注意完整回报估计可能方差更大，改善并不保证普遍存在。

### 情况三：D 明显改善

如果独立 Actor–Critic 训练表现更好，则说明共享表征／联合优化机制值得进一步拆解。

需要进一步区分：

- 参数共享；
- 梯度裁剪；
- 独立优化器；
- 网络参数量增加。

本轮不能把这几个因素的贡献直接合并为一个因果结论。

### 情况四：三种方案都没有改善

不能据此立即否定强化学习。

但这会削弱“只需修正 Critic 训练就能解决问题”的解释。

下一步应优先考虑：

- $q$ 相关的结构化价值网络；
- 更合理的战略动作接口；
- 训练样本效率；
- 状态信息充分性；
- MILP 决策变化与长期奖励之间的统计关系。

### 情况五：利润改善，但 Actor 仍几乎固定

说明可能学到了更合理的固定价格校准，而非真正动态状态反馈。

需要明确区分这两种结果，不应把利润改善自动解释为动态控制学习成功。

---

# 九、正确性与复现要求

## 9.1 不修改历史实验

不能覆盖：

- 七轮历史实验；
- MC+PPO；
- 原 Direct PPO v3；
- 原失败机制诊断；
- 任何冻结训练 checkpoint、日志或测试结果。

建立独立目录：

`reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v1/`

## 9.2 训练数学核验

必须检查：

- 24 个时段的完整回报；
- 终端冷却奖励；
- 奖励望远镜求和；
- B 的奖励／价值预测单位一致；
- C 的完整 return-to-go 与时间基线；
- 当前 rollout 不进入其自身的历史时间基线；
- PPO 旧/新 log probability；
- `tanh` Jacobian 修正；
- 优势标准化；
- KL early stopping；
- 梯度裁剪；
- D 的独立计算图与独立优化器；
- checkpoint 恢复、随机种子与模型推理一致性。

不允许因为某方案训练不稳定，就关闭关键 QA、删除异常场景或扩大求解器容差。

## 9.3 业务合法性核验

继续复用完整测试：

- MILP OPTIMAL 状态；
- 容量与路线；
- 已承诺车辆任务；
- 订单时间窗；
- 实际现金账本；
- 完整冷却结算；
- 独立轨迹回放。

训练、验证和测试都不得让 Actor 读取隐藏未来订单。

---

# 十、交付文件

至少生成：

- `experiment_protocol.md`
- `protocol/preregistration.json`
- `training/`
- `validation/`
- `evaluation/`
- `paired_results.csv`
- `training_diagnostics.csv`
- `gradient_diagnostics.csv`
- `critic_diagnostics.csv`
- `actor_state_sensitivity.csv`
- `action_probe_results.csv`
- `figures/`
- `qa/`
- `final_manifest.json`

最终中文报告：

`direct_ppo_training_ablation_report.md`

至少绘制五组图：

1. 四种 PPO 与 Pure MC 的正式利润比较；
2. 四种 PPO 的确定性验证学习曲线；
3. Critic Explained Variance 随更新变化；
4. 共享主干 Actor/Critic 梯度尺度和方向；
5. Actor 状态敏感性与随机／确定性策略的表现差异。

图表均使用真实实验数据，并保存绘图代码或配置。

---

# 十一、最终报告必须回答的问题

1. 调整奖励尺度，是否改善了 Critic 的学习质量？
2. Critic 的预测偏差和梯度量级是否随训练更新改善？
3. 不依赖 Critic bootstrap 的完整回报优势估计，是否改善了 PPO？
4. Actor–Critic 解耦是否提高了实际利润？
5. 哪种方案获得了更明显的状态依赖，而非仅仅更好的固定价格？
6. 随机训练收益改善，是否能转化为确定性部署收益？
7. 哪些方案真正超过原始 Direct PPO？
8. 哪些方案缩小了相对 Pure MC 的利润差距？
9. 改善是否在三个工况和三个训练 seed 上稳定？
10. 在当前证据下，应该继续改善 PPO 训练机制，还是进入带 $q$ 的方案 B？

结论必须明确区分：

- 实验事实；
- 统计证据；
- 合理但未验证的解释；
- 需要进一步干预才能确定的因果机制。

---

## 十二、执行指令

请从实际代码检查开始，先实现 B、C、D 的最小变体并完成 smoke test，然后运行正式训练与测试。

优先复用原有 Direct PPO 平台、冻结场景池、轨迹回放和统计工具。

不要重新开发整个实验框架，也不要引入新强化学习库或大规模重构。

本轮必须实际执行训练和独立测试，不接受只有架构文档、空白表格或未执行脚本的交付。

**最重要的是保持控制变量：本轮只研究 PPO 的训练机制，不同时改变动作空间、价格尺度、MILP 或运营模型。**

本轮需要检验的核心假设是：

> 当前 Direct PPO 未能学习有效动态战略价格，是否主要由于不准确的 Critic 估计与 Actor–Critic 联合训练机制，而不是战略价格动作接口本身必然无效？

如果训练机制改善能够显著提高利润，就继续深入相应方向。

如果三种干预都不能带来可靠改善，再考虑将研究资源转向更丰富的 $V_\theta(o,h,\tau,q)$ 动作表示。

无论结果正负，均保留全部冻结结果和失败记录，不以必须超越 MC 为实验成功条件。