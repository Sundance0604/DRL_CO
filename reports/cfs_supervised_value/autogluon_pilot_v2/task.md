# Codex 任务：监督机器学习资源价值实验 I

## 1. 研究目标

此前 DRL_CO 已经完成：

- MC Table 资源边际价值估计；
- Chebyshev、Cubic Spline、Ridge 经典函数逼近；
- MC 局部残差修正；
- MC 策略迭代；
- PPO、Direct PPO 和结构化 PPO20。

目前 Pure MC 仍然是非常有竞争力的强基准，而 PPO 尚未建立稳定收益优势。

本轮暂时停止强化学习实验，转而研究一个更基础的问题：

**能否通过监督机器学习，利用当前公开运营状态，改善车辆资源边际价值的估计，并进一步改善 MILP 的实际调度利润？**

本轮使用 AutoGluon Tabular。

不再手工逐个调试 XGBoost、LightGBM、CatBoost 等基础模型，而是让 AutoGluon 使用其自动模型训练和集成机制。

本次为轻量探索，不追求一次性证明机器学习优于 MC。

---

## 2. 数学定义

当前 MC Table 估计：

$$
V_{\mathrm{MC}}(t,h,q)
=
\mathbb E[\Delta J^\pi\mid t,h,q],
$$

其中 $\Delta J^\pi$ 是通过配对反事实计算的资源边际回报。

我们希望进一步估计：

$$
\boxed{
V_\theta(o_t,h,\tau,q)
=
\mathbb E[\Delta J^\pi\mid o_t,h,\tau,q]
}
$$

这里：

- $o_t$：决策时公开的运营状态；
- $h$：查询枢纽；
- $\tau$：未来资源释放时刻；
- $q$：该释放组的资源槽位；
- $\pi$：明确冻结的后续控制策略；
- $\Delta J^\pi$：两个配对反事实世界的完整冷却利润差。

注意区分当前时刻 $t$ 和资源释放时刻 $\tau$，以及 MC 资源注入数量与在线 MILP 槽位序号之间已有的语义差异。

不得在新模型中悄悄重新定义这些量。

### 需要特别注意历史实验

第三轮使用 MC 边际标签进行经典函数逼近。

第四轮引入过 action-conditioned local correction，但其部分标签采用 freeze-until-release 机制。

第五轮发现 freeze 与真实 rolling continuation 可能明显改变动作价值排序。

因此，本轮必须先核查现有标签的完整来源，不能把不同 continuation policy、不同资源价值语义或系统动作优势混在同一个监督学习数据集中。

如果可用标签的语义无法对齐，必须明确报告，不得强行训练。

---

# 3. 本轮比较四种方法

只使用一个工况：

**W40/8：Western Multicluster，40个订单，8辆车。**

选择 W40 的原因是该工况存在较多实际可行的多资源槽位结构，适合研究状态条件化的资源价值。

| 编号 | 方法 | 目的 |
|---|---|---|
| M0 | Myopic，未来价值为零 | 最低基准 |
| M1 | 冻结 Pure MC Table | 强基准 |
| M2 | AutoGluon Basic | 学习基础时间、枢纽、q 信息 |
| M3 | AutoGluon State | 额外学习当前公开运营状态 |

本轮只新训练两个 AutoGluon 模型。

暂不加入：

- PPO / SAC / DQN；
- GNN、Deep Sets、Transformer；
- 单独的大型 MLP；
- 新的 MC 策略迭代；
- 复杂贝叶斯优化；
- 多层大规模 Stacking；
- 新的动作相关残差模型。

如果 AutoGluon 自动训练了神经网络基模型，可以在 leaderboard 中保留其结果，但不额外启动深度学习实验。

## 3.1 M2：AutoGluon Basic

输入仅包含现有基础资源价值查询所需的信息：

$$
x_{\mathrm{basic}}
=
(t,\tau,h,q).
$$

若 MC Table 实际使用的基本变量比上述不同，按真实冻结接口对齐。

拟合：

$$
\widehat V_{\mathrm{basic}}
=
f_{\mathrm{AG}}(x_{\mathrm{basic}}).
$$

作用：

检验 AutoGluon 在不增加运营状态信息的前提下，能否通过平滑、泛化或有限样本估计改善 MC 的价值预测。

## 3.2 M3：AutoGluon State

在 Basic 特征基础上，加入决策时已经知道的运营信息。

建议优先使用：

- 当前时间及剩余运营时间；
- 查询枢纽、资源槽位；
- 各枢纽当前空闲车辆数；
- 已承诺车辆的预计释放数量；
- 各枢纽已知 backlog 数量；
- 即将过期订单数量；
- 当前待服务订单负载；
- 历史训练集估计的需求强度；
- 各枢纽供需失衡摘要。

定义：

$$
x_{\mathrm{state}}
=
(x_{\mathrm{basic}},z(o_t)),
$$

并拟合：

$$
\widehat V_{\mathrm{state}}
=
f_{\mathrm{AG}}(x_{\mathrm{state}}).
$$

第一轮控制在约10–20个具有明确经济含义的特征，不进行大规模自动特征组合。

其中历史需求强度只能通过训练数据估计，不能使用评价场景的真实未来订单。

**禁止使用：**

- 未发生的真实订单；
- 未来真实需求；
- 本次 MILP 最优解产生后的状态；
- 完整日最终利润；
- 标签计算所需的未来随机种子；
- scenario ID 作为模型输入；
- 任何只有执行动作之后才知道的信息。

---

# 4. 第一阶段：先审计已有 MC 训练数据

这是本轮最重要的预检。

请优先查找第三至第六轮已经保存的：

- 原始 paired MC marginal-return records；
- construction、validation 场景；
- 公开运营状态快照；
- continuation policy 配置；
- MC Table 的训练来源；
- 资源槽位与价格接口；
- 数据哈希和场景身份。

不要重新生成已经存在且可安全复用的数据。

需要确认已有记录能否构建：

$$
(x_i,Y_i)
$$

其中：

$$
Y_i=\Delta J_i^\pi.
$$

如果同一个公开状态有多次配对 MC 采样，可以保留原始样本与平均标签，但必须记录每个标签的 MC 重复次数和估计噪声。

特别强调：

**如果只有已经汇总好的 MC Table，而没有对应公开状态的独立边际回报标签，不得把同一表格值复制到不同状态上，伪造状态条件化训练集。**

如果现有数据不能支持 M3，可在预先限定的预算内补充少量标签。

但不能为了本轮训练 AutoGluon，就重新执行大规模 MC 数据生成。

### 预检输出

生成：

`supervised_data_audit.md`

记录：

- 可用监督样本总数；
- 独立 scenario group 数；
- 不同 $(t,h,q)$ 的覆盖；
- q≥2 样本数；
- 标签均值、方差、极端值；
- 当前状态特征可用性；
- 标签对应的 continuation policy；
- 与 Pure MC Table 的数据来源关系；
- 训练、验证和评价是否存在交叉。

样本规模要同时按记录数和独立场景数报告。

如果标签来源混乱、场景隔离失败，或者样本数量明显不足以训练与比较，应停止正式数值阶段，提交预检报告。

---

# 5. 第二阶段：训练两个 AutoGluon 模型

## 5.1 数据划分

按照完整 scenario group 划分数据，而不是随机拆分逐条记录。

同一完整仿真日及其共享随机未来、配对反事实样本，不得同时进入训练集与验证集。

建议：

- 训练：约80%的独立 construction scenario groups；
- 验证：约20%的独立 construction scenario groups；
- 后续闭环开发评价：完全不同的场景。

上述比例按独立场景组计算，而不是按数据行数。

在训练前冻结具体场景 ID、哈希和划分。

若样本过少，不应为了满足比例而让同一场景的数据泄漏到其他集合。

## 5.2 AutoGluon 配置

本轮主要使用已有安装环境中的：

`autogluon.tabular`

先检查实际版本和可用模型依赖。

建议采用轻量设置：

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label="target",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    path=model_path,
).fit(
    train_data=train_df,
    tuning_data=val_df,
    presets="medium_quality",
    time_limit=180,
    num_bag_folds=0,
    num_stack_levels=0,
)
```

代码仅供示意；应根据当前安装版本的官方接口核验参数。

这里故意不启用多层 Bagging / Stacking，以避免轻量实验发生计算预算失控或内部随机拆分造成场景泄漏。

允许 AutoGluon 构建默认的轻量 Weighted Ensemble。

如果实际版本和配置不支持这种组合，应采用等价且经核验的轻量设置，并完整披露差异。

**注意：本轮只能检验 AutoGluon 轻量配置，不能将其结果解释为 AutoGluon 全部模型与集成机制的性能上限。**

两个模型必须使用：

- 相同监督标签；
- 相同训练 scenario groups；
- 相同验证 scenario groups；
- 相同 AutoGluon 训练时间上限；
- 相同模型选择指标；
- 相同随机性控制原则。

唯一主要区别是是否加入公开运营状态特征。

不使用新的正式测试利润来选择超参数。

## 5.3 训练预算

每个 AutoGluon 模型最多训练180秒。

总 AutoGluon fit 预算：

$$
2\times180=360\text{ 秒}.
$$

不采用 extensive hyperparameter tuning，不启动 `extreme_quality` 或大型基础模型组合。

训练后保存：

- AutoGluon predictor；
- `leaderboard()`；
- `model_best`；
- 各基础模型验证指标；
- Weighted Ensemble 的信息；
- 训练时间；
- 推理时间；
- 模型磁盘占用。

如果最佳模型不是 Weighted Ensemble，也如实报告，不强制选择集成模型。

---

# 6. 第三阶段：监督预测质量比较

先不运行 MILP。

在完全隔离的监督评价样本上，比较：

- MC Table；
- AutoGluon Basic；
- AutoGluon State。

指标至少包括：

$$
\mathrm{MAE},\quad
\mathrm{RMSE},\quad
R^2.
$$

另外重点计算：

- 相同状态下不同资源候选的价值排序；
- 不同枢纽的相对价值排序；
- q≥2 的预测误差；
- 预测符号错误率；
- 资源价值极端值；
- 不同状态下的预测变化。

如果监督评价标签来自有限 MC 采样，应报告其噪声或标准误。

不能把带噪声 MC 标签当成无误差的真实 Bellman 价值。

### 关键消融

比较：

$$
\widehat V_{\mathrm{basic}}(t,h,q)
$$

和：

$$
\widehat V_{\mathrm{state}}(o,t,h,q).
$$

本轮最重要的机器学习问题是：

**加入公开运营状态信息，是否确实改善了资源边际价值的条件估计？**

AutoGluon State 即使拥有更低的 RMSE，也还需要通过下一阶段 MILP 验证。

---

# 7. 第四阶段：接入原 MILP

保持原 H1 single-level MILP 完全不变。

不同方法唯一改变资源价值来源：

```text
M0:
    resource_value = 0

M1:
    resource_value = frozen_MC_table(...)

M2:
    resource_value = AutoGluon_Basic.predict(...)

M3:
    resource_value = AutoGluon_State.predict(...)
```

要求：

1. 所有 Value Provider 返回实际美元单位。
2. 保持原有 terminal gate。
3. 保持原有 supply-slot/prefix 语义。
4. 不改变 MILP 约束或真实现金核算。
5. 同一查询在同一公开状态下应返回确定且可复现的价格。
6. 不根据测试利润进行价格尺度校准。
7. 如需限制异常预测值，必须只使用训练数据或已知经济约束预先定义范围，并记录限制触发次数。
8. 将模型预测转换成求解前的固定 MILP 目标系数，不引入神经网络约束。

建议批量构造特征、批量预测，并缓存重复的同状态价值查询。

同时测量批量和实际线上调用的推理延迟。

---

# 8. 第五阶段：少量完整运营日对照

只在 W40 执行。

建议使用六个此前未参与训练、验证或正式测试的新开发场景。

可优先检查：

`21028–21033`

但必须先核验历史使用记录与场景哈希。如果已被使用，按预先固定的递增规则选择其他未使用场景。

本轮不触碰已经查看过的正式测试场景。

对每个场景配对运行：

- M0；
- Pure MC；
- AutoGluon Basic；
- AutoGluon State。

总运行量：

$$
6\times4=24\text{ 个完整运营日}.
$$

若每个运营日有24次主动 MILP 决策，主策略评价预计产生：

$$
24\times24=576
$$

次 MILP 调用。

完整运行后保留冷却结算。

### 主要指标

$$
\boxed{
P_T=\text{Full-Cooldown Realized Cash Profit}
}
$$

输出：

| 场景 | M0 | Pure MC | AG Basic | AG State |
|---|---:|---:|---:|---:|
| 1 | 实测 | 实测 | 实测 | 实测 |
| 2 | 实测 | 实测 | 实测 | 实测 |
| 3 | 实测 | 实测 | 实测 | 实测 |
| 4 | 实测 | 实测 | 实测 | 实测 |
| 5 | 实测 | 实测 | 实测 | 实测 |
| 6 | 实测 | 实测 | 实测 | 实测 |
| 均值 | 实测 | 实测 | 实测 | 实测 |

同时报告：

- 每个场景的配对利润差；
- 完成订单数；
- 收入和成本；
- 空驶成本；
- 规范化整数业务决策切换次数；
- q≥2 的有效查询和占用情况；
- 资源价格查询次数；
- 模型推理与 MILP 求解时间。

六个开发场景不用于进行正式显著性论证。

本轮的目的只是判断是否出现值得进一步确认的收益趋势。

---

# 9. 研究中特别需要避免的问题

## 9.1 不要追求机器学习一定超过 MC

允许出现：

$$
P_{\mathrm{AG}}<P_{\mathrm{MC}}.
$$

如果预测改善但利润下降，这是有研究价值的负向结果。

不要因此临时调整模型或继续加训练时间。

## 9.2 不能混淆标签与真实目标

机器学习训练的是：

$$
\mathbb E[\Delta J^\pi\mid x].
$$

但真实调度优化目标是：

$$
\max_\pi\mathbb E[P_T(\pi)].
$$

两者不是同一个数学对象。

尤其不能把资源边际价值 RMSE 当成调度策略质量的充分证据。

## 9.3 不要混淆 MC 监督与 RL

这里的 AutoGluon 通过监督学习拟合反事实资源价值，不使用 PPO、GAE 或策略梯度。

即使最终 AutoGluon 优于 MC，也不能说明强化学习成功。

## 9.4 不要忽视计算成本

MC Table 的一次构建可能成本较高，但在线查表十分便宜。

AutoGluon 需要离线训练，在线预测也有额外耗时。

必须将：

- MC 标签生成；
- AutoGluon 训练；
- 在线模型推理；
- MILP 求解；

的成本分开报告。

不能因为复用了历史 MC 数据，就把该数据的构建成本记为零。

---

# 10. 硬预算与停止规则

这是轻量探索实验。

建议：

- 只执行 W40；
- 最多训练两个 AutoGluon；
- 每模型最多180秒；
- 最多6个闭环开发场景；
- 新增 MILP 求解原则上不超过1,500次；
- 新增数值运行时间控制在15分钟以内；
- 新增产物尽量不超过2 GiB。

新增 MC 标签必须单独记账，并优先复用旧实验资产。

如果原始标签不足，而补充标签预计会导致预算超标，停止训练并提交数据审计结果。

如果预估超过预算，不得静默减少某个表现不利的方法或事后删掉部分场景。

本轮不追加更多工况、训练种子或更复杂架构。

---

# 11. 最终交付

建立独立目录：

`reports/cfs_supervised_value/autogluon_pilot_v1/`

至少保存：

- `experiment_protocol.md`
- `supervised_data_audit.md`
- `dataset_manifest.json`
- `training/`
- `models/`
- `autogluon_leaderboards.csv`
- `prediction_comparison.csv`
- `policy_profit_comparison.csv`
- `decision_switches.csv`
- `runtime_comparison.csv`
- `qa/`
- `final_manifest.json`

最终中文报告：

`autogluon_mc_value_pilot_report.md`

报告主要回答：

1. 既有 MC 反事实数据是否足以训练状态条件化价值模型？
2. AutoGluon 能否准确拟合 MC 边际价值？
3. 加入公开运营状态，是否比只使用 $(t,h,q)$ 更好？
4. AutoGluon 的最佳模型是单模型还是 Weighted Ensemble？
5. 预测误差改善是否出现在 MILP 的关键决策区域？
6. 预测变化是否真正改变整数调度？
7. AutoGluon 能否在六个开发场景中取得比 Pure MC 更高的完整冷却利润？
8. 训练成本、在线推理成本与可能的利润收益是否值得？
9. 下一阶段是否值得投入 Deep Sets、MC Residual Learning 或更多训练场景？

---

# 12. 最终执行要求

请直接阅读已有源码与实验档案，按以下顺序执行：

**阶段一：数据审计**

确认已有 MC 边际标签、公开特征、后续策略语义和训练划分。

**阶段二：AutoGluon 训练**

在预算内训练 Basic 和 State 两个模型，保留真实 leaderboard。

**阶段三：预测评价**

在场景隔离的监督数据上比较 MC、Basic 和 State。

**阶段四：MILP 接入**

保持原优化器与现金核算不变，仅替换资源价值提供器。

**阶段五：轻量运营比较**

在六个新开发场景中运行 M0、Pure MC、AutoGluon Basic 和 AutoGluon State。

**阶段六：研究总结**

分析预测能力、决策变化、运营利润与计算成本。

不要因为 AutoGluon 集成能力较强，就假定其一定改善调度收益。

**本轮核心研究问题：**

$$
\boxed{
\text{公开运营状态信息 + AutoGluon}
\overset{?}{\Longrightarrow}
\text{更好的资源估值与调度利润}
}
$$

这是一项探索性试验。如果数据不足、标签语义不一致或实现预算超限，应明确说明并停止，不要为了完成模型训练而牺牲实验正确性。

本轮结束后，不自动启动第二轮实验。根据实测结果再决定是否研究：

- MC + AutoGluon Residual；
- Deep Sets；
- 更完整的状态条件化资源价值；
- 决策导向监督学习。

请实际执行允许范围内的数值实验、独立核验并生成完整报告，而不是只提交实验设计。