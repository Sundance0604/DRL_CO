# 第七轮接入语义审查

本文件先于新正式收益生成。结论是源码语义和受控再现，不是政策效果结论。机器证据在 `processed/query_semantics`、`bucket_base_alignment`、`fallback_action_audit` 及原始状态/动作中，运行后才填充。

## 已确认一致的部分

1. `cfs_resource_policy_iteration.protocol.resource_world` 移除查询 hub 的自然 idle 车辆，保留其他 hub 资源和所有承诺，再注入 q。共同保留车辆用同一前缀重编号。`pair_audit` 对 q/q−1 除末尾新增一辆真正 idle 车辆外逐字段核对。q 反事实的总车队可能大于原 K，不能解释成固定 K 的可部署状态。
2. exact B 的 `targets.target_job` 先在自然 M0 状态计算 pressure/bucket，再构造配对资源世界。q 与 q−1 共享原始标签，不是在注入后的世界重新分桶。在线 `pressure.summary` 使用当前自然状态。第七轮记录自然底座、移除后、q−1、q 的不同 R 和桶，绝不把原始标签改成后两者。
3. `value.contracts.add_release_slots` 将每个 (time,hub) 的车辆路径出口数约束为前缀 slot 数。负值不裁剪。每辆车只有一个 sink；不同时间/位置的组分别从 q=1 开始。q 是该组的释放数，不是包括所有在途资源的完整库存。这是已有可分接入的准确实现，但其经济适用性尚待本轮检验。
4. `BucketValueProvider` 任一非终端插值角样本少于10或 q>K 时，对整个查询使用同一 base 表。不是角值混合回退。主报告分清潜在查询、动作占用系数、选择分歧系数的分母及金额规模。终端自由车辆 V=0，不意味着终端路线的完整系统相对 advantage 为0。
5. `rolling.audit.incoming_supply` 只以最终 `idle_release` 为近期待释放资源，保留车辆 ID。当前已处理的到达不再次处理。`pressure` 不使用旧摘要的 last-delivery 字段作为自由时点。

## 已声明的代理近似，不列为已证实源码故障

- 训练桶观察时点是资源标签自然状态的 t。在线为未来 release tau 查表时，仍使用 decision t 的当前 hub 桶，不预测 tau 的未来桶。这是 regime 转移代理，不是相同条件分布。
- 桶阈值来自工况内 construction 分位数。历史名称 surplus/balanced/scarce 只表示相对低/中/高压力，不能声称有绝对盈余或供需平衡。
- 自然底座 Vres 标签是投影后的资源边际均值，在线按独立释放组加总。不同车辆/时间共享需求的机会可能相关；唯一 assignment 约束只确保本次物理分配不重复，不能证明价格项没有重复计价未来机会。
- `J−g` 的 g 是当期 booked 利润，realized 账本按真实占用/交付核算。二者差不等于纯释放之后的现金。资源 V、系统相对 Y_roll 和完整 Q 不互换。

## 本轮执行核算和公平性

新动作均调用 `planning.compile_action` 独立验证金额。原 M0/约束候选通过 exact B 的 `reconcile_integer_action` 保留原选中整数动作并重算现金，保留 raw floating profit/objective。新世界不使用 raw 浮点金额记账。独立现金差阈值1e−6 USD；不放宽 solver 容差或二次优化。

`actions.candidates` 只用当前状态：M0、原 A 首次 MC 改进 pi1、alpha=.25 的 D4、exact B2 和最多两个同 H1 可行域的确定性 MILP 备选。备选生成允许固定一条路线然后联合求解，但必须把返回的整个联合动作保存，不能在 rollout 中重新优化其余车辆或订单子集。等价物理动作合并并保留来源标签，车辆 ID 不商去、不任意重排改变物理动作。

`worlds.evaluate` 的第一期只编译已冻结整个动作，不调用优化器。下一期才使用同一 continuation，主实验全为 M0。全部冷却到承诺完成；每个世界保存全 trace、初态、未来 stream、账本及独立 replay。

## 尚未解决或未实施的边界

没有确认需修复的线上/训练代码一致性错误，因此不同时修改 target、bucket 或接入。若出现现金、动作、泄漏或非最优状态错误，停止受影响阶段，另存失败记录，不回写历史或混合 attempt。

在线未来为明确的 surrogate-distribution rollout：冻结 B construction 8300–8359 的经验订单 marks 和平均到达率，采用独立 Poisson 新订单数、均匀 booking 和构造数据的 ready/slack 分布。只保留已公开的订单和完整承诺，不读取正式场景的隐藏货物、剩余 N、实际未揭示 OD 配额或货重。它不等于原始按 OD 分层、不放回、固定 N 生成过程的严格条件分布。

自由资源交互只实施当前时点独立注入，跨时点注入未实施并单列。非零交互不能自动解释策略损失或证明 neural 必要。四个构造银行只提供训练银行敏感性描述，不声称精确训练方差或 Bellman fixed point。
