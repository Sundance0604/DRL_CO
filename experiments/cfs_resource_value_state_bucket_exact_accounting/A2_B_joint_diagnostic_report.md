本报告来自用户授权的新独立 exact-accounting B attempt。原 B 在 seed8405 触发金额一致性停止，完整保留且只读。唯一修正是从原求解器提取的整数动作独立重算执行金额，保留原浮点profit/objective诊断；不改变动作、求解器容差、物理、种子、分桶和统计规则，全部B数值重新运行，未混入原attempt缓存。A2仍为原始不可变冻结结果，跨A2/B解释须保留这一数值核算修正的边界。

# A2 + B 独立机制诊断联合汇总

## 结果摘要

A2 的全局冻结 alpha=0.25：CA 的 D4−U4 利润改善为 +8,274.62 USD，Holm p=0.001700，动作与表变化证据也支持改善；但 W80 的非劣性未建立，整体联合稳定化标准未通过。B 的 CA、W40、W80 闭环利润差均显著为负（四工况主比较 Holm 校正），不是仅仅缺少正向证据。W80 独立标签 MSE 改善而利润下降，揭示标签预测与闭环部署效果不能等同。

Case IV 仅编码两项联合标准均未通过。它既不抹去 A2 的局部正向结果和 B 的明确负向结果，也不证明低维可分价值架构达到结构极限。此轮没有据此启动 B-PI、联合阻尼或 RL。

A2先完整执行并冻结，B随后独立执行；不同正式seed银行不相互配对，不共享正式结果调参。两项实验分别只有阻尼更新和三桶状态扩展，未同时改变两个因素。

此处“有效”仅表示达到预注册联合证据阈值；“未建立”不是证明无效。任务书2×2矩阵作为组织证据的框架，不是足以排除所有竞争解释的因果识别。

## 任务书第35节：12问逐项回答

### 1. CA的后续policy deterioration能否被damping明显缓解？

正式D4−U4=+8274.62 USD，t95=[3929.720416256593, 12619.515146341077]，Holm=0.00169999；利润证据=True，动作证据=True，表证据=True。联合稳定化=False，不能只以利润增加替代全部判据。

### 2. W40的table/action instability能否被damping缓解？

最后两步动作差damped/undamped=0.0385/0.1219；relativeL2=0.1016/0.4241；动作/表证据=True/True。部署表变化变小部分由阻尼公式机械产生，仍需利润和独立动作证据。

### 3. Damping是否破坏W80/3原本较好的表现？

D4−U4=-215.36 USD，t95=[-2780.4523512430596, 2349.740177638001]。预注册非劣容差449.50 USD，非劣支持=False；未通过意味着不能排除超过容差的损失，不等于证明发生损害。

### 4. 最优alpha是否跨工况一致？

只冻结了一个全局alpha=0.25，没有各工况独立alpha选择或不同alpha的formal评估。诊断D4利润均值最大者（仅描述，不是预注册稳定化最优）：CA=0.5、W40=0.5、W80=0.25。这不足以认定存在跨工况共同最优alpha；全局选择遵循动作/表优先的lexicographic规则，不能在formal后重选。

### 5. Coarse imbalance bucket是否降低同一(t,u,q)内的resource-value heterogeneity？

Construction加权桶内方差下降是分组机械结果。独立20诊断scenario的冻结表预测残差MSE佐证：california_corridor-n80-k3: base−bucket MSE=+575311.736 USD², t95=[-361090.7407412246, 1511714.2130541499], Holm=0.427843, supported=False；western_multicluster-n40-k8: base−bucket MSE=+38795.057 USD², t95=[-7117.028718793394, 84707.14179442337], Holm=0.279052, supported=False；western_multicluster-n80-k3: base−bucket MSE=+918975.877 USD², t95=[248668.8587072374, 1589282.895654073], Holm=0.0392573, supported=True；western_multicluster-n20-k8: base−bucket MSE=+5821.277 USD², t95=[-4400.3367179691295, 16042.891504567622], Holm=0.427843, supported=False。不对diagnostic标签重新拟合或中心化。

### 6. Bucket policy是否稳定优于base MC Table？

同标签B2−B1正式配对证据：california_corridor-n80-k3: -10650.02 USD, t95=[-14103.788939334145, -7196.2552140522985], Holm=9.70396e-07, diagnostic=-5194.18, stable profit=False；western_multicluster-n40-k8: -1619.40 USD, t95=[-3003.596692052073, -235.20675907730106], Holm=0.046049, diagnostic=-2486.80, stable profit=False；western_multicluster-n80-k3: -3592.99 USD, t95=[-5792.38565159383, -1393.5945094486], Holm=0.00614296, diagnostic=-3246.98, stable profit=False；western_multicluster-n20-k8: +10.99 USD, t95=[-610.620972397656, 632.6106596022239], Holm=0.971643, diagnostic=+127.76, stable profit=False。整体联合有效性=False；未达到阈值不等于证明无效。

### 7. Bucket是否主要帮助CA/W40，而对W80影响小？

CA/W40增量分别-10650.02/-1619.40 USD；W80=-3592.99 USD，t95=[-5792.38565159383, -1393.5945094486]，非劣支持=False。需要结合各自区间、Holm与诊断方向，不能凭均值大小或不显著判断影响小；也未预注册跨工况差异的因果交互检验。

### 8. W20是否仍表现为资源宽松、value signal弱？

W20的B1−B0=+1042.60 USD，t95=[392.1942051372632, 1693.0143584720029]，Holm=0.026738；B2−B1=+10.99 USD。Construction绝对base边际价格均值（原始非终端tuq等权）为california_corridor-n80-k3=4315.24 USD；western_multicluster-n40-k8=761.31 USD；western_multicluster-n80-k3=5997.43 USD；western_multicluster-n20-k8=279.87 USD；W20压力分位阈值c33/c67=0.00583333/0.0145833。这些是资源压力/价格与收益的描述，结合原始count/MC_SE审查；condition-specific桶相对占比不能单独证明绝对资源宽松，也未对“信号弱”预注册二元判据。

### 9. A2与B哪个解释更强？

A2联合证据=False，B联合证据=False，组织性编码Case IV。两者均未建立完整联合证据，不能据此选出更强机制。A2/B使用不同formal场景和不同干预；不直接配对其利润增量，更不能仅比较p值大小来给机制强度排序。

### 10. 剩余问题更像update instability、state aliasing、separability还是有限MC噪声？

阻尼改善动作/表与利润支持更新机制，独立桶MSE/利润改善支持状态混叠机制，但它们不排除其他解释。当前查询回退daily mean：california_corridor-n80-k3=0.2876；western_multicluster-n40-k8=0.1556；western_multicluster-n80-k3=0.2752；western_multicluster-n20-k8=0.2910。高回退、有限条件MC_SE与利润区间均可能限制识别；没有独立MC-bank重复、E(D4)固定点残差、完整联合资源交互或未来桶预测实验，不能确定剩余问题主要是哪一个，也不能把Case IV证明为可分架构结构上限。

### 11. 是否已有证据进入bucketed self-consistent PI？

尚未达到预注册bucket联合证据；先审查支持、回退、独立预测误差和利润区间，不凭某一正均值自动进入B-PI。

### 12. 是否已有足够证据进入Strategic RL？

本轮不自动满足。没有完整固定点、经过认证的全日最优gap或joint interaction/nonseparability独立证据；PreviewH8差不是全日上界或RL可回收利润。即使Case IV也先审查MC支持与精度，并另行预注册机制实验，不能由未达判据推导RL必要。

## 补充证据与复现细节

### CA 后续路径比较

同新formal场景D4−U4=+8274.62 USD，t95=[3929.720416256593, 12619.515146341077]，Holm=0.00169999；联合稳定化证据=False。
D1=U1，因此D4−U4也等于两条路径从共同第一步到第四步的利润变化差；不能用新formal与A旧formal直接作paired差。正利润差但动作/表不稳定只能称profit improvement。

## 2. W40 的 table/action 不稳定是否减少？

最后两步same-state动作差damped=0.0385，undamped=0.1219；relativeL2分别=0.1016/0.4241；动作证据=True，表证据=True。
动作在共同U0完整状态库上独立solve，自己的闭环状态分布另报；未计算E(D4)或固定点残差，不把四步趋缓叫理论收敛。

## 3. 最终选择哪个 alpha，是否受 formal 影响？

全局alpha=0.25；exploratory_selection=False；选择只用8200–8219诊断集，lexicographic筛查/近似容差和.5优先在结果前固定，pilot/formal均在alpha冻结后启动。
共同raw V0/V1与A一致；raw V2/V3评估实际alpha-specific D2/D3，允许与A不同，是用户明确确认的真正自一致协议。

## 4. 是否破坏原先表现较好的 W80？

A2非劣证据=False，容差=449.50 USD，paired t95下限=-2780.45；B非劣证据=False，容差=421.02 USD。
未通过非劣门槛只能说不能排除超过容差的损失，不能把不显著解释为安全。

## 5. 阻尼是否减少符号变化、方向反转和格振荡？

| 工况 | undamped平均range USD | damped平均range USD | undamped方向反转均值 | damped方向反转均值 |
|---|---:|---:|---:|---:|
|california_corridor-n80-k3|3148.199|1059.746|1.333|0.486|
|western_multicluster-n40-k8|636.425|213.444|1.064|0.400|
|western_multicluster-n80-k3|2279.132|849.788|1.286|0.371|
|western_multicluster-n20-k8|223.437|67.353|1.175|0.486|

这些是原始native非终端格的四步描述，不是独立formal样本或极限环证据。完整sign_flip_count、raw/deployed路径和自然idle-q占据率均保存。

## 6. Bucket 是否提供了不同的资源价值信号？

| 工况 | c33 | c67 | tied |
|---|---:|---:|---|
|california_corridor-n80-k3|0.075|0.125|False|
|western_multicluster-n40-k8|0.023125|0.04625|False|
|western_multicluster-n80-k3|0.0958333|0.15625|False|
|western_multicluster-n20-k8|0.00583333|0.0145833|False|

同tuq三桶raw mean、conditional MC_SE及实际supported/fallback查表价格逐格保存；有raw均值差不等于有部署价格差。仅supported桶对照才支持regime-dependent marginal value，不强制单调。

## 7. Bucket 对利润是否有稳定增量？

| 工况 | B2−B1 USD | t95 | Holm | diagnostic USD | 稳定利润证据 |
|---|---:|---|---:|---:|---|
|california_corridor-n80-k3|-10650.02|[-14103.788939334145, -7196.2552140522985]|9.70396e-07|-5194.18|False|
|western_multicluster-n40-k8|-1619.40|[-3003.596692052073, -235.20675907730106]|0.046049|-2486.80|False|
|western_multicluster-n80-k3|-3592.99|[-5792.38565159383, -1393.5945094486]|0.00614296|-3246.98|False|
|western_multicluster-n20-k8|+10.99|[-610.620972397656, 632.6106596022239]|0.971643|+127.76|False|

B1/B2同construction标签、同M0父策略和同formal未来；B1相对旧A1还改变了共同hub采样密度，不能将旧A1到B2全部收益归因于bucket。

## 8. Within-cell heterogeneity 是否减少？

| 工况 | 独立MSE改善 USD² | t95 | Holm | corroborated |
|---|---:|---|---:|---|
|california_corridor-n80-k3|575311.736|[-361090.7407412246, 1511714.2130541499]|0.427843|False|
|western_multicluster-n40-k8|38795.057|[-7117.028718793394, 84707.14179442337]|0.279052|False|
|western_multicluster-n80-k3|918975.877|[248668.8587072374, 1589282.895654073]|0.0392573|True|
|western_multicluster-n20-k8|5821.277|[-4400.3367179691295, 16042.891504567622]|0.427843|False|

Construction加权桶内方差下降是分组的机械结果，仅描述。上表用独立20诊断场景及冻结construction表的预测残差，先scenario内平均，再paired t95/bootstrap/Holm；不从diagnostic重新拟合均值。

## 9. Support 是否充分，fallback 是否过高？

| 工况 | query每日均值率 | pooled率 | 非终端pooled率 | 支持回退门槛 |
|---|---:|---:|---:|---|
|california_corridor-n80-k3|0.2876|0.2882|0.2882|True|
|western_multicluster-n40-k8|0.1556|0.1541|0.1541|True|
|western_multicluster-n80-k3|0.2752|0.2743|0.2743|True|
|western_multicluster-n20-k8|0.2910|0.2888|0.2888|True|

分母是所有pre-solve coefficient查询，不是已选择资源；终端查询不回退会影响总体分母，非终端率单列。阈值ties、空桶、任何角点count<10及q>K都显式可见。高回退时阴性结果可能是支持不足，不是状态信息无用。

## 10. 对应2×2诊断矩阵哪一格？

按“达到证据阈值/未达到证据阈值”的编码，A2=False，B=False，组织性编码Case IV。
Case I支持两种机制都值得继续；II优先继续状态扩展；III优先稳定更新；IV仅表明两种有限诊断均未建立正面联合证据。特别是IV不能直接证明低维资源价值达到结构极限，仍需检查支持、效应精度和未测joint interactions。

## 11. 下一步是否需要 bucket PI、damped ADP 或联合方法？

先审查稀疏支持、误差区间、资源价格对未来regime的转移和联合资源交互，再决定新的状态或更新实验；本轮不自动加复杂度。
本轮不将两变量同时改动以追求更高收益，也不复用formal结果选择新的阈值/alpha。

## 12. 是否已经满足进入 Strategic RL 的门槛？

没有自动满足。MC table与有限self-consistent迭代、阻尼和最小状态扩展已按各自协议检验，但没有完整固定点、理论结构极限或joint interaction/nonseparability的独立因果证据。
PreviewH8−B2只能描述有限提前信息基准差，不能视为全日最优gap或RL可回收利润。即使编码IV，也需充分支持及精度、稳健剩余政策差、联合交互证据与新的预注册实验，才讨论Strategic RL。
如果未来进入RL，MILP仍负责可行性、matching/routing/capacity/commitments，学习层仅提供状态依赖价格或粗粒度目标；本轮没有实现或启动任何RL训练。

## 交付与复现

A2/B各有独立报告、config、源码/数据/policy/table/query冻结、QA和最终manifest。正式统计以每工况40独立scenario配对，图表为本地600dpiPNG/嵌入字体矢量PDF/SVG，使用数据与config一并保存。
是否完整完成最终交付，须检查两目录final_manifest状态和内容hash。没有Git提交/推送、平台默认模型替换或对旧实验写入。
