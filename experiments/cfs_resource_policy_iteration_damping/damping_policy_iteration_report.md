# A2：真正的自一致阻尼资源价值策略迭代

数值与分析阶段已完成。完整测试、视觉审阅、历史只读校验与最终冻结状态，以 `manifests/final_manifest.json` 为准。

## 1. 已确认的研究口径

用户于2026-10-07明确确认真正的自一致更新。共同 raw V0=E(M0)、raw V1=E(D1)=E(U1)保留历史一致性；各 alpha 后续 raw V2、V3重新评价其实际 D2、D3，不能要求它们继续等于 A 的 undamped表。
Vtilde0=V0；Vraw_n=E(Dn)；Vtilde_n=(1−alpha)Vtilde_(n−1)+alpha Vraw_n；D(n+1)=G(Vtilde_n)。E是完整冷却J(q)−J(q−1)，没有J−g或ACTION correction。
唯一变量alpha=.25/.50/.75。物理、MILP、参数、construction场景、time/hub/q网格、线性插值、负值及终端规则与A相同。不同release(time,hub)可分计价仍是架构限制，不是完整联合状态Bellman价值。

## 2. 历史复现、样本与冻结顺序

A的223个交付文件校验、16张表从全保存标签精确重建、20次旧政策独立求解、16组旧资源配对标签独立重算通过。前六轮只读索引记录全部原文件内容hash，未调用历史writer。
Construction8000–8059；diagnostic8200–8219；pilot8950–8951；formal9100–9139。每工况正式n=40，独立单位scenario；policy seed=0，training seed=null。相同场景跨算法配对，t/hub/q/alpha不是额外独立replication。
共同E0/E1在来源一致性证明后复用；3 alpha×2次新E×15,600=93,600反事实世界，均保存完整逐t状态、动作、ledger、冷却及回放。D1=U1精确别名：formal1440标签、1280独立仿真，不增加场景统计自由度。

## 3. 诊断alpha选择

全局冻结alpha=0.25，exploratory_selection=False。主排序只使用CA/W40诊断数据，W80利润安全筛查。formal和pilot在选择冻结后才运行。
利润崩溃定义：D4−D1均值损失超过D1绝对均值5%，且配对t95上限<0；表末步relativeL2比首步上升>10%；动作step4比step2上升>2个百分点。最后两步动作差优先，1个百分点内近似；表L2其次，.02内近似；利润最后，1%U0尺度内近似；最终优先.5。全部筛查不合格时按预注册最少失败项继续固定矩阵，但不宣称稳定化。

| alpha | 筛查失败数 | 最后两步动作差 | 最后两步relativeL2 | 诊断D4均值USD |
|---|---:|---:|---:|---:|
|0.25|0|0.0281|0.0910|53491.15|
|0.5|0|0.0510|0.1524|53802.30|
|0.75|0|0.0682|0.2454|52180.39|

## 4. 新正式集利润与配对比较

| 策略 | california_corridor-n80-k3 | western_multicluster-n40-k8 | western_multicluster-n80-k3 | western_multicluster-n20-k8 |
|---|---:|---:|---:|---:|
| U0 | 60194.54 | 37135.82 | 34861.62 | 17651.94 |
| U1 | 63727.73 | 40719.65 | 41059.18 | 18858.89 |
| U2 | 62143.38 | 41591.55 | 42154.06 | 18567.94 |
| U3 | 58129.64 | 41329.61 | 44676.55 | 18408.15 |
| U4 | 55752.28 | 39882.50 | 44950.12 | 18233.78 |
| D1 | 63727.73 | 40719.65 | 41059.18 | 18858.89 |
| D2 | 65362.23 | 41062.93 | 43446.58 | 18922.64 |
| D3 | 66183.69 | 41317.95 | 45076.99 | 18949.28 |
| D4 | 64026.90 | 41486.61 | 44734.76 | 18748.78 |

主family跨4工况的D4−U4统一两侧Holm；20项路径比较和8项中间跨路径差分别校正。t95和10,000次paired bootstrap未作多重区间校正，与Holm p分列。SD、median、skewness和逐scenario差全部保存。

| 工况 | D4−U4 USD | t95 | bootstrap95 | Holm p | 诊断D4−U4 |
|---|---:|---|---|---:|---:|
|california_corridor-n80-k3|+8274.62|[3929.720416256593, 12619.515146341077]|[4246.634354171551, 12585.216650819595]|0.00169999|+5670.86|
|western_multicluster-n40-k8|+1604.11|[143.7005992304355, 3064.5166554870375]|[202.4495896185705, 2989.5763323698698]|0.0965252|+1265.32|
|western_multicluster-n80-k3|-215.36|[-2780.4523512430596, 2349.740177638001]|[-2649.0630877106337, 2240.3672622212976]|0.866031|+2546.58|
|western_multicluster-n20-k8|+515.00|[-118.0636529020245, 1148.065433501179]|[-83.08087347569206, 1129.2129355190614]|0.215821|+588.29|

## 5. 稳定化联合证据

预注册damping effectiveness证据：False。不满足证据阈值并不证明阻尼无效，更不证明结构极限或DRL必要。
W80非劣容差=449.50 USD；D4−U4 t95下限=-2780.45；支持非劣=False。不显著不等于无伤害。

| 工况 | profit证据 | action证据 | table证据 | 方向一致 | 联合稳定化 |
|---|---|---|---|---|---|
| california_corridor-n80-k3 | True | True | True | True | False |
| western_multicluster-n40-k8 | False | True | True | True | False |
| western_multicluster-n80-k3 | False | True | True | False | False |
| western_multicluster-n20-k8 | False | True | True | True | False |

动作使用固定U0完整状态库、双方独立求解、全部24期，top-level排序不改变路线序列和车辆ID；正式40场景只是冻结后泛化佐证。自己的闭环状态分布单独报告SMD、Wasserstein和支持范围外比例。SMD零方差且均值不同不填零。
振荡全部原始/部署单格路径、符号变化、方向反转、range和自然idle-q占据率保留；图中展示选定alpha各工况range最大的5格。这是有限四步描述，不证明极限环。
负值保留，q非单调不修正；表差同时包含parent政策和自然状态分布变化，不能解读为固定完整状态上的纯operator因果差。MC_SE条件于有限经验parent，不包括整条PI链重新估计不确定性。

阻尼公式机械地缩小单步部署表更新，因此不能仅凭deployed relativeL2/RMSE下降声称稳定化；raw/deployed变化同时导出，动作、正式利润与W80安全证据不可省略。

状态分布中的release_delay_mean使用只读rolling.audit最终idle_release ETA，历史last-delivery摘要offset另存以审计。此处仅后处理，不改冻结的E/G或仿真物理，不参与alpha选择；diagnostic开始前已固定分析源码。

## 6. 运行时、QA、交付与限制

offline wall与worker累计分列，共同E0/E1复用不伪造新的计算时间；在线context/lookup/build/solver/transition/ledger与完整pipeline分开，嵌套计时不能重复相加。Gurobi均为受限MILP OPTIMAL，不是全日全局最优证明。
运行时是本地共享机器的流程实测；alpha=.25的部分construction曾与实现预检并行，不是隔离硬件的微基准。小幅耗时差不能直接归因alpha；QA保存实际测试起止与elapsed供审计。
源码、查询、parent table/policy、共同future与配对资源底座身份逐世界检查；保存系数从实际parent表独立复算；T后无新dispatch、完整承诺冷却、全部现金与物理回放通过。阻尼表逐格精确复算，D1/U1别名明确。
九组论文图提供600dpi PNG、字体嵌入矢量PDF、SVG、source CSV和plot config，并提供九页本地图册。测试和渲染检查完成后再冻结全部A2，之后才启动B数值实验。
没有E(D4)、没有固定点残差或理论收敛证明。没有bucket、ACTION correction、神经网络、DRL、SAC/PPO、平台默认算法更新或Git提交/推送。剩余机制需独立B实验及支持充分性审查，不能仅凭本轮失败宣称Strategic RL必要。

详细文件导航、字段及恢复入口见README。processed CSV可由Python直接读取；所有统计要按scenario配对，不按文件行数扩大样本。
