# 第六次数值实验 A：Self-Consistent MC Resource-Value Policy Iteration

研究日期：2026-10-07。模型族：single。独立目录：`experiments/cfs_resource_policy_iteration/`。
数值矩阵与研究QA已完成；最终测试/视觉审查/交付状态以 `manifests/final_manifest.json` 为准。

## 1. 研究设计与数学含义

本轮固定四次改善：pi0=M0/Zero；V0=E(pi0)、pi1=G(V0)；V1=E(pi1)、pi2=G(V1)；V2=E(pi2)、pi3=G(V2)；V3=E(pi3)、pi4=G(V3)。
唯一目标是 $V^{pi}_{res}(t,u,q)=E[J^{pi}(S_t^{u,q})-J^{pi}(S_t^{u,q-1})]$。J 使用完整 realized ledger，包括 T 后承诺冷却；没有扣 g，没有 ACTION correction、其他拟合器、Fluid 更新、神经网络或 DRL。
G 保留原 H=1 资源 slot-prefix MILP。pi0 保留原 M0；pi1..pi4 只用对应前一张 V 表，每期重规划。所有价值系数为求解前标量，没有新增非线性 joint post-state 或 route-slot 联合变量。
接口细节：每辆车的路径流只进入一个sink，按其release(time,hub)分组；每组sum(prefix slots)=sum(release indicators)，且z_q≤z_(q−1)。因此负价不能通过少计车辆或跳选后槽被暗中截断。不同释放时刻的组分别计价，G并未重建跨时刻完整联合库存或重叠需求机会；这是保持不变的可分接入近似，不是完整状态Bellman价值。
按任务书只执行四次E/improvement，未额外计算E(pi4)或第五次G。因此即使最后两步达到经验稳定标准，也没有直接测量V3−E(pi4)的固定点残差，更不能宣称已验证精确自一致或Bellman固定点。
q 仍是查询 hub 的第 q 辆空闲车：共同底座移除该 hub 空闲车，其他 hub 车和全部承诺保留，再加入共同资源前缀。配对世界仅多一辆空闲车。引擎按数组索引寻址，因此共同车使用同样的机械编号映射，承诺内容/物理字段不变。
正式车队 K 不变；反事实世界可大于 K。固定 q 端点外推扩到2K，q>K 显式标记 unsupported_endpoint，不借缺槽回退到M0。不把这些格算进主表变化指标。

## 2. 数据、split、信息与局限

四工况 CA80/3、W40/8、W80/3、W20/8；沿用 CFS 2022 PUMS、kNN1+MST、stretch1.2、50mph、capacity40、loaded/empty cost2/2.2、fare4*miles+.03*pounds、backlog5、expiry100、waiting0、maxdelay2、T24、reposition=false。Network-rho_max 与前五轮不变。CFS货物价值不是fare，实验 USD 利润不是实测经营预测。
CFS SHA-256：`c74292117a81082582f26bb1c54618f84e4248b58389280f04a3fb3f8e6236f2`。数值源码身份：`8934975ffb3a922fa2c58d6f44f44adef4b38b42c86f15b545dd272e7747c95e`。
Construction8000–8059（60/工况）；diagnostic8100–8119（20）；pilot8900/8901（2）；formal9000–9039（40）。Scenario seed与policy seed0分开；training seed=null。所有正式政策运行完全相同场景，以40个独立场景的配对差统计，不能把参数或t/hub/q行当样本。
为避免第五轮共享 donor 的跨父场景依赖，本轮每场景只使用自身未来路径；不同迭代共用相同生成场景实现 CRN。状态在每轮用当前 pi 重新采集，未来仅给 scheduler，online policy 不预览。此 MC 是有限 sampled-trajectory 估计，不是精确条件期望，也不是强信息状态的 Bellman evaluation。
七个时间节点、查询hub旋转和q网格沿用旧MC；线性time/q插值、负值保留、T后free resource为0，所有四轮一致。正式集未用于 table、iteration、damping或alpha选择。四轮固定完成；可选 damping 未执行，第六轮B未启动。
表格跨轮同格比较虽然支持域相同，但父策略自然状态不同；差异同时反映 evaluation operator 和 state distribution shift。没有将其解读成固定完整状态上的纯策略因果差。动作比较另用固定pi0诊断状态库、每个t的同一个完整状态，双方独立求解，而不是比较两条不同闭环轨迹的相同时间。
Construction scenario原始生成相互独立，但后续父table共享并由同一个60场景bank构造。各格MC_SE仅是给定经验父policy的描述性离散度，不包括重估整条PI链的不确定性，也不提供全链无条件置信保证。重复有限construction bank可能过拟合；正式40场景独立于此bank，用来评价冻结政策的泛化表现。
本轮没有额外固定状态资源-evaluation bank，因此表值变化来源的进一步分解仍待单独研究。固定pi0动作库也不能代表所有后期策略可能到达的状态；各政策自己的state distribution单独记录。

## 3. Phase 0与来源关系

第五轮16次M0/MC Table/RH4/RH8独立重新求解，利润和物理/动作轨迹完全一致；四工况旧MC资源标签重建一致。不是读取旧摘要后自称复现。
A–B人工例子：M0拿当前fare6，V0在t1的A资源价值10使pi1等待并赚10；重新评估t0的q1/q2边际从[6,10]变成[10,6]，pi2动作稳定。它证明当前代码确实重新评估改变的政策，也展示q非单调未被强行修正。不能由toy推导一般精确PI定理。
T后新增空闲车与共同在途承诺的配对toy：无新dispatch、完整冷却、利润差0；原已有承诺仍赚fare/付成本并在两边抵消。
另一个可手算estimand toy（仅历史语义诊断，不进入E/G）：T=2，当前短路A→B赚20，t1的B→A赚100；强制当前长路A→C赚10且release=2。baseline J=120、g=20；长路J=10、g=10，因此terminal candidate的Yroll=(10−10)−(120−20)=−100。与此同时，t=T只增加free resource的配对利润差仍为0。完整独立物理回放通过，见qa/terminal_estimand_distinction.json。
逐世界 lineage 审查 62,400 个反事实世界；67,552,184 次价值系数按父table重新计算一致。每期明确纯slot formulation，没有隐藏M0 fallback、Oracle训练或ACTION correction。每工况每轮还做一次独立重新求解，共16次，利润/动作/最终物理一致。

## 4. 正式政策结果

| 策略 | CA80/3 | W40/8 | W80/3 | W20/8 |
| --- | --- | --- | --- | --- |
| P0 | 56044.59 | 36271.39 | 36782.66 | 18255.71 |
| P1 | 61386.29 | 41473.18 | 43708.61 | 18892.76 |
| P2 | 58567.95 | 41425.05 | 43796.28 | 19015.39 |
| P3 | 57928.19 | 42211.71 | 44154.23 | 19037.42 |
| P4 | 49842.08 | 41657.29 | 44546.22 | 18870.01 |
| R0 | 58045.49 | 37812.65 | 37313.99 | 19316.45 |
| R1 | 59294.11 | 38436.33 | 37912.78 | 20015.80 |
| D | 64941.41 | 52068.90 | 44872.12 | 24553.94 |

每格40个独立场景 mean realized profit (USD)，不等于显著最优。RH4/RH8是known-supply参考；PreviewH8非可部署诊断，不是全日认证上界，也不能把gap宣称为RL可回收潜力。
预注册主family共36项：每工况相邻4组、P2/P3/P4相对P0的3组、P4−RH4/RH8的2组。统一两侧Holm；t95/10000次paired bootstrap未作多重区间校正。4个Preview−P4描述性另列。完整SD、median、skewness和逐scenario差均保存。

| 工况 | 比较 | mean差USD | paired t95 | Holm p |
| --- | --- | --- | --- | --- |
| CA-n80-k3 | P1-P0 | +5341.71 | [1020.99,9662.43] | 0.30073 |
| CA-n80-k3 | P2-P1 | -2818.35 | [-7284.30,1647.61] | 1 |
| CA-n80-k3 | P3-P2 | -639.76 | [-4678.10,3398.59] | 1 |
| CA-n80-k3 | P4-P3 | -8086.12 | [-12628.68,-3543.55] | 0.0212457 |
| CA-n80-k3 | P2-P0 | +2523.36 | [-2339.78,7386.51] | 1 |
| CA-n80-k3 | P3-P0 | +1883.61 | [-2558.49,6325.70] | 1 |
| CA-n80-k3 | P4-P0 | -6202.51 | [-10546.34,-1858.68] | 0.125852 |
| CA-n80-k3 | P4-R0 | -8203.42 | [-12464.66,-3942.17] | 0.00976406 |
| CA-n80-k3 | P4-R1 | -9452.03 | [-14068.59,-4835.46] | 0.00519324 |
| CA-n80-k3 | D-P4 | +15099.33 | [11099.41,19099.26] | 描述性 |
| W-n40-k8 | P1-P0 | +5201.79 | [3589.95,6813.62] | 3.26328e-06 |
| W-n40-k8 | P2-P1 | -48.13 | [-1447.68,1351.42] | 1 |
| W-n40-k8 | P3-P2 | +786.66 | [-291.98,1865.31] | 1 |
| W-n40-k8 | P4-P3 | -554.43 | [-1840.07,731.22] | 1 |
| W-n40-k8 | P2-P0 | +5153.66 | [3543.91,6763.40] | 3.73812e-06 |
| W-n40-k8 | P3-P0 | +5940.32 | [4188.73,7691.91] | 1.20264e-06 |
| W-n40-k8 | P4-P0 | +5385.90 | [3721.29,7050.51] | 3.18448e-06 |
| W-n40-k8 | P4-R0 | +3844.64 | [2182.90,5506.38] | 0.00106064 |
| W-n40-k8 | P4-R1 | +3220.96 | [1334.31,5107.60] | 0.0310125 |
| W-n40-k8 | D-P4 | +10411.61 | [8739.99,12083.24] | 描述性 |
| W-n80-k3 | P1-P0 | +6925.95 | [4020.14,9831.77] | 0.000703768 |
| W-n80-k3 | P2-P1 | +87.67 | [-3027.33,3202.67] | 1 |
| W-n80-k3 | P3-P2 | +357.95 | [-2524.46,3240.36] | 1 |
| W-n80-k3 | P4-P3 | +391.99 | [-2175.74,2959.72] | 1 |
| W-n80-k3 | P2-P0 | +7013.63 | [3530.27,10496.99] | 0.00594727 |
| W-n80-k3 | P3-P0 | +7371.58 | [4155.66,10587.49] | 0.00117463 |
| W-n80-k3 | P4-P0 | +7763.57 | [3944.04,11583.09] | 0.00548895 |
| W-n80-k3 | P4-R0 | +7232.24 | [3381.02,11083.45] | 0.0124445 |
| W-n80-k3 | P4-R1 | +6633.45 | [2671.99,10594.90] | 0.0357564 |
| W-n80-k3 | D-P4 | +325.90 | [-2694.89,3346.68] | 描述性 |
| W-n20-k8 | P1-P0 | +637.04 | [-57.10,1331.19] | 1 |
| W-n20-k8 | P2-P1 | +122.63 | [-476.07,721.34] | 1 |
| W-n20-k8 | P3-P2 | +22.03 | [-551.36,595.42] | 1 |
| W-n20-k8 | P4-P3 | -167.41 | [-611.40,276.59] | 1 |
| W-n20-k8 | P2-P0 | +759.68 | [155.16,1364.19] | 0.287129 |
| W-n20-k8 | P3-P0 | +781.70 | [100.30,1463.10] | 0.435859 |
| W-n20-k8 | P4-P0 | +614.30 | [-115.94,1344.54] | 1 |
| W-n20-k8 | P4-R0 | -446.44 | [-1118.02,225.14] | 1 |
| W-n20-k8 | P4-R1 | -1145.79 | [-1903.14,-388.44] | 0.0838134 |
| W-n20-k8 | D-P4 | +5683.93 | [4724.05,6643.81] | 描述性 |


## 5. 表、动作、状态、q与振荡

| 工况 | 表差 | RMSE USD | relativeL2 | Spearman | sign change |
| --- | --- | --- | --- | --- | --- |
| california_corridor-n80-k3 | V1-V0 | 2442.82 | 0.4258 | 0.8411 | 6.67% |
| california_corridor-n80-k3 | V2-V1 | 2241.33 | 0.3349 | 0.9099 | 8.57% |
| california_corridor-n80-k3 | V3-V2 | 2695.34 | 0.4120 | 0.8467 | 8.57% |
| western_multicluster-n40-k8 | V1-V0 | 580.84 | 0.4858 | 0.6679 | 31.43% |
| western_multicluster-n40-k8 | V2-V1 | 543.88 | 0.4295 | 0.7818 | 21.43% |
| western_multicluster-n40-k8 | V3-V2 | 539.61 | 0.3936 | 0.7435 | 23.33% |
| western_multicluster-n80-k3 | V1-V0 | 1933.80 | 0.2813 | 0.9254 | 0.95% |
| western_multicluster-n80-k3 | V2-V1 | 1563.41 | 0.1973 | 0.9536 | 0.95% |
| western_multicluster-n80-k3 | V3-V2 | 1237.81 | 0.1503 | 0.9551 | 0.00% |
| western_multicluster-n20-k8 | V1-V0 | 285.26 | 0.5155 | 0.5601 | 38.57% |
| western_multicluster-n20-k8 | V2-V1 | 271.04 | 0.4730 | 0.6370 | 30.48% |
| western_multicluster-n20-k8 | V3-V2 | 262.39 | 0.4574 | 0.6905 | 26.19% |

| 工况 | 动作改善步 | same-state日均变化比例 | 20场景t95 |
| --- | --- | --- | --- |
| california_corridor-n80-k3 | 1 | 14.79% | [12.84%,16.74%] |
| california_corridor-n80-k3 | 2 | 8.33% | [6.33%,10.33%] |
| california_corridor-n80-k3 | 3 | 8.33% | [6.23%,10.43%] |
| california_corridor-n80-k3 | 4 | 7.50% | [5.44%,9.56%] |
| western_multicluster-n40-k8 | 1 | 26.88% | [23.94%,29.81%] |
| western_multicluster-n40-k8 | 2 | 12.71% | [9.99%,15.43%] |
| western_multicluster-n40-k8 | 3 | 12.92% | [10.17%,15.67%] |
| western_multicluster-n40-k8 | 4 | 10.21% | [7.49%,12.93%] |
| western_multicluster-n80-k3 | 1 | 13.12% | [11.31%,14.94%] |
| western_multicluster-n80-k3 | 2 | 3.33% | [1.98%,4.69%] |
| western_multicluster-n80-k3 | 3 | 2.50% | [1.33%,3.67%] |
| western_multicluster-n80-k3 | 4 | 0.83% | [0.03%,1.63%] |
| western_multicluster-n20-k8 | 1 | 18.54% | [15.68%,21.40%] |
| western_multicluster-n20-k8 | 2 | 11.88% | [8.28%,15.47%] |
| western_multicluster-n20-k8 | 3 | 14.37% | [11.44%,17.31%] |
| western_multicluster-n20-k8 | 4 | 11.04% | [8.92%,13.17%] |

| 工况 | 动作改善步 | 正式集same-state变化比例 | 40场景t95 |
| --- | --- | --- | --- |
| california_corridor-n80-k3 | 1 | 14.69% | [13.18%,16.20%] |
| california_corridor-n80-k3 | 2 | 11.67% | [10.09%,13.24%] |
| california_corridor-n80-k3 | 3 | 9.38% | [7.79%,10.96%] |
| california_corridor-n80-k3 | 4 | 9.48% | [7.88%,11.08%] |
| western_multicluster-n40-k8 | 1 | 26.46% | [23.72%,29.20%] |
| western_multicluster-n40-k8 | 2 | 13.65% | [11.58%,15.71%] |
| western_multicluster-n40-k8 | 3 | 11.77% | [9.58%,13.97%] |
| western_multicluster-n40-k8 | 4 | 10.10% | [8.08%,12.13%] |
| western_multicluster-n80-k3 | 1 | 11.15% | [9.75%,12.54%] |
| western_multicluster-n80-k3 | 2 | 3.65% | [2.55%,4.74%] |
| western_multicluster-n80-k3 | 3 | 3.65% | [2.59%,4.70%] |
| western_multicluster-n80-k3 | 4 | 2.19% | [1.33%,3.04%] |
| western_multicluster-n20-k8 | 1 | 19.69% | [17.51%,21.86%] |
| western_multicluster-n20-k8 | 2 | 11.88% | [9.77%,13.98%] |
| western_multicluster-n20-k8 | 3 | 13.65% | [11.56%,15.74%] |
| western_multicluster-n20-k8 | 4 | 9.90% | [8.07%,11.72%] |

两组动作诊断分开统计，主稳定性判据使用预声明的pi0诊断状态库，正式状态库只作泛化佐证。正式pi0状态通过冻结源码独立重算采集，并与保存的正式运行核对利润、逐期动作和最终物理hash精确复现。排序规范忽略数组序列化顺序但保留vehicle ID；正式集没有参与table、iteration或模型选择。
| 工况 | V轮次 | q边际非递增违例/比较格 | 最大违例USD | 负值格 |
| --- | --- | --- | --- | --- |
| california_corridor-n80-k3 | 0 | 12/70 | 2045.86 | 2 |
| california_corridor-n80-k3 | 1 | 3/70 | 1209.73 | 6 |
| california_corridor-n80-k3 | 2 | 6/70 | 1425.47 | 4 |
| california_corridor-n80-k3 | 3 | 8/70 | 2532.35 | 7 |
| western_multicluster-n40-k8 | 0 | 45/245 | 1271.02 | 27 |
| western_multicluster-n40-k8 | 1 | 70/245 | 1155.11 | 102 |
| western_multicluster-n40-k8 | 2 | 62/245 | 2002.54 | 98 |
| western_multicluster-n40-k8 | 3 | 72/245 | 1742.06 | 104 |
| western_multicluster-n80-k3 | 0 | 6/70 | 3299.89 | 1 |
| western_multicluster-n80-k3 | 1 | 1/70 | 693.26 | 0 |
| western_multicluster-n80-k3 | 2 | 6/70 | 2949.81 | 1 |
| western_multicluster-n80-k3 | 3 | 1/70 | 1130.07 | 1 |
| western_multicluster-n20-k8 | 0 | 24/245 | 511.94 | 39 |
| western_multicluster-n20-k8 | 1 | 55/245 | 695.06 | 85 |
| western_multicluster-n20-k8 | 2 | 74/245 | 1439.73 | 96 |
| western_multicluster-n20-k8 | 3 | 57/245 | 891.61 | 95 |

q违例是诊断，不做isotonic；负值可能来自政策动作改变、机会竞争、成本、MC噪声或稀疏支持，不自动等于代码错。每格保留支持状态、MC_SE、持久性和原始配对利润；q之间依赖，不能把两个SE当独立作显著性检验。`negative_value_components.csv`对直接观察的负表价格逐对重建fare/loaded/empty/backlog/expiry增量及首次业务利润/动作改变，核对利润identity；插值产生的负q明确标记没有该q直接标签，不伪造归因。成分重建是核算分解，不是独立识别唯一行为机制。
振荡全部格保存V0..V3、符号变化、更新方向反转、range和idle-q占据频率。图中仅按range显示各工况最大的5格，选择规则透明，不冒充显著性。高q在自然轨迹很少占据时，其变化不能与常用资源格同等解释。
State shift逐hub/time提供idle、incoming、backlog、committed、utilization、已知release delay的SMD/Wasserstein及旧范围外占比。零方差且均值不同的SMD保持null并计数，绝不写0表示“无变化”。

## 6. 稳定性与任务书10个问题

### california_corridor-n80-k3

pi1−pi0=+5341.71 USD；pi4−pi0=-6202.51 USD，Holm p=0.125852。最后两步联合小变化标准：False；描述性类别：H-B-compatible corrected adjacent deterioration; not a proof of a cycle。收益t95全部落入容差带（描述性，未作多重等价检验）：False。Preview−pi4=15099.33 USD，仅信息参考。
- step3: gain=-639.76, relativeL2=0.3349, same-state action=8.33%, meanabsSMD=0.3974, undefinedSMD=0。
- step4: gain=-8086.12, relativeL2=0.4120, same-state action=7.50%, meanabsSMD=0.4347, undefinedSMD=0。

1. **第一次MC是否有增益？** P1−P0：+5341.71 USD，t95[1020.99,9662.43]，Holm=0.30073。方向和多重校正证据分开解释。
2. **V1−V0是否改变？** RMSE=2442.82 USD，relativeL2=0.4258，Spearman=0.8410636533278042，三类sign change=6.67%。
3. **pi2是否进一步增益？** P2−P1：-2818.35 USD，t95[-7284.30,1647.61]，Holm=1；不以累计差替代增量差。
4. **表变化是否逐渐减小？** relativeL2序列=[0.4258, 0.3349, 0.412]；连续非增=False。RMSE序列=[2442.82, 2241.33, 2695.34]。
5. **同状态动作差异是否减小？** 四步日均变化比例=[0.1479, 0.0833, 0.0833, 0.075]；连续非增=True，固定pi0状态库、20场景。
6. **状态分布是否稳定？** 各步meanabsSMD=[0.5601, 0.3639, 0.3974, 0.4347]；最后两步small判据和undefined分母计数见上。Wasserstein/support另存，不把时间格当独立样本。
7. **振荡在哪些格？** 95/105个部署支持格有更新方向反转。最大range三格：hub=06-472,t=7,q=1,range=9374.63,sign_flips=0,idle_q_frequency=0.287；hub=06-41740,t=3,q=1,range=9094.91,sign_flips=0,idle_q_frequency=0.000；hub=06-41740,t=7,q=1,range=7970.61,sign_flips=0,idle_q_frequency=0.200。仅有限步诊断，非周期极限证明。
8. **q违例和负值？** 四张表合计q违例29/280次，负值19格，其中mean+2SE仍负的条件描述格3。没有裁剪或isotonic；SE不含整条PI重估不确定性。
9. **pi4是否empirically stabilized？** 固定协议的联合small+nonincreasing判据=False；没有exact PI/convergence/最优V证明。小mean变化与宽CI不等价于收益等价。
10. **还有什么限制？** P4−RH4：-8203.42 USD，t95[-12464.66,-3942.17]，Holm=0.00976406；P4−RH8：-9452.03 USD，t95[-14068.59,-4835.46]，Holm=0.00519324。V仍忽略完整backlog/other-hub/incoming条件，slot可分与有限MC支持仍在；本轮不能判定主要误差来源，也不能从Preview gap推断DRL必要。

### western_multicluster-n40-k8

pi1−pi0=+5201.79 USD；pi4−pi0=+5385.90 USD，Holm p=3.18448e-06。最后两步联合小变化标准：False；描述性类别：not empirically stabilized in four fixed improvements。收益t95全部落入容差带（描述性，未作多重等价检验）：False。Preview−pi4=10411.61 USD，仅信息参考。
- step3: gain=+786.66, relativeL2=0.4295, same-state action=12.92%, meanabsSMD=0.2002, undefinedSMD=0。
- step4: gain=-554.43, relativeL2=0.3936, same-state action=10.21%, meanabsSMD=0.1447, undefinedSMD=0。

1. **第一次MC是否有增益？** P1−P0：+5201.79 USD，t95[3589.95,6813.62]，Holm=3.26328e-06。方向和多重校正证据分开解释。
2. **V1−V0是否改变？** RMSE=580.84 USD，relativeL2=0.4858，Spearman=0.667895135923701，三类sign change=31.43%。
3. **pi2是否进一步增益？** P2−P1：-48.13 USD，t95[-1447.68,1351.42]，Holm=1；不以累计差替代增量差。
4. **表变化是否逐渐减小？** relativeL2序列=[0.4858, 0.4295, 0.3936]；连续非增=True。RMSE序列=[580.84, 543.88, 539.61]。
5. **同状态动作差异是否减小？** 四步日均变化比例=[0.2688, 0.1271, 0.1292, 0.1021]；连续非增=False，固定pi0状态库、20场景。
6. **状态分布是否稳定？** 各步meanabsSMD=[0.3362, 0.2143, 0.2002, 0.1447]；最后两步small判据和undefined分母计数见上。Wasserstein/support另存，不把时间格当独立样本。
7. **振荡在哪些格？** 220/280个部署支持格有更新方向反转。最大range三格：hub=04-429,t=0,q=1,range=3883.19,sign_flips=0,idle_q_frequency=0.000；hub=04-429,t=3,q=1,range=2949.79,sign_flips=0,idle_q_frequency=0.688；hub=32-332,t=3,q=3,range=2296.60,sign_flips=2,idle_q_frequency=0.000。仅有限步诊断，非周期极限证明。
8. **q违例和负值？** 四张表合计q违例249/980次，负值331格，其中mean+2SE仍负的条件描述格19。没有裁剪或isotonic；SE不含整条PI重估不确定性。
9. **pi4是否empirically stabilized？** 固定协议的联合small+nonincreasing判据=False；没有exact PI/convergence/最优V证明。小mean变化与宽CI不等价于收益等价。
10. **还有什么限制？** P4−RH4：+3844.64 USD，t95[2182.90,5506.38]，Holm=0.00106064；P4−RH8：+3220.96 USD，t95[1334.31,5107.60]，Holm=0.0310125。V仍忽略完整backlog/other-hub/incoming条件，slot可分与有限MC支持仍在；本轮不能判定主要误差来源，也不能从Preview gap推断DRL必要。

### western_multicluster-n80-k3

pi1−pi0=+6925.95 USD；pi4−pi0=+7763.57 USD，Holm p=0.00548895。最后两步联合小变化标准：False；描述性类别：not empirically stabilized in four fixed improvements。收益t95全部落入容差带（描述性，未作多重等价检验）：False。Preview−pi4=325.90 USD，仅信息参考。
- step3: gain=+357.95, relativeL2=0.1973, same-state action=2.50%, meanabsSMD=0.2566, undefinedSMD=0。
- step4: gain=+391.99, relativeL2=0.1503, same-state action=0.83%, meanabsSMD=0.2370, undefinedSMD=0。

1. **第一次MC是否有增益？** P1−P0：+6925.95 USD，t95[4020.14,9831.77]，Holm=0.000703768。方向和多重校正证据分开解释。
2. **V1−V0是否改变？** RMSE=1933.80 USD，relativeL2=0.2813，Spearman=0.925399129172714，三类sign change=0.95%。
3. **pi2是否进一步增益？** P2−P1：+87.67 USD，t95[-3027.33,3202.67]，Holm=1；不以累计差替代增量差。
4. **表变化是否逐渐减小？** relativeL2序列=[0.2813, 0.1973, 0.1503]；连续非增=True。RMSE序列=[1933.8, 1563.41, 1237.81]。
5. **同状态动作差异是否减小？** 四步日均变化比例=[0.1312, 0.0333, 0.025, 0.0083]；连续非增=True，固定pi0状态库、20场景。
6. **状态分布是否稳定？** 各步meanabsSMD=[0.5718, 0.3932, 0.2566, 0.237]；最后两步small判据和undefined分母计数见上。Wasserstein/support另存，不把时间格当独立样本。
7. **振荡在哪些格？** 95/105个部署支持格有更新方向反转。最大range三格：hub=32-332,t=7,q=1,range=8407.83,sign_flips=0,idle_q_frequency=0.013；hub=04-429,t=3,q=1,range=6728.41,sign_flips=0,idle_q_frequency=0.575；hub=06-488,t=0,q=2,range=5635.97,sign_flips=0,idle_q_frequency=0.000。仅有限步诊断，非周期极限证明。
8. **q违例和负值？** 四张表合计q违例14/280次，负值3格，其中mean+2SE仍负的条件描述格0。没有裁剪或isotonic；SE不含整条PI重估不确定性。
9. **pi4是否empirically stabilized？** 固定协议的联合small+nonincreasing判据=False；没有exact PI/convergence/最优V证明。小mean变化与宽CI不等价于收益等价。
10. **还有什么限制？** P4−RH4：+7232.24 USD，t95[3381.02,11083.45]，Holm=0.0124445；P4−RH8：+6633.45 USD，t95[2671.99,10594.90]，Holm=0.0357564。V仍忽略完整backlog/other-hub/incoming条件，slot可分与有限MC支持仍在；本轮不能判定主要误差来源，也不能从Preview gap推断DRL必要。

### western_multicluster-n20-k8

pi1−pi0=+637.04 USD；pi4−pi0=+614.30 USD，Holm p=1。最后两步联合小变化标准：False；描述性类别：not empirically stabilized in four fixed improvements。收益t95全部落入容差带（描述性，未作多重等价检验）：False。Preview−pi4=5683.93 USD，仅信息参考。
- step3: gain=+22.03, relativeL2=0.4730, same-state action=14.37%, meanabsSMD=0.1869, undefinedSMD=0。
- step4: gain=-167.41, relativeL2=0.4574, same-state action=11.04%, meanabsSMD=0.1786, undefinedSMD=0。

1. **第一次MC是否有增益？** P1−P0：+637.04 USD，t95[-57.10,1331.19]，Holm=1。方向和多重校正证据分开解释。
2. **V1−V0是否改变？** RMSE=285.26 USD，relativeL2=0.5155，Spearman=0.5601276337911616，三类sign change=38.57%。
3. **pi2是否进一步增益？** P2−P1：+122.63 USD，t95[-476.07,721.34]，Holm=1；不以累计差替代增量差。
4. **表变化是否逐渐减小？** relativeL2序列=[0.5155, 0.473, 0.4574]；连续非增=True。RMSE序列=[285.26, 271.04, 262.39]。
5. **同状态动作差异是否减小？** 四步日均变化比例=[0.1854, 0.1187, 0.1437, 0.1104]；连续非增=False，固定pi0状态库、20场景。
6. **状态分布是否稳定？** 各步meanabsSMD=[0.1517, 0.1469, 0.1869, 0.1786]；最后两步small判据和undefined分母计数见上。Wasserstein/support另存，不把时间格当独立样本。
7. **振荡在哪些格？** 220/280个部署支持格有更新方向反转。最大range三格：hub=04-429,t=3,q=1,range=1397.67,sign_flips=0,idle_q_frequency=0.375；hub=04-429,t=7,q=2,range=1349.22,sign_flips=3,idle_q_frequency=0.375；hub=04-429,t=3,q=2,range=1347.76,sign_flips=0,idle_q_frequency=0.613。仅有限步诊断，非周期极限证明。
8. **q违例和负值？** 四张表合计q违例210/980次，负值315格，其中mean+2SE仍负的条件描述格8。没有裁剪或isotonic；SE不含整条PI重估不确定性。
9. **pi4是否empirically stabilized？** 固定协议的联合small+nonincreasing判据=False；没有exact PI/convergence/最优V证明。小mean变化与宽CI不等价于收益等价。
10. **还有什么限制？** P4−RH4：-446.44 USD，t95[-1118.02,225.14]，Holm=1；P4−RH8：-1145.79 USD，t95[-1903.14,-388.44]，Holm=0.0838134。V仍忽略完整backlog/other-hub/incoming条件，slot可分与有限MC支持仍在；本轮不能判定主要误差来源，也不能从Preview gap推断DRL必要。


## 7. 运行时与核算

| E轮次 | wall秒 | worker秒 | paired worlds | provider calls |
| --- | --- | --- | --- | --- |
| 0 | 384.64 | 1142.18 | 15600 | 0 |
| 1 | 4328.48 | 12962.07 | 15600 | 21954579 |
| 2 | 3629.00 | 10869.02 | 15600 | 21933457 |
| 3 | 1707.44 | 5108.70 | 15600 | 21950472 |

| 工况 | 额外E轮次 | worker秒 | 相邻gain USD | gain USD/离线worker小时 |
| --- | --- | --- | --- | --- |
| california_corridor-n80-k3 | 0 | 153.49 | +5341.71 | +125287.13 |
| western_multicluster-n40-k8 | 0 | 460.46 | +5201.79 | +40668.87 |
| western_multicluster-n80-k3 | 0 | 155.69 | +6925.95 | +160142.79 |
| western_multicluster-n20-k8 | 0 | 372.54 | +637.04 | +6156.05 |
| california_corridor-n80-k3 | 1 | 2852.40 | -2818.35 | -3557.02 |
| western_multicluster-n40-k8 | 1 | 5169.54 | -48.13 | -33.52 |
| western_multicluster-n80-k3 | 1 | 2092.42 | +87.67 | +150.84 |
| western_multicluster-n20-k8 | 1 | 2847.71 | +122.63 | +155.03 |
| california_corridor-n80-k3 | 2 | 2744.19 | -639.76 | -839.27 |
| western_multicluster-n40-k8 | 2 | 5767.53 | +786.66 | +491.02 |
| western_multicluster-n80-k3 | 2 | 1198.79 | +357.95 | +1074.94 |
| western_multicluster-n20-k8 | 2 | 1158.51 | +22.03 | +68.45 |
| california_corridor-n80-k3 | 3 | 1095.51 | -8086.12 | -26572.08 |
| western_multicluster-n40-k8 | 3 | 2042.90 | -554.43 | -977.01 |
| western_multicluster-n80-k3 | 3 | 805.64 | +391.99 | +1751.60 |
| western_multicluster-n20-k8 | 3 | 1164.65 | -167.41 | -517.46 |

目标世界含独立物理回放和完整逐期存储，因此offline时间不是只解MILP；三个worker的wall与summed-worker不同，不能相加。offline parent/simulation/serialization/aggregation/interpolation/QA分别记录，worker为包含这些部分的总量。纯provider lookup、context、build、solver、transition、postprocess、trace、ledger都有独立计时。lookup是build子集，release-slot assembly含lookup，cooldown是transition子集；不能重复相加。原M0内部build/extraction不改源代码，只在planner overhead合并标识，不冒充纯build。USD/离线worker小时仅摊销描述，不是现金ROI。

## 8. QA、文件导航和未做内容

全反事实独立回放62,400世界、1,917,505步；全部policy运行1,744次、71,604步，物理精确，cash容差1e-6 USD，承诺完整冷却。
`README.md`逐模块/目录/字段/统计含义导航；`targets`为资源标签，`tables`为四轮价值面，`policies`为五个政策的反向来源，`raw/worlds`与`raw/traces`为逐t完整记录，`processed`为分析表，`qa`为审查，`figures`含16组PNG/PDF/SVG与source/config。
初次尝试为补齐独立计时而主动中止，完整归档于raw/failed_attempts/timing-v1，不混入最终样本；仅观测计时与等价聚合分段，物理/候选/控制/目标/种子不变。重做Phase0并全量重跑，四张新旧V0文件SHA必须精确相同，结果见qa/timing_repair_parity.json。修订前没有本轮formal outcomes可选。
前五轮与CFS最终内容校验、pytest、字体矢量渲染/视觉检查以及交付SHA在final_manifest。未做：可选damping、第六轮B、ACTION correction、joint post-state、neural/DRL/SAC/PPO、分支切换/提交/推送。MC资源评估和近似policy iteration属于经典ADP思想，不能因没有DRL就说没有数据驱动价值估计。不能把本实验新增代码解释为已经更新平台默认框架。
