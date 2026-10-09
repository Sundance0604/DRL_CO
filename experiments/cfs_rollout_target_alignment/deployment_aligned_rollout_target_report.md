# 第五次数值实验：Deployment-Aligned Rollout Target

研究日期：2026-10-07。模型族：single-level。独立目录：`experiments/cfs_rollout_target_alignment/`。

**最终状态：完成。** 2240次正式运行、112次pilot全部成功；2352个政策运行独立物理回放通过。没有训练RL，没有修改前四轮实验或主模型物理语义。

本轮最重要的结论是：**freeze控制确实显著扭曲参考M0下的未来优势标签，但仅修正标签并添加ACTION特征，尚未证明能带来跨工况稳定的实现利润增益。** 不能由此认定完整target问题已解决：系统相对优势与绝对资源价格的接入、terminal gate、可加性和q-mixture仍有明确限制。

## 1. 本轮究竟识别了什么

本轮不增加函数逼近家族，不使用 DRL、SAC、神经网络或 MCMC，不更改网络、最短路、rho、车辆容量、候选物理路线、订单截止期、承诺执行、日终冷却期或实现利润记账。

研究分成两个不同问题：

1. 在完全相同的当前期干预与未来收益定义下，后续“冻结到候选释放”是否扭曲正常 rolling control 下的候选未来优势？
2. 使用同一新的 rolling target、同一数据、同一 MC base、同一拟合方法与超参数点，仅增加 ACTION 特征，能否改善排序并提高闭环实现利润？

第二个问题不能用不同轮次的利润直接识别。第五轮与第四轮同时有新的种子和新的标签含义；本轮的严格单因素对照是**第五轮内部 ACTION−OLD**，不是“第五轮−第四轮”。

### 1.1 必须先纠正的旧、新标签差异

检查源码后，第四轮原始 action label 实际是：先固定一个预选的可行订单子集，冻结新分配到候选释放，再在释放状态计算有限车辆资源边际：

\[
Y^{fourth}=\mathcal J(q)-\mathcal J(q-1).
\]

第五轮任务书要求的是当前候选相对无干预 M0 联合动作的系统未来优势：

\[
Y^{roll,M0}(S_t,k,r)
=\bigl[\mathcal J_t^{r,M0}-g_t^r\bigr]
-\bigl[\mathcal J_t^{0,M0}-g_t^0\bigr].
\]

两者不仅 continuation control 不同，连 estimand、当前订单分配与 baseline 都不同。直接拿它们作 freeze−rolling 主比较，无法把差异归因于 freeze。

因此，本轮经用户明确同意，主识别采用以下严格配对设计；第四轮原始资源边际另列复现诊断，不进入 Phase-1 gate。

| 项目 | 严格 freeze | rolling |
|---|---|---|
| 当前初始状态、车辆、候选、未来需求 | 同一冻结查询 | 完全相同 |
| 当前期路线约束 | 仅 `y[k,r]=1`，wait 则仅固定该车等待 | 完全相同 |
| 当前订单与其他车辆 | 原 MILP 联合优化，不固定 served subset | 完全相同 |
| baseline 当前动作 | 原始无候选约束的 M0 | 完全相同 |
| 当前业务利润 g | 收入减原业务成本和罚款；不含 tie-break/V | 完全相同 |
| t+1 到候选释放前 | 全体车辆执行已有承诺/等待，不新增分配 | 每期重求解 M0 |
| 候选释放以后 | 每期重求解 M0 | 每期重求解 M0 |
| 配对世界冻结阈值 | candidate 与 baseline 均用**查询候选**释放期 | 不冻结 |
| 标签 | 相同的 `(Jr−gr)−(J0−g0)` | 完全相同定义 |

对 canonical baseline 候选，调用原无约束规划得到已满足该候选的最优解，避免重复求解引入无关平局差异；其相对标签严格为零。非 baseline 候选不固定订单 x，也不固定其他车辆的动作。

### 1.2 J 与 g 的核算边界

- J：从同一个当前状态出发，运行到 T，并继续冷却直到全部承诺交付完成的 realized ledger；扣掉同一个过去已实现利润常数。
- g：当前 MILP 的业务 operating profit，排除 delay tie-break 与价值系数。
- `J−g`：从完整系统回报中移除当前已计业务利润。它还包含“当前分配时整段路线计费”和“实际运行载客/空驶成本”的核算差额，**不应被简化成车辆释放后单独赚的钱**。
- 当前新分配收入归属一次；重复分配、未交付承诺、未完成冷却和利润 identity residual 均有断言。

当前完整动作改变多个车辆/订单，因此 Y 是**系统级相对动作优势**，不是该车一辆资源的绝对边际价格，也不是完整联合动作空间上的已拟合 Q。

## 2. 冻结材料、数据与实验环境

已阅读前四轮报告、目标构造、ValueProvider、候选、原始 MILP、runner、trace、ledger、rolling 边界控制及回放逻辑。任务书中的 `main_v2_DRL` 在此 checkout 没有同名字面文件；实际冻结数学文档是 `doc/main_v2.tex`，并结合 single 引擎核对。

历史只读索引覆盖 **81,632 个文件**，包含前四轮源码、模型、raw/processed/results/report。第一次冻结前已验证第四轮数值源码身份，并对 CFS 内容校验。本轮所有新数据和结果只写到第五轮目录。源码快照、查询清单、未来路径清单、模型冻结、正式统计预注册分别保存，不依靠 Git 提交代替研究版本身份。

CFS 来源：本地 CFS 2022 PUMS SQLite。SHA-256：

```text
c74292117a81082582f26bb1c54618f84e4248b58389280f04a3fb3f8e6236f2
```

需求 OD/权重来自 CFS；实际车辆、预订/窗口、网络 stretch、每英里成本、fare 与 travel time 是冻结的实验生成假设，不是观测经济价格或实测运程。尤其 **CFS 货物价值不是模型 fare**。不能把本报告中的 USD 利润解释成真实经营预测。

四工况：CA80/3、W40/8、W80/3、W20/8；含义为区域、订单数 N、车辆数 K。网络为 kNN1+MST、stretch=1.2，唯一一致网络路径；网络边 travel time 按 50mph 向上取整。capacity=40，loaded/empty cost=2/2.2，fare=4×miles+0.03×pounds，backlog penalty=5，expiry penalty=100，waiting penalty=0，max delay=2，tie penalty=1，T=24，reposition=false。

rho 保留 network-rho 的 rho_max：Western=1001.6888930413795 miles；California=660.2997451231745 miles。全矩阵没有借改善政策修改这些物理或经济参数。

环境：Windows，Python 3.11.14，Gurobi 10.0.3，NumPy 2.2.5，SciPy 1.15.3，Matplotlib 3.10.8。Python hash seed=0，solver seed=0、threads=1、gap=0、单次求解时限30s，最多3个实验 worker，日级 wall budget=600s。target M0 不接受仅 feasible-limit 的非最优 incumbent。数值拟合确定性，无 RL training seed；scenario seed 与 policy seed=0 分开记录。

## 3. Phase 0：人工检查和历史基线复现

### 3.1 三个人工 toy

**Toy A：即时收益扣除。** 在相同物理动作、释放状态和未来机会下，扰动已经归属的即时 fare：ΔJ=10、Δg=10，因而未来优势=0。这里是核算干预，不冒充真实数据同一状态的两个可行订单集。

**Toy B：相同当前收益、不同释放。** 两动作当前 g 都为10，但释放差异导致未来优势100；说明不是将当前收益再加一次。

**Toy C：A--B--C rolling-control 扭曲。** 当前长路线 g=50、完整 J=250；短路线 g=40、完整 J=310。M0 当前偏向50，但短路线的 rolling 未来优势为70。另一延迟长路线 rolling 优势=−100，严格 freeze 标签却为0；冻结隐藏了后续控制损失。候选与 baseline freeze 世界共用查询释放阈值。5个 toy 世界均通过无求解器回放。

### 3.2 只读复现

四工况 × M0/RH4/RH8 共12个历史基线复现，利润差=0，物理与动作轨迹一致。另独立重建第四轮 W40/8、seed4000 的24个资源/action标签检查，原正常 rolling MC 资源边际与原 fixed-subset/release label 均差=0。不是仅加载旧 JSON 后宣称复现。

新冻结模型另取四工况的 MC Table 与 Bayes-ACTION，各重新求解一次，共8次；利润差=0、每期动作与物理轨迹一致。参见 `qa/baseline_regression.json`、`qa/historical_target_regression.json`、`qa/independent_reproduction.json`。

## 4. Phase 1：先识别 freeze 是否扭曲目标

### 4.1 查询设计

W40/8，seed6000--6019，共20个独立场景。固定观察 t=3/8/12/16/20；每个时点考虑全部 deciding vehicles，每车最多8个候选。先选 baseline、wait、时长极值、未覆盖释放 hub，再由 deterministic hash 补齐；只使用当前可见状态与路线元信息，不查看未来订单或 rollout outcome。

无车/无合法 alternative 的时点保留原因，不替换为“更有趣的时点”。冻结查询356条，其中 baseline77条、非 baseline279条、77个 state-vehicle candidate group，另保留182条 missing 记录。Missing 是时点/车辆记录，不能以182除以100解释成“时点缺失率”。

查询 digest：

```text
37259e2ce51c8b8ce3682658826e210861829b6ec60e4f48d522a0b3529a1b62
```

### 4.2 主结果

| 量 | 结果 | 统计口径 |
|---|---:|---|
| 平均绝对 mismatch | 1918.40 USD | 非 baseline279条描述性行平均 |
| mismatch RMSE | 3390.86 USD | 同上 |
| Pearson(Yfreeze,Yroll) | 0.3053 | 描述性，不用行数作独立样本 |
| Spearman | 0.2037 | 同上 |
| 包含零转换的 sign change | 46.24% | 三类符号，1e-6 USD tie zone |
| 严格正负对立 | 25.81% | 不将 zero→nonzero 当正负反转 |
| 严格候选排序反转 | 252/621=40.58% | 包含 baseline；仅两种标签下均严格排序的 pair |
| canonical top1 不一致 | 30/77=38.96% | 平局内按最小 candidate hash 定义 |
| 最优候选集合完全不相交 | 23/77=29.87% | 不把平局内换选全算作严格改变 |

所有356条（包含定义上为零的 baseline）描述性平均绝对差1503.47、RMSE3001.84、Spearman0.1942。它们不是主标量结果，不能借添加零标签稀释 mismatch。

以独立场景为单位，先算各场景平均绝对差，再等权汇总：mean=1581.57、SD=1114.60、n=20、t95 CI=[1059.93,2103.22] USD。行加权1918.40与场景等权1581.57不同，是候选数不等导致，不是计算冲突。

预注册 T1 被 Spearman<0.90、strict pair flip≥10%、top1 disagreement≥10%、inclusive sign change≥10%和特定 regime inversion 多项触发。**不是 gray/exploratory gate；第二阶段在查看正式政策收益之前启动。**

### 4.3 duration / route / hub / time

duration 与绝对 mismatch 的 pooled Pearson=0.2425，scenario-cluster bootstrap10000次95%区间=[0.1257,0.3813]；Spearman=0.4204。存在正相关，但**没有达到预注册 duration-specific gate 的 Pearson≥0.30**。

描述性分组平均绝对差：duration≤1 为0；(1,4] 为1242.04；(4,8] 为1799.29；(8,12] 为2849.14；>12 为2486.37 USD。最后一档下降，所以不是“路线越长，差异必然单调越大”，也没有控制 time/hub/供需混杂。

跨 hub：202条，meanabs=2417.03；同 hub：23条，2043.25；wait：54条，0。Wait 只占1期，下一期即释放，所以两个控制协议在此分支重合。

时点 t3 的描述性 meanabs=2467.35，t16=2059.43，t8=1721.89，t12=1425.31，t20=529.53 USD；t12/t20只有8/4个有该分组的独立场景，不能把小样本组写成稳定规律。本段及duration/hub表是候选行加权；图02/03先按各场景取组内均值再等权画t95，因此图中点值与行加权表不要求相同。

释放 hub `32-332` meanabs=2673.92、`04-429`=2452.58，是本样本中较大两组；`04-429` 内严格 comparable pair 的反转率90.63%是**条件分母**，不代表该 hub 所有路线90.63%都错。分组只用于描述和预声明 engineering gate，不以大量 candidate 行构造伪显著性。

完整12维分组在 `processed/target_mismatch_by_group.csv`；原始 frozen queries、四世界目标、pair ranking 与 top1 在 `phase1_identification/`。

## 5. Phase 2：严格同目标的 OLD/ACTION 对照

### 5.1 数据 split 与样本单位

| split | 原始 scenario seeds | 目的 | action target行数 |
|---|---|---|---:|
| construction | 6200--6239 | construction prior、MC Table、拟合 | 3660 |
| validation | 6300--6314 | 共享超参数选择 | 1522 |
| heldout | 7000--7011 | 冻结后的目标预测/排序评价 | 1158 |
| formal | 7000--7039 | 40个独立场景/工况的闭环政策比较 | 非 target row 样本 |

同一个 parent query 使用本场景未来和同 split固定offset7 donor未来，共2条冻结路径；两个世界共享同一未来。两条路径先平均到 parent query，不把它们或 q 扩展当独立统计样本。Construction原始查询分别CA289/W40 681/W80 163/W20 697，共1830条，乘2为3660目标行。

Heldout 每工况请求12个场景，但固定时点实际有合法查询的场景为 **CA9、W40 12、W80 6、W20 12**；对应candidate数88/223/34/234。未补选时点或替换场景。Formal每工况仍严格40个场景；heldout覆盖减少意味着预测比较统计功效不足，特别是W80/3。

另一个依赖限制：第二条路径复用offset7 donor，使一个原始场景的future同时进入自己的标签和另一父场景的标签。因此主heldout的parent-scenario bootstrap/t95忽略了donor跨父场景依赖，只是**条件父场景聚类近似**，不能当无条件独立簇确证。该问题不影响Phase1单自有future和formal40自有scenario的利润统计。另补结构性、明确post-hoc的own-future-only分析：固定模型和预测，仅用原先已保存rep0真实标签；各原始scenario又成为独立单位。未为了得到更有利结果选择路径，也没有改标签重新拟合。

### 5.2 特征与拟合约束

OLD 保留旧 frozen-summary：remaining、other_idle、incoming、backlog、expected、q和release hub。ACTION在此基础上加duration、deadhead、loaded、compatible_load、expiring_load、release_idle、origin_idle、wait、same_hub。

任务书 ACTION 的 remaining/expected 与 OLD 完全相同，incoming=OLD incoming+1。完整字段仍记录，但等价列不重复进入设计矩阵，避免仅复制信息就改变正则化强度。公共 construction 变换、hub/q类别、q-mixture、目标、样本、MC base均逐数值相同。

同 family 的两个模型使用同一候选网格，并以两个 feature set 的 validation ranking loss、top1 loss、RMSE平均顺序**共同选择同一个超参数点**；不是分别给ACTION一个更有利的 degree/penalty。Ridge3点；Chebyshev degree2--6的additive基；Spline knots4/6/8/10、ordinary/natural cubic B-spline和二阶差分惩罚；Bayes共享EB noise/prior scale、hub/q partial pooling和pooling multiplier，闭式Gaussian posterior mean。所有4个EB优化均收敛。

q 行是同一 action label 的固定原slot integration扩展，每条权重1/nq。没有新增route-slot联合变量、full tensor basis、高维nonlinear joint V或MCMC。Bayes posterior covariance是给定EB超参数/似然的条件不确定性，**不是将q/candidate行当独立场景得到的研究结论置信区间**。

Model freeze：2026-10-07 01:52:50 UTC（北京时间09:52:50）；42个模型及配套冻结文件。正式比较预注册更早完成。模型冻结之后才生成heldout target、运行pilot/formal；没有根据它们调参。

数值 QA：Spline正规方程最大condition number约1.458×10¹⁰，来自中心化基与category/intercept等冗余方向；原拟合保留1e-6 diagonal jitter。为避免把“Cholesky返回了数值”当作稳定性证明，另对相同冻结惩罚目标构造augmented-design SVD解，只做独立检查、不部署或选模。32个模型均有限；非Bayes模型relative backward error<1e-10，heldout预测与独立SVD的最大差仅2.645×10⁻⁷ USD；Bayes covariance的对称特征值检查通过。这支持本次样本内数值计算一致，但不能证明未来外推或所有扰动都稳健。逐模型记录见 `processed/fitting_numerical_stability.csv`。

### 5.3 保留的在线结构与不能回避的支持域问题

仍部署 `Vtotal=Vslot+C(route)`，q效果用原固定q-mixture。Residual为 `Yroll−Bslot`。但：

1. Yroll是整个系统相对基线的joint-action优势；Bslot是绝对有限资源边际，两者**不是同一个estimand**。本轮按任务书测试该残差接入，但不宣称存在无误差的逐车可加分解。
2. 多个车辆候选各自的系统优势可能覆盖同一后续机会；将它们加进联合MILP可能产生重叠/非可加性。当前实验没有识别该项到底贡献多少误差。
3. 原 `release≥T → V=0` gate保留。真实Yroll在这个区域可能不为零，因为当前联合动作影响其他车辆在T前的接单，也有booked/realized成本差额。原始标签没有改成假零；训练只用nonterminal释放，heldout同时给出all/nonterminal/terminal。
4. 在heldout中terminal候选：CA40条中33条非零、meanabs4048.72；W40 87条中57条非零、827.48；W80 16条中13条非零、1073.06；W20 92条中47条非零、420.39 USD。因此**完整支持域的精确target alignment尚未完成**。
5. Baseline相对标签严格为零，但未新增每个baseline预测强制归零约束；相关偏离另存 `target_support_audit.csv`，避免在本轮同时改原在线结构。
6. 训练continuation是M0，而部署后的M0+V未来仍每期用V。本轮是对参考M0的滚动动作评价及一步策略改进，不是最终增强策略的自一致Bellman固定点。

这些是数学/集成限制，不是数据泄漏或物理回放失败。MC Table对Yroll的RMSE亦属于异口径诊断；不能据此说MC Table“学错了自己本来的资源目标”。

### 5.4 正式策略与统计规则

P0 M0/Zero；P1 LegacyFluid；P2 MC Table；P3/P4 Linear OLD/ACTION；P5/P6 Cheb OLD/ACTION；P7/P8 Spline OLD/ACTION；P9/P10 Bayes OLD/ACTION；R0 RH4；R1 RH8；D OraclePreview H8。没有RH4+V，不能把M0-trained target称为RH4-aligned。

正式矩阵：4工况×40个scenario×14策略=2240；pilot为4×2×14=112。Scenario hash逐策略相同。正式比较用40个场景配对差，保存mean、SD、median、t95、skewness、10000次paired bootstrap；预声明comparison family内跨工况做Holm两侧p校正。t95区间未做多重校正，和Holm p分列。

Value prediction按scenario-cluster作配对比较；candidate/q/time不是独立replication。纯Y排序按truth严格pair比较，预测tie计半分；top1使用固定hash平局规则，并提供tie-set版本。另记录g+V候选效用诊断；但它也不是完整多车辆MILP的精确排序。

OraclePreview H8是受限窗口的信息参考，不是可部署策略或已认证全日上界。Oracle gap capture只在Oracle−M0分母为正时定义，负/零分母不硬填。

## 6. 预测、政策、运行时与最终问题解答

### 6.1 Held-out：预测改善不等于决策改善

以下是all-scope、两条未来路径先平均后的候选描述性RMSE。单位USD；每格为OLD→ACTION。置信推断另按独立scenario配对，不使用候选行作样本。

| 方法 | CA80/3，n=9 | W40/8，n=12 | W80/3，n=6 | W20/8，n=12 |
|---|---:|---:|---:|---:|
| Linear | 7261.88→6583.70 | 2042.54→1972.70 | 3462.82→2171.74 | 927.12→913.42 |
| Cheb | 7117.17→7304.16 | 2135.61→1980.39 | 3412.67→2983.04 | 922.07→935.11 |
| Spline | 6598.65→6318.78 | 2111.04→1949.05 | 3255.42→2226.53 | 934.13→932.35 |
| Bayes | 7224.57→6379.63 | 2038.31→1965.75 | 3226.01→2082.35 | 942.05→927.57 |

ACTION在14/16个描述性RMSE比较改善；例外为CA Cheb和W20 Cheb。按父场景等权的paired RMSE差，CA Linear=−658.72，t95=[−1014.06,−303.38]、Holm p=0.0433；W80 Bayes=−1414.29，t95=[−2079.88,−748.70]、Holm p=0.0433。CA Linear的MAE也改善，Holm p=0.00792。**这些p/区间仅属于前述忽略共享donor依赖的条件近似，不能据此作无条件确证。** 在独立own-future-only敏感性分析中，pooled RMSE改善为12/16；所有prediction/ranking/top1比较均未通过Holm 0.05。应报告预测改善趋势及条件支持，不应选择较有利的两路径区间隐去这项限制。

候选行pooled ranking accuracy在14/16比较提高，CA Cheb持平、W80 Cheb微降。但场景等权的16个ranking improvement t95均跨零，Holm也均未达0.05；W80 Bayes的场景等权ranking差还是负值。这说明row-pooled趋势与“每个场景等权”的统计问题并不相同。Canonical top1和tie-set accuracy亦单独保存，不能只挑RMSE最有利的口径。

完整MAE/RMSE/R²/Spearman/sign/ranking/top1、all/nonterminal/terminal、场景级差与bootstrap分别见 `heldout_value_metrics.csv`、`heldout_paired_metrics.csv`、`heldout_pooled_RMSE_intervals.csv`、`heldout_candidate_ranking_metrics.csv`。独立路径敏感性在 `heldout_own_future_*`，依赖解释在 `qa/heldout_donor_dependency_audit.json`。额外g+V候选效用排序见 `heldout_g_plus_value_ranking.csv`，没有用其结果重新选模。

### 6.2 正式实现利润

下表每格为40个独立场景的mean realized profit，单位USD；**均值最大不自动意味着显著更优**。

| 策略 | CA80/3 | W40/8 | W80/3 | W20/8 |
|---|---:|---:|---:|---:|
| P0 M0/Zero | 61996.31 | 36548.53 | 37178.68 | 18181.07 |
| P1 LegacyFluid | 34511.78 | 31441.72 | 18549.08 | 15218.47 |
| P2 MC Table | 62022.56 | 41143.69 | 41795.34 | 18908.24 |
| P3 Linear OLD | 60093.88 | 41335.10 | 39655.21 | 18510.67 |
| P4 Linear ACTION | 63582.89 | 41196.17 | 37026.88 | 19388.08 |
| P5 Cheb OLD | 55619.36 | 36824.34 | 40723.01 | 18547.90 |
| P6 Cheb ACTION | 53588.88 | 39032.39 | 37994.12 | 19379.43 |
| P7 Spline OLD | 61102.07 | 37728.26 | 41614.18 | 18511.37 |
| P8 Spline ACTION | 57638.48 | 39003.30 | 38617.04 | 18410.38 |
| P9 Bayes OLD | 59282.61 | 41741.70 | 39165.11 | 18374.68 |
| P10 Bayes ACTION | 61126.61 | 40497.37 | 40431.22 | 19196.73 |
| R0 RH4 | 63618.99 | 38324.93 | 37254.58 | 18819.35 |
| R1 RH8 | 63330.24 | 40225.27 | 36731.06 | 19966.88 |
| D OraclePreview H8 | 67645.59 | 53630.34 | 45624.19 | 24096.08 |

正式ACTION−OLD结果如下；区间为未做多重校正的paired t95，Holm p针对预声明16个feature increment比较。

| 工况 | 方法 | mean差USD | t95 CI USD | Holm p |
|---|---|---:|---|---:|
| CA80/3 | Linear | +3489.01 | [418.92,6559.11] | 0.3775 |
| CA80/3 | Cheb | −2030.48 | [−5667.61,1606.65] | 1.0000 |
| CA80/3 | Spline | −3463.59 | [−6517.38,−409.80] | 0.3775 |
| CA80/3 | Bayes | +1844.00 | [−1642.81,5330.81] | 1.0000 |
| W40/8 | Linear | −138.93 | [−1912.66,1634.81] | 1.0000 |
| W40/8 | Cheb | +2208.05 | [228.81,4187.30] | 0.3775 |
| W40/8 | Spline | +1275.03 | [−660.45,3210.51] | 1.0000 |
| W40/8 | Bayes | −1244.33 | [−2983.73,495.07] | 1.0000 |
| W80/3 | Linear | −2628.33 | [−6314.59,1057.92] | 1.0000 |
| W80/3 | Cheb | −2728.89 | [−5234.84,−222.94] | 0.3775 |
| W80/3 | Spline | −2997.14 | [−6200.39,206.10] | 0.6586 |
| W80/3 | Bayes | +1266.12 | [−2016.52,4548.75] | 1.0000 |
| W20/8 | Linear | +877.41 | [289.39,1465.44] | 0.0715 |
| W20/8 | Cheb | +831.53 | [−140.21,1803.27] | 0.8225 |
| W20/8 | Spline | −100.98 | [−765.83,563.87] | 1.0000 |
| W20/8 | Bayes | +822.05 | [175.74,1468.35] | 0.2102 |

16个mean恰好8正8负。4个正向t95不跨零，2个负向t95不跨零，**所以不能写“所有利润区间都跨零”**；但16个Holm均>0.05，且无方法在四工况保持全正方向，不能宣布可部署的稳定feature increment。

Paired bootstrap多数方向与t95一致；W80 Spline的bootstrap上界约−45.58、t95上界+206.10，边界结论对区间方法敏感，Holm也未支持稳定差异。完整SD、median、skewness（最大绝对值约2.23）、bootstrap及逐场景差均保留，而非仅输出平均数。

### 6.3 MC Table 与复杂V

Cheb/Spline/Bayes-ACTION相对MC Table的12个mean比较中，10个为负；只有W20 Cheb(+471.19)、W20 Bayes(+288.49)为正，二者区间均跨零且Holm p=1。

CA Cheb−Table=−8433.68，t95=[−12404.63,−4462.74]、Holm p=0.00134，是明确的负向证据。CA Spline−Table=−4384.08，未校正t95为负，但Holm p=0.0628，不能按校正后0.05标准宣布差异。没有复杂ACTION方法在任何工况对Table取得校正后的显著正增益。

**Table不是所有continuation V的均值冠军。** 全部V策略中，CA与W20均值最高是Linear-ACTION；W40是Bayes-OLD；W80才是MC Table。如果只比较Table与三个复杂ACTION模型，Table在3/4工况均值领先。这两个集合不同，不能混写“Table四工况最强”。

与Zero比较，W40 Table差+4595.16、t95=[2369.94,6820.39]、Holm p=0.00547；W40 Bayes-ACTION也比Zero提高+3948.84、Holm p=0.0212。后者说明Bayes可以比零V有效，**并不意味着ACTION比Bayes-OLD更好**；它在该工况反而低1244.33。CA Cheb-ACTION比Zero低8407.43、Holm p=0.00604。LegacyFluid在本新矩阵四工况均值均较低，不能将其当作默认最可靠V。

复杂家族也不是“所有差异都没有证据”：预声明的exploratory direct-method family中，CA Bayes-ACTION优于Cheb-ACTION(+7537.73，Holm p=0.00130)及Spline-ACTION(+3488.13，p=0.0261)。但其他工况未形成同样校正后的稳定优势，W20的Cheb均值还略高于Bayes。因此结论是**存在局部优势，未发现全工况稳定赢家**。

### 6.4 RH与Oracle参考

RH8−RH4：CA−288.75（未区分）；W40+1900.34，t95=[469.42,3331.26]、Holm p=0.0317；W80−523.52（未区分）；W20+1147.52，t95=[666.39,1628.66]、p=0.0000871。更长horizon不是四工况必然更好。

OraclePreview均值最高只表示该受限H8预览诊断有更多信息，不能将其当成真实可部署策略或已经证明的全日最优利润上界。Gap文件保留逐场景分母和是否可定义，没有对负/零分母填造capture比率。

### 6.5 运行时

固定查询生成wall：Phase1=1.350s；construction=2.448s；validation=1.566s；heldout=1.328s。MC正常rolling目标构造wall为construction180.053s、validation66.730s。Action target整条pipeline（含回放/存储）wall为Phase1 13.862s、construction57.015s、validation22.756s、heldout17.608s。

Phase1 unique-world求解/执行计时：rolling5.751s、freeze9.907s，总比0.5805。但rolling复用329个世界、freeze因释放阈值不同有563个世界；**总比不是同一世界rolling比freeze便宜42%**。平均每world分别0.017480/0.017597s，比0.9933；即便这个比也有state/threshold mix区别，仅是描述性，不是受控速度因果比较。

四工况OLD/ACTION共同拟合与validation总耗时：Linear1.257s、Cheb3.519s、Spline6.335s、Bayes1.337s；共享网格的一对方法计时，不伪装成单模型时间。正式矩阵wall2311.061s，pilot103.736s，含并发、回放与存储，不能与单日CPU/solver计时简单相加。

| 策略 | CA平均在线日wall秒 | W40 | W80 | W20 |
|---|---:|---:|---:|---:|
| M0 | 0.052 | 0.071 | 0.058 | 0.067 |
| MC Table | 1.768 | 1.928 | 1.519 | 1.045 |
| Linear ACTION | 1.894 | 2.133 | 1.669 | 1.137 |
| Cheb ACTION | 1.948 | 2.189 | 1.690 | 1.180 |
| Spline ACTION | 2.014 | 2.210 | 1.761 | 1.183 |
| Bayes ACTION | 1.947 | 2.171 | 1.667 | 1.148 |
| RH4 | 3.976 | 3.966 | 3.516 | 2.133 |
| RH8 | 5.473 | 5.558 | 4.962 | 3.111 |
| Oracle H8 | 18.471 | 21.403 | 14.264 | 12.581 |

Context preparation/provider/slot/build/solve/day六部分逐运行记录。Provider timer包含slot调用；build为exclusive计时，因此并非六列均可相加。在线runtime也是当前机器并发负载下的wall，不是硬件无关benchmark。完整分解在 `offline_runtime_breakdown.csv`、`runtime_breakdown.csv`。

## 7. 对任务书12个问题的明确回答

1. **freeze与rolling差多少？** 非baseline meanabs1918.40、RMSE3390.86、Spearman0.2037；场景等权绝对差1581.57 USD，t95不含零。测试W40/8内明显不一致。
2. **是否改变candidate ranking？** 是，严格可比较pair反转40.58%；canonical top1不一致38.96%，最优集合完全不相交29.87%。不是仅改变尺度。
3. **是否随duration增大？** 有正相关，Pearson0.2425、cluster区间为正，但不单调，未达预注册0.30 duration gate；不能作独立因果解释。
4. **哪些route/hub/time严重？** 跨hub、duration9--12及>12、早期t3，以及32-332/04-429在该样本中较大；按各组实际scenario覆盖和strict-pair分母解释，不能泛化所有route。
5. **能否解释第四轮RMSE好而profit不稳？** 它证实freeze-control distortion是可信机制，但不能证明是第四轮失败的唯一或主要原因。原第四label与主识别的estimand不同；本轮修正后仍有预测/利润脱节与集成限制。需有相同feature/target语义的额外policy factorial才能量化其因果贡献。
6. **同target下ACTION是否优于OLD？** 有多处预测改善：两路径14/16 pooled RMSE下降；CA Linear/W80 Bayes只有忽略donor依赖的条件校正支持，独立own-future-only为12/16改善且所有Holm未达0.05。Ranking未被独立场景统计明确区分，profit 8正8负，16个Holm均未达0.05。不能作总体“优于”的结论。
7. **预测改善是否转成realized profit？** 未发现稳定转换；W80 Bayes预测改善较大，但其ACTION−OLD利润区间跨零。CA Spline RMSE降低而利润mean下降3463.59，未校正t95为负。少数正向未校正区间存在，但不足以满足任务书跨工况、排序、正式利润共同标准。
8. **MC Table是否仍最强baseline？** 仍很强且无复杂ACTION对其校正后正胜；不过全部V中只在W80均值最高，不能说四工况都最强。结果支持将低维(t,u,q)继续作为第一基线，不证明其信息已充分。
9. **Cheb/Spline/Bayes是否没有稳定赢家？** 全工况没有；CA Bayes对Cheb/Spline有明确局部优势，W20 Cheb均值较高但差异未明确区分。需同时保留局部证据与总体不确定性。
10. **主要误差是否已经转移到q/route integration？** 尚未识别“主要”误差来源。终端gate与系统优势/绝对slot/逐车可加/q平均的不相容已有直接诊断；本轮禁止修改coupling，也没有其单因素ablation，不能凭profit不稳就宣布coupling主导。
11. **是否无需立即实现完整joint post-state？** 当前不支持立即升级，也不能证明永远无用。Solve-after残差与post backlog/idle/incoming/imbalance有描述性相关，最大绝对Spearman约0.377；小样本且未控制混杂，更不是joint V增益实验。优先做低维/集成对照。
12. **下一轮做什么？** 先把标签语义、baseline零锚与terminal支持域、以及slot接入协议明确一致，再将route-slot或粗离散供需bucket设为**各自单独**的预注册对照；保留现MC Table。扩大有合法query的独立heldout覆盖。不要同时改target、coupling和state然后归因一个因素；不应直接进入neural/DRL。

工程决策：Phase1=G1/T1；就“rolling target+ACTION是否产生稳定增益”更接近**有支持域保留条件的G3**，不满足G4。下一轮不能假设所有target语义已经修妥，也不能依据本轮直接宣布(t,u,q)充分、joint V无用或DRL必要。

## 8. QA、异常与交付完整性

- 12个历史M0/RH基线、24个原目标复现、8个新冻结模型独立重新求解均通过。
- 2352个pilot/formal运行、97,300个逐期记录独立无求解器物理回放通过；利润差均0。所有当前分配订单book_time≤t，未完成承诺为0。
- 6,660个主目标世界、201,613个逐期记录单独建立内容hash索引，J/g/fare ownership/冷却完成检查通过。
- 56,448个policy优化期均报告OPTIMAL（是各自受限MILP的最优，不是全日全局最优）；无FEASIBLE_LIMIT。
- 40,320个context-before/after hash检查通过；103,566条selected pre-solve context记录合法，online系数没有actual-post字段；无quadratic value terms。Oracle仅D策略允许preview，不能把它的预览当其他策略信息。
- 前四轮81,632文件、第四轮150交付文件、CFS内容在全平台测试后再次校验未变；第五轮99个Phase1/106个Phase2冻结源码与42个模型及配套文件身份未变。
- 最终全平台pytest：154 passed、0 failed；1条已有Starlette/anyio弃用警告。旧SAC的微型兼容性回归属于全平台测试，不是第五轮数值实验或新增RL学习。
- 32个拟合模型数值稳定性检查通过；独立augmented-SVD未用来替换模型或挑赢家。
- Heldout共享donor依赖已单独披露；新增own-future-only独立敏感性，不把条件parent-block区间伪装成无条件确证，不改变主formal比较。
- 16组论文图有600dpi PNG、embedded-font矢量PDF、SVG、源CSV、plot config；图册16页。图形数据与plotting config可用于本地重绘，非网页截图。

**首个target尝试并非被悄悄删除。** 无求解器回放的物理状态精确相等，但累积cash存在约7.3×10⁻¹²的浮点求和差，触发当时过严的现金精确相等断言。108个partial worlds、一个已完成任务和旧版本身份归档在 `raw/failed_attempts/replay-v1/`。只把cash检查改为1e-6 USD，物理字段仍exact；查询digest不变，在查看分析outcome前重新冻结源码并全量重跑Phase1。旧尝试不混入最终样本。

**首轮全平台测试：152 passed、2 failed。** 原因分别是生成产物时源码身份发生变化，以及Windows读取仍在写入metrics文件的共享冲突。通过第五轮局部`.gitignore`将大量raw/QA渲染排除出平台代码打包，保留初次XML；两个失败项单独复测成功，随后稳定文件下完整154项全过。没有为消除回归失败改写旧研究源码、SAC实现或平台测试断言。

最终状态、交付hash、raw索引、源码分析快照及QA见 `manifests/final_manifest.json`。单个结果必须通过scenario/model/code/observations身份联查，不建议只按相似文件名引用。

## 9. 文件导航与复现

- 本报告：研究结论、比较定义及边界。
- `README.md`：详细模块导航、参数/字段/单位和恢复运行入口。
- `phase1_identification/`：任务书的查询、freeze/roll、差异、ranking/top1和summary快捷文件。
- `targets/`：construction/validation/heldout rolling labels、MC资源边际、残差与provenance。
- `models/`：OLD/ACTION各家族、MC/Fluid、construction prior、共同selection。
- `raw/worlds/`：每个候选/共享baseline四世界的逐t状态、动作、可见订单、求解期、J/g、实现ledger和回放。
- `raw/runs/<run_id>/`：`spec.json`、`summary.json`、`observations.json.gz`。后者含完整逐期trace、decision ledger、metrics和replay；不只有日汇总。
- `processed/`：配对统计、value预测、排序、post-state残差、runtime、图源数据、run hash索引。
- `figures/`：16组600dpi PNG、矢量PDF/SVG、plotting config及图册。
- `qa/`：人工toy、旧baseline/target复现、信息边界、历史身份、独立重新求解、回放、测试与PDF渲染审查。
- `manifests/`：各阶段源码/查询/未来路径/模型/正式比较冻结，以及最终状态与校验清单。

原始压缩JSON可用 `gzip.open(path,'rt',encoding='utf-8')`读取。CSV标量一列一字段；嵌套context/provenance为JSON字符串，保留scenario/policy/model/hash身份，允许后续Python重绘而不用重新仿真。

逐期覆盖边界：6,660个主action/baseline rollout世界保存201,613个逐期记录，全部policy运行也保存完整trace。MC slot底座保持冻结旧构造，保存paired profits、state/future hashes及可重建数据/源码，没有另外给每个MC资源反事实保存完整逐期trace。因此不能宣称“所有offline MC资源世界都已各自保存逐期原始轨迹”。

执行顺序见README。恢复只允许在源码/配置/模型/查询身份完全匹配时使用缓存。本次任务不进行分支切换、提交或推送；已有未提交改动保留。
