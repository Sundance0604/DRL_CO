本报告来自用户授权的新独立 exact-accounting B attempt。原 B 在 seed8405 触发金额一致性停止，完整保留且只读。唯一修正是从原求解器提取的整数动作独立重算执行金额，保留原浮点profit/objective诊断；不改变动作、求解器容差、物理、种子、分桶和统计规则，全部B数值重新运行，未混入原attempt缓存。A2仍为原始不可变冻结结果，跨A2/B解释须保留这一数值核算修正的边界。

# B：最小供需状态扩展的独立数值实验

## 主要结果

本轮三桶策略 B2 没有改善闭环利润。在预注册的四工况主比较中，CA、W40、W80 的 B2−B1 配对利润差分别为 −10,650.02、−1,619.40、−3,592.99 USD；对应 95% t 区间全部位于零以下，四项 Holm 校正 p 分别为 9.70×10⁻⁷、0.04605、0.006143。这不只是“未发现正向收益”，而是本轮固定部署协议下有统计证据的负向利润效应。W20 的 +10.99 USD 区间跨零，不能声称改善。

W80 的独立诊断标签预测 MSE 得到改善（base−bucket = +918,975.88 USD²，Holm p=0.03926），但其闭环利润反而下降。预测标签更准不保证部署策略更好，尤其当前桶用于未来释放价格的转移近似、可分资源价格和策略改变后的状态分布仍未单独识别。不能由这一结果断言所有状态扩展无效，也不能自动推导必须使用 RL。

数值与分析已完成；全部测试、图表视觉审阅、历史只读校验及最终冻结，以 manifests/final_manifest.json 为准。

## 1. 实验边界与数学定义

A2完成并冻结后才启动B。B不使用阻尼，不做bucket PI，只在共同M0父策略下构造一次base与bucket MC value；MILP约束、收入成本、资源替换与完整冷却均不变。
R=(Bload+Dnear)/(Midle+Inear+1)，Hb=4、eta=1。订单load按capacity=40折成车辆当量，资源是车辆数。Bload只含可见ready backlog；Dnear=已知且近4期ready bookings+construction历史均值的尚未预订近期待到load，互不重复。Inear只含已锁定承诺的近4期最终release，不是优化预测供给。
Inear复用现有rolling.audit的最终idle_release事件，不把历史last-delivery时间当释放，也不重复处理已到达的当前期事件。实际优化/推进产生的边界反例测试通过，历史物理与摘要代码保持只读。
每工况construction的M0原始状态上固定c33/c67：R<=c33 surplus；c33<R<=c67 balanced；R>c67 scarce。阈值相等时不人为抖动或重分桶，balanced可为空。
Y=J^M0(q)−J^M0(q−1)是完整冷却后的finite-resource边际值，不是route advantage、ACTION bonus或Q(S,a)。在原始共同状态计算一次桶，再移除查询hub的idle并分别注入q/q−1，两世界共用原始桶、原始可见状态和同一未来订单。
在线当前观测hub桶作为未来release(time,hub)的查表上下文；没有candidate-dependent post-state或未来桶预测。因此它仍是可分slot价格近似，当前桶向未来释放时点的转移假设是明确限制，不能当作完整条件Bellman价值。

## 2. 数据、共享标签与样本单位

Construction8300–8359；diagnostic8400–8419；pilot8960–8961；formal9200–9239。四工况正式各40个独立scenario，policy seed=0，training seed=null。B0/B1/B2/R0/R1/D全部用完全相同场景配对，共960次formal仿真。
B0=M0；B1=共同标签的base V(t,u,q)；B2=相同标签按pre-injection桶分组的V(t,u,q,b)；R0/R1=RH4/RH8；D=OraclePreviewH8，不是经过认证的全日upper bound。
为使三桶有可用支持，base与bucket都采用同一7个历史time节点、全部5hub和原q节点；每原始tuq先有60construction场景再分桶。这是B内部共同采样设计，相对A的hub轮换加密，不能把B1−旧A1的差单独归因于状态扩展。唯一直接扩展对照是同标签B2−B1。
54,600个construction和18,200个独立diagnostic资源世界全部保存逐t动作、状态、ledger、冷却及回放。诊断标签仅用于独立预测误差佐证，不拟合表、不选阈值、eta或算法。

## 3. Support与回退

原始cell记录count、sample_count、conditional MC_SE、bucket、terminal anchor。查询需要的任何非终端线性插值源角count<10，或q>K，则整个查询明确回退到同一base V(t,u,q)。终端t>=24恒为0，不要求抽样支持。负值保留；不施加q或bucket单调性。
部署整数t/q全网格另存source corners、最小源count、interpolated/unsupported/fallback标识。插值后的联合MC_SE未估计，不将源格SE伪装成插值SE。
count>=10是最低样本门，不是完整状态或桶内R范围充分覆盖的证明。原targets兼容legacy feature_row不进入两种MC均值拟合；权威桶输入是pre_injection_pressure。经济收入成本与网络坐标/载荷转换仍是固定实验假设，利润不是CFS直接观测财务结果。
fallback rate主口径是每场景全部pre-solve coefficient调用中的回退比例，再跨40场景取均值。另报汇总池化率、非终端池化率、各桶/原因计数。它包含最终未选用的slot及终端零查询，不能解释为已执行资源的回退占比。

## 4. 正式利润与配对统计

| 策略 | california_corridor-n80-k3 | western_multicluster-n40-k8 | western_multicluster-n80-k3 | western_multicluster-n20-k8 |
|---|---:|---:|---:|---:|
| B0 | 57606.42 | 37643.55 | 34947.35 | 18124.18 |
| B1 | 68615.23 | 41796.40 | 42102.15 | 19166.79 |
| B2 | 57965.21 | 40177.00 | 38509.16 | 19177.78 |
| R0 | 61267.61 | 39207.47 | 35739.46 | 19398.28 |
| R1 | 60550.13 | 40083.93 | 35815.47 | 20014.21 |
| D | 66239.50 | 52207.42 | 46529.88 | 24200.13 |

主family是4工况B2−B1，统一两侧Holm；16项其他算法比较单独Holm；Preview差只作描述。t95与10,000次paired bootstrap是未校正区间，Holm p单列。SD、median、skewness与逐scenario差保留。t/hub/q/bucket/query不是独立replication。

| 工况 | B2−B1 USD | t95 | bootstrap95 | Holm p | diagnostic方向 | fallback均值 |
|---|---:|---|---|---:|---:|---:|
|california_corridor-n80-k3|-10650.02|[-14103.788939334145, -7196.2552140522985]|[-13892.741268739472, -7323.2916484021725]|9.70396e-07|-5194.18|0.2876|
|western_multicluster-n40-k8|-1619.40|[-3003.596692052073, -235.20675907730106]|[-2891.95220197287, -256.8689887535298]|0.046049|-2486.80|0.1556|
|western_multicluster-n80-k3|-3592.99|[-5792.38565159383, -1393.5945094486]|[-5680.913303065397, -1474.4051092839593]|0.00614296|-3246.98|0.2752|
|western_multicluster-n20-k8|+10.99|[-610.620972397656, 632.6106596022239]|[-611.6163224221367, 568.319554992412]|0.971643|+127.76|0.2910|

## 5. 异质性、动作与机制证据

Construction采用同tuq总体方差=加权桶内方差+桶间方差的精确分解。分组后加权桶内方差必然不增，不能单靠这项下降证明状态有用。
独立diagnostic用冻结construction表预测Y，计算base/bucket squared residual；先将一个scenario的全部cell误差差取平均，再做20独立scenario的paired t95/bootstrap及4工况Holm。要求base−bucket MSE均值>0、t95下限>0、Holm<=.05，才视为异质性减少的独立佐证。不用diagnostic分组均值重新中心化或拟合。
同状态动作分析以B0完整24期状态为固定基准，两方分别求解。输出精确完整hash、route、assignment、wait/reposition及release-hub差；不舍入，不把不同车辆ID合并。桶占比按每场景24期×5hub计算后跨scenario统计。

预注册bucket effectiveness证据=False；关键工况多数正均值=False；问题工况联合利润/独立MSE/支持证据=False；W80非劣证据=False，容差=421.02 USD。
问题工况需至少CA或W40达到>=1%base均值的正式配对改善及Holm<=.05、diagnostic方向一致、独立MSE佐证和fallback均值<=.5；关键工况fallback也须<=.5。W80要求配对t95下限>-1%base均值，不以不显著代替非劣。未达联合阈值只能说证据不足，不能证明bucket无效或不存在state aliasing。

| 工况 | 独立MSE改善USD² | 方差佐证 | 稳定利润佐证 | 非终端回退率 |
|---|---:|---|---|---:|
|california_corridor-n80-k3|575311.736|False|False|0.2882|
|western_multicluster-n40-k8|38795.057|False|False|0.1541|
|western_multicluster-n80-k3|918975.877|True|False|0.2743|
|western_multicluster-n20-k8|5821.277|False|False|0.2888|

## 6. QA、运行时与交付

阈值与期望prior从construction独立复算；所有世界的原始pre-injection压力、桶、资源初始状态、同future及边际标签校验。Base/bucket从相同标签逐格重算，unsupported占位值必须经查询回退门。在线所有桶query和保存MILP系数由逐t合法状态与冻结表独立重算。
所有派发期求解OPTIMAL、无超时静默接受，T后只完整冷却无新dispatch，cash与physical replay一致。传统策略与base adapter先在历史场景独立回归；历史A、A2与更早研究全部内容hash最终再检查。
十一组600dpi PNG/vector PDF/SVG、source CSV和plot config，并有十一页本地图册。Offline wall/worker与在线context/lookup/build/solver/transition/ledger分开，嵌套计时不相加。
单次parent M0改进不意味着自一致或收敛；不使用神经网络、DRL、ACTION correction或阻尼，没有更新平台默认算法。仍有有限样本、稀疏桶、当前桶向未来释放时间转移、可分resource prices及有限PreviewH8的限制。失败不自动支持Strategic RL：需独立确认充分支持、剩余全日政策gap、joint interaction/nonseparability。

具体文件导航、字段和恢复方式见README。A2与B分别冻结，不共享formal数据做参数选择，不自动启动B-PI或RL。
