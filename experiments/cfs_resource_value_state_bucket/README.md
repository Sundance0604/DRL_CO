# B：最小供需状态扩展实验

**当前已按停止条件停止，尚未完成。** 诊断seed8405的myopic资源世界触发金额一致性检查；详情见 `B_stop_condition_diagnostic.md` 和 `qa/stop_condition_failure.json`。数值/分析冻结源码未改，construction数据保留，正式实验未启动，等待是否另建修正attempt的用户确认。

这是独立的 single 科研实验包，不是实验平台默认算法，也不包含 DRL/SAC、BHH、阻尼或 ACTION correction。A2已于2026-10-08 03:04北京时间完成全部QA和不可变冻结；289个最终交付文件逐字节校验通过后，B主流程于03:05启动。入口仍严格要求A2的 `manifests/final_manifest.json` 为 `COMPLETED`，所有旧实验、A2与CFS只读。B当前进度查看本目录 `manifest.json` 和各阶段manifest；正式研究结论必须等本目录最终核验、报告和 `COMPLETED` 冻结，不能把启动或数值完成当作全部完成。

## 研究问题和严格对照

只检验增加一个合法当前供需桶是否能减少 `V(t,u,q)` 的状态混叠。共同父策略 `M0` 下生成同一组完整冷却资源配对标签 `Y=J(q)-J(q-1)`：B1 使用无条件均值；B2 使用三桶条件均值。禁止重新做多轮 bucket PI、加入 damping 或更改 MILP 约束、收入成本、承诺和路由执行。

策略为 `B0=M0`、`B1=base MC`、`B2=bucket MC`、`R0=RH4`、`R1=RH8`、`D=OraclePreviewH8`。D 只提前查看8期，并不是经过认证的全日最优上界。

### 新增唯一状态

`R=(Bload + Dnear)/(Midle + Inear + 1)`；固定 `Hb=4`、`eta=1`。

- `Bload`：当前可见且已经 ready 的待服务订单载荷，除以 capacity=40 转为车辆当量。
- `Dnear_known_bookings`：当前已经预订、将在近4期 ready 的可见订单载荷；不重复计入 ready backlog。
- `Dnear_prior`：仅由 construction 历史场景生成的未预订近期待到载荷均值，不查看当前场景实际未来订单。
- `Midle`：当前 hub 的闲置车辆数。
- `Inear`：既有锁定承诺将在近4期最终释放到此 hub 的车辆数；不是优化器假设会获得的供给。
- `+1`：固定平滑项，避免零分母；不是额外物理车辆。

`Inear` 复用现有 `rolling.audit.incoming_supply` 的最终 `idle_release` 事件；不复用旧摘要的 last-delivery 时间，也不重复处理已完成的当前期到达。原始摘要和物理代码保持只读。记录每辆近期待释放车辆的ID/hub/eta，边界反例测试核验此口径。

Ready 判断复用现有 `ready_time(order,t)`：如果未来接入有未揭晓批次ready时间的订单，cutoff前使用合法上界，不偷看实际ready结果。当前CFS场景的ready在booking中已知，此保护不改变本轮既定数据与模型。

Construction 的 M0 原始状态形成每工况的 R 分布，按 1/3、2/3 分位数固定 `c33/c67`。桶0 surplus 为 `R<=c33`；桶1 balanced 为 `c33<R<=c67`；桶2 scarce 为 `R>c67`。相同阈值保留，balanced 可能为空，不人为调整阈值追求三等分。

### 配对世界与在线语义

桶在原始共同状态计算一次，再移除查询 hub 的原 idle 并注入 q 或 q-1；不能让额外车辆改变所属桶。保留其他 hub 资源和所有承诺，两世界用完全相同的未来订单与车辆共同前缀。

在线查表使用当前 hub 桶为未来 release(time,hub) slot coefficient 选择价格，不预测未来桶，也不计算每候选路线的 post-state 桶。这是明确的当前 regime 转移近似，不是完整 Bellman value、Q(S,a) 或 route advantage。多个释放组仍按独立资源价格计价。

## 样本、冻结顺序与采样设计

| split | seeds | 用途 |
|---|---|---|
| construction | 8300-8359，60场景/工况 | historical prior、阈值、共同 M0 标签和两种表 |
| diagnostic | 8400-8419，20场景/工况 | 冻结表的闭环诊断及独立标签预测误差核验；不重新拟合或调参 |
| pilot | 8960、8961 | 冻结后的运行完整性检查 |
| formal | 9200-9239，40场景/工况 | 6策略严格配对评估，共960次仿真 |

工况仍为 CA80/3、W40/8、W80/3、W20/8，真实 CFS 来源与既有经济/网络假设固定。policy seed=0，training seed=null。独立统计单位是 scenario；time、hub、q、bucket 和查询不是额外实验样本。

Base/bucket 共同采样每场景7个历史时间节点、全部5 hub、原始 q 节点。每原始 tuq 先有60场景再分桶。相比 A 的 hub 轮换，B 的两种表一起加密：不能把 B1 与旧 A1 的差归因于 bucket，直接扩展对照必须是同标签 B2-B1。

A2 完成后先逐字节冻结 A2 和前六轮只读产物，再独立历史回归、冻结数值源码、生成 construction 数据、采集 M0 压力、冻结阈值、生成共同标签、冻结表。诊断开始前冻结分析源码；诊断后不调阈值/参数，冻结策略，然后 pilot/formal。正式集不能用于任何 alpha、阈值或表选择。

## Support 与显式 fallback

原始节点样本数不足10，或查询 q 超过观测 K，则回退到同一 base 表。线性 t/q 插值需要多个源节点时，任何非终端源节点不足10，整个查询均回退；不混合部分不受支持的桶占位值。t>=24 恒为0，是物理终端规则，不是缺样本回退。负值、q及桶非单调保留。

每次调用记录实际来源、桶、压力、角点支持与回退原因。主回退率是每场景所有 pre-solve coefficient 查询的回退比例，再跨场景统计；包含最终没被选择的 slots 和终端查询。另报非终端率、池化率、按桶与原因计数，不能当作实际执行资源占比。

插值点没有伪造联合 MC_SE；原始角点条件 MC_SE 在支持表中保存。它不包含阈值/prior/父策略整体重新估计的不确定性。

`>=10` 是样本数门，不等于覆盖了全部完整状态或桶内R范围。原始targets为兼容保留的legacy feature_row不参与base/bucket均值拟合；桶的权威输入是 `pre_injection_pressure`，尤其不能把legacy incoming特征当作精确Inear。

## 模块导航

| 文件 | 职责 |
|---|---|
| `common.py` | B 独立配置、种子、A2完成门、源码与历史文件冻结 |
| `pressure.py` | construction expected prior、合法在线压力、阈值与桶 |
| `provider.py` | 冻结 bucket/base 表查询、支持角点、显式回退和完整 provenance |
| `simulation.py` | 保留原物理控制，仅将 summary 换成增加合法桶的上下文；不改MILP |
| `protocol.py` | base/bucket provider adapter、完整冷却与回放、同状态独立求解 |
| `targets.py` | 原始 M0 压力调查、先冻结阈值、共享配对标签与两类表 |
| `policies.py` | 6策略独立身份、闭环仿真、逐t state/ledger/query 和固定B0同状态比较 |
| `phase0.py` | 历史 M0/base/RH4/RH8 与全unsupported强制base回退的独立回归 |
| `run_experiment.py` | 严格construction→diagnostic→freeze→pilot→formal；数值完成不等于QA完成 |
| `analysis.py` | scenario配对统计、support/occupancy/fallback、独立预测误差、动作与runtime |
| `figures.py` | 11组论文图、600dpiPNG/矢量PDF/SVG/数据/config/图册 |
| `delivery.py` | 逐世界和在线系数核验、历史只读核验、证据阈值报告与最终冻结 |

## 数据目录和字段

| 目录/文件 | 内容 |
|---|---|
| `config/main.json` | 参数、split、采样、支持门槛、研究边界 |
| `manifests/analysis_preregistration.json` | 在数据结果前固定比较family、独立方差佐证与回退率口径 |
| `manifests/historical_readonly.json` | 前六轮与A2全部既有文件SHA；历史文件只读 |
| `manifests/threshold_freeze.json` | construction阈值/prior hash；标签与formal均未见 |
| `manifests/*_query_freeze.json` | 时间/hub/q、场景、M0父策略、solver等冻结查询身份 |
| `manifests/table_freeze.json` | 8张base/bucket表和共同targets SHA；diagnostic/formal未见 |
| `manifests/policy_freeze.json` | 诊断后的固定策略声明；未见formal，无超参数选择 |
| `manifests/final_manifest.json` | 完整QA后 `COMPLETED`、所有最终交付文件SHA |
| `raw/source.snapshot.zip` | 数值源码/config复现快照 |
| `raw/datasets/` | 独立场景JSON和真实CFS/抽样来源 |
| `raw/survey/` | 原始construction M0完整轨迹和native-node压力 |
| `raw/construction/`、`raw/diagnostic/` | 每资源世界initial、trace、ledger、future身份、原始压力/桶与回放 |
| `raw/runs/` | 每闭环run的完整spec和summary |
| `raw/traces/` | 全24期state packets、动作、现金及T后冷却、B2全部query日志 |
| `targets/construction/` | 共同训练标签JSON/CSV；base和bucket严格同源 |
| `targets/diagnostic/` | 独立M0标签，只核验固定表，不再拟合 |
| `tables/priors.json` | base prior、construction expected pressure prior、condition-specific cuts |
| `tables/base_tuq/` | 4工况无条件MC均值；线性time/q插值 |
| `tables/bucket_tuqb/` | 同源base、3桶表、原始节点counts、观测q范围与支持规则 |
| `policies/` | 6策略独立record/policy/table/context哈希；无training |
| `processed/` | 下表可直接供Python按scenario配对分析；不是平台数据库 |
| `qa/` | 物理回放/lineage/support/pressure/历史/测试/图表视觉核验 |
| `figures/` | 11组图和11页图册、source CSV、plot config |

`policy_hash` 表示行为策略身份；B0 与共同M0标签严格一致。`record_hash` 是包含上下文等完整登记信息的身份；不要把二者混用。`scenario_hash` 保证跨算法同数据，`table_hash` 锁定实际部署表，`identity` 锁定完整run缓存输入。

### processed 中的关键文件

- `pressure_construction.csv`：原始状态R及Bload/Dnear/Midle/Inear/visible IDs/window，形成阈值前采集。
- `bucket_thresholds.csv`：每工况c33/c67、count、ties、边界定义。
- `support_audit.csv`：原始time/hub/q/bucket的sample_count、raw mean、MC_SE、terminal/interpolated/unsupported/fallback。
- `deployment_query_provenance.csv`：全部整数time=0..24、q=1..2K、hub/bucket实际price、source角点、最小sample_count、回退原因；MC_SE未估计处为null。
- `bucket_contrast.csv`：同tuq三桶raw conditional mean、MC_SE与实际supported/base-fallback价格，便于识别“原始桶差异未被部署”。
- `within_cell_variance.csv`：construction总体/加权桶内/桶间方差精确分解，USD²；分组必然使加权桶内方差不增，仅描述。
- `diagnostic_target_predictions.csv`：独立标签与冻结base/bucket预测、两种squared residual、差、fallback。
- `diagnostic_MSE_daily.csv`：先在每scenario内部平均cell误差，保留20独立样本，避免伪重复。
- `variance_corroboration.csv`：base-bucket MSE改善的scenario级mean/SD/median/t95/bootstrap/skewness/Holm；必须独立正向才作为异质性减少佐证。
- `formal_policy_summary.csv`：6策略每工况利润统计、service与online时间；同场景跨算法配对。
- `paired_policy_differences.csv`：B2-B1主family4比较，其余16比较独立校正；D只描述。CI未多重校正，Holm_p单列。
- `formal_paired_scenario_differences.csv`：逐scenario配对差；绘图/复算必须通过seed连接，不能按任意行号连接。
- `action_disagreement.csv`：固定B0完整状态双方独立solve的精确动作差和hash；t不是独立样本。
- `formal_action_daily.csv`、`formal_action_summary.csv`：先按24期求scenario比例，再跨40scenario统计。
- `bucket_occupancy.csv`：每策略桶的scenario平均比例，全部time×hub等权；t95/bootstrap仍以scenario为单位。
- `formal_fallback_queries.csv`：每次B2调用的实际R/bucket、price、corner counts、fallback/reason、当前桶语义。
- `fallback_audit.csv`：各scenario/bucket全部与非终端查询数、回退率和原因计数。
- `formal_fallback_summary.csv`：daily mean/t95、pooled rate、nonterminal pooled rate，分母不能混用。
- `runtime.csv`：逐formal scenario/policy在线、solver、lookup/context/build/transition/ledger和pipeline计时。嵌套项不能全部相加。
- `evidence.json`：预注册联合机制证据，不满足不自动证明bucket无效。

## 证据标准和图表规范

关键CA/W40/W80多数B2-B1正均值；CA或W40至少一个达到>=1%base利润、跨4条件Holm<=.05且诊断方向一致，并在同工况得到独立diagnostic MSE改善证据；W80配对t95下限>-1%base利润；关键工况回退daily mean<=.5。W20不决定主要成败。

“桶内方差下降”不能单独作为有效性依据：分组数学上自然下降。独立MSE以冻结construction预测，不用diagnostic均值重新中心化或拟合。正式profit和diagnostic MSE是不同证据，不能合并成额外样本。

绘图使用 Matplotlib 科学后端，可替换；颜色与平台模型accent独立。PDF使用嵌入字体的矢量对象，PNG600dpi，SVG文本保留；hatch/marker/line style支持黑白打印。每组source CSV/config保存数据hash、尺寸、样本与选格规则。极端cell图只从construction选择，不按formal收益选图。实际PDF渲染必须逐页视觉检查后才标记PASSED。

## 运行与恢复

使用项目已有 `.venv`，不切换共享平台环境；Gurobi版本/solver规则与A一致：单worker solver线程1、seed0、gap0、最多3worker。环境固定PYTHONHASHSEED=0、PYTHONDONTWRITEBYTECODE=1、UTF-8和单线程BLAS。主数值入口：

```powershell
.venv/Scripts/python.exe -m experiments.cfs_resource_value_state_bucket.run_experiment --stage all
```

`--stage phase0/construction/diagnostic` 可停在指定阶段；代码/hash一致的缓存可安全恢复。完整数值完成后还需analysis/worlds/history/图表/视觉/full tests/freeze，不能仅见正式CSV便宣布完成。数值源码冻结后如需修改，先保留原attempt并重启独立attempt，禁止覆盖缓存掩盖改变。

独立交付入口 `delivery --stage analysis/worlds/history/freeze`。图表以 `figures.generate(cfg)` 生成；PDF QA复用A2的通用只读渲染脚本，输出只写本目录QA，不改变A2。所有最终报告以实际数值填充，不生成预设研究结论。

科学计算CSV由数值环境生成，不强制改写成Excel；PDF技术核验用Codex bundled runtime/pypdf/Poppler，不安装额外TeX。大型raw/processed/QA/figure产物本地保留并最终hash，按本目录.gitignore排除Git及平台源码打包，避免将海量仿真数据混进100MiB源码包。

## 停止条件与下一步边界

任何未来未预订实际订单进入online bucket、资源注入后重新分桶、阈值/参数根据formal选择、静默fallback、物理/replay不一致、non-optimal静默接受都会停止。任何bucket+阻尼/ACTION/RL混合改变均不属于此实验。

本轮不自动启动B-PI或Strategic RL。联合报告需区分证据支持和证据不足；双双未达阈值不等于证实结构极限。RL还需要充分MC支持、剩余全日policy gap、joint interaction/nonseparability等独立证据。这里的PreviewH8差不能单独当作RL可回收收益。
