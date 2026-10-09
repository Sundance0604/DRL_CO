# 第六次数值实验 A：自一致 MC 资源价值策略迭代

本目录是独立研究实验，不改平台默认算法，不改前五轮结果。当前状态以 `manifest.json` 和 `manifests/final_manifest.json` 为准，不能仅凭本说明认定完成。

## 研究对象

`pi0=M0/Zero → V0=E(pi0) → pi1=G(V0) → V1=E(pi1) → pi2 → V2 → pi3 → V3 → pi4`。

E 的标签只有 **完整 realized ledger 的 J(q)−J(q−1)**，不是 J−g 动作优势，不是 route correction。q 表示查询 hub 的第 q 辆空闲车辆。每对世界相同的网络、订单、参数、当前 pool、其他 hub 车辆和在途承诺，仅多一辆查询 hub 空闲车。仿真在每期继续使用对应 pi；到 T 停止新控制，承诺继续冷却至全部完成。所有负值保留，不强制 q 单调。

这是近似、可分资源价格的 empirical policy iteration，不是精确 Bellman/PI，不承诺最优性或理论收敛。

四次评价只覆盖E(pi0)..E(pi3)，生成pi4；没有额外E(pi4)或第五次改善。经验稳定诊断不等于已测量V3−E(pi4)固定点残差。

G使用原release(time,hub)分组：每辆车进入一个sink，每组前缀slot总量严格等于释放流数量，z_q≤z_(q−1)。负价不能通过漏计车辆或跳选后槽被暗中截断。跨释放时刻分别计价仍是可分近似，没有重建完整联合库存和重叠需求机会；E中query hub的free count条件与G的分组接入应分别理解，不能因此宣称完整状态Bellman价值。

## 模块导航

| 文件 | 唯一职责 |
|---|---|
| `common.py` | 冻结配置、分组种子、源代码/历史内容校验、只读 CFS 来源、新目录数据管理 |
| `protocol.py` | 纯 slot provider、父策略调用来源、配对资源状态、无求解器回放和同状态决策 |
| `simulation.py` | 冻结旧模拟器的机械复制，只增加独立上下文/查表/求解/状态更新/trace/ledger计时；通过旧轨迹与V0精确对照 |
| `phase0.py` | 独立复现第五轮 M0/MC Table/RH、旧 MC 标签、人工两步迭代和 T 后资源零边界 |
| `targets.py` | 每轮重新采集当前父策略状态，计算全冷却配对世界，按(t,u,q)聚合和插值，冻结 V 表 |
| `policies.py` | G(V)及政策反向 lineage；固定策略矩阵；同一 pi0 诊断状态上比较相邻政策动作 |
| `run_experiment.py` | 固定四轮执行顺序、恢复检查、诊断/pilot/formal 调度 |
| `analysis.py` | 独立场景配对统计、表值/动作/分布/q/负值/振荡/运行时诊断 |
| `action_analysis.py` | 规范集合顺序与等价JSON表示，不舍入、不消除车辆ID差异；定义在诊断/正式结果前单独登记，原比较保留于raw |
| `figures.py` | 读取结果生成16组论文图、矢量图册、600dpi PNG、SVG、图源CSV和每图配置 |
| `delivery.py` | 最终 lineage、配对世界、信息边界、回放、历史完整性审查与中文报告 |
| `pdf_qa.py` | 用文档运行环境检查图形字体/矢量/分辨率/渲染及视觉审查状态 |

`tests/test_resource_policy_iteration.py` 在仓库 tests 中，只测试协议，不通过预设利润方向制造“实验通过”。

## 数据、种子和冻结规则

- 工况：CA80/3、W40/8、W80/3、W20/8。四工况物理参数沿用前五轮，不改变 network-rho、容量、fare/cost、deadline、ready-time、路线、extension、wait/reposition 或 solver。
- Construction：8000–8059，每工况60个场景。四次 E 都使用这些固定生成场景，但重新采集当前 pi 的状态。
- Diagnostic：8100–8119，每工况20。不是正式选模集；四次 improvement 固定完成，无按收益停止或选 iteration。
- Pilot：8900/8901。Formal：9000–9039，每工况40。所有政策使用相同场景，正式样本是 scenario，不是 t/hub/q/参数行。
- Policy seed=0。没有训练组件，因此 training seed=null。
- 每个 construction 场景只使用自身未来路径，配对世界共用此路径。没有跨场景共享 donor。它是 sampled-trajectory MC evaluation，有限样本、有限支持，不能解释为充分条件状态的精确期望。
- 时间节点沿用旧 MC 七节点投影网格，固定种子旋转选择 hub，q 查询集合沿用旧配置。节点取法不意味着使用 Chebyshev 拟合器。
- 每个当前 query hub 的空闲资源替换成 q 辆；其他资源保留。冻结引擎要求 `vehs[id]`，所以共同保留前缀做同样编号映射，再依次追加自由资源；配对共同部分字段严格相等。
- q 对照可能使反事实总车数超过 K。为避免缺槽时退回 M0，表覆盖查询上限2K；q>K 为同一线性插值的端点常值，明确标记 `unsupported_endpoint`。正式车队 K 不变。查询 q≤K 中缺测整数槽为 `interpolated_q`。
- time 线性插值；time≥T free resource=0；不变更原 release slot-prefix 约束。没有 ACTION correction、coupling 新变量、RH 训练、RL/神经网络或 Fluid 更新。

## 结果目录与字段

- `config/main.json` 是不可更改的本轮主协议；三个 `.yaml` 文件采用合法 YAML 1.2 的 JSON 子集。记录物理参数、construction/diagnostic/formal seeds、策略、比较族、样本单位、插值与终端规则。
- `manifests/source_manifest.json`：数值源码 SHA、源码压缩快照、配置 SHA、CFS SHA。
- `manifests/historical_readonly.json`：前五轮文件逐项 SHA。最终再次校验，不修改旧目录。
- `iteration_n_query_manifest.json`：在该轮目标运行前冻结场景、time/hub/q规则、父政策 hash；状态由对应政策运行产生，不按目标结果挑状态。
- `iteration_n_manifest.json`：父政策、parent table、construction seed 集、network/code/provider/target-generator/interpolation/terminal/table hash、created_at，以及运行规模和计时。
- `policies/piN.json`：每工况政策反向指向前一张 V 表和 source hash，不能把 pi2 偷换成 pi0 continuation。
- `tables/VN/<condition>.json`：原始 MC 平均资源价值加固定插值。`cells.csv` 每格 value、support_status、independent_scenarios、MC_SD、MC_SE。MC_SE 是该格 construction scenario 离散度，不是全策略性能置信区间。
- `targets/iterN/targets.csv|json`：每原始场景/time/hub/q 一行。`profit_q`、`profit_previous`、`target` 单位 USD；`state_hash` 是父策略原状态，`paired_common_hash` 是共同资源世界；`continuation_policy_hash/table_hash` 标明实际父策略。terminal anchor 单独标识，不能当作新增统计样本。
- `raw/worlds/iterN/*.json.gz`：每个 q 世界的完整逐 t trace、源状态、可见订单、动作、solver、ledger、冷却、回放、policy/table hash 和 provider calls。共享 q 世界只运行/存储一次，不重复仿真虚增样本。
- `raw/formal_runs/<id>/`：正式 spec/summary，summary 引用逐期压缩 trace 与 SHA。`raw/runs/` 是 diagnostic/pilot。
- `raw/traces/`：完整逐 t 原始记录，diagnostic 和 construction parent 还保存可恢复的全状态 packets。包含决策期和 commitment-only 冷却期。gz可通过标准 Python `gzip.open(...,'rt',encoding='utf-8')` 读取。
- `formal_policy_summary.csv`：n、mean、SD、median、t95、bootstrap95、skewness、服务率、在线时间。mean 最大不是显著最优。
- `paired_policy_improvements.csv`：相邻4组、相对 pi0 的后续3组、pi4−RH4/RH8、Preview−pi4。36项主比较跨四工况统一 Holm 校正；4个 Preview gap 描述性单列，不声称最优上界或可学习潜力。
- `table_change_metrics.csv`：shared observed 与 deployment grid 两口径；RMSE、relative_L2、Spearman、符号转换，含 early/middle/late。终端0和未支持 q 不混入主指标。
- 此处 `deployment_grid` 辅指标指**存储时间节点 × 全整数q≤K**，不是声称所有时间都有直接样本。主 `observed_shared` 仍只用直接测得的共同节点/q。
- `deployment_query_provenance.csv`：补充所有整数时间0..T、hub和q=1..2K的实际provider值，明确observed、interpolated_time/q、unsupported_q_endpoint及terminal_exact_zero，保存源时间/q端点与权重。插值/外推不虚构MC_SE或独立样本；直接节点与冻结表逐项核对一致，可用time/hub/q/table_hash关联原始系数。此导出不修改既有主统计或E/G。
- `policy_action_disagreement.csv`：**同一原状态**求解 pi_n/pi_n+1；full/routes/assignment/wait-reposition/release-hub 是否不同。固定 pi0 diagnostic 状态库、全部t；daily 先算每场景变化比例再对20个场景等权统计。不将不同政策各自轨迹在相同 t 的动作差直接称为同状态差。
  分项wait/reposition按车辆的route参与/接受的出发delay或reposition目的hub比较；没有动作的车辆表示idle/wait。因此同一辆车的立即发车和延迟发车不会被误算为相同等待控制。分项定义在看到诊断/正式结果前登记，主full-action hash未改变。
  Canonical序列化保留路径顺序，但统一tuple/list及数值1/1.0、-0.0/0等完全等价的表示，二元bool表示统一为0/1；没有浮点舍入或误差容差，不忽略vehicle ID。每对比较都要求canonical equality与记录hash一致。原始序列标记保留，表示层修订在任何diagnostic/formal输出出现前登记，不修改物理或数值策略。
- `formal_policy_action_disagreement*.csv`：正式集40场景的独立佐证，定义与诊断集相同但分开统计、不用于选策略。正式trace原本不额外收集state packet，因此事后独立重算冻结pi0轨迹，要求利润、每期动作hash和最终物理hash完全复现，状态库保存于 `raw/analysis/formal_state_banks/`。定义在看到任何diagnostic/formal结果前单独登记；仅增加事后分析，不改变E/G数值源码。
- `state_distribution_observations.csv`：政策自己轨迹的 idle/incoming/backlog/committed/utilization/已知 release delay，原始场景/time/hub一行。
- `release_time_observations.csv`、`release_time_histograms.csv`：逐incoming资源的已知release-context时间/延迟与按hub/time/iteration的描述性直方图。不只保存均值。复用冻结 `summary` 的既有承诺投影，不看unbooked订单；它是接口的context语义，不将其宣称为重新设计的释放预测。资源/时间观察并非新增独立统计样本。
- `state_distribution_shift.csv`：SMD、Wasserstein、原范围外占比。描述性重复观察，不能把全部时间格当独立样本。`SMD=null` 表示方差为0且均值不同，绝不伪装成0。
- `q_monotonicity_audit.csv`：违例大小、次数、支持域、MC SE；q估计间相关，不能用独立SE公式直接显著性检验。
- `negative_value_audit.csv`：负值、持久性、支持和噪声检查；`negative_value_components.csv`逐对重建fare/cost/业务增量及first action change。插值负q没有该q直接标签时明确缺失，不伪造因果归因。MC_SE条件于经验parent policy，不包含整条PI重估不确定性；不删掉负值或把所有负值都叫程序错误。
- `oscillation_cells.csv`：各格 V0..V3、sign flip、range、更新方向反转、当前 idle-q 占据频率。occupancy 不是 provider查表次数，也不是选择的 release binary slots 数。
- `runtime_breakdown.csv`：offline wall/worker/parent/world simulation/serialization/aggregation/interpolation/QA；online context、纯provider查表、MILP build、solver、state transition、postprocess、trace、realized accounting和日级总计。lookup是build子集，release-slot assembly含lookup，cooldown是transition子集，不能叠加嵌套计时。pi1..pi4独立计时；原M0内部build/extraction保持原代码，只以planner overhead合并标识，不把代理当纯build。
- `offline_evaluation_gain_efficiency.csv`：每次额外E的分工况worker耗时、对应相邻配对收益及USD/离线worker小时；保留负收益，不解释成真实现金ROI。
- `qa/`：基线、资源配对、lineage、terminal、信息边界、独立物理回放、pytest、图形/字体/矢量/视觉审查与旧结果完整性。
- `figures/`：16组 PDF/SVG/600dpi PNG，`source_csv/` 和 `plot_config/`，`figure_atlas.pdf` 汇总。论文图来源于 Python，不是网页截图。
- 生成的 `figures/` 不属于平台源码快照，因此本目录 `.gitignore` 排除它们，避免实验论文图将平台复现包推过既有100MiB安全上限。文件仍完整保存在本地，并纳入本实验 `final_manifest` 交付SHA；没有删除图，也没有放宽平台安全限制。可用本目录 `figures.py` 从分析数据重绘。

## 恢复与重绘

在 `_review_repo` 根目录运行，Windows PowerShell 设置 `PYTHONHASHSEED=0`、`PYTHONIOENCODING=utf-8`、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。

```powershell
.venv/Scripts/python.exe -m experiments.cfs_resource_policy_iteration.run_experiment --stage phase0
.venv/Scripts/python.exe -m experiments.cfs_resource_policy_iteration.run_experiment --stage all
.venv/Scripts/python.exe -m experiments.cfs_resource_policy_iteration.analysis
.venv/Scripts/python.exe -m experiments.cfs_resource_policy_iteration.figures
.venv/Scripts/python.exe -m experiments.cfs_resource_policy_iteration.delivery
```

源码/配置/模型身份不一致时不得复用缓存。初次构造为补齐计时主动中止，完整保留于 `raw/failed_attempts/timing-v1`，重做Phase0并全量重跑；新旧V0四张表SHA必须完全相同。异常保留原始记录并停止，不能暗中换标签、放松 solver、截短冷却或选有利 formal seed。只有所有 QA 和最新图视觉审查通过，final_manifest 才允许标记完成。可选 damping 在本协议中未运行，不能按正式结果反选 alpha；第六轮B尚未授权。
