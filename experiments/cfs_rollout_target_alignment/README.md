# 第五次数值实验：Deployment-Aligned Rollout Target

本目录是一个独立数值研究，不是对实验平台界面或主模型的修改。前四轮源码、冻结模型、原始结果和报告均不写入。当前研究没有 DRL、SAC、神经网络或 MCMC。

## 先读什么

- `deployment_aligned_rollout_target_report.md`：最终结论、证据、12 个研究问题、限制与下一步。
- `figures/figure_atlas.pdf`：论文图册。各图还有单独的矢量 PDF、SVG 和 600dpi PNG。
- `processed/phase1_summary.json`：第一阶段识别结果及 T0/T1 决策。
- `processed/policy_summary.csv`、`processed/paired_policy_differences.csv`：正式利润和配对比较。
- `manifests/final_manifest.json`：最终交付状态、文件校验、测试与回放数量。

## 与第四轮的关键区别

第四轮 action label 实际上是在预先选择一个可行订单子集、冻结其他新分配到候选释放后，计算该释放状态下 `J(q)-J(q-1)`。这是**资源边际收益**。

第五轮新标签是**候选当前动作相对原始 M0 baseline 的未来收益优势**：

\[
Y^{roll}=(J^r-g^r)-(J^0-g^0).
\]

这两种量不能直接用于识别“控制是否冻结”的影响。因此经用户明确确认，本轮主比较采用相同的当前联合分配、相同的 `J-g`、相同的初始状态与未来订单，仅改变后续控制：

- `rolling`：当前只约束该车的一个路线变量；从下一期起，每期重求解原始 Myopic M0。
- `freeze`：当前同样联合求解；从下一期到该**查询候选的**释放时间之前，全体车辆仅执行承诺/等待；之后恢复 M0。
- 两个 freeze 世界使用同一个查询释放阈值，不按 baseline 自身路线再选择一个不同阈值。
- 旧第四轮 `q=1` 资源标签另列在目标文件中，标记 `fourth_diagnostic_only=true`，不参与 Phase-1 gate。

`g` 是业务利润，不是含 delay tie-break 的求解器目标。`J` 使用完整冷却期的实际交付与车辆运行账本。`J-g` 也包含“分配时整条路线计费”与“实际载客/空驶成本”的核算差额，不能简单解释成释放后的车辆利润。

## 文件与模块导航

| 路径/文件 | 职责 |
|---|---|
| `common.py` | 隔离输出、历史只读文件校验、CFS 校验、源代码快照、基线复现 |
| `forced.py` | 独立复制原始单期 M_t；仅增加一个 `y=1` 或可行 wait 约束；订单 x 和其他车辆仍联合优化 |
| `protocol.py` | 未来订单仅由调度器揭示；逐期执行、J/g 核算、冷却期、无求解器物理回放、压缩原始记录 |
| `queries.py` | 只依据当前可见状态和候选元信息选查询，先冻结、后计算目标；状态打包与恢复 |
| `targets.py` | Phase 1 的四世界严格配对：候选/基线 × freeze/roll；旧资源标签诊断 |
| `diagnostics.py` | 预注册的 sign/ranking/top1、分组、场景级区间与 gating |
| `toys.py` | 小型 A--B--C 人工网络，检查即时收益扣除、释放状态与冻结控制扭曲 |
| `mc_slots.py` | 独立重建正常 rolling 的有限资源边际 MC Table；不冻结到释放 |
| `phase2_targets.py` | Phase 2 的滚动 action target，两条同 split 未来样本路径 |
| `features.py` | OLD/ACTION 信息边界、共享基函数、固定 q-mixture、在线 ValueProvider |
| `fitting.py` | MC base、残差、Ridge/Chebyshev/Spline/Gaussian EB；OLD/ACTION 共同选超参数 |
| `policies.py` | 14 策略矩阵；只读冻结模型；逐期 trace、日终 ledger、独立回放 |
| `phase2.py` | 新 split 及 construction → validation → freeze → heldout → pilot → formal 编排 |
| `statistics.py` | 预声明配对比较、场景级统计、Holm 校正、solve-after post-state 审计与历史标签复现 |
| `figures.py` | 从分析数据生成论文级矢量图，与实验存储协议分离 |
| `delivery.py` | 任务书目录快捷副本、历史/CFS 二次校验、冻结身份、原始 trace 全量审计、独立重求解复现和最终文件索引 |
| `run_experiment.py` | Phase 0、冻结查询、Phase 1 与自动 gate 的入口 |
| `config/main.json` | 实验参数、种子、候选上限、阈值；冻结后不得就地改参数继续复用结果 |
| `manifests/` | 旧文件校验清单、来源版本、预注册、查询冻结、模型冻结、最终交付校验 |
| `raw/datasets/` | 本轮独立抽取的 CFS 订单场景与 observed/generated provenance |
| `raw/query_states/` | 每个场景固定时点的完整可恢复状态，JSON gzip |
| `raw/worlds/` | 每条候选或共享 baseline 的 full-cooldown rollout；每期前后状态、动作、可见订单、求解期列表、最终账本 |
| `raw/targets/` | 可恢复的逐场景/未来路径任务记录，不以分析表替代观测 |
| `raw/runs/` | 正式和 pilot 运行的 `spec.json`、`summary.json`、`observations.json.gz` |
| `raw/failed_attempts/replay-v1/` | 首次回放现金精确相等检查失败的完整归档；不是最终样本；原因是约 7e-12 的浮点求和差异 |
| `targets/raw/` | MC 资源目标原始任务记录 |
| `targets/` | 配对目标、rolling target、资源目标、残差训练表 |
| `models/` | 四工况各 8 个 OLD/ACTION 冻结模型、MC Table、旧 Fluid 只读副本、construction prior |
| `processed/` | 分析结果及每张图的数据；不覆盖原始观察 |
| `phase1_identification/` | 任务书要求的 Phase-1 查询/目标/排序/汇总文件快捷副本 |
| `figures/` | 单图与图册，以及 plotting config |
| `qa/` | baseline、toy、历史目标复现、信息边界、回放、测试、交付完整性、PDF 渲染检查 |

## 样本、种子与比较单位

条件是 `模型/算法 × 工况 × 参数`；独立统计样本是 scenario seed，不是候选行、时间点、q 行或参数个数。

- Phase 1：W40/8，6000--6019；固定 t=3/8/12/16/20；全体 deciding vehicles，每车最多 8 个候选。
- construction：6200--6239；validation：6300--6314。
- formal：7000--7039；heldout 从 7000--7011 生成值预测目标，但不得改变冻结模型。
- 每条 Phase-2 查询使用本场景及同 split 固定 offset=7 的 donor future，两条路径先冻结，不根据目标选择。
- policy seed=0；training seed=null，因为这是确定性数值拟合，不是 RL 训练。
- 缺少 deciding vehicle 或 alternative 的时点保留 missing_reason，不替换时点。Heldout 请求 12 个场景，实际有合法查询的样本数以结果表为准。
- q 扩展是固定 slot integration 的加权建模行，不是额外独立样本；每条 q 行权重为 1/q 数量。

Heldout 两路径均值有共享 donor 导致的跨父场景依赖，原 parent-block 区间只能作条件近似。另有固定模型、仅用本场景 rep0 future 的 `heldout_own_future_*` 独立敏感性分析（post-hoc、不是重新选模）；详见 donor dependency audit。Phase1 与正式政策利润比较未使用共享 donor outcome。

## 严格特征对照

OLD 保留旧 frozen-summary 的 remaining/other_idle/incoming/backlog/expected/q 与 release hub 信息。ACTION 在此基础上增加合法的 duration/deadhead/loaded/compatible_load/expiring_load/release_idle/origin_idle/wait/same_hub。

原任务书 ACTION 的 remaining、expected 与 OLD 相同；incoming 只是 OLD incoming 加一个常数。完整字段仍保存，但等价列不重复进入拟合，避免仅重复同一信息而改变正则化强度。对应关系保存在 `features.ALIASES` 和模型文件中。

公共列的 construction 变换在两个模型中逐数值相同。同 family 的 OLD/ACTION 使用同一网格、同一 MC base、同一批 target、同一 q-mixture，并共同选择同一个超参数点。Bayes 共享 EB noise/prior scales 和 pooling multiplier，采用闭式 posterior mean，不使用 MCMC。

## 重要边界

在线结构仍是 `V_slot + C(route)`，没有新增 route-slot 联合变量。新的系统动作优势不能被证明可无误分解为逐车可加系数。因此本实验检验的是这一冻结近似架构中的实际政策效果，不宣称精确求出了联合 Q 或 V。

原来 `release >= T` 时 V=0 的 gate 保留。但系统相对优势在该区域不一定为 0：其他车辆仍可在 T 前继续接单，账本也可能有成本差额。原始标签不改成假零；拟合只用非终端释放；heldout 同时报告 all/nonterminal/terminal，明确这一支持域限制。

原始 M0 baseline 的相对标签严格为 0，但拟合器没有新增“每个 baseline 候选预测强制归零”的约束，因为这会同时改变第四轮在线结构。`processed/target_support_audit.csv` 单独记录 baseline 预测偏离和 terminal 非零标签。MC Table 的原始目标是绝对资源边际，其对系统动作优势的 RMSE 只作为异口径诊断，不能据此判断 MC Table 的政策质量。

`processed/heldout_g_plus_value_ranking.csv` 另提供 `g_current_joint_action + future advantage` 的候选效用排序诊断，因为实际决策不仅比较未来价值。它仍不等价于多车辆可加 MILP 的完整动作排序。`processed/post_state_residual_audit.csv` 包含实际 post backlog、idle、committed incoming、imbalance；这些字段只在求解之后分析，不进入在线特征。

本轮对齐的是“每期重新优化”的控制机制及参考 continuation `pi=M0`，不是求出了最终 `M0+V` 策略自身的固定点价值。每期使用 V 后得到的是基于 M0 评价的一步策略改进，未来闭环策略与参考 M0 仍可能不同。

OraclePreview H8 是受限窗口的信息诊断，不是可部署策略，也不是已认证的全日全局上界。没有用 Myopic-trained V 运行 RH4+V。

## 主要字段与单位

- `scenario_seed`：原始当前状态场景；`donor_seed`：该 MC 未来路径来源；`mc_replication`：路径编号。
- `state_hash`：完整初始车辆/订单/承诺/历史状态；`future_hash`：共享的完整订单实现；`candidate_hash`：物理路线元信息；`baseline_action_hash`：原始 M0 联合动作。
- `time`：查询的当前决策期；`candidate_release_time`：原计划释放期；`candidate_duration`：释放期减当前期，单位 period。
- `candidate_deadhead`/`deadhead_distance`/`loaded_distance`：网络 miles，不是欧氏距离。
- `candidate_current_stage`、`baseline_current_stage`：当期 g，模型货币 USD；不含 tie-break/V。
- `candidate_total_*`、`baseline_total_*`：完整 realized ledger 的 J_t，扣除同一个过去现金常数。
- `Y_roll`、`Y_freeze`、`difference`：相对未来优势及两种控制的差额，USD。
- `fourth_resource_marginal`：旧定义 q=1 与 q=0 的差额，USD；只能作诊断。
- `compatible_load`/`expiring_load`/`backlog`：载荷除以 capacity，不是人数或订单数；capacity=40。
- `release_idle`、`origin_idle`、`other_idle`、`incoming`：车辆数；incoming 只依据已知承诺。
- `solve_count_after_intervention`：从 t+1 到 T-1 实际重求解次数，不把冷却期当优化期。
- `realized_minus_booked`：实际账本和期间 booked profit 总和的差额；成本口径差异单独审计。
- `CI95_lower/upper`：场景配对差的未校正 t95 区间；`holm_p` 是同预声明 family 跨条件的多重比较校正。
- `daily_wall_seconds`：在线整日；其他 context/provider/slot/build/solve timer 不全可相加，尤其 provider 包含 slot 调用。

CSV 中嵌套的 context、route、trace provenance 为合法 JSON 字符串，标量保持一列一字段；Python 可用 `csv`/pandas 读取，对 JSON 字段用 `json.loads`。原始压缩文件用 `gzip.open(path, 'rt', encoding='utf-8')` + `json.load`。

## 运行

在仓库根目录的 PowerShell 中：

```powershell
$env:PYTHONHASHSEED='0'
$env:PYTHONIOENCODING='utf-8'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.venv/Scripts/python.exe -m experiments.cfs_rollout_target_alignment.run_experiment --through all
.venv/Scripts/python.exe -m experiments.cfs_rollout_target_alignment.statistics
.venv/Scripts/python.exe -m experiments.cfs_rollout_target_alignment.figures
.venv/Scripts/python.exe -m experiments.cfs_rollout_target_alignment.delivery prepare
.venv/Scripts/python.exe -m experiments.cfs_rollout_target_alignment.delivery history
.venv/Scripts/python.exe -m experiments.cfs_rollout_target_alignment.delivery reproduction
```

最终结果可按身份校验恢复运行，但不能改源码/配置后悄悄沿用已冻结标签；修复应归档旧尝试并记录版本。后处理源码有单独快照；图形输出不反向参与模型选择。本次任务不执行提交、推送、分支切换或 PR。
