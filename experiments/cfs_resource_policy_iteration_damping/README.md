# A2：真正的自一致阻尼资源价值策略迭代

本目录仅研究 single 的阻尼更新，不更新实验平台默认策略，不修改旧 DRL/SAC、BHH 或前六轮研究产物。数值、分析及全部交付核验已通过：93,600个新增资源世界、131,421,408个存储价值系数、185,116个历史文件与CFS内容校验，全量回归测试197项，以及九组论文图和九页图册的技术与逐页视觉核验。最终完成与不可变冻结状态以 `manifest.json` 和 `manifests/final_manifest.json` 的 `COMPLETED` 为准；冻结后本目录只读，B才可启动。正式结果见 `damping_policy_iteration_report.md`，不能把数值完成视为全部交付核验完成。

## 已确认的协议修订

2026-10-07 用户明确选择真正的自一致阻尼迭代。任务书中“raw table 与 A 不一致则停止”限定为历史重建及共同起点的校验，不能用于要求改变后的策略仍产生旧标签。

固定 `Vtilde0=V0=E(M0)`，`D1=G(V0)=U1`。对 n=1、2、3：重新评价实际 Dn 得到 `Vraw_n=E(Dn)`，再计算 `Vtilde_n=(1-alpha)Vtilde_(n-1)+alpha Vraw_n`，生成 D(n+1)。E 的标签是全冷却 J(q)−J(q−1)，不是 J−g、route advantage 或 ACTION correction。

因此 raw V0、raw V1 与 A 相同，允许有证据的复用；raw V2、raw V3 必须针对各 alpha 的 D2、D3 重新采集父状态和配对世界。不能用旧 A 的 V2、V3 平滑代替。

## 唯一变量与样本

alpha=.25/.50/.75，其余沿用 A：4 工况、construction8000–8059、7 个时间节点、原 hub 旋转、原 q 网格及2K端点范围、线性插值、负值保留、不强制 q 单调、原 MILP 和日终完整承诺冷却。

diagnostic8200–8219，仅用于选择一个全局 alpha；pilot8950–8951；formal9100–9139。Scenario seed 是统计样本单位，policy seed=0，training seed=null。t/hub/q 和价值格不是独立 replication。

正式 U0–U4、D1–D4 共 1440 个策略标签，D1 精确复用 U1 的已核验轨迹，实际独立仿真1280次。复用不会增加独立场景数。相同正式场景下进行策略配对，不与 A 的9000–9039旧正式场景直接作配对统计。

## 预注册选择和解释

详见 `config/main.json` 和 `manifests/analysis_preregistration.json`。主选择在 CA、W40 等权，另用 W80 筛查利润崩溃。按利润崩溃、表变化上升、动作变化上升筛选；随后依次比较最后两步动作差、表 relativeL2、诊断利润。接近阈值预先固定，优先 .5。

若三个 alpha 全未通过筛查，仍按预先规定的最少失败项及同一排序选一个完成固定矩阵，但明确为 exploratory，不宣称稳定化。正式结果不用于重新选 alpha。

正式主 Holm family 是4工况 D4−U4。20项路径比较、8项中间跨路径比较分别校正。均报告配对 t95 和10000次 paired bootstrap，区间未作多重校正，与 Holm p 分列。W80“不明显破坏”需要 t95 下限高于 U4 平均利润的负1%容差；不能用“不显著”充当等价或无伤害证明。

只有利润、动作趋势、表变化、W80安全与诊断/正式方向共同满足预注册标准才称 damping stabilization。没有额外 E(D4)，不声称已证明固定点、理论收敛或最优 V。表跨轮同时受父策略状态分布变化影响，格 MC_SE 不包括整条 PI 链重估的不确定性。

阻尼公式本身就会缩小单步部署表更新，不能仅凭 deployed relativeL2/RMSE 较小宣称稳定化。raw 与 deployed 两种变化同时保存，仍需独立的同状态动作、正式利润和安全性证据。

状态分布的 `release_delay_mean` 只在后处理中使用既有 rolling audit 的最终 idle_release ETA；历史摘要的 last-delivery 时间另存 `legacy_incoming_timestamp_offset_mean`，不能视为真实最终释放时点。此诊断口径在 diagnostic 开始前固定，不参与alpha选择，不改冻结的仿真/求值算子或旧产物。

## 模块和目录

| 路径 | 职责 |
|---|---|
| `common.py` | 独立配置、种子、源码冻结、历史只读内容身份、数据路径；不调用 A 的 writer |
| `phase0.py` | 16张旧表全部重聚合；20次旧政策独立求解；16组旧资源配对标签独立重算 |
| `evaluation.py` | 原 A worker 的机械复制，仅改为显式新输出路径；E(实际 Dn)、全逐期配对世界、原始表及阻尼复算 |
| `policies.py` | U/D 政策来源、闭环仿真、D1精确别名、固定 U0 状态库的同状态动作比较 |
| `run_experiment.py` | 按历史→construction→diagnostic→freeze alpha→pilot→formal 顺序执行，断点身份核验 |
| `analysis.py` | 预注册 alpha 排序、场景配对统计、表/动作/状态/振荡/运行时分析 |
| `delivery.py` | 逐世界父政策/表与保存系数核验、精确阻尼复算、历史只读校验、证据门控报告及最终冻结 |
| `figures.py` | 九组论文图及图册，矢量PDF/SVG、600dpi PNG、平坦图源CSV与plot config |
| `pdf_qa.py` | 使用bundled工具检查字体/矢量/分辨率、渲染所有页；真实逐页检查后才标记视觉通过 |
| `config/` | outcome-blind 参数、正式比较和阈值 |
| `manifests/` | 历史索引、源码、查询、父policy/table、各轮表、alpha冻结及各阶段运行身份 |
| `targets/alpha_*/raw_V*/` | 每次实际 E 的 target JSON/CSV、单格支持和 MC_SE |
| `tables/alpha_*/raw_V*/` | 未阻尼资源价值表；V0/V1有共同起点复用证明 |
| `tables/alpha_*/deployed_V*/` | 真正部署的阻尼表；每格可用 alpha、上一张部署表和 raw 表复算 |
| `policies/U*/` | 相同新场景中使用的历史 undamped 控制策略 |
| `policies/alpha_*/D*/` | 各 alpha 的 D1–D4，与实际部署表的反向来源 |
| `raw/alpha_*/targets/` | 每个construction任务缓存和所有配对世界索引 |
| `raw/alpha_*/raw/worlds/` | 每个反事实资源世界完整逐 t 状态、计划、实现ledger和无求解器回放结果 |
| `raw/alpha_*/raw/traces/` | 实际父 Dn 的construction闭环轨迹及采样状态 |
| `raw/runs/` | 每个诊断/pilot/formal实验的spec和summary；D1明确alias_of |
| `raw/traces/` | 新政策完整决策期与冷却期轨迹、每期实现利润、U0同状态比较库 |
| `raw/action_comparisons/` | 固定完整状态下的双方独立求解，不是两条闭环轨迹同t直接比较 |
| `processed/` | 可供Python读取的平坦CSV和对应JSON；参数组合与统计样本不混淆 |
| `qa/` | 历史回归、回放、测试和最终图形核查；失败不静默接受 |
| `figures/` | 最终论文图600dpi PNG、矢量PDF/SVG、图源CSV与plot config（本地保留，避免进入平台源码快照） |

`raw V0/V1` 的原始标签和世界保留在只读 A 目录，不伪造重复世界。本目录manifest给出对应文件及内容hash。

## 主要字段

- condition：区域/N/K工况；scenario_seed：独立需求场景；alpha：策略路径，不是额外统计样本。
- iteration：表 n 和策略 n+1 的对应关系必须按字段名区分。
- policy_hash/table_hash/code_hash：策略、部署表和数值源码身份；不能只按文件名复用缓存。
- profit_q/profit_previous/target：同一未来路径配对世界利润与差值，单位 USD。
- time/hub/q：资源查询期、位置、第 q 辆查询hub空闲车。其他hub资源与原承诺保留。
- state_hash/future_hash/paired_common_hash：原父状态、同场景未来、只相差额外空闲车的共同底座。
- MC_SE/sample_count/support_status：当前父策略下的直接标签支持；插值格不能伪造SE。
- raw_V0..3/damped_V0..3：单格未阻尼/部署路径；range及方向反转为有限步描述，不证明周期。
- occupancy_frequency：诊断自然状态中该hub恰有q空闲车的频率，不是优化器选择slot的频率。
- alias_of：D1复用的U1运行；不是多一份独立实验。
- replay_steps/replay_profit_difference：包括T后冷却的独立物理回放；现金容差1e-6，物理状态精确。
- relative_L2/RMSE_USD：相同部署格的表变化；不同轮自然状态会变化。
- full_action_disagree：固定U0完整状态下canonical动作不同，忽略顶层序列化顺序但保留路径顺序和车辆ID。
- SMD：分布标准化均值差；零方差且均值不同保持缺失，不能填0。
- wall/worker/lookup/build：并发wall与worker合计分开；嵌套计时不能重复相加。

## 运行

从仓库根目录使用现有 `.venv`，设置 `PYTHONHASHSEED=0`、`PYTHONDONTWRITEBYTECODE=1`、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`PYTHONIOENCODING=utf-8`。

```text
python -m experiments.cfs_resource_policy_iteration_damping.run_experiment --stage history
python -m experiments.cfs_resource_policy_iteration_damping.run_experiment --stage construction
python -m experiments.cfs_resource_policy_iteration_damping.run_experiment --stage all
```

数值仿真需要现有已配置Gurobi环境；通用交付/图形检查尽可能使用bundled工具。恢复只接受相同源码、配置、场景、父政策和表身份。A2最终报告、图形及QA完成并冻结前，不启动B数值实验。

数值入口退出时的 `NUMERICAL_COMPLETE_ANALYSIS_QA_PENDING` 只表示固定矩阵完成。后续交付入口为：

```text
python -m experiments.cfs_resource_policy_iteration_damping.delivery --stage analysis
python -m experiments.cfs_resource_policy_iteration_damping.delivery --stage worlds
python -m experiments.cfs_resource_policy_iteration_damping.delivery --stage history
python -m experiments.cfs_resource_policy_iteration_damping.delivery --stage freeze
```

`freeze` 必须在测试、全部图形技术/视觉检查和所有核验均真正通过之后使用；缺少任一必需QA会拒绝完成。已冻结的 `analysis.py` 不能改变预注册筛查或排序。图形布局修正不改数值、统计定义或alpha选择；调整后必须重新渲染并检查最新输出。`COMPLETED` 后不得再运行会重写阶段manifest的数值入口，也不得向本目录写入B的测试或图形核验结果。
