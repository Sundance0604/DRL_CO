# 第四次数值实验：Local Action-Conditioned Value 与 Hierarchical Bayesian V

本目录是已完成的独立科研实验，不修改前三轮模型、结果或平台默认算法，也不训练 DRL/SAC。完成 2560 次正式、128 次 pilot、320 次旧方案对照；3008 次政策运行独立回放通过，全套 142 tests 通过（1 个依赖弃用 warning）。正式状态与运行数量以 `manifests/final_manifest.json` 为准。

## 从哪里开始查看

- [中文研究报告](local_post_bayesian_value_report.md)：统计口径、限制、12 类核心表和十个研究问题。MC Table 保持强基线；Bayesian 无稳定领先；新旧方案政策差区间均跨零，不作因果字段改进声明。
- `figures/`：13 类论文图，每类有 PNG、PDF、SVG；`plotting_config.json` 记录尺寸、字体、symlog 阈值、源 CSV 的 SHA256。
- `processed/`：Python 可读取的平面分析表。一个参数工况不是一个独立统计样本；政策推断以独立 scenario seed 为配对单位。
- `targets/`：新生成的资源槽目标、动作条件目标、残差与小样本 continuation 敏感性诊断，不复用第三轮标签。
- `models/`：table、linear、chebyshev、spline、bayesian 及原样冻结的 `legacy_fluid_reference`；JSON 包含系数、变换、超参数和选择证据。
- `raw/runs/<run-id>/`：每次运行的场景和解析配置、完整逐 t trace、决策账本、实现账本、指标、成功或失败状态。包含 cooldown；失败不删。
- `raw/datasets/`：CFS 抽样场景和 observed/generated provenance；原始 SQLite 只读，不复制。
- `raw/source.snapshot.zip`：冻结数值源码和配置。`.gitignore` 只排除大体积原始记录，不影响本地保存。
- `raw/analysis-source.snapshot.zip`：冻结后分析、测试与报告的独立快照；不冒称为原始数值源码。`raw/traces/index.json` 按 run ID 索引已有 trace、账本与 SHA，不重复复制大文件。
- [论文图集](figures/figure_atlas.pdf)：13 页 PDF，各单图另有 PDF/SVG 和 600 dpi PNG。
- `manifests/`：历史基线回归、数值源冻结、模型冻结、最终交付清单。
- `qa/`：全量测试 XML、信息边界、线性/槽语义、Bayesian 数值性、物理独立回放和独立重跑检查。

## 实现模块

平台模型包：`model_families/single/local_value/`，与第三轮 `single/value/` 独立，避免修改旧 provider。

| 文件 | 职责 |
|---|---|
| `local_value/providers.py` | 有限字段白名单的候选动作上下文、12 个统一特征、固定 q 混合设计矩阵、旧 provider 兼容适配器、局部 provider |
| `local_value/route_milp.py` | 独立的价值窗口规划器；保留原车辆流/订单分配/容量/锚点与资源槽 prefix，附加末端动作常数修正 |
| 本目录 `baseline_regression.py` | 四工况 × Myopic/RH4/RH8，在第三轮 seed 3000 上同时重跑冻结模拟器与第四轮模拟器，逐物理 trace 比较 |
| `common.py` | 隔离目录、历史文件保护、配置和代码身份、源码压缩包、CFS 数据准备、JSON 化结构 CSV 字段 |
| `targets.py` | 自然 Myopic 状态上的 fresh slot labels；真实候选动作干预后再做 paired resource marginal；两条相同配对未来路径 |
| `fit_models.py` | 共用残差集；additive Linear/Cheb/Spline；Gaussian empirical Bayes evidence 和闭式 posterior；仅验证集选择 |
| `simulation.py` | 继承 full-cooldown 物理和账本；调度未来只供 scheduler/显式 Oracle；解后记录真实 post summary |
| `run_policies.py` | 16 策略正式阶梯、20-seed 旧 summary 配对对照、失败记录和模型身份检查 |
| `run_experiment.py` | 严格分阶段编排、可恢复但身份变动立即报错 |
| `analysis.py` | 模型冻结后的 held-out、配对统计、解后残差诊断、独立物理回放和论文图；不参与模型调参 |
| `tests/test_local_post_bayesian.py` | 信息隔离、Oracle 拒绝、槽语义、旧接口兼容、固定上下文、线性、Gaussian PSD/确定性/稀疏收缩 |
| `tests/test_local_post_analysis.py` | 真实 joint-post 执行语义、常数相关性缺失、匹配 sensitivity 与统计导出回归测试 |

## 正式设计与运行

工况固定为 California N80/K3、Western N40/K8、Western N80/K3、Western N20/K8，均采用 rho_max。CFS、geometry、抽样、车辆部署、容量、成本、费率、时间生成、已承诺任务与日终 full-cooldown 语义继承前三轮。

- construction：4000–4039，每工况 40 场景。
- validation：4100–4114，每工况 15 场景。
- held-out value：5000–5011，仅冻结后评估，禁止调参。
- pilot：4100–4101，仅工程检查，所有方法保留。
- formal：5000–5039，每工况 40 场景，16 方法，共 2560 次。
- old-summary：formal 前 20 seeds，Myopic/RH4 × Cheb/Spline，共 320 次新种子重跑；local 对应结果直接复用正式运行，禁止拿第三轮历史均值代替配对。
- continuation sensitivity：Western N40/K8，4000–4004，小样本 RH4 continuation，绝不混入拟合集。

在仓库根目录的 PowerShell 中运行（已有本地 Python 和 Gurobi）：

```powershell
$env:PYTHONHASHSEED='0'
$env:PYTHONIOENCODING='utf-8'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.venv/Scripts/python.exe -m experiments.cfs_local_post_bayesian_value.baseline_regression
.venv/Scripts/python.exe -m pytest -q --basetemp=workspace/pytest-fourth --junitxml=experiments/cfs_local_post_bayesian_value/qa/pytest-full.xml
.venv/Scripts/python.exe -m experiments.cfs_local_post_bayesian_value.run_experiment --through all
.venv/Scripts/python.exe -m experiments.cfs_local_post_bayesian_value.analysis
```

`--through` 支持 inputs、targets、fit、heldout、pilot、formal、all。结果可恢复，但配置、数值源码和冻结模型必须完全一致。前三轮的报告、模型、原始结果不重写。新增分析代码另存哈希；不能拿当前新增源码重新计算旧实验身份再冒称旧 frozen identity。

## 局部信息与防泄漏

原 route candidate 只有兼容订单集合，不绑定最终订单子集。订单 x 与路线 y 仍联合求解。任何“真实剩余 backlog、真实服务量、真实 post supply”只能在解后审计，绝不进入当期价值系数。

Provider 只接收当前可见 summary 和候选动作上下文。候选上下文含 origin/pickup/release、release 时刻、等待、距离/时长、兼容订单/负载/即将过期负载、已承诺容量以及确定的局部供给变化；字段白名单拒绝额外的求解结果字段。整个候选列表在 optimize 前固定 hash，求解后再次检查。MILP 的二次项、二次约束和 general constraints 必须为零。

统一拟合的 12 个原始特征如下；此外所有方法均有 release-hub 类别和离线 q 效应。

| 特征 | 单位与含义 | 信息来源 |
|---|---|---|
| remaining | max(0,(T−release)/T) | 已知时钟与候选释放 |
| duration | release−候选出发决策时刻，period | 候选路线或末端 wait |
| deadhead | origin→pickup，miles | 固定 network 距离 |
| loaded | pickup→endpoint，miles；是 backbone 长度，不是实际载客距离 | 候选路线 |
| compatible_load | 当前兼容订单 load/capacity，不代表最终服务量 | 当前 booking pool |
| expiring_load | 下一期失效的当前兼容订单 load/capacity | 已知 ready/deadline |
| incoming | 已知最终 idle release 至 release+4 的辆数，再加候选确定释放 1 辆 | 现有 commitments 和局部假设 |
| release_idle | release hub **当前** idle，未偷算联合更新 | 当前 summary |
| origin_idle | origin 当前 idle 加该候选确定的 −1；未来窗口节点为条件 origin，不是全局真实 post supply | summary 与局部假设 |
| expected | 构造集每期 booking prior × 剩余决策期 | construction-only |
| wait | 候选 delay 或末端单期 waiting，period | 候选动作 |
| same_hub | origin==release，0/1 | 候选几何 |

上下文还记录兼容订单数、残余容量等，但**本轮统一 12 特征预算不将其额外加入任何方法**。它们用于 provenance/检查，不给 Bayes 额外信息。

在 RH 的未来假设节点，`origin_idle` 是“当前观察库存 + 条件局部 depletion”的**有符号特征**，可能为 −1，不是声称已经知道未来的实际闲置车辆数。`incoming+1` 是候选释放提示，不会删除 frozen summary 中旧 commitment 的释放提示；延长既有路线时它并非去重后的物理供给库存。真实库存与 pipeline 只在解后审计。Cheb/Spline 将连续协变量裁剪至构造域，Linear/Bayes 保留线性外推；这种分布与整合差异须在报告中披露，不能用这些代理特征替代物理状态或资源槽计数。

q 不能依赖最后被选中的 slot 去改变 route 系数，否则需要 route-slot assignment variables。因此 C 在训练中允许有 q effect，部署时使用构造集频率固定平均；所有局部方法采用相同平均。目标严格是 `sum Vslot(g,q) z(g,q) + sum C(context_r) y_r`。同一车辆只在其窗口末端 sink 领取修正，锁定的不可控到达没有局部修正，终端 release>=T 的自由资源价值为零。

旧 provider 通过 LegacyAdapter 增加 `action_context=None`，忽略 context 并保留旧数值；旧第三轮源文件不改。新的 local provider 在 None 时只输出 MC 基础槽项；有 context 时输出 base+C，槽建模只调用 None，避免修正重复。

## Target 的准确统计含义

每个自然状态在固定投影时刻查询两个轮换 hub，基础表用相同未来下 q 与 q−1 的 Myopic V0 实现收益差。所有 committed vehicles 保留，只替换被查询 hub 的当前自由车辆。不同 q 是同一场景的相关反事实，不是独立样本。

动作目标另外执行固定干预：先从当前可见候选中用固定 hash 选取候选，确定可行的锚点加贪心订单组合；另一候选为可用车辆的等待。第一期固定该动作，其他车辆等待/继续承诺；直到候选 release 不再接新分配，但正常处理已承诺移动、已公开 booking 和 expiry。到 release 后，两世界共同使用原 Myopic V0、完全相同 future、network、cost、solver seed，执行到 full cooldown。

这样得到的是**该固定局部干预/期间控制规则下的有限资源边际价值**，并非完整 joint optimal post-decision V，也非实际联合 MILP assignment 的唯一结果。中间禁止新分配带来 distribution shift，是明确的 target-construction 限制。两条经验路径是本场景未来及同 split offset=7 的 donor 未来；不是已证明校准的条件需求后验。val/held-out 的 donor 不跨 split。

所有回归拟合 fresh action residual `R=Y−construction MCslot(time,hub,q)`；不是把候选字段贴到旧 Y 上。结构字段在 CSV 中用合法 JSON，其他列是标量，可直接供 Python 使用。原始目标在 targets/raw 保留，聚合/残差在 targets 与 processed，绘图导出独立数据表，不混写。

## Bayesian 和统计限制

数值库版本另见 `manifests/environment_manifest.json`，与第三轮的冻结环境逐项核对。为保留冻结文件的原始证据，不事后回写原 `model_freeze.json` 的选择内容。

Hierarchical Gaussian residual regression 分别有 global、hub、q prior variance；用 construction marginal evidence 做 empirical Bayes，再在 validation 选固定 partial-pooling multiplier，闭式计算 Gaussian posterior。没有 MCMC。主策略只部署 posterior mean。

预测区间包含 posterior coefficient covariance 和 iid observation noise；不包含 base-table 与 empirical-hyperparameter uncertainty，相关 donor/q 标签也不独立，因此必须看 held-out 50/80/95 coverage，不能把 posterior interval 当作政策利润的置信区间。

政策利润用独立场景的 mean、SD、median、t95 CI；配对差要求相同场景 hash。|skew|>1 增加固定 10000 次 bootstrap。直接方法比较预先列出但没有多重检验校正。compression 仅当 RH8−RH4 的 t95 下界>0；Oracle gap ratio 同样要求正分母证据，不能解释成信息恢复率或 DRL 可达到的收益。

old-summary 对照同时改变 target、base table 和拟合模型，能判断本轮完整方案是否更好，**不能单凭它把收益差归因于某一个新字段**。Ranking 表明确区分同 parent decision 的 action-spatial 排名和严格 same-q、same-release 的 spatial 排名；无匹配对时记缺失，不能造出 0 或 100%。

解后记录真实 backlog、served、idle、incoming pipeline、imbalance 和 route releases。对 held-out intervention residual 做 omitted-state 相关、scenario-cluster bootstrap CI、leave-seed-out residual regression。imbalance=backlog−supply，不能把三者同时当作可识别独立线性系数；联合回归用 backlog+supply，imbalance 单独报告相关性。这里的 association 不证明因果。
