# DRL_CO · 本地实验平台

当前分支：`react-experiment-platform`。文档更新：2026-10-03。功能版本：v0.2。

## 本分支负责什么

这是目前推荐使用的实验工作台分支，不是把三篇研究拼成一个算法。公共平台只负责调度、文件/状态、结果存储、通用统计与绘图；旧 DRL/SAC、BHH、single-level 各自拥有参数、数据接口、数学展示、框架、runner、metrics、trace 和模板。

已实现：

- 网页在“诊断”下方选择研究模型族；名称和固定蓝/橙/青绿 accent 同时标识上下文，数据、参数、框架、公式、运行和结果随模型切换。
- 三族共用范围/间隔、离散值及批次校验协议；同一实验条件下区分 scenario seed、policy seed、training seed。
- single 的 myopic、fluid、rollout、oracle 直接执行，默认不出现训练流程；可选 learned hub value 是监督拟合，不是 SAC 强化学习。
- single 逐 t 状态、观察、决策、收益/成本/服务指标与不变式记录。
- single 多算法、多场景、参数敏感性、热力图、时间序列、配对差值；本地 Python 导出 PDF/SVG/600 dpi PNG、数据、配置与可离线重绘脚本。
- 冻结数据、独立进程任务、源快照、复现 ZIP、本机 token/Origin 防护与脱敏诊断。

明确边界：DRL/SAC 与 BHH 尚未开放论文分析 adapter；response surface、训练随机性方差分解、大规模统计实验、原生 macOS 尚未验收。single 计划成本不等于实际现金支出，oracle 不虚构运营 trace。不要把旧 SAC 的历史数字视为平台三族共同排名。

## 三个分支的区别

| 分支 | 用途 | 推荐场景 |
|---|---|---|
| [main](https://github.com/Sundance0604/DRL_CO/tree/main) | 旧 DRL/SAC 研究基线；调度环境、候选共享 SAC、下层匹配与历史训练评估 | 阅读或复现旧 RL 研究 |
| [single-level-dispatch-model](https://github.com/Sundance0604/DRL_CO/tree/single-level-dispatch-model) | 单层匹配模型原型、上界、在线前瞻与审阅；保留旧 DRL/SAC 作对照 | 单独研究 single 原型及与旧模型的差异 |
| [react-experiment-platform](https://github.com/Sundance0604/DRL_CO/tree/react-experiment-platform) | 当前本地 React 实验平台；三族独立模型包、统一批实验、single 记录与论文分析 | 新实验、参数扫描、逐 t 分析与论文图导出 |

分支是代码版本路线，不是模型族：平台分支内同时有 legacy、BHH、single 三族。其他分支没有平台三族拆包结构；不能照搬平台的路径或启动命令。


本次跨分支文档更新仅修改各自 README，不把本分支的功能代码合并进 main 或 single-level 分支。

## 找模块：先看真正实现，不要误入兼容入口

```text
仓库根目录/
├─ model_families/                 三族真正的研究实现
│  ├─ single/                     当前单层模型，青绿
│  │  ├─ parameters.py            参数与约束
│  │  ├─ generation.py / data.py  数据 Schema 与领域接口
│  │  ├─ plugin.py                框架/组件/颜色/训练需求
│  │  ├─ mathematics.py           当前框架公式摘要
│  │  ├─ runner.py / metrics.py   正式执行、逐 t、指标定义
│  │  ├─ analysis.py              single 结果到分析协议的提取器
│  │  ├─ templates/               本族网页实验模板
│  │  ├─ scenarios.py             独立场景生成
│  │  ├─ scenario_support/        本族图/订单/车辆辅助对象
│  │  └─ engine/                  单层 MILP、前瞻、需求、上界原型
│  ├─ legacy/                     旧 DRL/SAC，蓝色
│  │  ├─ parameters.py / generation.py / data.py
│  │  ├─ plugin.py / mathematics.py / runner.py / metrics.py
│  │  ├─ frameworks.py / templates/
│  │  └─ engine/                  domain/environment/optimization/rl/simulation
│  └─ bhh/                        BHH，橙色
│     ├─ parameters.py / generation.py / data.py
│     ├─ plugin.py / mathematics.py / runner.py / metrics.py
│     ├─ optimization.py          稳态、有限时域、滚动承诺
│     ├─ spatial.py               空间校准
│     └─ templates/
├─ experiment_core/               公共协议、调度、冻结存储、统计与绘图
├─ platform_api/                  本地后端 API
├─ frontend/src/                  React 源代码
├─ configs/                       CLI 示例配置，不是 frozen datasets
├─ schemas/generated/             自动生成的族命名空间 Schema/OpenAPI
├─ scripts/                       安装/诊断/启动/停止/CLI 薄入口
├─ tests/                         模型与平台回归
├─ docs/                          使用说明、验收、模型审阅与历史报告
├─ doc/                           单层数学说明 LaTeX
├─ model/                         single 旧导入/命令行兼容入口
├─ drl_co/                        legacy 旧导入兼容入口
├─ experiments/                   旧 SAC 训练/评估/分析命令
├─ checks_on_original/            原模型核验
├─ data/ / results/               Git 保存的旧样例与历史精简结果
└─ workspace/                     本机运行产物，默认不推送 Git
```

| 想找/修改的内容 | 现行文件 |
|---|---|
| single 的 MILP、车辆/订单推进 | [model_families/single/engine/mt_prototype.py](model_families/single/engine/mt_prototype.py) |
| single 的前瞻/rollout 原型 | [model_families/single/engine/online_lookahead.py](model_families/single/engine/online_lookahead.py) |
| single 实验正式入口、逐 t 数据 | [model_families/single/runner.py](model_families/single/runner.py) |
| single 参数、数学、指标 | [parameters.py](model_families/single/parameters.py)、[mathematics.py](model_families/single/mathematics.py)、[metrics.py](model_families/single/metrics.py) |
| SAC 神经网络 | [model_families/legacy/engine/rl/candidate_sac.py](model_families/legacy/engine/rl/candidate_sac.py) |
| 旧 DRL 环境与下层优化 | [dispatch.py](model_families/legacy/engine/environment/dispatch.py)、[lower_layer.py](model_families/legacy/engine/optimization/lower_layer.py) |
| BHH 优化与空间校准 | [optimization.py](model_families/bhh/optimization.py)、[spatial.py](model_families/bhh/spatial.py) |
| 网页页面、菜单与模型切换 | [frontend/src/main.tsx](frontend/src/main.tsx) |
| 批实验表单、公式展示、论文分析界面 | [frontend/src/family-components.tsx](frontend/src/family-components.tsx) |
| 网页样式 | [frontend/src/style.css](frontend/src/style.css) |
| 后端路由、下载与访问校验 | [platform_api/app.py](platform_api/app.py) |
| 队列、状态、批次计划 | [experiment_core/service.py](experiment_core/service.py) |
| 范围/间隔网格 | [experiment_core/batching.py](experiment_core/batching.py) |
| 公共契约、注册与持久化 | [contracts.py](experiment_core/contracts.py)、[plugins.py](experiment_core/plugins.py)、[storage.py](experiment_core/storage.py) |
| 样本分组、均值/SD/CI、配对 | [experiment_core/analysis.py](experiment_core/analysis.py) |
| 绘图输入协议 | [experiment_core/analysis_contracts.py](experiment_core/analysis_contracts.py) |
| 本地 Python 论文图渲染 | [experiment_core/paper_renderer.py](experiment_core/paper_renderer.py) |
| source snapshot、独立 worker | [experiment_core/worker.py](experiment_core/worker.py) |
| 复现 ZIP | [experiment_core/reproduction.py](experiment_core/reproduction.py) |
| 本地启动/停止与 CLI | [launch.py](experiment_core/launch.py)、[cli.py](experiment_core/cli.py)、[scripts/](scripts/) |

所有族均按自身 package 维护 `parameters.py`、`generation.py`、`data.py`、`plugin.py`、`mathematics.py`、`runner.py`、`metrics.py` 与模板。新增研究逻辑不要写进其他族或公共调度层。

### 兼容目录不是另一套需要同时修改的模型

`model/mt_prototype.py` → `model_families/single/engine/mt_prototype.py`。

`drl_co/rl/candidate_sac.py` → `model_families/legacy/engine/rl/candidate_sac.py`。

`experiment_core/bhh.py`、`spatial.py` → `model_families/bhh/optimization.py`、`spatial.py`。

旧模块导入与相应 `python -m ...` 命令保留转发；平台分支应改右侧真正实现。single-level 研究分支没有这个迁移，它的 `model/` 仍是实现主体。外层原始代码包若有同名 `model/`，也不是本仓库的实时副本。

## 安装与日常入口

全部从检出本分支的仓库根目录执行。平台采用独立 Python 3.11 环境与 `uv.lock`，不使用旧 SAC 的历史 conda 环境或原始代码包的依赖清单。

```powershell
.\scripts\setup.ps1
.\scripts\doctor.ps1
.\scripts\start.ps1
```

打开 <http://127.0.0.1:8765/>。停止：`.\scripts\stop.ps1`。端口与本机资源配置见 [platform.local.json](platform.local.json)。

本次验收环境：Python 3.11.14、Node 24.18.0、Gurobi 10.0.3（可用许可证）。安装成功不代表优化许可证可用，先运行 doctor。macOS/Linux 对应 `bash scripts/setup.sh`、`doctor.sh`、`start.sh`；原生 macOS 未验证。

命令行也使用同一实验核心：

```powershell
.\scripts\exp.ps1 plugins
.\scripts\exp.ps1 dataset generate --config configs/datasets/matching-demo.json
.\scripts\exp.ps1 experiment validate --config configs/experiments/matching-myopic.json
.\scripts\exp.ps1 run --config configs/experiments/matching-myopic.json
```

`latest` 在同名数据存在多个修订时会拒绝歧义，需选择具体 hash。`*.template.json` 的 checkpoint 占位项需填真实训练运行 ID。网页模板由各族 `templates/` 提供，`configs/` 则是保留的 CLI 示例，二者不自动同步。

改网页后，在 `frontend/` 执行 `npm run build`，启动服务读取的是 `frontend/dist/`；不要修改打包产物或 `node_modules/`。参数 Schema 从模型包源定义生成，不要直接修改 `schemas/generated/`。

## 数据、逐 t 结果和论文图在哪里

默认在仓库根目录的 `workspace/`；如设置 `DRL_WORKSPACE`，以指定路径为准。

| 路径 | 作用 |
|---|---|
| `workspace/datasets/<dataset_id>/<hash>/` | 正式冻结数据，含 manifest/raw/normalized/derived |
| `workspace/runs/<run_id>/spec.resolved.json` | 实际执行参数与框架 |
| `workspace/runs/<run_id>/manifest.json` | family、framework、condition、data/source hash、scenario/policy/training seeds |
| `workspace/runs/<run_id>/metrics.json` | 各场景运行级指标；结合 metric-definitions 阅读 |
| `workspace/runs/<run_id>/trace.json` | 本族逐 t 记录；single 包含完整状态、观察、决策、measurement、不变式 |
| `workspace/runs/<run_id>/dataset.snapshot.json`、`code.snapshot.zip` | 本次数据与源码快照 |
| `workspace/runs/<run_id>/state.json`、`events.jsonl`、`stdout.log`、`stderr.log` | 运行状态、过程和错误 |
| `workspace/runs/<run_id>/checkpoint.json` 等 | 仅需要拟合/训练的框架产生 checkpoint，不是所有运行都有 |
| `workspace/runs/<run_id>/reproduction.zip` | 请求导出后产生的复现包 |
| `workspace/analyses/<analysis_id>/figure.pdf / .svg / .png` | Python 论文图，不是网页截图 |
| `workspace/analyses/<analysis_id>/observations.csv / periods.csv / states.jsonl` | 多实验样本、逐期数值、完整嵌套状态；原生 JSON 保留 null 和原因 |
| `workspace/analyses/<analysis_id>/statistics.csv / paired-differences.csv` | 均值、SD、n、CI 与配对差值 |
| `workspace/analyses/<analysis_id>/plotting-config.json / plot-data.json` | 保存的图形配置与聚合绘图数据 |
| `workspace/analyses/<analysis_id>/plotting.py / requirements.txt / analysis.zip` | 离线重绘与整套导出；ZIP 也含所选 runs/datasets |
| `workspace/index.sqlite*` | 查询索引及 SQLite 运行辅助文件 |
| `workspace/qa-*/` | 测试临时工作区，不是正式运行结果 |
| `data/`、`results/` | 旧样例和历史精简结果，不是平台新产物位置 |

raw-export 只含数据，不生成 figure 或 plotting.py。不要期待没有 operational trace 的 oracle 生成时间序列。

参数配置不是样本。独立 frozen scenario 才是统计样本；同一物理场景的 policy seeds 先平均，重复运行不增加 n。不同数据/物理参数/源码/口径默认分 panel，不自动池化；n<2 不生成置信区间。具体协议及统计假设见 [平台说明](docs/EXPERIMENT_PLATFORM.md)。

workspace、环境和第三方依赖默认不推送 Git：**push 不等于备份本地实验**。重要数据、checkpoint 与论文产物应独立备份。不要删除整个 workspace 来清理缓存，也不要公开含 token 的 server.json/launcher.json 或未经检查的复现包。

## 文档阅读顺序

1. [平台说明](docs/EXPERIMENT_PLATFORM.md)：使用、模型兼容矩阵、批次/统计/导出协议与边界。
2. [平台验收报告](docs/PLATFORM_VALIDATION.md)：真实运行和未验证项，历史记录已标明。
3. [Single 模型说明](docs/SINGLE_LEVEL_MODEL.md)、[模型审阅](docs/SINGLE_LEVEL_MODEL_REVIEW.md)、[数学源文档](doc/main_v2.tex)：原型研究背景，路径按当前导航理解。
4. [旧 SAC 诊断](docs/DIAGNOSIS.md)、[旧实验报告](docs/EXPERIMENT_REPORT.md)：旧研究记录，不替代平台验收。

## 验证状态与维护提示

功能代码验收：48 项测试通过、TypeScript/生产构建通过、npm audit 0 已知漏洞；三族批实验和 single 五类图形均实际运行。完整 420 场景原型入口回归正常结束，容量、订单生命周期等不变式通过；发布源快照完成导出后重跑，4 个场景的收益、成本、订单计数一致。tiny 验收不证明算法优越性。详细证据见验收报告。

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

首次/小规模统一验收工具 `python -m experiment_core.phase1_validation` 需要独立工作区，不能与现有 Coordinator 同时占用同一个 workspace。本次 README 文档提交仅检查路径、链接与分支差异，没有再次启动训练或大规模模型实验。

切换分支前停止服务并检查未提交修改；不要为了阅读另一分支覆盖当前环境。旧实验源码快照应保留，README 文档也可能参与 source hash，不能手改旧 manifest 来绕过复现校验。

## 附录：旧 DRL/SAC 研究记录（历史）

以下保留原 README 的 SAC 研究记录，避免丢失实验依据。这里的历史 conda 环境、测试数、研究计划及旧逻辑路径不代表 v0.2 平台验收；本分支当前源代码位置和启动方式以上文为准。

### 原模型为何不收敛，后来如何收敛

最致命的问题不是超参数，而是因果链断裂。旧 notebook 生成了 SAC 动作，却没有在 Gurobi 建模前可靠地写回 `order.virtual_departure`；即使调用旧 `test_step`，城市中的订单桶也未刷新。因此奖励几乎与动作无关，critic 可以把 loss 拟合到很小，actor 却没有可学信号。

重构依次完成了这些修复：

1. 用 `DispatchEnv.apply_actions` 作为策略进入优化器的唯一入口，同时更新订单和城市快照。
2. 用稳定 `order_id` 对齐动态订单集合中的 `(s, a, r, s')`，不再按易错的数组下标拼 transition。
3. 修复 masked discrete SAC 的 alpha loss 符号、target critic dropout、空动作 mask、critic 梯度污染与随机种子反复重置。
4. 修复下层模型的订单互斥约束、虚假跨城市车辆匹配，以及不必要的二次目标项。
5. 补回订单人数、收益、截止时间、电量/供给等特征，并统一尺度。
6. 将固定城市 ID 的 MLP 改成对每个 `(订单, 候选城市)` 共享参数的打分器；它不使用绝对城市编号，并通过测试验证节点重编号后的置换等变性。
7. 降低目标熵（ratio `0.2`、初始 alpha `0.1`），避免高熵策略在 greedy 评估时 argmax 退化。
8. 在高压训练中加入 `normal → moderate → high → extreme → high → extreme` 课程，并默认使用 supply 配对反事实奖励改善信用分配。

更完整的代码级审计见 [docs/DIAGNOSIS.md](docs/DIAGNOSIS.md)，全部实验和负结果见 [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)。

### 主要实验结果

所有关键比较使用独立随机图；多模型结果先在同一场景内平均，再做场景间配对，避免把同一场景重复计权。

| 阶段 | 关键结果 | 结论 |
|---|---:|---|
| 固定训练图，低熵 SAC | 270,092（greedy）/ 277,970（sampling） | 超过 no-op 和 random，证明修复后确实能学；仍低于 supply 296,627 和 myopic 298,133 |
| 旧 fixed-ID SAC，held-out | 236,403 | 跨图失败，主要是城市 ID 归纳偏置 |
| 候选共享 SAC，固定图训练 | 303,236 | 去除绝对 ID 后，泛化显著恢复 |
| 候选共享 SAC，多随机图训练 | **307,622** | 距 supply 309,500 约 0.61%，距 myopic 312,412 约 1.53% |
| 多图 + behavior cloning | 303,501 | 一步 MILP 示范与长期目标不一致，BC 反而略降 |
| high 压力，零样本 SAC | 185,205 | 高于 myopic 177,201，低于 supply 199,069 |
| extreme 压力，零样本 SAC | 115,307 | 高于 myopic 109,673，低于 supply 117,738 |
| high 压力课程 SAC | 189,904 | 比原 SAC +3.35%，低于 supply 191,857 |
| extreme 压力课程 SAC | 117,835 | 比原 SAC +5.14%，低于 supply 121,407 |
| high 反事实 SAC | 191,464 | 比原课程 SAC +294；4/7 场景获胜，趋势很小 |
| extreme 反事实 SAC | 118,584 | 比原课程 SAC +773；5/7 场景获胜，方差大 |

因此目前最严谨的判断是：强化学习已经从“实现上不可能收敛”变成“能学习且能跨图泛化”，资源稀缺时跨期策略优于短视 MILP；但在当前问题规模上，RL 尚未稳定优于利用领域结构的 supply 启发式。

还验证了两条没有成功的路线：

- 把 critic Q 直接加入 MILP 目标，在一组 extreme 场景有效，但压力课程后的 critic 在新场景退化，说明 Q 的跨订单标定不可靠。
- actor top-k 候选剪枝能删除 30%–46% 的合法城市，却最多只减少 5.6% 的真实可行变量；当前小规模问题的瓶颈不是候选空间，因此没有继续接 rolling horizon。

### 快速开始

以下是旧实验路线的历史启动方法，曾使用 Python 3.9 的 `pavane` conda 环境验证。新平台请使用上方的项目独立环境。Gurobi 需要本机可用许可证。

```powershell
conda activate pavane
python -m pip install -r requirements.txt
python -m pytest -q
```

最小训练检查：

```powershell
python -m experiments.training.train_candidate --episodes 3 --horizon 8 --batch-size 32
```

正式三种子候选 SAC：

```powershell
python -m experiments.benchmarks.candidate `
  --seeds 11,22,33 `
  --episodes 20 `
  --horizon 24 `
  --output-dir runs/candidate-reproduction
```

高压课程 + supply 配对反事实奖励：

```powershell
python -m experiments.benchmarks.candidate `
  --seeds 11,22,33 `
  --episodes 36 `
  --horizon 24 `
  --pressure-curriculum `
  --counterfactual-reward `
  --output-dir runs/candidate-counterfactual
```

反事实奖励现在默认启用。复现旧的绝对奖励时使用 `--no-counterfactual-reward`。每个真实时间步只增加一个隔离的 supply 控制分支求解，不是对每个订单分别 leave-one-out。

候选剪枝仅建议用于诊断或更大规模场景：

```powershell
python -m experiments.evaluation.stress `
  --checkpoint-root runs/candidate-curriculum `
  --levels moderate,high,extreme `
  --prune-topks 1,2 `
  --output runs/stress/pruning-validation.csv
```

默认会同时求解完整 MILP 以计算 oracle recall；真实部署计时应加 `--no-pruning-diagnostics`。

### 反事实奖励

对订单 `o` 使用实际策略和同一动作前状态下 supply 控制策略的差值：

\[
r_o = \frac{u_o(a)-u_o(a^{supply})}{1000}
+ 0.25\frac{J(a)-J(a^{supply})}{1000N_t}.
\]

`u_o` 是订单级匹配收益/等待或取消惩罚，`J` 是完整下层目标，`N_t` 是活跃订单数。该奖励降低了不同场景规模造成的绝对回报漂移，但现有七场景独立测试只显示小幅正向趋势，不能声称显著提升。

### 仓库结构

| 路径 | 作用 |
|---|---|
| `drl_co/domain/` | 城市图、城市、订单和车辆领域对象 |
| `drl_co/environment/` | Gym 调度环境及动作落地边界 |
| `drl_co/optimization/` | Gurobi 车辆—订单联合匹配模型 |
| `drl_co/rl/` | candidate SAC、masked discrete SAC、特征和 fixed-ID 兼容模型 |
| `drl_co/simulation/` | 场景生成、城市快照、状态推进和通用仿真工具 |
| `drl_co/data_io.py` | 样例场景加载和旧 pickle 模块路径兼容 |
| `experiments/training/` | candidate SAC 与 fixed-ID SAC 训练入口 |
| `experiments/evaluation/` | rollout、held-out、压力测试、Q-MILP 与剪枝评估 |
| `experiments/benchmarks/` | 多随机种子基准实验入口 |
| `experiments/analysis/` | 各阶段实验结果分析脚本 |
| `docs/` | 根因审计和完整实验报告 |
| `data/` | 版本化的小型样例场景 |
| `tests/` | 环境、SAC、置换等变、场景与反事实测试 |
| `results/` | 精简后的最终 CSV 与图，不含 checkpoint |

`runs/` 被 `.gitignore` 忽略，训练生成的 checkpoint、逐 episode 日志和 smoke 结果不进入仓库。历史 notebook、旧 PPO/MARL 草稿、数千个文本输出、临时 ILP/图片和大体积中间数据已经删除；如需考古仍可从 Git 历史找到。

### 后续实验计划

优先级从高到低：

1. **扩大问题规模再测剪枝。** 增加城市、车辆、同批订单和真实可行边，只有当可行 MILP 变量减少至少 40%、oracle recall 至少 98% 时才继续。
2. **做压力条件化 critic。** 将车辆/订单压力作为显式输入，并使用 pairwise ranking 或 calibration loss；在独立验证集定权重，在完全隔离测试集报告。
3. **增强反事实信用分配。** 保留一次 supply 控制分支，再只对真正偏离 supply 的少量订单抽样做 leave-one-out，而不是全面 rolling horizon。
4. **加入图消息传递。** 当前候选共享网络已置换等变，但只靠手工最短路/供需特征；可加入轻量 GNN 后做同预算消融。
5. **提高统计可信度。** 至少 10 个训练种子、30 个独立场景，报告场景配对 bootstrap 置信区间、wall-clock、Gurobi gap 与失败率。
6. **重新定义胜负标准。** 同时比较目标值、求解时间和最坏场景表现；只有在预注册测试集上稳定超过 supply，才宣称 RL 带来净优势。

### 已验证状态

- 环境：`conda env pavane`
- 回归测试：`14 passed`
- 下层求解：训练与正式反事实评估均无 Gurobi solve failure
- 推荐配置：候选共享 SAC + 多随机图 + 压力课程 + supply 配对反事实奖励
