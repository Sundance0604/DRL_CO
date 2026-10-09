# DRL_CO · 本地实验平台

当前分支：`react-experiment-platform`。文档更新：2026-10-03。功能版本：v0.2。

2026-10-10 数值研究同步：[数值实验与报告索引](reports/README.md) 汇集多轮 CFS 与 PPO 的源码、配置、可读报告和汇总图。数据集、原始轨迹、权重、价值表及完整运行目录留在本地；复现所需本地输入见索引说明。

完整逐字段使用说明：[Single 实验平台详细手册](docs/SINGLE_PLATFORM_USER_GUIDE.md)（2026-10-04）。覆盖全部 11 个页面、条件出现的框架参数、批实验 JSON、逐 t trace、统计口径和 Python 导出；首次使用建议按手册第 15 节操作。

2026-10-04 CFS 限范围实验：新增 single 的物理网络 `pickup_radius`（英里，默认0仅own hub）、双账本与无再次优化的日终完整执行。正式比较显式选择 `single-realized-v1 / cooldown_realized`；旧 booked 模式仅供复现。CFS货值不当运费，几何网络不当真实公路；详见 [实验复现说明](experiments/cfs_rho_endofday/README.md)、[修改前审计](reports/cfs_rho_endofday/implementation_audit.md) 与 [仿真报告](reports/cfs_rho_endofday/simulation_report.md)。现有逐字段手册描述修改前默认核算，本次新增口径以上述实验说明为准。

2026-10-05：single 新增独立的 rolling-horizon 框架，默认使用已知订单的多期联合路线 MILP、`V=0` 和 realized 日终核算，不是强化学习，也不是原 sampled rollout。原 CFS 实验结果未重写；其源代码快照记录的是当时的版本，不含本次新增框架。

第二次数值仿真：运行入口与数据索引见 [cfs_dynamic_lookahead](experiments/cfs_dynamic_lookahead/README.md)。压力扫描使用20/40/80订单、3/5/8车、旧报告三档rho；比较myopic、合法known-supply RH与明确标为**不可部署oracle diagnostic**的未来订单preview。所有结果仍采用full-cooldown realized ledger，旧CFS产物不覆盖；大体积raw本地保存并从Git排除。实验完成状态与源码/配置身份以新结果目录的manifest为准，不能拿pilot均值冒称正式30-seed结论。

## 本分支负责什么

这是目前推荐使用的实验工作台分支，不是把三篇研究拼成一个算法。公共平台只负责调度、文件/状态、结果存储、通用统计与绘图；旧 DRL/SAC、BHH、single-level 各自拥有参数、数据接口、数学展示、框架、runner、metrics、trace 和模板。

已实现：

- 网页在“诊断”下方选择研究模型族；名称和固定蓝/橙/青绿 accent 同时标识上下文，数据、参数、框架、公式、运行和结果随模型切换。
- 三族共用范围/间隔、离散值及批次校验协议；同一实验条件下区分 scenario seed、policy seed、training seed。
- single 的 myopic、fluid、rollout、oracle 直接执行，默认不出现训练流程；可选 learned hub value 是监督拟合，不是 SAC 强化学习。
- single 的 rolling 联合安排窗口内车辆路线、订单与后续车辆复用；窗口控制、规划器、执行校验相互独立。
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

## Windows 双击启动

已有环境时，直接双击仓库根目录的 [start.bat](start.bat)。它会定位自身所在目录、检查项目 Python 和网页构建、启动或复用服务，健康检查通过后自动打开默认浏览器。网址读取实际服务端口，不写死为 8765。

控制台会保留启动结果；出现 `ready` 后可关闭窗口，后台服务继续运行。自动打开浏览器失败时，复制输出的本地 URL 手动打开。关闭网页或控制台并不等于停止服务，停止仍用 `scripts/stop.ps1`。

首次尚未安装时，先运行下方的 `scripts/setup.ps1`。BAT 不自动安装依赖、不修改 PowerShell 执行策略、不要求管理员权限，也不验证优化许可证；运行实验前仍建议执行 doctor。

命令行验证可用 `start.bat --no-browser --no-pause`（不打开浏览器、不等待按键）；平时双击无需参数。原有 `scripts/start.ps1` 继续只启动服务，不自动打开网页。

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
│  │  ├─ planning.py              共享路线可行性、领域动作、执行前校验
│  │  ├─ rolling/                 H/k 控制、规划接口、可替换窗口算法
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
├─ start.bat                      Windows 双击启动并打开浏览器
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
| single 的 rolling 窗口控制 | [controller.py](model_families/single/rolling/controller.py)：时间边界、可见状态、冻结执行前缀与重规划 |
| single 的 rolling 规划算法 | [route_milp.py](model_families/single/rolling/route_milp.py)：联合多期路线宏动作 MILP |
| rolling 算法接口 / 注册 | [contracts.py](model_families/single/rolling/contracts.py)、[registry.py](model_families/single/rolling/registry.py) |
| single 动作生成与执行校验 | [planning.py](model_families/single/planning.py)：旧 M_t 与 rolling 共享物理约束 |
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
| Windows 双击、本地启动/停止与 CLI | [start.bat](start.bat)、[launch.py](experiment_core/launch.py)、[cli.py](experiment_core/cli.py)、[scripts/](scripts/) |

所有族均按自身 package 维护 `parameters.py`、`generation.py`、`data.py`、`plugin.py`、`mathematics.py`、`runner.py`、`metrics.py` 与模板。新增研究逻辑不要写进其他族或公共调度层。

## Single rolling horizon：使用与实现边界

在左侧选择 **Single-level model**，进入“实验编辑器”，选择 **Rolling-horizon joint route planning**；选择 single 数据、调整模型参数后校验并提交。该框架不出现训练步骤，自动选择 `single-realized-v1 / cooldown_realized`。也可加载 [matching-rolling 模板](model_families/single/templates/experiments--matching-rolling.json)。模板数据引用只是示例，需替换为本机已存在的数据版本。重启后端并使用最新网页构建后，新框架才会出现在目录中。

| 框架参数 | 默认 | 含义 |
|---|---:|---|
| `planning_horizon` | 6 | H，联合规划 `[t, min(t+H,T))` 内的新决策；范围1–100 |
| `commit_periods` | 1 | k，执行本窗口前 k 期后重规划；范围1–H。k越大，响应新订单越慢 |
| `completion_extension` | 6 | 完成时间诊断的额外 padding；范围0–100。不扩大新决策区间，不是行程截断上限；必要时自动延长 |
| `planner_id` | `route_milp` | 当前注册的算法。控制器只依赖 `WindowPlanner` 接口，不依赖 Gurobi 变量 |

可以通过现有批实验协议扫描上述数值参数。扫描 H 与 k 时，每个组合都必须满足 `k <= H`；非法组合会被拒绝，不会静默修正。

### 控制、算法与状态更新分层

1. `rolling/controller.py` 负责 H/k 边界、当前可见状态快照、规划后缀的丢弃，以及每一期动作的选择。窗口、执行前缀均使用右端不含的区间；如 T=5、H=4、k=2，窗口为 `[0,4)`、`[2,5)`、`[4,5)`，执行区间为 `[0,2)`、`[2,4)`、`[4,5)`。
2. `rolling/contracts.py` 规定 `WindowRequest -> WindowSolution`。请求只含隔离后的当前状态、窗口边界和中断检查；输出是按期组织的 `PeriodAction / VehicleAction`，以及算法诊断，不是求解器内部索引。
3. `rolling/route_milp.py` 是第一种规划器：在一个 MILP 中联合选择车辆时空流、等待、允许的调位、已承诺路线继续/兼容延长、新路线和订单分配。固定路线锚点、逐弧容量、网络 rho、ready/deadline 与原 single 物理规则一致。订单最多分配一次，完成一趟后的车辆可以在窗口后续期再次使用。
4. `planning.py` 在每一期执行前按真实状态重新校验：时钟、车辆可用性、订单唯一性、pickup/endpoint 锚点、路线前缀与剩余逐弧容量。未指定的闲置车辆等待；未指定的在途车辆必须继续已有承诺。动作编译为原 `DecisionPlan` 后，仍由原 `Sim.commit_plan` 统一推进，保留 revision 与 stale-plan 防护。
5. `runner.py` 仍负责订单出现、执行、逐 t trace、双账本与日终冷却。规划器不能修改真实状态，不接触 scenario 全部未来订单，也不能绕过统一执行器。

增加新算法时，在 single 的 `rolling/` 内实现 `.plan(request)` 并在 `registry.py` 显式注册；返回相同领域动作即可复用控制、批实验、执行、trace 与核算。可直接注入另一规划器做单元测试。若新算法有专属参数，应在本族 Schema 中扩展，不在公共平台硬编码算法分支。借鉴的是 BHH 的窗口/冻结思想，没有导入 BHH 的 AV/HV、往返业务变量或参数类。

### 信息、承诺与日终边界

- **仅当前已预订订单**：没有需求预测，没有偷看后续真实订单。未揭示的 batch ready 使用当前可见上界，并在整个窗口中冻结这个上界，不利用未来 cutoff 的实际结果。
- **执行 k 期**：k期内新出现的订单正常进入订单池、计罚、过期，但不插入冻结计划；下一次重规划才能考虑。推荐先用 k=1。
- **物理承诺不撤销**：已经分配的订单、在途空驶、装载、holding、骨干路线和逐弧容量随真实车辆带入新窗口；仅未执行的规划后缀丢弃。规划尾部不直接写入真实状态。
- **T是新决策硬截止**：决策只发生在 `t < T`。路线可以在 H 或 T 后完成；`cooldown_realized` 在日终仅执行已有承诺，不再优化、不接受新分配。即使 `completion_extension=0` 也不会截断长路线。
- **中断与失败**：窗口构建和求解期间均检查取消/超时。有可行 incumbent 时可以执行并记录窗口 gap；没有可行解则失败，不提交动作、不自动改成 myopic。执行前发现非法动作也会拒绝，不能静默丢弃已有订单。

### 第一版不是完整全局最优模型

当前算法使用**路线宏动作**：在一个窗口内部，新选路线按既定骨干执行到端点；中间到站只继续承诺，不额外规划中途接单/延长。真实下一次重规划时，又允许原 single 的前缀兼容延长与沿途接单。因此它是可行的受限多期路线模型，不是所有逐期 M_t 动作序列的精确等价展开，也不是 full-horizon oracle。

窗口目标是当前可见订单的 **booked** 收益/成本（包含本次承诺的完整路线成本、backlog/expiry、等待及原 delay tie-break）；没有预测价值，`V=0`。窗口后缀会被重规划，未来新订单也会出现，故不能将多个 `window_objective` 相加当作实际利润；正式比较使用完整执行后的 `realized_profit`。不同 H/k 不保证单调改善，需配对实验评估。

逐 t `plan.solver` 新增 `replanned`、`window`；样本 metrics 新增 `planning_windows`、`window_solve_count`、`planning_algorithm` 和 `planning_information`。每个窗口记录起止/冻结边界、可见订单、求解状态、时间、`window_objective`、bound、gap、完成边界与自动延长量。该 bound/gap 只针对本窗口的受限 MILP；`global_gap` 明确为 null。`candidate_route_count` 是窗口构建的路线列总数，包含最终未被车辆流选中的候选，不宜直接与单期 myopic 的候选数比较。

回归入口：[test_single_rolling.py](tests/test_single_rolling.py)。其中合成场景仅用于验证控制与物理约束，不是 CFS 数据，也不是算法效果的统计结论。

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
