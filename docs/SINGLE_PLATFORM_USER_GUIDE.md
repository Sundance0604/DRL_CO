# DRL CO 实验平台详细使用手册：以 Single-level model 为例

核对日期：2026-10-04。适用分支：`react-experiment-platform`；平台 v0.2。核对基线：`7b861b28a2d927e8919c1ba79fd3e3c6163a75a2`。本文根据当前网页、前端字段、后端校验和 single runner 编写，不是未来功能规划。

本文面向“需要知道每个框、每个数、每个按钮是什么意思”的使用者。先按页面阅读；遇到结果 JSON，再查字段字典。相同字段重复出现时引用同一定义，不另造一种口径。

**重要：示例参数是原型占位值，不是经过实证标定的交通参数。COMPLETED 也不等于所有 MILP 都被证明全局最优。**

## 目录

- [1. 平台、模型、框架与样本是什么](#s01)
- [2. 启动、公共界面与输入规则](#s02)
- [3. 总览：每张卡片与运行表](#s03)
- [4. 数据管理：生成、冻结、预览、导入](#s04)
- [5. 实验编辑器：模板、全部参数与价值函数](#s05)
- [6. 参数范围与批实验、完整 JSON](#s06)
- [7. 队列：状态、进度与取消](#s07)
- [8. 运行详情：指标、事件、文件](#s08)
- [9. 实验比较：两个运行的 A−B](#s09)
- [10. 回放：逐 t、订单、车辆、决策与记账](#s10)
- [11. 复现：导出与检查 ZIP](#s11)
- [12. 数学模型：每个符号与当前 V](#s12)
- [13. 实验分析：论文图、样本与统计](#s13)
- [14. 诊断：环境、错误与单步](#s14)
- [15. 从第一次实验到论文图的操作示例](#s15)
- [16. Python 导出数据与本地文件字典](#s16)
- [17. 常见误区、限制、排错与核对依据](#s17)

<a id="s01"></a>
## 1. 平台、模型、框架与样本是什么

### 1.1 这不是一套大算法

平台是共享的本地实验外壳，负责冻结数据、展开任务、记录状态、保存结果、复现、统计与绘图。研究逻辑放在三个独立模型包里：

| 名称 | 模型族 ID | UI 固定识别色 | 本手册范围 |
|---|---|---|---|
| Single-level model | `single` | 青绿色，`#138579` | 全部页面、组件、字段 |
| DRL / SAC | `legacy` | 蓝色 | 只解释切换与边界，不解释 SAC 参数 |
| BHH | `bhh` | 橙色 | 只解释切换与边界，不解释 BHH 参数 |

三个族共用 Python 环境、调度器和工作空间，但不共用研究参数、数学模型、领域数据解释、runner、指标语义与 trace。选择 BHH 不是把 single 的控制器换成 BHH。UI 颜色用于识别论文上下文，不是论文绘图的算法配色。

`single` 是平台族名；`single_level_matching` 是本族模型插件及数据物理模型名；Myopic/fluid/rollout 等是本族实验框架；`react-experiment-platform` 是 Git 分支。四者不是同一概念。

### 1.2 single 实际做什么

给定城市枢纽网络、车辆初始位置、预订订单流，每一期先处理车辆到达与新预订，再求解订单指派、线路、延迟出发和可选空车重定位。已有承诺必须保留，容量按线路每条弧检查，订单需要满足可见的准备时间和目的地截止时间。

single 默认是优化与仿真，不是强化学习。Myopic、fluid、rollout、Oracle 都无需训练。可选 learned value 是对 fluid 对偶值的**监督线性拟合**，不是 SAC，也不是神经网络 RL。

### 1.3 必须分开的五个层级

```text
模型族 single
└─ framework：例如 rollout
   └─ parameter configuration：例如 samples=2、empty_cost=10
      └─ frozen scenario：例如 s-10001
         └─ 某个 run 中的该场景结果与逐期 trace
```

平台通常把“一套参数条件 × 一个 policy/training seed”创建为一个 run；**一个 run 可以执行多个场景**，不是一个 run 必然只有一个样本。

| 词/字段 | 含义 | 不应怎样理解 |
|---|---|---|
| parameter configuration / condition | 固定模型、控制器、价值函数、求解与评估条件 | 3 个参数值不是 3 个独立统计样本 |
| scenario / instance | 一套完整网络、初始车队、订单事件 | 把同一场景复制改名不会得到新独立样本 |
| scenario seed | 生成冻结场景的随机种子，保存在数据里 | 编辑器 Policy seeds 不会重新生成场景 |
| policy seed | 算法内部随机性，主要影响 rollout 抽样 | 确定性 myopic 改 10 个 seed 不等于 10 套新数据 |
| training seed | 拟合组件的运行种子，仅拟合阶段适用 | 当前线性全批量拟合从零开始，改 seed 不保证不同权重 |
| solver seed | Gurobi 搜索随机种子，属于求解资源参数 | 不等于 policy seed、scenario seed |
| replication | 同一条件下独立生成的场景样本 | 不同成本、算法或网络规模不能自动混成一个 n |

第一版论文分析以冻结的物理场景为统计单位；同一条件、同一物理场景下多个 policy seed 的结果先取平均，再跨独立场景统计。

### 1.4 时间、距离、钱与利润口径

- `t` 是离散期，默认从 0 开始。没有统一绑定为小时、分钟或一天。
- 网络边权是模型距离单位；`speed` 是距离单位/期。单边行驶期数是 `max(1, ceil(边权/speed))`，路径期数为各边期数之和，不是把全路径距离一次除以速度再取整。
- `currency` 是模型货币单位，没有默认指定人民币或美元。
- `operating_profit` 按**指派时收入与计划线路成本**记账，不是已到账现金流，不是订单全部送达后才记收入。
- `business_cost` 是同一记账口径的计划线路/重定位成本及周期性罚项，不是实际运行里程计算的财务支出。
- `augmented_solver_objective` 加入 V、扣除求解 tie-break；不是经营利润，不适合跨价值组件做收益排名。

<a id="s02"></a>
## 2. 启动、公共界面与输入规则

### 2.1 本机从哪里启动

当前真正的平台仓库：

```text
D:\mycodelife\workshop\AI\DRL_CO_new_model\DRL_CO_new_model_20261002\_review_repo
```

双击外层 `start.bat` 或仓库内 `start.bat`。外层 BAT 转发到真正仓库；不要用外层原型 `model/` 当平台程序。已有 `.venv` 和前端构建时，BAT 启动或复用本地服务，通过健康检查后打开默认浏览器。通常地址为 `http://127.0.0.1:8765/`，以启动结果中的实际 URL 为准。

关闭网页或启动提示窗口不会自动停止后台服务。停止入口为仓库 `scripts/stop.ps1`。首次没有环境或网页构建时先按[仓库 README](../README.md)完成安装；BAT 不自动安装，也不替代 Gurobi 许可证检查。

### 2.2 公共区域每项说明

| 界面文字/控件 | 说明 |
|---|---|
| `DRL CO / LOCAL RESEARCH PLATFORM`、研究工作台 | 产品标识，不是算法或结果字段 |
| 页面标题 + `Single-level model` | 当前页面及当前论文模型族；切换后名称和 accent 同步变化 |
| `LOCAL ONLY · v0.2` | 本机平台及版本提示；不是论文模型版本号 |
| 左侧菜单 | 进入总览、数据管理、实验编辑器、队列、运行详情、实验比较、回放、复现、数学模型、实验分析、诊断 |
| 诊断下方“论文模型族” | 选择 Single-level model、DRL / SAC、BHH；切换整个工作区，清空本次页面上下文，重新载入本族数据/运行/参数 |
| `参数 · 数据 · 框架 · 数学 · 结果完全切换` | 上述切换范围；不是自动把已有实验转换到另一个模型 |
| 本地工作空间、冻结数据 · 独立任务 · 可追溯 | 存储/执行方式提示；不是云端同步状态 |
| 红色错误条 / `role=alert` | 最近请求或校验错误，常包含 `error.code/message/path/details`，见第 14、17 节 |
| 提示条与 `×` | 操作成功说明；`×` 只关闭提示，不撤销操作 |
| “操作结果”展开区 | 最近一次生成、校验、预览、提交、比较、诊断的原始返回 JSON；切页通常清空，不等于已保存文件列表 |
| `DRL CO · 每个数字都有数据、配置和执行记录。` | 页脚说明，不是完整性验证结果 |

Single 导航没有“BHH 分析”；只有切到 BHH 才显示该页。实验分析目前只有 single 接入论文分析，其他族显示暂未开放的说明。

### 2.3 输入规则与保存规则

| 输入形式 | 如何填写 | 注意事项 |
|---|---|---|
| 数字框 | 直接填数字；整数参数只用整数 | 显示的下限/上限是提示，最终以服务端 Schema 和联合约束为准；不能输入 NaN/Infinity |
| `true / false` | 下拉选择布尔值 | JSON 中使用 `true`，不能写字符串 `"true"` |
| 固定枚举 | 选择支持的字符串 | 某些枚举因只有一个合法值被画成文本框，但不代表允许自定义 |
| 数组/对象框 | 合法 JSON；例如 `[10001,10002]` | **离开输入框时**才解析并更新表单值；非法 JSON 不会替换原值，不要在未失焦时认定新内容已生效 |
| 资源 ID | 1–96 字符，以字母/数字开头，后续字母、数字、`_ . -` | 不可含中文、空格、斜杠；实验显示名称可以中文，资源 ID 不可以 |
| 完整 JSON 大框 | JSON 对象，键名不能重复 | 不支持注释、尾随逗号；从网页发出时会先经过浏览器 JSON 解析，不要依靠重复键进行覆盖 |

模型族选择和编辑器表单草稿保存在浏览器 localStorage，按族分别保存。同一浏览器重新打开可能看到上次修改值，并非出厂默认值；它也不是已经提交的实验。数据生成表单、批次扫描行和分析选项不具有相同的持久化保证。

**上方表单 → 下方完整 JSON 是单向更新。** 手工改大框、或者“应用到批次 JSON”，不会把所有改动反向同步到表单/数学卡片；提交、校验、计划预览读的是大框，而诊断“当前配置单步”读的是上方表单 `spec`。再改上方任意表单可能把手改的批次 JSON 覆盖回单次配置。详见第 6 节。

数据与运行列表约每 3 秒刷新，不是每毫秒实时流；环境能力通常在工作区加载时获取。刷新整个页面可重新检查，但先保存重要 JSON 草稿。

<a id="s03"></a>
## 3. 总览：每张卡片与运行表

| 卡片/按钮 | 数值来源/功能 | 解释与边界 |
|---|---|---|
| `REPRODUCIBLE BY DESIGN`、主标语 | 介绍性文字 | 不表示每项环境依赖均自动可复现 |
| `创建实验 →` | 打开实验编辑器 | 不生成数据，不直接创建 run |
| 冻结修订 | 当前族数据修订列表长度 | 同一数据集多个 hash 分别计数；列表包含归档修订，不是独立场景数 |
| 已完成 | 当前族 `COMPLETED` run 数 | 不是送达订单数、样本数或最优证明数；可包含拟合与 Oracle |
| 执行中 | 当前族 `RUNNING` run 数 | 不包含 `QUEUED` |
| 需检查 | `FAILED / TIMEOUT / INTERRUPTED` run 数 | 不包含用户主动 `CANCELLED`；也不是数据校验错误次数 |
| 最近运行 | 当前族最近最多 6 个运行 | 展示顺序来自运行列表，通常最近提交在前，不是利润排序 |
| 查看全部 → | 进入队列 | 不修改运行 |
| Gurobi 许可证：已验证/未验证 | 能力检查中实际建立并求解小模型 | `已验证` 不保证所有大模型规模都在许可证额度内，也不保证未来许可证一直有效 |
| CPU 运行等提示 | 已知能力边界 | 当前无断点续算；macOS 原生启动未验证，不应解释为全平台已验收 |
| 研究口径 | 监督拟合≠SAC、Oracle/在线/理论口径分开 | 是必须遵守的解释约束，不是可关闭筛选 |

总览、队列共用运行表：

| 列 | 每个显示项是什么 |
|---|---|
| 实验 | `spec.name` 为人类名称；下面 `r-...` 是唯一 `run_id`；点击名称进入详情 |
| 模型 / 控制器 | 上行模型插件 `single_level_matching`，下行 `myopic/rollout/train_value/oracle_lp/oracle_mip`；fluid 的控制器仍是 myopic，需看 value/framework 才能区分 |
| 状态 | 生命周期状态，见第 7 节 |
| 进度 | 最后一个含 `progress` 的事件 ×100%，四舍五入；主要为完成场景比例，不是 MILP 最优性 gap |
| 打开 → | 选择 run 并进入运行详情 |
| 取消（仅队列符合条件时） | 请求取消 QUEUED/RUNNING；完成 run 不显示此按钮 |

<a id="s04"></a>
## 4. 数据管理：生成、冻结、预览、导入

### 4.1 “生成可重复的示例数据”全部字段

下面“网页初始值”指新载入当前工作区的数据生成表单；“Schema 默认值”指直接调用接口时省略该字段的默认值，两者确实有区别。

| 字段 | 网页初始值 / Schema 默认值 | 合法值 | 作用及注意事项 |
|---|---|---|---|
| `schema_version` | `dataset-generation/v1` / 同 | 必须为此值 | 生成请求协议版本；不改变随机分布，不要当备注修改 |
| `dataset_id` | `single-demo` / 必填无默认 | 合法资源 ID | 数据集容器名；同名容器可有多个冻结修订；并非实验名 |
| `family` | `single_level_matching` / 同 | 只能此值 | 当前族的数据接口；不能写 `single` 或其他族名；跨族被拒绝 |
| `seeds` | `[10001,10002,10003,10004]` / `[10001,10002,10003]` | 1–100 个互不重复整数；0–2,000,000,000 | 每个 seed 生成一个完整场景，包括网络、车队和需求；场景 ID 为 `s-<seed>` |
| `splits` | `["test"]` / `["train","validation","test"]` | train/validation/test；长度为 1 或等于 seeds 长度 | 一个值应用于所有场景；逐个值按 seeds 顺序对应。不会复制数据，也不会随机重新划分 |
| `horizon` | `6` / `8` | 整数 1–200 | 预订与正常决策期数，运行 t=0…horizon−1；不是所有已接订单都能在 horizon 前到达 |
| `num_vehicles` | `5` / `5` | 整数 0–100 | 初始车数，允许零车的基准；位置按 seed 生成；车辆异质性不是这里的可选项 |
| `num_cities` | `8` / `8` | 整数 2–20 | 随机场景的枢纽节点数；不是地图行政市个数，城市位置不直接画地理坐标 |
| `orders_per_step` | `2` / `2` | 整数 0–100 | 每个预订期固定生成的订单数；总生成数通常 horizon×此值，**不是 Poisson 强度**；订单乘客数另行随机 |
| `first_mile` | `batch` / `batch` | batch/direct/none | 接驳准备时间机制，见下表；这是数据生成项，不是运行参数 |

`seeds` 不同可能同时改变网络、车辆和需求。若研究问题是“同一固定网络上的需求变动”，仅在此框换 seed 并不足以保证固定网络；需要专门构造/导入固定网络场景。

| `first_mile` | 真实实现 | 可见信息与限制 |
|---|---|---|
| `batch` | 同城市、同接驳收集窗口的预订组成 HV 接驳 tour | 收集窗口默认 2 期、接驳容量 7；收集结束前使用准备时间上界，结束后揭示实际准备时间 |
| `direct` | 每订单有随机直接 first-mile 时长 | 起点在圆盘内均匀抽样、hub 在边界；不是调用真实道路地图 |
| `none` | 预订时即在 hub 准备好 | 去掉接驳等待，不是“没有订单” |

接驳的 `H_BATCH=2、Q_HV=7、K_CA=0.9、TAU_S=0.1`、城市半径/速度比例 0.5–1.5 期、乘客原始抽样 1–7、deadline slack 10–20 期属于当前生成器固定假设，**不是网页隐藏可编辑字段**。改变运行 `capacity` 不会重新生成乘客分布或改变接驳容量。

“生成冻结修订”会生成本族数据、校验、写本地文件、计算 hash，并把新修订选入编辑器。它真的写数据，但尚未提交求解实验。同样 payload 具有同样 hash；生成配置/provenance 也是 payload 一部分，因此改容器名等配置可能影响 hash。不要仅凭 ID 猜数据一致。

### 4.2 “数据修订”表格与操作

| 列/按钮 | 行为 | 易混点 |
|---|---|---|
| 数据集 | `dataset_id`；归档项附“已归档” | 一行是一个修订，不一定一个独立数据集 |
| 版本 hash | SHA-256，表格缩写前 12 位，悬停可看完整值 | 比较/复现要用完整 hash；不是 Git SHA |
| 物理模型 | `single_level_matching` | 不是 controller/framework |
| 预览 | 读取完整修订，显示场景摘要 JSON | **每场景只显示前 10 个订单**；预览少于真实订单不代表丢数据；同时设置导入网络模板 |
| 选用 | 当前修订进入编辑器并切页，清空限定 scenario_ids | 拟合控制器倾向 train split，其他 test；仍需自行确认 split 中真的有场景 |
| 校验 | 使用 all split 读取并核验 hash、网络、订单等 | 校验数据结构不保证订单一定可服务，也不启动优化 |
| 归档 | 给整个 dataset_id 写 archived 标记 | 影响该容器所有修订；不是删除，既有记录仍在；编辑器普通下拉不再列出它。当前网页没有“取消归档”按钮 |
| 导出 JSON | 打开修订 JSON 链接 | 浏览器可能显示 JSON 而非自动下载；这是精确数据，不是只有生成配置 |

### 4.3 “数据预览”中每个字段

生成/冻结后的“操作结果”是修订manifest：`dataset_id`为容器ID，`revision`和`hash`为同一完整内容摘要，`family`为物理模型，`scenarios[]`只列每场景`id/split/horizon`索引。校验按钮返回`valid`、`scenarios`（场景数）、`orders`（所有订单条数）。这些数量是数据规模，不是完成实验数或独立统计性证明。

预览 JSON 顶层为场景数组，字段含义如下；包外的完整数据 envelope 也同时列出。

| 路径/字段 | 说明 |
|---|---|
| envelope.`schema_version` | `dataset/v1`，是冻结数据协议，不是 dataset-generation/v1 |
| envelope.`family` | `single_level_matching` |
| envelope.`scenarios` | 场景数组；平台上限 100 个 |
| envelope.`provenance` | 原始来源；生成时 generator 与 config，导入时 import 与 mapping；不能当算法结果 |
| scenario.`id` | 场景 ID，在数据修订内唯一；自动生成 s-seed |
| `seed` | 用于生成及路径一致性随机扰动的场景种子 |
| `split` | train/validation/test，属于数据，不是提交时临时重标记 |
| `horizon` | 正常决策时域 |
| `nodes` | 节点 ID 字符串列表，至少两个且唯一 |
| `edges` | `[起点ID,终点ID,正距离]` 列表；当前是无向连通网络，非经纬度 |
| `hubs` | 每辆初始车的位置列表；顺序对应运行车辆 ID `0,1,…`；允许多辆在同 hub |
| `radii` | 每个城市的 radius/city-speed 时间尺度；不是地图半径公里数 |
| `orders` | 全场景订单；预览截前 10 条，正式导出不截断 |
| order.`id` | 同一场景唯一的订单字符串 ID |
| `departure` | 起点 hub ID，必须在 nodes 中 |
| `destination` | 终点 hub ID，不能等于 departure |
| `passenger` | 正整数需求载荷；容量使用此数，不是订单条数 |
| `book_time` | 非负整数预订期，必须早于 horizon；订单从此期进入池 |
| `start_time` | 实际可接载准备时间，允许小数；不早于 book_time |
| `end_time` | 到达目的地截止时间；不早于 start_time，不是预订结束时间 |
| `revenue` | 指派成功时收入，有限数；生成器用 distance×100+passenger×50；当前导入校验未强制非负，需自己保证研究合理性 |
| `penalty` | 每期未指派 backlog 罚项，非负；生成器用 passenger×5；不是一次性 expiry_cost |
| `slack`（batch） | 生成 deadline 所用余量，默认抽样 10–20 期 |
| `cutoff`（batch） | 接驳收集结束期；在此之前实际 start_time 未供在线策略使用 |
| `bound`（batch） | 接驳可准备时间保守上界；`t<cutoff` 时在线 ready_time 使用该值 |

冻结数据和 before/arrivals 可包含仿真真实 start_time，这供审计使用，不表示在线决策偷看了它。在线决策使用 `ready_time(o,t)`；对外观察 `observation.orders.start_time` 在 cutoff 前改为 bound。

### 4.4 “导入订单表”所有字段

| 控件 | 用法 | 边界 |
|---|---|---|
| 格式下拉 | csv/json/jsonl，初始 csv | 文件扩展名不自动切换该值；两者必须匹配 |
| 选择文件 | 选择 .csv/.json/.jsonl 文本，读到下面文本框 | 单纯选择只读文本，点击冻结按钮才提交后端 |
| 粘贴订单数据文本框 | 可直接粘贴；选择文件会替换文本 | CSV 用表头；JSON 用对象数组；JSONL 每非空行一个对象 |
| 校验并冻结导入 | 当前预览的**第一个场景**作为网络/车队/seed/split/horizon 模板，替换 orders，生成 import-时间戳数据集 | 没有预览时禁用；不继承所有场景、不生成多 seed 副本；导入后仍应在修订列表选择它，按钮不等同于选用 |

网页固定列映射：`id,departure,destination,passenger,book_time,start_time,end_time,revenue,penalty`，逐项含义见上表。UI 不显示自定义列名映射；API 才可自定义 mapping。网页也未提供 cutoff/bound/slack 的额外列映射；**导入保留 batch 网络模板不意味着订单仍具备 batch 揭示机制**，没有这些元数据的订单按普通 start_time 使用。

示例（节点 `0→1` 必须存在于模板；数值只是格式示例）：

```csv
id,departure,destination,passenger,book_time,start_time,end_time,revenue,penalty
o-1,0,1,2,0,1.5,12,1200,10
o-2,1,0,3,1,2,13,1300,15
```

限制：原始订单文本 ≤10 MiB、≤20,000 条订单；冻结总数据 ≤100 场景且合计 ≤200,000 订单。网页/API 请求还有 15 MiB 总体限制。导入是一个完整新场景，不是给原场景“追加几条”。

<a id="s05"></a>
## 5. 实验编辑器：模板、全部参数与价值函数

### 5.1 模型包实验模板

| 下拉模板 ID | 设置内容 | 需自行检查 |
|---|---|---|
| `experiments--matching-myopic` | myopic + zero，普通基线 | 数据 split/test 及真实 hash |
| `experiments--matching-fluid` | myopic + fluid_dual，模板把 recompute 设 true，预期订单数 2 | **Schema 默认 recompute 是 false，模板不一样** |
| `experiments--matching-rollout` | rollout + zero；模板 samples=1、预期订单数 2 | 普通框架默认 samples=2；不是相同配置 |
| `experiments--matching-bound` | oracle_lp、oracle 信息集 | 不是执行车辆的 perfect-information policy |
| `experiments--value-train` | train_value + zero，train split，epochs=30 | 需要独立训练场景，完成后取得 run ID |
| `experiments--value-evaluate.template` | learned_hub_time；checkpoint_run 是替换占位符 | 必须替换成真实已完成拟合 run ID；不能直接提交 |

数据生成模板 `datasets--matching-demo` 不在此实验模板下拉里。模板只属于本族，保留当前选中的冻结修订，不使用模板里的 `latest` 作为真实 hash；但可能重置名称、框架、参数、split 与其他默认项。选模板后重新核对完整 JSON。当前框架由组件推断，不应靠名称判断。

### 5.2 名称、数据及框架

| 字段/区域 | 对应路径 | 解释 |
|---|---|---|
| 实验名称 | `name` | 1–200 字符，允许中文，显示用途；不是文件夹名，重复名称合法 |
| 数据集修订 | `dataset.dataset_id/revision` | 选择未归档的当前族冻结修订；标签为 ID·hash 前 8 位；必须有真实修订 |
| split 下拉 | `dataset.split` | train/validation/test/all；筛选冻结场景，不改变数据自身 split；all 不意味着允许 train 当评估样本 |
| 当前实验框架 | `framework_id` 及三个组件引用 | 从本族框架选择；切换会重置 controller/value 参数和评估口径，通常保留相同模型的物理参数 |
| 学习组件阶段 | 仅 learned 框架出现 | Training/拟合切 train_value+zero、train；Held-out evaluation/评估切 myopic+learned_hub_time、test |
| 数学卡片 | 当前表单 spec | 随框架显示公式；不是已运行结果，也不一定反映手改大框 JSON |
| 模型参数 / 框架参数 / 价值组件参数 | `model/controller/value_function.parameters` | 由插件 Schema 自动识别；标题下显示插件 ID；无参数时“当前组件无可调参数” |

| 框架菜单名称 | framework / controller / value | 用的信息 | 训练与输出 |
|---|---|---|---|
| Myopic | myopic / myopic / zero | online | 无训练；V=0，运营 trace/metrics |
| Analytical / fluid value | fluid / myopic / fluid_dual | online | 无训练；LP 对偶价值，运营 trace/metrics |
| Sampled rollout | rollout / rollout / zero | online | 无训练；抽样前瞻，运营 trace/metrics |
| Perfect-information LP bound | oracle_lp / oracle_lp / zero | oracle | 连续松弛上界，无实际运营 trace/profit |
| Perfect-information MIP benchmark | oracle_mip / oracle_mip / zero | oracle | pooled 网络整数 benchmark，仍松弛车辆级部分限制；非真实 policy |
| Learned hub value (optional supervised component) | learned / train_value 或 myopic / zero 或 learned_hub_time | online | 只有此组件暴露拟合阶段；评估要 checkpoint |

直接选择 learned 框架通常先进入拟合；不是已经拿到了权重。切阶段也会重置相关组件参数，先记录 checkpoint ID，再在评估阶段填入。

### 5.3 模型参数：`single_level_matching`

以下是 Schema 默认值；保存草稿、模板与批次 variants 可以覆盖它们。

| 字段 | 默认 / 合法值 | 实际意义与影响 |
|---|---|---|
| `speed` | 20；有限数 >0 | 主干网络行驶速度；通过逐边取整改变 travel periods。不是城市接驳速度，改它不会重写冻结准备/截止时间 |
| `capacity` | 7；整数 1–100 | 主干车辆每弧载荷上限 Q，包含旧承诺占位。不是每车最多订单数，也不是生成器固定 HV 接驳容量 |
| `loaded_cost` | 10；≥0 | 新计划主干线路单位距离成本；延长旧路线只计新增尾段。与 actual_distance 无关 |
| `empty_cost` | 10；≥0 | 跨 hub 接载空驶及显式重定位的单位距离成本；不同空驶来源都受影响 |
| `waiting_cost` | 0；≥0 | 空闲车等待一期及新线路延迟出发的计划等待成本；不是订单池每订单 penalty |
| `expiry_cost` | 300；≥0 | 当订单若继续拖到下一期已不可满足截止时间、且本期未指派时的附加过期罚款；不是所有 pending 在终点统一罚此数 |
| `max_delay` | 3；整数 0–12 | 空闲车候选线路可延迟出发 0…此值期；不是允许迟到交付的期数，deadline 仍必须满足 |
| `tie_break` | 1；≥0 | 求解目标扣 `tie_break × 所选线路 delay`；运营利润不扣这项。它是软目标项，并非严格词典序，在收益很接近时可能改变经营收益最优解 |
| `cross_hub` | true；布尔 | 空闲车候选 pickup hub 可含当前位置邻居；不是任意全网 teleport；false 只从自身 hub 出发接载，仍可能在线路下游接单 |
| `reposition` | false；布尔 | 允许空闲车选择无订单锚点的空车重定位；与 cross_hub 为不同动作，true 不代表保证会发生 |

改变 speed/capacity/costs 是实验条件变化，不是数据样本增加。即使 capacity 大于 7，既有生成数据载荷仍按原始 1–7 抽样；即使 speed 改了，数据生成时写好的 deadline 不会重新生成，因此这是在同一需求上的物理参数敏感性。

### 5.4 框架参数：Myopic、Oracle 无可调项

`myopic/oracle_lp/oracle_mip` 的 `controller.parameters` 必须是空对象 `{}`；它们没有 hidden epochs、learning_rate。CPU/求解限时统一在 solver 里设置。未知参数通常被 `extra=forbid` 拒绝，不会被默默接受。

### 5.5 框架参数：Sampled rollout

| 字段 | 默认 / 约束 | 实际意义 |
|---|---|---|
| `samples` | 2；整数 1–16 | 每期、每候选规则使用 K 个随机未来；这些不是论文实验统计 n；增加 K 通常增加求解次数和耗时 |
| `candidate_cross_hub` | `[false,true]`；非空布尔数组 | 比较哪些首期 cross_hub 规则；数组整体填 JSON。只 `[true]` 就没有 own/cross 二选一；允许重复布尔但没有科学收益 |
| `expected_orders_per_step` | 2；整数 0–100 | 前瞻抽样每个未来预订期生成订单数；不改变冻结真实订单；不是自动读取数据生成 orders_per_step |

同一期的候选规则使用同一组 sampled futures，有利于降低比较噪声。先各解首期计划，再复制状态模拟未来，用最后累计利润 `clone.J` 的 K 次平均选候选，提交赢家首期。未来 base rule 使用模型参数 cross_hub，value 必须 zero。

这是有限候选规则前瞻，不是穷举所有动作的动态规划；首个受影响的 first-mile batch 条件抽样仍是近似。`policy_seed` 控制抽样。`samples` 与 policy seeds、scenario seeds 是三类不同数量。

未来抽样器目前按有无batch元数据选择batch或direct机制；没有单独复现`first_mile=none`的零准备时间未来分布选项。因此在none冻结数据上使用rollout，前瞻仍可能按direct first-mile抽样，这是分布模型近似/错配，需要在研究报告说明，不可称为完全匹配的需求模型。

### 5.6 价值组件：zero 与 fluid_dual

zero 无参数，V(h,t)=0。

| fluid 字段 | 默认 / 约束 | 解释 |
|---|---|---|
| `recompute` | false；布尔 | false：每个场景第一期算 LP 并缓存；true：每期从当前状态重新算 LP。前者不是每天用旧数据库统一 V |
| `multiplier` | 1；≥0 | V 缩放系数 m_V，最终 V=m_V×对应车辆流守恒约束对偶值；设 0 等于不使用该价值项，但代码仍可能计算 LP，不保证节省耗时 |
| `expected_orders_per_step` | 2；整数 0–100 | LP 对未来需求期望的假设；均匀 OD，平均载荷 4、平均 slack 15、典型 batch 准备时间近似；不是实际未来预订 |

具体 V：当前车队变成各 hub/未来可用时刻供给；当前订单池用在线可见 ready_time，未来用预期订单；建立连续 time-expanded 流 LP，含载客流、空驶流、等待流、服务需求份额、容量与车辆流守恒；最大化收入+避免罚款−线路成本。解到最优后读 `flow[hub,time].Pi`，乘 multiplier；缺失键返回 0。

它是车辆可用位置/时间的边际启发价值，**不是 LP 总目标值**、不是单订单 revenue、不是精确 Bellman V，更不是严格运营上界。fluid LP 当前成本近似主要使用 loaded/empty cost；没有把在线所有罚项/等待结构逐项完整搬过去。

### 5.7 可选拟合：train_value

| 字段 | 默认 / 约束 | 解释 |
|---|---|---|
| `epochs` | 30；整数 1–1000 | 对收集到的 fluid-dual 标签全批量梯度更新次数；不是模拟期数 |
| `learning_rate` | 0.01；有限数 >0 | 名义学习率；实际除以“1 + X 的谱范数平方/n”做尺度调整；太大仍可能数值不稳定 |

拟合运行要求 dataset.split=`train`、value=`zero`。模拟走 myopic reference 并收集 fluid 标签，拟合从零初值开始的 7 维线性模型，均方误差及 `1e-4` L2 正则更新，不是 RL reward 更新。训练 MSE 是监督标签拟合误差，不是运营 profit。`epoch` 从 0 开始；最后 12 个事件不一定包含整条训练曲线。

### 5.8 可选评估：learned_hub_time

| 字段 | 默认 / 约束 | 解释 |
|---|---|---|
| `checkpoint_run` | 必填，无合法默认；资源 ID | 填本机已完成 train_value run 的 `r-...`，不是 weights 文件路径、batch ID、显示名称或 REPLACE 占位符 |
| `multiplier` | 1；≥0 | 缩放预测 V=m_V×wᵀφ；与 fluid multiplier 作用类似 |

φ 有 7 项：常数 1、归一化剩余时间、归一化节点度、归一化平均 travel time、该 hub 空闲车比例、该 hub backlog load/总容量、平均城市 radius-time。权重存在 `weights.json`，不是 pickle 神经网络。

评估检查 feature schema、权重 hash、物理模型参数一致；在同修订里选中训练场景会触发泄漏拒绝。**这不是全面的数据去重防泄漏**：如果把物理相同训练场景改名放进另一修订，使用者仍需自己避免泄漏。建议在一份修订中用明确不重叠 seeds 分好 train/test，保留全部来源证据。

### 5.9 “求解资源”全部字段

| 字段 | 默认 / 约束 | 解释与注意 |
|---|---|---|
| `time_limit_seconds` | 30；后端有限数 >0，界面提示下限 0.1 | 每次 Gurobi optimize 的限时，不是整个 run 总限时；rollout 一期会调用多次，fluid 也另解 LP |
| `mip_gap` | 0；0–1 | 相对最优性 gap 允许值；0 请求尽可能精确，仍受限时。0.01 约为 1%，不是 0.01% |
| `threads` | 1；整数 1–16 | 单个 Gurobi 模型线程；不是同时运行实验数；需满足平台 cpu_budget |
| `seed` | 0；整数 0–2,000,000,000 | Gurobi 搜索 seed；不重新生成场景，不控制 rollout futures |

single solver.backend 固定 `gurobi`。即使只用线性 learned value，在线决策仍解 MILP，不能改 CPU 假装实现 Gurobi 替代算法。默认并发在 platform.local.json；不是这里 threads。

### 5.10 “评估与终点”全部字段

| 字段/值 | 默认 | 解释 |
|---|---|---|
| 信息集 `online` | 普通框架默认 | 只能用当前可见状态及未来分布；不是关闭保存未来真实数据 |
| 信息集 `oracle` | Oracle 框架自动选 | 全时域真实订单用于 pooled bound；普通在线框架宣称 oracle 会被拒绝 |
| 终点 `report_pending` | 默认 | 到正常 horizon 截止，分别记录未指派、已承诺未完成；不会强制所有车送完 |
| 终点 `drain_committed` | 可选在线框架 | horizon 后清空未指派池，只推进已承诺车辆/订单，记录额外 drain 期；不再接新单。有基于最晚 deadline 的终止上限，不应理解为无限续跑 |

drain 不会把截止时未指派订单当作新任务服务；`pending_unassigned` 保留截止数量。终点不同，会改变 delivered、pending、成本/利润及时间曲线，不可混为同口径比较。

Oracle 必须 oracle+zero+report_pending；warmup 当前只支持 0；accounting_version 固定本族口径。界面仍有选项不意味着所有组合合法，校验拒绝后不会自动换算法。

<a id="s06"></a>
## 6. 参数范围与批实验、完整 JSON

### 6.1 “参数范围与批实验”所有字段/按钮

| 字段/按钮 | 初值/选项 | 如何使用 |
|---|---|---|
| 添加扫描参数 | 无扫描行；新行默认首个参数、linear、1→3 间隔1 | 增加一维网格；新行默认数字不保证适合所选参数，须更改 |
| 扫描参数 N | 当前模型/controller/value Schema 参数路径 | 只列本族当前组件；例如 `model.parameters.empty_cost`；切框架后原路径可能失效，需要重选 |
| 数组项路径 | 例如 `controller.parameters.candidate_cross_hub.0` | 扫某个数组元素；数组整体扫描则列表中每个值本身又是数组 |
| 扫描方式：线性间隔 | `linear` | start、start+step、…≤stop；step>0、stop≥start |
| 扫描方式：对数倍率 | `log` | start、start×step、…≤stop；start>0、step>1；不是 log10 指数步长 |
| 扫描方式：离散 JSON 值 | `list` | 在离散值填非空 JSON 数组，可扫描布尔、数组、枚举/字符串或非等距数字；仍经 Schema 校验 |
| 起点 | 默认1 | 包含的第一个值；整数参数最终每一项须为合法整数 |
| 终点 | 默认3 | 允许最大边界，只有序列正好达到才包含它；1→4、step2 生成1和3，不强行补4 |
| 间隔 / 倍率 | 默认1 | linear 用加法，log 用乘法；log=1 非法 |
| 离散值 | 默认 `[1,2,3]` | 布尔用 `[false,true]`；候选数组整体用 `[[false],[true],[false,true]]` |
| 移除 | 每行 | 只移除此扫描轴，不删除 run 或数据 |
| Policy seeds（不是 scenario seeds） | 文本 `0` | 逗号分隔不同非负整数；一条 run 对应一个 policy seed；一般确定性框架只用0 |
| Training seeds | 拟合控制器时替代上项 | 生成 training_seeds，policy_seeds 置 `[0]`；仅拟合适用 |
| 场景 ID（留空…） | 空 | 逗号分隔 s-ID，空用所选 split 全部场景；不填 seed 数字，不在该 split 的 ID 会拒绝 |
| 应用到批次 JSON | 无 | 从**当前表单**建立 batch-spec，覆盖下面大框；不提交、不自动校验后端、不回写上方参数 |

扫描范围不能直接扫描 dataset.seed、horizon、dataset ID 或 execution seeds。生成器参数是数据条件，应先生成不同冻结数据；Policy/Training seeds 有专门协议。网页扫描选择器不列 solver 参数，但高级 JSON 可扫描 `solver.parameters.*`。目前没有可视化“多框架批选”，需高级 variants 或分次提交。

### 6.2 提交前正确顺序

1. 选数据和框架、填上方单次参数；数组字段先失焦。
2. 配扫描轴和 seeds，点击“应用到批次 JSON”。
3. 需要不同 framework 时在大框加 variants；之后**不要再动上方表单**，否则可能覆盖大框。
4. 点“校验配置”，读错误/操作结果。
5. 点“预览计划”，核对 count、全部 runs、场景集合、真实 hash、policy/training seed、最终参数。
6. 保存 JSON 草稿后点“提交实验 →”。成功后进入队列。

“校验配置”也会展开计划、读数据并检查组件/联合约束，但不执行正式优化；不等于校验许可证或保证求解可行。“预览计划”也不执行正式优化。

提交真正在本地创建 batch/run、源快照、文件和队列任务。网页没有给连续点击自动加幂等键；重复提交可以创建重复任务，避免连点。成功提示不等于实验已经完成。

### 6.3 完整单次 JSON 所有字段

字段中省略的可选项由后端填默认；未知协议字段一般不允许。显示的解析后配置以 `spec.resolved.json` 为准。

| 路径 | 值/默认 | 解释 |
|---|---|---|
| `schema_version` | experiment-spec/v1 | 单次配置协议 |
| `family` | single；可省略由 model 推断 | 显式声明当前族，不能与 model 矛盾 |
| `framework_id` | 根据 controller/value 推断 | 人类框架标签的机器 ID；若手改组件也要删除旧 framework_id 或改成匹配值 |
| `name` | 必填1–200字符 | 实验显示名称 |
| `dataset` | 对象 | 冻结数据引用 |
| `dataset.dataset_id` | 必填 ID | 数据容器 |
| `dataset.revision` | 必填真实 hash | 不能把模板 latest 当实际修订；读时内容 hash 必须吻合 |
| `dataset.split` | test | train/validation/test/all；拟合必须 train |
| `dataset.scenario_ids` | [] | 场景过滤；与 split 是交集，不是并集 |
| `model.id` | single_level_matching | 物理模型插件 |
| `model.version` | "1" | 当前插件协议版本，仅 "1"；不是论文版本 |
| `model.parameters` | 第5.3节 | 模型参数对象，默认会补齐 |
| `controller.id` | myopic | myopic/rollout/train_value/oracle_lp/oracle_mip |
| `controller.version` | "1" | 同上 |
| `controller.parameters` | 对象，依当前插件 | 对应第5.4/5.5/5.7节 |
| `value_function.id` | zero | zero/fluid_dual/learned_hub_time |
| `value_function.version` | "1" | 同上 |
| `value_function.parameters` | 依当前价值组件 | 第5.6/5.8节 |
| `solver.backend` | gurobi | 本族必须 gurobi |
| `solver.parameters` | 第5.9节 | 求解设置 |
| `evaluation.information_set` | online | 信息集合 |
| `evaluation.accounting_version` | legacy-assignment-v1 | 本族固定指派时记账；名字含 legacy 不表示用了 SAC runner |
| `evaluation.terminal_policy` | report_pending | 终点策略 |
| `evaluation.warmup_periods` | 0，非负整数 | 契约有字段，但 single 非0会拒绝；目前未实现 warm-up 统计 |
| `evaluation.metrics` | 公共默认 profit/assigned/delivered/pending；新族草稿可能列全9项 | 指标请求/声明列表；当前 runner 会生成本族适用完整指标，不是只录此列表。分析页面自己选择图指标 |
| `execution.policy_seeds` | [0]；1–100个不同整数，0–2e9 | 多个值展开多个 run；场景仍复用冻结集合 |
| `execution.training_seeds` | null/省略；提供时1–100不同整数 | 只允许 train_value，非拟合框架提供会报 TRAINING_NOT_APPLICABLE |
| `execution.save_trace` | true | 保存全部已成功完成 runner 的逐期 trace；false 则 trace.json空数组，无法回放/时间序列/完整状态导出 |
| `execution.timeout_seconds` | 300；>0有限数 | 整个 run 的总墙钟超时；与每次 optimize 限时不同，多场景合计受它限制 |

`training_seeds` 不存在与空数组不等价：不要为非学习框架填 `[]` 占位；用省略或 null。默认已有草稿可能是早先的4项 metrics，当前族默认构造可能是9项，这不导致数据记录缺字段。

### 6.4 batch-spec/v1 每个额外字段

| 字段 | 说明 |
|---|---|
| `schema_version` | batch-spec/v1；单次不包这一层 |
| `base_spec` | 完整单次配置，各条件共同起点 |
| `variants` | 1–100个覆盖对象；逐项递归合并 base_spec，不是顺序互相覆盖；普通单框架用 `[{}]` |
| `ranges` | 参数路径→{start,stop,step,scale}；含边界展开规则见第6.1节 |
| `sweeps` | 参数路径→非空离散值数组 |
| ranges 项的 `scale` | linear/log，默认linear |

各轴是笛卡尔积，不是按行配对。2 个 variants × 3 个 empty_cost × 2 个 loaded_cost × 2 个 policy seeds = **24 runs**；如果每 run 有 10 场景，就是240次场景执行，但每条件统计 n 仍通常为10，不是24或240。

单轴≤1000值、最终整个批次≤1000 runs；同一路径不能同时出现在 ranges 和 sweeps。离散列表不要重复值，否则只会重复执行，分析也不会把相同物理 instance 变成独立样本。

不同框架的 variants 应完整替换各自 component.parameters，例如从 fluid 切 zero 时写 `"parameters": {}` 未必能清除递归合并继承的旧字段；最安全是 base 用 myopic+zero 空参数，再向 fluid/rollout variants 添加本组件参数。必要时单独提交，不要盲目拼所有组合。框架不支持的参数路径会拒绝，不会跳过。

### 6.5 “预览计划 / 校验配置”的返回字段

| 返回路径 | 含义 |
|---|---|
| `valid` | 校验按钮成功返回 true；计划按钮直接返回 plan，没有这层 |
| `plan.schema_version` / 根 schema_version | execution-plan/v1 |
| `count` | 将创建的 run 数，不是独立场景数 |
| `runs[]` | 展开的每个 run 条目，还没有真正分配 r-ID |
| 条目.`spec` | 已补默认、匹配族/框架的单次精确配置 |
| `policy_seed` | 执行器运行 seed；拟合预览里也可能为选定训练 seed，不能只看字段名判断统计含义 |
| `training_seed` | train_value 时的 seed，否则 null |
| `scenario_ids` | 此 run 真正执行的场景列表 |
| `scenario_seeds` | ID→冻结场景 seed 映射 |
| `dataset_hash` | 冻结修订完整 hash |
| `condition_id` | 模型/框架/value/solver/评估配置摘要，不含数据与运行 seed；分析另用 context/instance 分组 |
| `statistical_unit` | independent-scenario；不是声明“所有 seed 一律独立” |
| `resource_policy` | spawn isolated processes; no nested pools，任务独立进程执行 |

提交成功返回 `batch_id=b-...` 和 `runs[{run_id,status}]`；这些才是创建后的资源 ID。

<a id="s07"></a>
## 7. 队列：状态、进度与取消

“独立任务队列”使用第3节相同五列表。卡片小字“默认并发1；单个求解器默认1线程”描述默认值；实际并发/cpu_budget 以诊断为准。

| 状态 | 说明 | 接下来做什么 |
|---|---|---|
| QUEUED | 已保存/入队，等待并发与 CPU 预算 | 等待，不要重复提交；可取消 |
| RUNNING | 独立 worker 运行中 | 读事件/日志；可请求取消 |
| COMPLETED | runner 正常结束并写完整结果 | 检查每次 solver.status/gap，再做统计 |
| FAILED | 校验后实际执行异常、无 incumbent、数值问题等 | 看 error、stderr.log、diagnostic.zip |
| CANCELLED | 用户请求取消，可能为协作退出或强制停止 | 不当作成功评估；当前不是断点 |
| TIMEOUT | 整体 run 超时 | 检查总timeout和solver限时；不能悄悄丢弃失败样本 |
| INTERRUPTED | 服务重启/意外中断等恢复识别 | 不支持从中间 trace 续算，修复环境后重新执行 |

progress 是按已完成场景推进，首个场景长时间求解时可一直0%；训练阶段可能模拟全场景后进度已100%但拟合仍在运行。100%并不代替状态COMPLETED，也不是每个 MILP gap=0。

取消 RUNNING 时先写请求文件，正常在检查点响应；长求解未响应时调度器会在宽限后终止其拥有的进程。不要当作“暂停”。run 的 metrics/trace 通常在整个 runner 结束才写；失败或取消不能保证有已完成场景的完整文件，事件中可能有部分 metrics。`save_trace=true` 不是逐期崩溃安全落盘保证。

任务执行提交时捕获的源码快照，不会因你之后编辑工作目录而自动改变正在排队的实验代码。

<a id="s08"></a>
## 8. 运行详情：指标、事件、文件

### 8.1 所有可见区域与操作

| 项 | 含义 |
|---|---|
| 选择运行 | 当前族名称+r-ID，下拉包含各状态；选择后读详情与 trace，不重新运行 |
| 大标题 | `detail.spec.name` |
| 状态标签 | 第7节生命周期状态 |
| 重跑 | 在当前源码 content hash 与原运行一致时创建新 batch/run；不是覆盖旧结果。若 hash 改变会拒绝；原spec中的seed列表仍会展开，不保证只创建1 run |
| 复制配置 | 把旧 resolved spec 载入当前编辑器；不自动提交，可以在新代码版本下建立新实验；它不是严格复现 |
| 场景指标 | `run-metrics/v1`完整 JSON，多场景为 rows；不是自动求均值的榜单 |
| 时间序列 · 指派时利润 | 当前 trace 场景的 `after.profit`累计曲线；x为step.period，y为累计利润 | 
| 图上悬停提示 | period/profit 等图值；图的平滑连接只用于网页展示，不是新求解期 |
| 没有适用的利润轨迹 | 没选 run、trace关闭、Oracle 无trace、未生成结果等 | 
| 事件与产物 | 文件下载链接 + 最近最多12条事件JSON，不一定完整日志 |
| 精确配置与数据引用 | 展开显示 resolved spec，定义见第6.3节 |

详情利润图没有独立场景选择器：使用与“回放”共享的当前场景索引，通常初始第一个。要多场景/多算法均值与 CI，使用实验分析；不要将此一条曲线当总体均值。

### 8.2 metrics.json 顶层、每行

| 字段 | 解释 |
|---|---|
| `schema_version` | run-metrics/v1 |
| `elapsed_seconds` | 本 runner 场景及拟合执行耗时；不含完整排队时间，不是每期 solve runtime之和 |
| `rows` | 每场景一行 |
| 行.`scenario_id` | 冻结场景 ID |
| `scenario_seed` | 场景生成 seed |
| `policy_seed` | 评估运行算法 seed；拟合行为 null |
| `training_seed` | 拟合 seed，或 learned 评估所用checkpoint的训练seed；普通非学习评估null |
| `family` | single |
| `framework_id` | 规范框架ID，历史文件可能未完整填充；看组件核对 |
| `metrics` | 该场景指标对象，见下一表 |

### 8.3 single 运营结果每个字段

| 字段 | 单位/方向 | 实际口径 |
|---|---|---|
| `operating_profit` | currency，越大越好 | 各期plan.operating_profit累加，不含V/tie-break |
| `business_cost` | currency，越小越好（同数据口径） | 所有期accounting.business_cost合计 |
| `business_cost_reason` | 文本 | 明确指出指派时计划成本及周期罚项，不是实测支出 |
| `augmented_solver_objective` | currency，诊断项 | 各期实际求解目标累加；不同V尺度不可直接排名，非经营利润 |
| `arrivals` | orders | 全冻结场景订单数；不是乘客数 |
| `assigned` | orders | 已指派订单总数，收入在此发生，不表示已上车或送达 |
| `delivered` | orders | 到达目的地并卸载完成总数，受终点策略影响 |
| `cancelled` | orders | 未指派且到不可再继续等待的订单移出数量；不指整run的CANCELLED状态 |
| `pending` | orders | pending_unassigned+pending_committed |
| `pending_unassigned` | orders | horizon截止未指派池大小；drain后仍保留截止数量 |
| `pending_committed` | orders | 最终仍在车上承诺集合中的未完成数，包括还没上车订单 |
| `assigned_order_rate` | fraction | assigned/arrivals；无订单时null，不是0 |
| `delivered_order_rate` | fraction | delivered/arrivals；无订单时null |
| `assigned_load_rate` | fraction | 已指派订单passenger之和/全部passenger之和 |
| `delivered_load_rate` | fraction | 已指派且不再 outstanding 的订单load/全部load；不同于订单条数比例 |
| `cross` | orders | 空闲车选择其他pickup hub后指派的订单计数；不是车辆跨城次数或道路边数 |
| `downstream` | orders | 空闲车自身hub线路，在下游pickup的订单指派计数 |
| `enroute` | orders | 已在trip状态的车新增指派订单计数 |
| `extended` | events | trip路线新增尾段/延长的事件数，不是新增订单数 |
| `delayed` | events | 新启动线路选择delay>0的事件数，不是总delay期数，不是迟到送达数 |
| `repos` | events | 显式空车重定位动作次数，不是cross-hub空驶次数 |
| `planned_loaded_distance` | 距离单位 | 新载客线路/延长尾段的计划距离；不是逐边实测里程 |
| `planned_deadhead_distance` | 距离单位 | cross-hub空驶及reposition计划距离 |
| `planned_departure_delay` | 期 | 新计划线路delay的合计，区别于delayed事件数 |
| `actual_distance` | null | 当前没有重建最终在途实际里程，不能填0代替 |
| `actual_distance_reason` | 文本 | 未计算实际距离的原因 |

无订单时rate=null可能导致该指标无法绘图，分析会明确拒绝，不会默认填0。订单比例和乘客载荷比例不应写成同一个“服务率”。

### 8.4 Oracle 专属指标

| 字段 | 含义 |
|---|---|
| `upper_bound` | LP最优值或MIP求解器的最大化上界；MIP未收敛时不等于当前incumbent |
| `bound_incumbent` | pooled松弛模型当前最好可行值；不是原车辆政策可行profit。LP最优时与upper_bound相同 |
| `bound_direction` | maximize-upper |
| `bound_scope` | full horizon pooled time-expanded network，放松车辆不可分指派/backbone/锚点等限制 |
| `operating_profit` | null；Oracle不是执行policy，不伪造运营利润 |
| `operating_profit_reason` | a relaxation is not an executed policy |
| `status` | OPTIMAL/FEASIBLE_LIMIT/NO_SOLUTION_LIMIT 等底层求解状态，不是run状态 |
| `runtime` | 该bound solve秒数 |

Oracle `trace`为空。LP需要最优才返回；MIP可在限时有上界或无incumbent情况下返回已知信息，字段可能null。不要把没有incumbent的COMPLETED当原问题成功服务，或把pooled MIP称为完整在线MILP的精确先知最优策略。

### 8.5 事件每个字段

| 字段/事件type | 解释 |
|---|---|
| `sequence` | 同run递增事件序号，从1；不是时间戳 |
| `type` | started/scenario_started/scenario_completed/training/completed/error |
| `started.policy_seed` | worker执行seed；拟合时执行器也传训练seed，但manifest区分其角色 |
| `scenario_id` | 场景事件所属ID |
| `progress` | 0–1，主要场景完成比例 |
| `scenario_completed.metrics` | 该场景完整指标快照，即使整run后续失败也可能在事件中保留 |
| `training.epoch` | 从0开始的拟合迭代索引 |
| `training.value_mse` | 监督标签均方误差；当前每次更新使用的误差计算点，不是held-out评估 |
| `training.method` | supervised fluid-dual regression; not reinforcement learning |
| `error` | code/message/path，有时details；见诊断 |

当前事件文件没有统一墙钟timestamp。不能从sequence推断真实秒级时间；用runtime、elapsed_seconds、日志及manifest证据分别解释。

### 8.6 产物链接每个名称

| 文件 | 内容/何时存在 |
|---|---|
| `spec.submitted.json` | 原始提交配置，可能整份batch包含其他variants |
| `spec.resolved.json` | 此run最终生效的单次配置，已补默认 |
| `dataset.refs.json` | 冻结数据引用 |
| `dataset.snapshot.json` | 提交时完整冻结payload快照 |
| `manifest.json` | run/batch/条件/seed/源码hash/限制信息，字段见第16节 |
| `environment.json` | Python/OS/依赖版本/CPU；同诊断environment结构 |
| `plugins.lock.json` | 本族组件及参数Schema、capabilities快照，字段见第16节 |
| `code.snapshot.zip` | 提交时源文件快照，含受Git跟踪/未忽略的文件；源码不可用时能力受限 |
| `state.json` | status及可能error/forced等；不是车辆状态 |
| `stdout.log` / `stderr.log` | worker标准输出/异常日志；空stderr不代替数值正确性检查 |
| `events.jsonl` | 完整逐行事件，比页面最后12条更多 |
| `metrics.json` | 完整场景数值与null原因 |
| `metric-definitions.json` | 指标单位、方向、适用族、记账口径 |
| `trace.json` | 场景→steps的逐t结构；关闭trace或Oracle时无适用steps |
| `charts.json` | 声明式图表说明，data_ref/encodings/units等；不是PDF论文图 |
| `weights.json` | 仅train_value产物；feature_schema、feature_names、weights |
| `checkpoint.json` | 仅拟合checkpoint描述；hash、训练数据、参数及方法 |
| `reproduction.zip` | 手动导出复现包后出现 |
| `diagnostic.zip` | 生成脱敏诊断后出现 |

每次运行只显示实际存在文件，未完成时不保证全表存在。浏览器文件链接可能打开新标签或下载；它们不自动运行其中代码。

<a id="s09"></a>
## 9. 实验比较：两个运行的 A−B

这是简洁的两运行检查页；多算法、多seed、多指标、论文PDF和严格物理instance去重应使用第13节“实验分析”。

| 控件/显示项 | 说明 |
|---|---|
| 同场景配对比较说明 | 要求相同源码hash、冻结修订、模型物理参数、信息集、终点、记账及warmup；不能拿所有旧run任意拼 |
| 每个复选框 | 名称·run_id·状态；此页会列各状态，但后端拒绝不完整运行 |
| 比较 A − B | 必须恰选2个才可点击；single 网页固定比较 operating_profit，没有指标下拉；A/B按选择ID顺序，先勾为A |
| 差值柱图 | x=scenario_id，y=difference=A−B；是逐场景差，不是原始利润或百分比 |
| 操作结果 | 比较结果JSON；下面逐项解释 |

| 返回字段 | 说明 |
|---|---|
| `metric` | operating_profit（网页single固定）；API可选适用运营指标 |
| `direction` | A-B，不是自动排序后“最好−最差” |
| `metric_sense` | maximize或minimize；profit最大化，cost最小化 |
| `pairs[]` | 每场景paired条目 |
| pair.`scenario_id` | 两run共同场景ID |
| `a` / `b` | 两run该场景指标；组内多行先均值 |
| `difference` | a-b；profit为正有利A，cost为负有利A |
| `n` | 配对场景数；旧比较页按scenario_id配，不做新版分析的物理内容改名去重 |
| `mean_difference` | 配对差的平均，而不是平均percent |
| `confidence_interval_95` | 配对差均值95% Student-t区间，n<2为null |
| `ci_reason` | 未算CI的原因；有CI通常null |
| `percent_improvement` | 利润：(meanA−meanB)/meanB×100；成本反号；只在B平均基线>0时计算 |
| `percent_reason` | B平均≤0时null百分比的解释；禁止把负基线分母当正常提升率 |
| `failed_runs` | 成功比较中为0，不是“失败run已被自动剔除” |
| `method` | scenario paired Student-t; policy seeds averaged within scenario |

该页不是训练统计保护的全部实现：不要手工勾拟合任务当评估。新版“实验分析”明确禁用拟合项，建议论文统一用新版。

源码内容hash不同会拒绝，即使只是修改文档导致hash变化也可能发生；这不是算法性能自动变化的证据。对新版本请复制配置并成套重跑，不要改历史manifest来绕过检查。

<a id="s10"></a>
## 10. 回放：逐 t、订单、车辆、决策与记账

### 10.1 页面每个控件与图上元素

| 控件/文字 | 解释 |
|---|---|
| 选择运行 | 同详情；需有适用trace |
| 场景下拉 | trace中的scenario_id；一个run的多个场景分别回放，切换后期数置0 |
| 期数滑块与数值 | 0至当前steps最大period；选择的是已记录的离散期；drain可能超过horizon−1 |
| 实际已提交状态 | step.committed=true时显示，网络用step.after |
| 求解计划 · 尚未提交 | 未commit的计划标识；正常成功single trace为committed=true，debug输出则false |
| 抽象网络与车辆位置 | 圆环布局的节点和边，不是地图，不是道路动画 |
| 节点圆圈文字 | nodes中的hub ID |
| 边中间数字 | frozen edge distance权重，不是速度、拥堵或当前载荷 |
| 节点下“X辆” | after.vehicles中hub==此节点的车辆数，不仅是idle；hub是最近到达/当前位置记录，在途车辆不显示实时道路坐标 |
| 决策与资源事件 | 当前完整step JSON，可查下方字典 |
| 无轨迹提示 | trace关闭、Oracle无operationaltrace、未完成写文件等；不是自动生成替代轨迹 |

这里没有播放/暂停按钮或连续动画；是按期滑块检查。网页节点计数不可替代`idle_vehicles`，地图上的最近hub不可解读为在途车已到计划终点。

### 10.2 一期的先后顺序

```text
进入t
→ 处理该时刻到hub的车辆、卸载/上车、订单新预订入池
→ before（仍为t）与在线observation
→ solve_plan（只求解）
→ plan（候选赢家决策、会计分解、状态hash）
→ 检查状态未变，commit_plan
→ after（clock变为t+1）+ measurements + invariants
```

`before` 已包含本期新预订和期初到达；不要把它当“上一期结束、还没入单”。`after.period=t+1` 是下一决策clock，不表示所有在途车瞬间移动到下一hub。下一个begin_period再处理到达。

### 10.3 trace.json 与 step 顶层字段

| 路径 | 解释 |
|---|---|
| trace根 | 场景数组；空数组代表没有保存适用trace |
| 场景.`scenario_id` | 场景ID |
| `network.nodes/edges` | 本场景冻结网络，与第4节一致 |
| `steps` | 每个已记录期step列表 |
| step.`schema_version` | single-trace-step/v2 |
| `period` | 求解/提交发生的期t |
| `before` | 期初处理之后、计划求解之前的仿真状态 |
| `observation` | 对策略的在线可见字段；batch的start_time截止前改为bound |
| `arrivals` | 本期新预订订单数组，不是已到目的地的车辆/乘客 |
| `plan` | decision-plan/v1已选计划，见第10.5节 |
| `after` | 提交计划后状态 |
| `measurements` | 为分析准备的当期/累计指标，见第10.7节 |
| `invariants` | 本期内部一致性检查，不是全部论文定理证明 |
| `committed` | true表示实际已向该仿真状态提交一次 |
| `drain` | 仅horizon后drain额外期出现true；不是新需求时段 |

### 10.4 before/after/observation 状态每个字段

| 字段 | 解释 |
|---|---|
| state.`period` | 当前clock；before=t，after=t+1 |
| `profit` | 累计指派记账经营利润sim.J |
| `pool` | 尚未指派订单数组，字段同第4.3节；已指派未送达不再在pool |
| `counts` | assigned/cross/downstream/enroute/extended/delayed/repos/cancelled/delivered累计计数，同第8.3节 |
| `vehicles` | 车辆状态数组，每辆字段如下 |
| vehicle.`id` | 从初始hubs位置序列得到的车辆ID，保存为字符串 |
| `hub` | 最近所在hub；在途时不是GPS坐标或计划终点 |
| `phase` | idle空闲、trip已承诺线路、repos显式重定位；trip也可能是等待延迟出发或跨hub空驶，并非一定已有乘客在车 |
| `route` | 尚待经过的hub序列，不含当前hub；不是所有历史行驶节点 |
| `dead` | cross-hub接载前剩余空驶hub段数量；不是车辆死亡标记。repos也可为0，判断重定位看phase |
| `arrive` | 下一hub到达期，null表示当前没有下一到达事件；不是整条路线结束时间 |
| `hold` | 等到哪个期才从当前hub发车，null为无此延迟；不是已等待时长 |
| `onboard` | 真正已上车订单ID集合；不能直接当乘客数，需查passenger |
| `assigned_orders` | 该车已承诺而未完成全部订单ID，含未来pickup；送达后移除 |
| observation.`period` | t |
| observation.`orders` | before.pool的在线版本；不列未预订未来订单 |
| observation.orders.`ready_time_information` | upper-bound：t<cutoff；revealed：普通订单或batch已截止 |
| observation.`vehicles` | before车辆状态，同上 |

`onboard⊆assigned_orders`，两者不能混用。表面上phase=trip而onboard=[]可能正在空驶、等接驳、延迟出发或刚完成一个载客段，不一定bug。

回放滑块按整数期选择。若记录不连续，界面当前实现可能退回第一条step而不是显示“缺期”；完整导出应核对steps.period，不要只凭滑块标签推断记录齐全。正常成功运营trace应逐期连续。

### 10.5 plan 每个字段与线路字段

| 字段 | 解释 |
|---|---|
| `schema_version` | decision-plan/v1 |
| `period` | 与step t一致 |
| `state_hash` | before规范序列化的hash；提交前核验没有状态改变 |
| `operating_profit` | 本期利润增量，不含V/tie-break |
| `augmented_solver_objective` | 本期MILP objective，包括V并扣tie-break |
| `assignments` | order_id→vehicle_id的新增指派列表，不含此前已承诺订单重复行 |
| `assignments[].order_id/vehicle_id` | 字符串资源ID；具体订单属性查数据/arrivals/pool |
| `routes` | 每个选中车辆线路描述；不列全部候选线路 |
| route.`vehicle_id` | 所属车 |
| `p` | 主干载客线路起点hub；可能不同于车辆当前hub |
| `e` | 所选线路结束hub，不是当前到达位置 |
| `B` | 起点p→终点e一致最短路径节点序列 |
| `delay` | 空闲车在原hub延迟出发的期数；旧trip线路通常0 |
| `t0` | 主干线路从p出发的时刻，可能包括原hubdelay和到p空驶 |
| `cost` | 该候选新增线路的计划成本；旧线路保留时可能0，不代表整车历史成本为0 |
| `new` | true为新启动或尾段扩展；false保留既有线路；不是“车辆第一次被使用” |
| `repositions` | 显式空车重定位列表，不含cross-hub路线前空驶 |
| reposition.`vehicle_id/hub` | 移动车与目标hub |
| `expired` | 本期判定若继续未指派会无法满足deadline的集合；列表内订单仍可能被本期救回指派，因此**不等于本期最终cancelled集合** |
| `solver` | 当前赢家MILP求解信息；rollout额外有candidate_scores；见下一节 |
| `accounting` | solver.accounting的同一分解，方便直接读取 |

routes中的内部`serve`候选订单对象未序列化；实际服务判断用assignments，不能把B上所有订单都算已指派。

### 10.6 solver 与 accounting 全部字段

| solver字段 | 含义 |
|---|---|
| `status` | OPTIMAL、FEASIBLE_LIMIT、INFEASIBLE、UNBOUNDED、NO_SOLUTION_LIMIT等；未找到incumbent的运营步骤不会正常commit |
| `runtime` | 此次Gurobi solve秒数；不含全部rollout未来或fluid建模时间 |
| `incumbent` | 当前最好可行的augmented目标值，不是纯profit |
| `bound` | 同一个MILP最大化目标的求解器上界；不是单独Oracle upper_bound，也不是fluid V |
| `gap` | 此MILP相对gap，可能null；0或容差内为求解证明信息 |
| `accounting` | 本期经营收支分解，下面逐项 |
| `controller`（rollout） | rollout |
| `cross_hub`（rollout） | 本期实际获选的首期cross-hub候选值，不一定等于base模型参数 |
| `candidate_scores`（rollout） | 所有候选的cross_hub和平均全未来累计profit score；不是候选首期profit，也不是独立实验样本 |
| candidate.`cross_hub/score` | 候选布尔与K条未来平均clone.J |

| accounting字段 | 含义 |
|---|---|
| `assignment_revenue` | 本期新增指派订单收入之和 |
| `planned_route_cost` | 选中线路计划cost之和，含延迟出发等待费/接载空驶/主干新增段等 |
| `planned_reposition_cost` | 显式重定位计划空驶成本 |
| `idle_waiting_cost` | idle车选择w等待本期的waiting_cost之和 |
| `backlog_penalty` | 本期池中未指派订单各自penalty之和；可每期重复发生 |
| `expiry_penalty` | expired且最终未指派订单数×expiry_cost，另加于backlog罚项 |
| `business_cost` | 除assignment_revenue之外上述成本罚项的和 |
| `profit_identity_residual` | assignment_revenue−business_cost−plan.operating_profit，应接近0 |

不要把planned_route_cost再拆出来额外加一遍loaded_cost/empty_cost，否则重复计成本。不要把V当收入，或把tie_break当实际等待财务成本。

### 10.7 measurements 全部字段

| 字段 | 单位/口径 | 网页论文时间序列可选？ |
|---|---|---|
| `period_profit` | 本期plan.operating_profit | 是 |
| `cumulative_profit` | after.profit | 是 |
| `period_business_cost` | 本期accounting.business_cost | 是 |
| `cumulative_business_cost` | 记录至本期的cost累计 | 是 |
| `period_objective` | 本期augmented目标，诊断用途 | 是 |
| `cumulative_objective` | 至此augmented目标累计，诊断用途 | 是 |
| `arrivals` | 本期新预订条数 | 否，原始导出中有 |
| `pool_before` | 本期入单后、指派前pool条数 | 是 |
| `pool_after` | 提交后尚未指派pool条数 | 是 |
| `assigned_delta` | after累计assigned−上次step.after累计assigned | 是 |
| `delivered_delta` | after累计delivered−上次step.after累计delivered，含本期begin_period卸载 | 是 |
| `cancelled_delta` | 本期过期移出数 | 否，原始导出中有 |
| `assigned_load` | 本期新assignments乘客载荷之和 | 是 |
| `delivered_load` | 上期未完assigned_orders集合减本期未完集合的乘客载荷和 | 是 |
| `idle_vehicles` | after.phase=idle的数量 | 是 |
| `trip_vehicles` | after.phase=trip的数量，含delay/接载空驶 | 是 |
| `reposition_vehicles` | after.phase=repos的数量 | 是 |
| `onboard_load` | after各车onboard订单passenger之和 | 是 |

当前每条“时间序列指标”是快照或本期增量，不是二者通用。`assigned_load`不是累计已服务load，`onboard_load`不是总capacity，`pool_after`不含已指派未送达。

### 10.8 invariants 全部字段

| 检查 | 为true意味着什么 |
|---|---|
| `state_hash_matches` | plan基于记录before |
| `clock_advanced_once` | 提交后clock恰t+1 |
| `unique_assignment` | 本期assignments同order只出现一次 |
| `capacity_valid` | after每车当前onboard载荷≤capacity；完整路线弧容量还依赖MILP约束，不由此字段单独证明 |
| `profit_reconciles` | 当期profit_identity_residual绝对值<1e-5 |

任一false会触发TRACE_INVARIANT失败。全部true不等于证明所有未来操作、需求分布近似或统计独立性正确；这些属于不同检验层。

<a id="s11"></a>
## 11. 复现：导出与检查 ZIP

| 控件 | 实际行为 |
|---|---|
| “复现不是一个hash”说明 | 保存精确数据、配置、环境与源码证据，而非只有摘要值 |
| 选择运行 | 指定待打包run，未选时导出禁用 |
| 导出复现包 | 创建/更新该run下reproduction.zip，不执行模型 |
| 下载复现包 | 下载已生成ZIP；未先导出可能NOT_FOUND |
| 检查复现包 / 选择.zip | 上传到本机后端做结构/校验和/数据Schema检查，返回检查JSON，不自动执行源码、权重或求解器 |

| 导出操作结果 | 说明 |
|---|---|
| `run_id` | 所选run |
| `artifact_id` | reproduction.zip |
| `data_included` | true，包含精确数据，**不是脱敏包** |

| 检查操作结果字段 | 说明 |
|---|---|
| `valid` | ZIP结构、协议、文件集合、hash及数据合法 |
| `data_included` | 是否包含数据 |
| `files` | 被checksum清单校验的文件数，不是场景或样本数 |
| `spec` | 包内单次配置，定义见第6节 |
| `source_code` | code.git_sha/dirty/content_hash/lock_hashes/snapshot等，见第16节 |
| `code_execution` | 明确not executed，环境和源码需核对 |

网页检查返回不直接显示整个dataset（后端内部核验它）。`valid=true`不表示当前源码已匹配、不表示许可证有效、不表示已实际重跑结果一致。

ZIP通常包含：dataset.json、spec.json、run/manifest.json、environment、plugins.lock、code.snapshot.zip、metrics/trace/events（若存在）、锁文件、schemas.json、checkpoint（若此评估使用）。checksums.json记录每文件size/sha256及dataset_hash。

网页目前没有“导入包后一键复现执行”按钮。“重跑”在运行详情且要求当前源码内容hash匹配；命令行reproduce可以校验/建立计划，但不自动执行包内代码。**修改本文、README等受跟踪文件也会改变当前content_hash**；以后旧run重跑被拒绝时使用旧源码快照对应版本，或复制配置开始新版本实验，不要篡改记录。

安全限制：拒绝绝对路径、`..`、反斜杠/冒号穿越及重复ZIP路径、未知/缺失文件、校验和不符；ZIP自身≤100MiB、解压总量≤200MiB、文件≤10,000。不过**网页请求总上限15MiB会先限制上传**，因此较大合法包也可能无法通过网页检查。源码和依赖的恢复需人工审查，不自动解pickle。

分享前检查包内数据、日志和源码。复现包可包含真实订单信息，不能因为在“本地平台”就默认可公开。

<a id="s12"></a>
## 12. 数学模型：每个符号与当前 V

数学模型页与编辑器MathPanel是同一组件，读取**当前编辑器表单**，不是所选run详情，也不是手改JSON大框。要看某run公式，先在详情“复制配置”再查看；不要误以为选择run会自动换数学spec。

| 区域 | 说明 |
|---|---|
| 标题 | Myopic single-level dispatch / Analytical / fluid dual value / Sampled look-ahead rollout / Optional supervised hub-time value / pooled LP/MIP bound，随框架切换 |
| family / model / framework 行 | 当前族、模型插件和框架标签 |
| 公式 | 实现结构摘要，不是完整论文定义、全部变量域与每一条约束 |
| 英文研究说明条目 | 容量每弧、锚点、承诺保留、指派记账、框架限制等 |
| 当前数学参数与组件 | 展开model/controller/value_function/evaluation JSON；字段同第5、6节 |

### 12.1 在线单层目标与约束符号

目标结构：

```text
maximize Π_t(x,y,z,w)
       + Σ_(v,r) V(h_r,t_r)y_vr
       + Σ_(v,h) V(h,t+τ_vh)z_vh
       + Σ_v V(h_v,t+1)w_v
       − ε Σ_(v,r) d_r y_vr
```

| 符号 | 含义 |
|---|---|
| t / v / o / r / h / e | 决策期、车辆、订单、候选线路、hub、线路弧（容量公式中e是弧；线路JSON e是终点，两处上下文不同） |
| `x_ov` | 是否将订单o指派给车辆v |
| `y_vr` | 是否选车辆v候选线路r |
| `z_vh` | 是否显式空车重定位到h |
| `w_v` | idle车辆本期等待 |
| `s_o` | 订单本期未指派，不是scenario ID或状态 |
| `Π_t` | 本期经营利润表达式：收入−计划线路/重定位成本−车等待−backlog−expiry |
| `V(h,t)` | 车辆在hub/time恢复可用的价值估计；公式以(h,t)写，代码callback可能按(when,hub)传参，不是两个不同V |
| `h_r,t_r` | 所选线路结束hub与可用时刻 |
| `τ_vh` | 车辆当前hub至h的网络行驶期数 |
| `h_v` | 车辆当前hub |
| `ε` | tie_break参数 |
| `d_r` | **此目标项中为线路出发delay，不是距离**；实现扣tie_break×delay |
| `q_o` | order.passenger载荷 |
| `Q_v` | 车辆容量；当前同族统一capacity参数 |
| `L_ve` | 此车此弧此前承诺载荷 |
| `P_o` | 订单起终点一致最短路径 |
| `a_o^visible(t)` | 当前可见准备时间；batch截止前为bound |
| `b_o` | end_time截止 |

`Σ_v x_ov+s_o=1`：本期订单要么唯一指派、要么保留/过期；`x≤Σy`：指派必须由选中线路覆盖；容量逐弧≤剩余座位；pickup≥可见准备时间，destination≤截止。网页未列全所有线路选择、锚点与extension约束，不能直接用这几行替代论文完整模型。

### 12.2 不同框架的 V 和比较对象

| 当前框架 | V怎么计算 | 目标应如何阅读 |
|---|---|---|
| Myopic | 精确为0 | 仍可能扣tie-break，所以objective未必等于profit |
| fluid | LP车辆流守恒约束对偶π_(h,t)×multiplier | V是一辆边际可用车的启发影子价值，不是LP总收益 |
| rollout | base V=0，外层K条抽样未来profit平均选择首期规则 | 回放objective是首期赢家MILP目标；candidate.score是全前瞻累计值，两者不同 |
| learned评估 | multiplier×wᵀφ | 7项线性监督预测；依赖checkpoint和状态features |
| learned拟合 | reference V=0用于模拟，fluid π做监督标签 | 展示learned数学profile只是研究方法摘要，不表示训练时已经使用最终预测V控制 |
| Oracle | zero value，整体离线pooled bound | upper_bound，不生成逐期执行profit |

fluid式中的 `λ^fluid_(h,t)` 表示对偶值；不是数据生成的demand_rate。learned式中的 `λ_reg` 是正则权重，不是Oracle上界或arrival rate；网页未暴露调节此正则的输入。rollout式的 `A_rules` 是当前有限候选规则集合，不是所有可能组合动作。

Oracle LP/MIP inequality是模型关系摘要，在保持相同数据、参数、记账/时域并满足松弛推导条件时理解。MIP限时incumbent、solverbound、LP最优不能混用；UI公式不替代逐场景数值核验。

<a id="s13"></a>
## 13. 实验分析：论文图、样本与统计

### 13.1 “实验选择 · 独立场景样本”每个字段

| 控件/显示项 | 初值/作用 | 注意 |
|---|---|---|
| 分析数据筛选 | 所有数据；按dataset_id筛可见运行 | **不是按hash过滤**；同ID的多修订仍可能在多个panel |
| 分析框架筛选 | 所有框架；使用framework_id，历史缺失时退controller.id | 旧fluid run可能显示myopic，核对value参数 |
| 选中筛选结果 | 用当前筛选的COMPLETED非train_运行替换已选集合 | 不是只“新增可见项”；Oracle也可选，但指标是否可用还需后端检查 |
| 清空 | 清空所有selected run IDs | 不删除分析或run |
| 每项复选框 | 勾COMPLETED且非拟合run | 拟合/失败/运行中可显示但禁用；禁用不等于失败已纳入统计 |
| 实验名称 · framework | 研究条件标识 | 名称相同不代表参数、代码或数据相同 |
| run_id / dataset_id / status | 具体资源、数据容器与状态 | 精确hash在runs.json/manifest中 |
| X场景 | metrics.rows.length | 此run含场景数，不是所有选中后独立n |
| policy seed / `—` | run.policy_seed，无值显示破折号 | `—`不是0，训练种子不能当policyseed |
| 参数与实验条件 | 展开model/controller/value参数对象 | UI未在此同时展示全部solver/evaluation，完整条件查runs.json |
| 已选 X 个运行 | selected.length | 是run数，不是样本n；改变筛选不会自动取消已隐藏的勾选，必要时清空再选 |

前端可见列表包含失败项但不能选。正式论文仍需报告实验设计的失败数/比例和原因；统计不会替你自动纳入“原计划但未选”的失败任务。后端收到不完整run会拒绝，而不是静默删除。

### 13.2 “论文绘图配置 · 本地 Python”所有字段

| 字段 | 初始值/选项 | 实际作用 |
|---|---|---|
| 比较方式 | comparison，下面5种 | 决定样本组织及图形，不改变模型/原结果 |
| 指标（可多选，最多6个panel指标） | operating_profit；切timeseries后重置cumulative_profit | 原生多选框，用Ctrl点选/Shift扩选；1–6个不同指标，0项非法 |
| 横轴参数 | 只在sensitivity/heatmap出现；初始model.parameters.empty_cost | 下拉收集所有已完成运行中model/controller/value的数值参数，不只是当前勾选；每个所选run都必须有该数字 |
| 纵轴参数 | 只在heatmap出现；初始model.parameters.loaded_cost | 必须与横轴不同，也必须各条件都存在 |
| 论文尺寸 | double：双栏7 inch；single：单栏3.35 inch | 是整张figure宽度；1 inch=2.54cm，约17.78/8.51cm；高度随panel行数变化 |
| Panel列数 | 2；整数1–3 | 行数自动ceil(panel数/实际列数)，实际列数不超panel数；不是每列算法数 |
| 95% pointwise Student-t CI提示 | 网页固定confidence=.95 | 没有confidence输入框；不是bootstrap或同时置信带 |
| PDF/SVG/600 dpi PNG提示 | 网页固定3格式、dpi600 | 没有网页dpi输入框；图由Python生成，不是网页截图 |
| 生成论文图并保存到本地 | 有选run且不busy时可点击 | 建统计、绘图、写分析目录/ZIP；不是后台队列实验，不需要重新求解模型 |
| 导出多实验完整数据 | 同上 | 只导出原始结果合同，不用图形指标/轴配置；不会生成figure、plotting.py和统计表 |
| 错误条 | 统计或渲染错误 | 当前选择不兼容会明确报错，不填0/插补/混口径 |

轴下拉不是万能的数据字段选择器：场景seed、horizon、数据集名称、布尔cross_hub、solver threads不在当前网页数值参数轴列表。想做这些研究先设计数据和分组，必要时用Python离线分析；不要把seed当连续研究变量。

### 13.3 五种比较方式逐项解释

| 名称 / kind | 图与统计单位 | 选择条件 |
|---|---|---|
| 多算法 / 多场景均值、散点与CI / comparison | 每条件独立场景散点、均值marker、t区间 | 同数据物理/源码口径的不同框架可同panel；不同参数条件保持分组，不自动平均 |
| 单参数敏感性与置信带 / sensitivity | 参数x→场景均值线、逐点95%带 | 只有所选轴维度可从物理分层中排除；其余条件保持不变，否则额外series/panel |
| 双参数热力图 / heatmap | x,y格子为样本均值，每算法单独panel，颜色条+数值 | 要两不同数字参数；未实验格空白，不插值；各格CI在统计表，不在图上单独显示 |
| 逐t动态曲线 / timeseries | t→每期指标场景均值与逐点CI | 必须save_trace=true且有对应measurements；不是同run单场景回放线 |
| 同场景配对差值 / paired | 完全相同物理场景下A−B散点、均值及CI，含0参考线 | 每一物理/数据分层必须恰2个条件、相同instance集合；只允许maximize/minimize运营指标 |

第一版没有response surface/3D、箱线图、自动统计检验p值、显著性星号或任意自定义画法的网页选项。底层结果合同允许以后加renderer/分析，而不是当前这些功能已经实现。

### 13.4 非时间序列指标菜单的9项

| 指标 | 选择用途 | 哪些情况不适用 |
|---|---|---|
| operating_profit | 核心利润比较，越大越好 | Oracle为null、拟合不当评估 |
| business_cost | 指派计划成本及罚项比较，越小越好 | Oracle无此运营成本；单看成本低可能只是少服务，应与profit/served一起看 |
| augmented_solver_objective | 求解诊断 | 不当运营收益排名，paired拒绝该诊断项 |
| assigned | 指派订单条数 | 不等于送达条数/乘客数 |
| delivered | 截止/续完口径送达数 | 终点不同不得直接混比 |
| pending | 未指派+已承诺未完 | 不含已cancelled |
| assigned_order_rate | assigned/arrivals，0–1 | 无订单null不可数值绘图 |
| delivered_order_rate | delivered/arrivals，0–1 | 同上 |
| upper_bound | Oracle松弛上界 | 在线policy无此字段；不能与profit放同一指标数组自动补值 |

菜单列出族定义并不表示所选每个framework都有全部9项。Oracle与在线框架应分开选适用指标、分图报告；要自行画“利润与上界差距”，用完整导出在Python按场景明确匹配计算。

时间序列菜单为第10.7节标“是”的16项；切模式会重置指标，重新检查不要沿用旧选择。

### 13.5 样本与分层：系统实际怎样算

1. 从完整COMPLETED评估run取场景指标及trace。
2. 用冻结场景物理内容算instance_key，排除id/split/seed元数据；改名复制同一事件不增加n。
3. 同condition+instance内部多policy seed/重复run数值平均，再将不同instance作为场景样本。
4. 按数据hash、model/evaluation/solver/source内容分context；指定敏感性轴时，仅移除对应model参数分层维度。controller/value轴不从physical里删，不混成物理改变。
5. 剩余framework及非轴参数保持series身份；同名称多条件附短hash后缀。
6. 每个条件/每个轴点/每期单独算mean、SD、CI，不把不同参数点混成“更多样本”。

同condition_id不保证同源码或数据，因为条件摘要不含两者；必须结合context、dataset_hash、source_hash。不同源码/求解设置/信息集/终点的数据默认分panel，即使运行名称完全相同。

不同训练checkpoint目前也是不同条件，不自动估计训练随机性方差。多训练seed实验不能仅把多模型权重当独立scenario样本；本版不做层级随机效应/方差分解。

### 13.6 CI、SD、散点与配对方向

设同条件n个独立场景指标为x_i：

```text
mean = Σ x_i / n
sd   = sqrt(Σ (x_i−mean)^2 / (n−1))
95% CI = mean ± t_(0.975,n−1) × sd / sqrt(n)
```

这是**均值**区间，不是所有个体95%落入的范围、不是prediction interval。n=1有mean/min/max、SD及CI为null并有原因；重复同场景policy seed不让n变大。点是场景层policy均值，不是每个原始seed都画一个独立点。

paired对每个instance计算d_i=A_i−B_i，再按相同公式估计差值均值CI。**新版分析的A/B由框架显示标签及条件key排序确定，不是网页勾选顺序**；务必读图图例和paired-differences.csv的condition_a/b确认符号。profit差>0有利A，cost差<0有利A。

CI不跨0只是这个t模型下的方向证据，不替代对实验设计、偏态/极端值、小n、独立性和多重比较的检查。敏感性/时间曲线的带是pointwise，不是整条曲线同时95%置信；缺失期不填补，末端仅存活/记录的场景可能导致n变化，需查每期statistics.n。

### 13.7 “本地输出”每项字段

| 可见项 | 说明 |
|---|---|
| 本地输出 · a-ID | 唯一analysis_id，普通绘图和raw-export都使用a前缀 |
| 绝对路径 | `local_path`，本机实际保存目录，不是浏览器下载目录 |
| figure.pdf / figure.svg / figure.png | 下载Python画的论文文件；PDF/SVG矢量，PNG高分辨率 |
| analysis.zip | 数据、配置、代码与图整体包；raw-export包没有图及绘图脚本 |
| 图片预览 | 本地figure.png的网页预览，不是另外一张dashboard截图 |
| 统计结果与口径 | 正常绘图显示statistics；raw-export显示config |
| 已保存分析 | 当前族历史manifest列表；按钮a-ID·kind，点击重新读取已保存输出，不重算模型 |

分析列表进入页面时和生成后刷新；不保证像运行列表一样每3秒刷新。切离再回来会重置未保存选项，但已生成本地目录仍在。

### 13.8 statistics 每个字段

| 字段 | 含义 |
|---|---|
| `context` | 物理/数据/源码分层hash，不是数据ID |
| `metric` | 指标名 |
| `condition_id` | 非paired条目的具体实验条件hash |
| `series` | 框架/自定义标签；图里同名series可能另附短hash，统计表还需靠condition_id识别 |
| `x` / `y` | 所选扫描参数数值；timeseries x为period；无参数轴常为0 |
| `n` | 该点独立物理instance个数 |
| `mean` | 样本均值 |
| `sd` | 样本标准差，ddof=1；n1null |
| `minimum` / `maximum` | 场景层数值极值，不是CI上下限 |
| `ci_low` / `ci_high` | 均值t区间端点 |
| `ci_reason` | n不足原因，正常null |
| `method` | scenario-level Student-t; policy seeds averaged within frozen instance |

paired-differences.csv额外含context/metric/instance_key/condition_a/condition_b/a/b/difference。它是每instance一行配对数值，不是statistics汇总；不含percent improvement，本版paired论文图不自动输出提升率或p值。

### 13.9 高级AnalysisSpec字段：网页未暴露，不要误找输入框

| API配置字段 | 默认/限制 | 说明 |
|---|---|---|
| schema_version | analysis-spec/v1 | 分析协议 |
| run_ids | 1–100不同合法ID | 所选评估runs |
| kind | comparison | 五种方式 |
| metrics | [operating_profit]，1–6不同指标 | 与网页对应 |
| x_parameter / y_parameter | null | 轴参数，敏感性必x、热力图x/y不同 |
| confidence | .95；0<数值<1 | API可调整，网页固定.95 |
| width / columns | double / 2；columns1–3 | 与网页对应 |
| dpi | 600；整数300–1200 | API可调整，网页固定600 |
| formats | [pdf,svg,png]非空 | 输出格式，网页固定三种 |
| renderer | matplotlib | 当前唯一实现，但统计合同与renderer解耦 |
| labels | {}；标签1–120字符 | framework ID→显示名；不是修改指标定义，网页无此编辑项 |

单分析≤24 panels、≤200,000逐期记录，超过需要过滤/拆组。图形宽度是整个figure，多指标×contexts×算法热力图可能迅速达到panel限制。

<a id="s14"></a>
## 14. 诊断：环境、错误与单步

### 14.1 “本地环境”全部字段

| 路径 | 说明 |
|---|---|
| `environment.python` | 当前后端Python完整版本/build信息；不是系统所有Python |
| `environment.os` | Windows等系统名 |
| `environment.architecture` | AMD64等机器架构 |
| `environment.dependencies` | 下列包版本或null；null是包不存在/未取得，非版本0 |
| `numpy` / `scipy` | 数值及统计依赖 |
| `torch` | legacy等使用；安装torch不表示single在做神经训练 |
| `fastapi` / `pydantic` | 本地API与参数契约依赖 |
| `gurobipy` | 优化器Python接口版本，不等于许可证状态 |
| `psutil` | 进程所有权和资源管理依赖 |
| `environment.device` | cpu；不是GPU占用率 |
| `gurobi.installed` | 检测到gurobipy版本 |
| `gurobi.license_verified` | 实际小模型试求解成功 |
| `gurobi.error` | 检查失败时出现异常说明，成功时可不存在 |
| `resume` | false，未支持断点续算 |
| `native_macos_verified` | false，未验证macOS原生启动 |
| `max_parallel` | 调度器最大并行run数，默认1；与solver.threads不同 |
| `cpu_budget` | 总Gurobi线程预算；配置未给时通常取本机逻辑CPU数 |

本机此次页面显示Windows、CPU、Gurobi已验证、max_parallel=1；cpu_budget是机器/配置相关值，不要把当前值硬编码成所有人的默认。

### 14.2 任务诊断与只求解单步

| 控件 | 实际行为 |
|---|---|
| 选择运行 | 只指定生成脱敏诊断的run；**不会把该run自动载为当前单步spec** |
| 生成脱敏诊断 | 写该run/diagnostic.zip，返回artifact_id等；再去详情文件列表下载 |
| 调试期数 | 共享period状态，初值0，界面min0；应填所选数据第一场景范围内整数0…horizon−1，不是steps数组位置 |
| 当前配置单步 | 用当前**编辑器表单spec**、其数据筛选后的**第一个场景**、第一个policy seed；在独立新仿真推进此前各期，再只解目标期但不提交该期 |
| 终端命令提示 | doctor/debug step的CLI说明，不是网页命令输入框；网页不能执行任意shell |

单步会调用求解器，耗时不一定只有“一期”：此前各期需要在临时仿真里求解/提交，rollout有内部多次求解。它不修改原run、冻结数据或生产车辆状态。返回before/plan/committed=false/network，字段定义见第10节，没有该期after/measurements。

该入口仅适合当前single在线匹配路径；不是Oracle bound求解的替代按钮，不进行完整批次计划级联合约束核验；合法框架先经正常校验。手改大框JSON不会进入单步；若想调某旧run先“复制配置”，再选择合适场景ID，不要只选诊断下拉。

### 14.3 diagnostic.zip字段与脱敏边界

| diagnostic.json字段 / 操作结果 | 解释 |
|---|---|
| run_id | 诊断对象 |
| state | 当前run state.json，含status/error/forced等 |
| environment | 当前环境，而非自动恢复历史环境 |
| errors | 完整events中type=error的列表，不含正常日志全部正文 |
| redacted | true，已执行平台脱敏规则 |
| raw_data_included | false，不包含原始数据 |
| artifact_id（返回值） | diagnostic.zip |

脱敏处理password/token/secret键、项目路径/home路径及部分文本凭据模式；**不保证识别所有个人信息、任意路径或嵌入数据**。分享前仍手动检查诊断包。不要发送workspace/server.json的token或owner.json内部凭据。

### 14.4 错误对象每个字段

| 字段 | 如何读 |
|---|---|
| error.`code` | 机器可定位原因，例如DATASET_SELECTION、CHECKPOINT_PARAMETERS |
| `message` | 具体自然语言错误 |
| `path` | JSON路径或参数点路径；空字符串表示未定位到单字段 |
| `details[]` | Schema校验可能包含多个错误 |
| details.`path` | 如/model/parameters/capacity，指向要修改字段 |
| details.`message` | 类型/范围/额外字段等说明 |
| details.`code` | Pydantic错误类型，不一定与顶层code相同 |
| state.`forced` | 若出现true，取消/超时过程中强制终止标志，不是求解强制最优 |

<a id="s15"></a>
## 15. 从第一次实验到论文图的操作示例

这些是使用步骤，本文核对过程没有替用户提交新的实验或改动既有结果。建议先做小规模功能核验，再设计正式实验；“用4个seed走通”不等于足够论文统计规模。

### 15.1 第一次Myopic评估

1. 双击BAT，打开平台，模型族选Single-level model。
2. 诊断确认Gurobi安装与许可证可用。
3. 数据管理：dataset_id=`single-first-test`；seeds=`[10001,10002,10003,10004]`；splits=`["test"]`；horizon=6、vehicles=5、cities=8、orders_per_step=2、first_mile=batch。
4. 数组填完离开框，生成冻结修订。生成后记录完整hash。
5. 编辑器：框架Myopic，确认value=zero；名称例如“single / myopic / first test”。默认物理参数先不变；终点report_pending、save_trace=true。
6. 不扫描时可直接使用单次JSON；先校验、预览，应该1 run、4 scenario IDs、policy seed0。点提交。
7. 队列等待COMPLETED，打开详情检查rows=4、solver.status/gap和trace invariants。
8. 回放选择每个场景，看pending/assigned/delivered，不能把有profit而未送完当成记账bug。

### 15.2 同场景比较Myopic、fluid、rollout

1. 保持同一数据hash、物理参数、求解设置、终点与代码版本。
2. 各选Myopic、Analytical/fluid、Sampled rollout运行；对fluid明确选择recompute=true或false，不能把两者混为同一配置。
3. rollout设置samples=2、candidate_cross_hub=[false,true]、expected_orders_per_step与研究假设一致。先policy seed0；若研究算法随机性，再按设计加policy seeds。
4. 三者完成后，实验分析只选这些同一版本条件，comparison，指标profit/business_cost/assigned_order_rate/delivered_order_rate，双栏、2列。
5. 生成论文图，读每panel、n=独立场景数、SD/CI，保存ZIP和配置。
6. 要看fluid−myopic的paired，改kind=paired，只选两个条件的runs，确认相同instance集合，读图例A/B顺序。

也可高级variants一次提交多框架，示例覆盖结构（**不是完整可提交配置，base_spec需填真实数据与完整基础spec**）：

```json
{
  "variants": [
    {"name": "myopic", "framework_id": "myopic", "controller": {"id": "myopic", "parameters": {}}, "value_function": {"id": "zero", "parameters": {}}},
    {"name": "fluid-R", "framework_id": "fluid", "controller": {"id": "myopic", "parameters": {}}, "value_function": {"id": "fluid_dual", "parameters": {"recompute": true, "multiplier": 1, "expected_orders_per_step": 2}}},
    {"name": "rollout-K2", "framework_id": "rollout", "controller": {"id": "rollout", "parameters": {"samples": 2, "candidate_cross_hub": [false, true], "expected_orders_per_step": 2}}, "value_function": {"id": "zero", "parameters": {}}}
  ]
}
```

base使用myopic+zero空参数，避免递归合并继承非法组件字段。这个例子不含Oracle，因为运营profit图不适用它。

### 15.3 单参数敏感性

1. 一个框架、相同冻结数据，其余参数固定。
2. 添加扫描路径`model.parameters.empty_cost`，linear start=5、stop=20、step=5，policy seed0。
3. 应用批次JSON→校验→预览：4 run，分别cost5/10/15/20，各运行同4个场景。
4. 全完成后分析kind=sensitivity、横轴empty_cost，选profit与cost等，查看每点n通常4，而不是把4参数×4场景说成n16。

要改变真实orders_per_step、horizon、车辆数等生成条件，应生成对应冻结数据并单独保留数据条件；当前分析不会把多个数据hash自动接成一条需求强度曲线。

### 15.4 双参数热力图

empty_cost `[5,10,15]` × loaded_cost `[5,10,15]`，同框架、同场景、同代码，预览9 runs。分析heatmap选这两轴；各格为场景均值，不是最优拟合曲面。格子没运行保持空白，要单元格CI/样本n查statistics.csv。

### 15.5 逐t资源曲线

只选运营trace的评估run；timeseries选cumulative_profit、pool_after、idle_vehicles、trip_vehicles、onboard_load等，最多6项。不能用period_profit去标“累计利润”，不能把trip count标“载客车辆”；在论文caption注明normal/drain终点及逐点CI。

### 15.6 可选learned组件

1. 用不重叠seeds生成train/validation/test，或单份修订逐seed指定split。
2. learned→Training/拟合，split=train，epochs/learning_rate明确，校验提交。
3. 完成后记录r-ID并检查weights/checkpoint/训练MSE；不把训练期间reference运营结果当learned held-out结果。
4. learned→Held-out evaluation，split=test，填checkpoint_run，保持与训练物理参数完全一致；校验并运行。
5. 只拿held-out评估run参与实验分析。当前validation split可作为独立手工评估集，但平台没有自动超参数搜索/early stopping/model selection流程。

<a id="s16"></a>
## 16. Python 导出数据与本地文件字典

### 16.1 文件保存在哪里

默认工作空间为仓库内workspace；设置DRL_WORKSPACE会改变实际根目录，以analysis.local_path和启动信息为准。

```text
_review_repo/workspace/
├─ datasets/<dataset_id>/<revision>/
│  ├─ normalized.json       精确规范数据
│  ├─ raw.json              原始生成payload或导入文本/映射
│  ├─ derived.json          validation统计
│  └─ manifest.json         修订索引
├─ runs/<run_id>/           配置、源码、指标、trace、日志、ZIP
├─ analyses/<analysis_id>/  多实验数据、统计、图与离线脚本
├─ index.sqlite            运行索引，非实验事实唯一来源
└─ server.json / 日志等     服务状态；server.json含token，不公开分享
```

workspace通常被Git忽略，代码推送不上传你的数据与图；你必须另行备份重要运行目录或导出ZIP。浏览器下载路径与原始local_path可以不同；图完成后已存本地，即使不下载网页链接也在目录中。

### 16.2 analysis目录每个文件

| 文件 | 用途 | raw-export有吗？ |
|---|---|---|
| `manifest.json` | analysis_id/family/status/config/local_path/plots/统计口径/warnings/source_hashes等 | 有，但raw信息较少 |
| `data-contract.json` | analysis-data/v1、trace_schema、sample_unit、numeric_table_layout | 有 |
| `runs.json` | 每run的完整spec+manifest，不只有名称 | 有 |
| `datasets.json` | dataset_hash→精确payload | 有 |
| `metric-definitions.json` | 9项本族指标定义 | 有 |
| `observations.csv` | 场景级数值长表，metric/value列 | 有 |
| `observations.json` | 场景级完整对象，保留null与reason | 有 |
| `periods.csv` | 逐t数值长表，增加period；一条step多个metric行 | 有，有trace才有数据 |
| `periods.json` | 完整逐t测量对象 | 有 |
| `states.jsonl` | 每行一run/scenario/t的完整嵌套trace、车辆/订单/决策/记账 | 有，有trace才有数据 |
| `plotting-config.json` | AnalysisSpec有效配置 | 普通绘图有，raw无 |
| `plot-data.json` | 统计后的panel/series/points/samples | 普通绘图有，raw无 |
| `statistics.csv` | mean/sd/n/CI及context/condition/axis | 普通绘图有，raw无 |
| `paired-differences.csv` | paired逐instance差；非paired可能空表 | 普通绘图有，raw无 |
| `plotting.py` | 可离线运行的renderer副本，无模型依赖 | 普通绘图有，raw无 |
| `requirements.txt` | 绘图时Matplotlib/NumPy精确版本 | 普通绘图有，raw无 |
| `figure.pdf/svg/png` | 论文输出图 | 普通绘图按formats生成，raw无 |
| `analysis.zip` | 打包上述产物，不包含自己 | 有 |
| `state.json`（可能出现） | 渲染失败时status=FAILED、message | 不是成功图标志，失败可能无正常manifest/list项 |

页面通用提示提到“运行python plotting.py可重绘”，**仅普通绘图包适用**；raw-export包只导数据，网页提示是通用文字，不代表其中一定有这个脚本。

### 16.3 observations / periods 的共同元数据字段

| 字段 | 说明 |
|---|---|
| `run_id` / `family` / `framework` | 运行ID、single、规范框架或组件回退名 |
| `condition_id` | 实验配置身份；与context合用 |
| `dataset_id` / `dataset_hash` | 容器与真实冻结修订 |
| `scenario_id` | 原场景ID |
| `instance_key` | 排除id/split/seed后的物理场景hash，跨改名同内容去重 |
| `scenario_seed` / `policy_seed` / `training_seed` | 三个角色分开记录 |
| `physical_key` | 原model+evaluation+solver+source物理上下文摘要 |
| `source_hash` | 当前run源快照内容摘要 |
| `parameters`（JSON） | model/controller/value.parameters展平路径→数值/布尔/字符串；数组展开.0/.1等 |
| `physical`（JSON） | model/evaluation/solver/source详细条件，CSV不把整个对象放一列 |
| `metrics`（JSON） | 第8.3/8.4或第10.7节完整字典，包括null/reason |
| `period`（periods） | 决策期t，非row序号 |
| `metric` / `value`（CSV） | 每行一个数值指标/值；同run/scenario/period重复多行正常 |
| `model.parameters.*`等（CSV） | 参数展开为独立列；看它们判断condition，不只看算法名 |

CSV只导有限类型中的数字（布尔不作为数值行）；null和文本reason不形成数值行，某些reason只在JSON保留。**缺行不等于value=0**。不要按CSV行数当统计n：同一场景会有多个metric、多个period、多个run行。

states.jsonl的根字段为run_id、scenario_id、instance_key、period、trace；trace完整含第10节所有内容。重构详细订单/车辆分析用它，不要要求所有嵌套状态强行塞入二维CSV。

### 16.4 导出包能否供自己的Python多实验绘图？

可以。普通绘图包提供原始长表+完整JSON+统计表+配置+渲染脚本；raw-export也提供可供自定义脚本处理的场景、逐期与状态资料。不同框架、数据与参数可选择同一导出，但必须按物理/源码/条件分层，不把所有行直接求mean。

离线重绘普通绘图包：解压到独立输出目录，用合适Python环境安装requirements.txt，在该目录执行`python plotting.py`。本项目已装绘图依赖时可用项目.venv运行文件。脚本以自身目录读config/plot-data并写同名figure，会覆盖该目录原图，修改前留副本。

这一步只重新**渲染已统计的plot-data**，不重新跑模型、不重新估计CI、不自动读取你修改过的observations重新统计。改变confidence或统计分组后应重新生成plot-data/统计，再重绘；只改plotting-config.confidence不会让旧CI自动变化。

如果自己的Python要重新统计：读取observations.json/CSV→按context/condition/instance聚合policy seed→以instance计算统计。时间序列加period；paired保证共同instance集合。建议保留null原因、失败报告、单位与source_hash，而不是只保存一列profit。

### 16.5 publication-ready到底包括什么

当前renderer用Matplotlib+NumPy，本地无图形窗口的Agg后端；统计计算使用SciPy Student-t。架构通过分析合同/renderer接口解耦，而非每新增图都改run存储。

- serif DejaVu Serif字体，轴/刻度/图例约7–8pt、细线、去上/右边框；无需外部TeX。
- Okabe–Ito色盲友好系列色、不同marker/线型；热力图cividis和数值注释帮助黑白阅读。UI族颜色不限制论文系列颜色。
- 多panel自动排版；PDF嵌入Type42字体，SVG保留文本，PNG默认600dpi。
- 每图配原始数据、统计、配置、脚本和版本锁；不是截图。

这是正式论文的绘图基础，不是自动符合任何期刊全部投稿规范。实际投稿还需检查目标期刊尺寸、字体授权/替换、caption、长标签和中文字体；DejaVu Serif不是通用中文字体，中文自定义标签可能需要自行配置字体。显示series过多时marker/颜色会循环，建议减少同panel曲线。

### 16.6 run manifest、checkpoint与指标定义字段

| manifest字段 | 说明 |
|---|---|
| run_id / batch_id | r/b唯一资源ID |
| policy_seed / training_seed / scenario_seeds | 执行角色与冻结ID→seed；learned评估训练seed在metrics行/checkpoint，不一定manifest同样填入 |
| condition_id / family / framework_id | 实验条件、族、规范框架 |
| statistical_unit | independent-scenario |
| dataset_hash / scenario_ids | 真正数据修订与所选场景 |
| platform_version | 0.2.0，不等于模型插件version |
| limitations | 已知限制字符串；rollout包括conditional first collection batch approximate等 |
| code.`git_sha` | 提交时HEAD，可能null；dirty状态还需snapshot/content_hash |
| `dirty` | 提交时Git工作区是否有改动，不表示算法结果“脏” |
| `code_available` | 是否取得Git SHA；true不表示环境已可一键复原 |
| `content_hash` | 当前纳入快照文件的内容hash，包含文档等；不是只有Python算法hash |
| `lock_hashes` | uv.lock、frontend/package-lock.json文件摘要 |
| `snapshot` | code.snapshot.zip（有快照时） |

| checkpoint/weights字段 | 说明 |
|---|---|
| feature_schema | hub-time-value/v1，特征契约 |
| feature_names | 第5.8节7项顺序，评估必须匹配 |
| weights | 与feature_names同顺序数值数组 |
| weights_hash | weights.json的SHA-256 |
| dataset_hash | 拟合数据修订 |
| train_scenario_ids | 实际拟合场景集合 |
| all_scenario_ids | 该修订全部场景ID，不代表全用于训练 |
| model_parameters | 拟合物理参数，评估需完全匹配 |
| training_seed | 拟合run种子 |
| training_method | supervised-fluid-dual-v1 |

| metric-definitions字段 | 说明 |
|---|---|
| key / label | 机器名 / 可读学术名 |
| unit | currency/orders/fraction等 |
| sense | maximize/minimize/diagnostic/upper-bound，决定能否当运营paired指标 |
| population | run/orders/period/full horizon等研究对象 |
| aggregation | scenario_mean声明，不代表metrics.json已经自动做了全场景均值 |
| time_scope | configured terminal policy |
| accounting_version | legacy-assignment-v1 |
| compatible_model_families | 当前single_level_matching适用集合，不表示legacy可混用 |

plugins.lock的每条含family/kind/id/version/parameters_schema/capabilities/ui_schema；kind=model/controller/value_function；parameters_schema.properties列参数、类型、默认、上下界，additionalProperties=false禁未知字段；capabilities.resume=false、training取决于train_、arbitrary_imports=false；ui_schema.section标表单分区。Schema的title/type/enum/const/minimum/maximum/exclusiveMinimum/items/required描述协议，而非run结果。

charts.json包含schema_version=chart-artifact/v1、type=scenario-bars、data_ref=metrics.json、encodings.x=scenario_id、y=operating_profit、units.y=currency、metric_defs=metric-definitions.json、javascript_allowed=false。它是通用声明，Oracle并没有profit可绘；真正论文图以分析可用指标判断。

### 16.7 analysis manifest / plot-data字段

| analysis manifest字段 | 说明 |
|---|---|
| schema_version | analysis-artifact/v1 |
| analysis_id / family / status | 输出身份，正常完成COMPLETED |
| config | 有效AnalysisSpec；raw-export仅run_ids/kind=raw-export |
| local_path | 输出绝对目录 |
| plots | 已生成figure文件名；raw为空 |
| panels | 实际panel数；raw不一定有 |
| sample_unit | frozen physical scenario; policy seeds averaged within instance |
| inference | pointwise Student-t及独立性假设说明 |
| warnings | n不足、口径分层、augmented诊断、缺失不插补等 |
| source_hashes | 所选run源码hash集合 |
| statistics | 第13.8节统计对象列表 |
| artifacts | 读取分析接口时列实际文件名，方便下载 |

plot-data.json是panel数组：title含dataset与context短hash；metric/context/xlabel/y_parameter/ylabel声明坐标；series数组含label、identity（部分图型）、points、samples；points含x/y及n/mean/sd/CI/min/max等统计，samples是用于画点的场景层数值。它不是原始trace，应配observations/states一起保存。

<a id="s17"></a>
## 17. 常见误区、限制、排错与核对依据

### 17.1 常见现象与应该检查什么

| 现象 / 可能code | 原因与检查 |
|---|---|
| 数据列表刚打开为空 | 初次请求尚未完成；等刷新；若有错误检查服务与当前族，不要立即重生成 |
| `DATASET_SELECTION` | 所选split无场景，或scenario_ids与split不匹配；查看完整数据 |
| `DATASET_HASH_MISMATCH` | 冻结文件被修改；不要手改normalized.json，重新冻结合法数据 |
| `DATASET_MODEL_MISMATCH/FAMILY_MISMATCH` | 用了别族数据或model；切族不自动转换数据 |
| `SCHEMA_VALIDATION` | 类型/范围/必填/未知字段；读details.path，尤其数组JSON、checkpoint、seed |
| `FRAMEWORK_MISMATCH` | 手改controller/value后还留旧framework_id；设正确ID或让核心推断 |
| `UNSUPPORTED_COMBINATION` | 组件不在本族或rollout非zero；不靠换显示名称绕过 |
| `TRAIN_SPLIT_REQUIRED` | 拟合必须train+zero；别用test训练 |
| `TRAINING_NOT_APPLICABLE` | 非拟合框架填training_seeds；省略/null而非空列表 |
| `TRAIN_TEST_LEAKAGE` | 同修订训练场景进入评估；选held-out，不篡改split或manifest |
| `CHECKPOINT_PARAMETERS/HASH/FAMILY` | 物理参数、权重内容、特征契约不匹配；检查正确train r-ID |
| `NOT_FOUND` | hash/run/文件不存在或包尚未生成；不是所有产物始终存在 |
| `SWEEP_PATH/SWEEP_RANGE/EMPTY_SWEEP` | 路径未注册、布尔误用线性、log倍率≤1、离散值空；重新选择正确方式 |
| `BATCH_SIZE` | 网格×variants×seeds超过1000run，或单轴超过限制 |
| `CPU_BUDGET` | 单求解器threads超预算；并发和thread分开配置 |
| `FEASIBLE_LIMIT`但COMPLETED | 有incumbent已提交，但没有该步最优证明；论文记录gap与runtime |
| `NO_SOLUTION_LIMIT/BACKEND_ERROR` | 运营MILP无incumbent或优化器异常；看stderr及solver/license；不填0profit代替 |
| `TIMEOUT` | 总run限时，尤其rollout×场景；与单solver限时区别 |
| `TRACE_UNAVAILABLE` | save_trace关闭、Oracle或无完整trace；不能事后凭metrics重建全部状态 |
| `METRIC_UNAVAILABLE` | 某framework无该指标、rate为null；选择适用指标，不插补 |
| `PAIRED_CONDITIONS/UNPAIRED_SCENARIOS` | 一个分层不是恰2条件，或instance集合不同；核对数据hash/条件/seed集合 |
| `NONNUMERIC_AXIS` | 轴在某run缺失或为bool/字符串；只选全条件存在的数字轴 |
| `CONDITION_POOLING` | 普通comparison选择了应作为扫描的多条件；改sensitivity或拆组 |
| `PANEL_LIMIT/ANALYSIS_SIZE` | >24panel或>200k期记录，按数据/指标拆分析 |
| `RENDER_FAILED` | 字体/数值/依赖/布局等渲染错误；原始数据可能已写，查对应state与日志 |
| `CODE_VERSION_MISMATCH/COMPARISON_CODE_MISMATCH` | 内容快照变了；本文等文档变化也可能触发；保留原记录，复制配置做新版本实验 |
| `WRITE_TOKEN/ORIGIN_DENIED/HOST_DENIED` | 会话令牌或来源/主机防护；从正确本地URL刷新，不关闭安全防护或泄露token |
| `UPLOAD_SIZE/BUNDLE_SIZE/BUNDLE_HASH/BUNDLE_PATH` | 请求/ZIP超限制或包校验不符；不要运行/修改不可信包来绕过 |

### 17.2 必须避免的解释错误

- 看到torch、learned、V或rollout，直接宣称single用了强化学习。
- 把objective当profit，把upper_bound或MIP incumbent当真实执行policy收入。
- 把assigned当delivered，把订单率当乘客率，把计划里程当actual distance。
- 把扫描参数值、K条sampled future、多个policy seed或CSV行数当独立n。
- 认为phase=trip一定已载客、地图“X辆”一定都是idle、step.after已经按计划到终点。
- 把expired数组大小直接当cancelled_delta，不检查订单是否本期已指派救回。
- 手改JSON后拿旧MathPanel/单步表单当最新配置，或上方改一项覆盖了批次大框而不自知。
- 开启save_trace就认为每期实时安全落盘、失败/取消可以断点恢复。
- 一键选择所有可完成运行就宣称它们处于同源码/数据/终点/求解环境下。
- 认为检查ZIP成功等于已重跑复现、下载raw-export一定有plotting.py、重绘脚本自动重估CI。

### 17.3 正式实验提交前/报告前清单

1. 明确研究问题、物理模型、framework与指标方向，不跨论文模型族混比。
2. 记录数据完整hash、各scenario seed及生成假设，确认train/test不重叠。
3. 明确parameter configuration、独立replications数量和算法随机性设计。
4. 比较算法复用相同冻结场景；明确求解资源、timeout、信息集、终点和记账口径。
5. 检查实际提交大框/展开计划，不只看上方可视化表单。
6. 保持同批研究源码快照稳定；先完成代码/文档修改，再成套正式运行。
7. 检查完成/失败数、每步incumbent/bound/gap、trace invariants和null原因，不静默剔除失败。
8. 分析读context、condition、instance、n和A/B符号，CI声明pointwise且小样本谨慎。
9. 保存精确数据、原始结果、统计、plotting config、图形、代码及版本锁；备份workspace。
10. 根据目标期刊再核对论文尺寸/字体、caption、单位、legend和方法，不用网页截图作最终论文图。

### 17.4 当前边界一览

| 已有 | 不应承诺为已有 |
|---|---|
| 三族独立领域包、共享本地平台 | 三族各自虚拟环境/数据库/服务隔离 |
| 自动Schema字段、统一有限网格批实验 | 任意生成参数直接在编辑器扫描、可视化多framework批选 |
| single完整成功run场景结果与逐期记录 | 失败run逐期崩溃安全持久化、断点续算 |
| myopic/fluid/rollout/oracle、可选线性监督拟合 | single DRL/neural value、自动训练评估搜索流水线 |
| 单参数、热力图、时间序列、comparison、paired论文图 | response surface、任意图形网页配置、多层随机效应推断 |
| 同场景统计、Student-t CI、物理instance去重 | 自动验证真实统计独立性、同网络需求生成、全修订防训练泄漏 |
| 本地PDF/SVG/PNG、数据和脚本 | 任意期刊自动合规、任意中文字体、DRL/BHH完整论文analysis |
| 复现ZIP、源码快照、校验和 | 自动执行包内代码、自动恢复任意环境或checkpoint |

### 17.5 本文核对依据与维护入口

本文对照运行中的single页面，并以以下现行源定义为依据；没有新提交实验、取消任务、归档数据或改动研究算法。运行数量/具体结果值不是本文固定标准，因为workspace会持续变化。

| 说明内容 | 实现依据 |
|---|---|
| 页名、输入、操作、结果显示 | [main.tsx](../frontend/src/main.tsx) |
| MathPanel、BatchBuilder、AnalysisPanel | [family-components.tsx](../frontend/src/family-components.tsx) |
| single框架、校验、颜色、训练需求 | [plugin.py](../model_families/single/plugin.py) |
| 生成Schema、物理/框架/value参数 | [generation.py](../model_families/single/generation.py)、[parameters.py](../model_families/single/parameters.py) |
| 数据生成/导入/验证 | [data.py](../model_families/single/data.py)、[datasets.py](../experiment_core/datasets.py) |
| 模板与数学显示 | [templates/](../model_families/single/templates/)、[mathematics.py](../model_families/single/mathematics.py) |
| 正式metrics/trace、监督拟合 | [runner.py](../model_families/single/runner.py)、[metrics.py](../model_families/single/metrics.py) |
| MILP/推进/会计与fluid V | [mt_prototype.py](../model_families/single/engine/mt_prototype.py)、[online_lookahead.py](../model_families/single/engine/online_lookahead.py) |
| 接驳可见信息、Oracle松弛 | [demand.py](../model_families/single/engine/demand.py)、[bounds_check.py](../model_families/single/engine/bounds_check.py) |
| JSON契约、范围展开、源码/队列 | [contracts.py](../experiment_core/contracts.py)、[batching.py](../experiment_core/batching.py)、[service.py](../experiment_core/service.py) |
| 分析提取、样本分层、统计 | [single/analysis.py](../model_families/single/analysis.py)、[analysis.py](../experiment_core/analysis.py) |
| 分析协议与Python画图 | [analysis_contracts.py](../experiment_core/analysis_contracts.py)、[paper_renderer.py](../experiment_core/paper_renderer.py) |
| 复现、诊断与本地接口 | [reproduction.py](../experiment_core/reproduction.py)、[diagnostics.py](../experiment_core/diagnostics.py)、[app.py](../platform_api/app.py) |

继续阅读：[简版平台说明](EXPERIMENT_PLATFORM.md)、[现有验收记录](PLATFORM_VALIDATION.md)、[单层模型审阅](SINGLE_LEVEL_MODEL_REVIEW.md)。历史说明的默认值/验收数字可能不同，当前UI/Schema/正式runner与本手册核对版本优先。
