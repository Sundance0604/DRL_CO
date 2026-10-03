# 本地实验平台说明

## 定位与启动

这是研究用的本地 React 平台，不是面向互联网的服务。网页、HTTP API 和终端入口共用 `experiment_core`；数据集、物理模型、控制器、价值函数与求解后端分别定义。所有计算来自真实代码与冻结数据，没有模拟成功结果。

Windows：

```powershell
.\scripts\setup.ps1
.\scripts\doctor.ps1
.\scripts\start.ps1
# 结束服务
.\scripts\stop.ps1
```

macOS/Linux：

```sh
bash scripts/setup.sh
bash scripts/doctor.sh
bash scripts/start.sh
bash scripts/stop.sh
```

`start.command` 是 macOS 双击入口，需先完成安装。脚本定位自身目录，支持从其他目录启动，不依赖绝对个人路径。**尚未完成原生 macOS 验收**，不能把 Windows 通过视为 Apple Silicon/Intel Mac 通过。

默认地址为 <http://127.0.0.1:8765>。`platform.local.json` 控制本地端口、网页服务并发上限和可选 CPU 线程预算；并发默认 1。后端只能绑定 `127.0.0.1`，端口冲突不会导致启动器终止其他进程。`scripts/dev` 使用前端构建监视，不自动重启正在计算的后端。

依赖通过 `uv.lock` 和 `frontend/package-lock.json` 锁定。平台使用 Python 3.11，验证环境为 Python 3.11.14、Node 24.18.0、Gurobi 10.0.3、CPU。安装需要网络；优化模型需要有效的对应版本许可证。稳态 BHH 与空间校准使用 CPU 数值方法，不调用 Gurobi。

## 目录与数据流

```text
React / CLI
    -> validated RunSpec / BatchSpec
    -> immutable dataset revision + compatible registered plugins
    -> Coordinator (workspace lock, SQLite index, isolated workers)
    -> metrics / events / trace / artifacts / source snapshot
```

`frontend/` 只展示与提交实验，不执行优化逻辑。`platform_api/` 提供本地 API。`experiment_core/` 管理契约、冻结数据、插件、执行、结果、复现和诊断。原有 `model/` 与 `drl_co/` 保留；适配器调用其实现，不让浏览器直接读取任意文件或加载任意模块。

默认工作区为仓库 `workspace/`，也可以设置 `DRL_WORKSPACE`。数据、权重、日志和结果均不纳入 Git。SQLite 是可重建索引；运行目录中的 JSON 事实记录才是恢复依据。每个工作区只能有一个协调器；终端检测到本工作区网页服务后连接现有 API，不启动第二个写入者。

## 模型兼容矩阵

| 模型族 | 控制器 | 价值函数 | 后端与语义 |
|---|---|---|---|
| `single_level_matching` | `myopic` | `zero`、`fluid_dual`、`learned_hub_time` | Gurobi，每期单层匹配 |
| `single_level_matching` | `rollout` | `zero` | Gurobi，克隆状态上的真实前瞻 |
| `single_level_matching` | `train_value` | `zero` | Gurobi，流体 dual 标签监督训练 |
| `single_level_matching` | `oracle_lp`、`oracle_mip` | `zero` | 完全信息松弛上界，不是可执行策略 |
| `legacy_dispatch` | `myopic` | `zero` | 原 supply 启发式 + 下层优化适配器 |
| `legacy_dispatch` | `train_sac`、`candidate_sac` | `zero` | CPU PyTorch + Gurobi，真实强化学习 |
| `bhh_steady` | `myopic` | `zero` | CPU，稳态解析/数值研究，不是在线决策 |
| `bhh_finite` | `myopic` | `zero` | Gurobi，全时域完全信息优化 |
| `bhh_finite` | `rolling_horizon` | `zero` | Gurobi，已知订单的滚动窗口与承诺延续 |
| `bhh_spatial` | `myopic` | `zero` | CPU，固定点样本与小规模精确 TSP |

不支持的组合在提交前报错。特别是：单层匹配尚不支持 rolling horizon，旧 SAC 权重不能作为枢纽价值函数，非零 warm-up 尚不支持，断点续跑能力为 false。当前设备固定为 CPU，不提供无效 CUDA/MPS 开关。

`train_value` 的小型线性模型使用剩余时间、枢纽结构、空闲供给、积压与城市半径特征，监督目标来自流体 dual；它只验收新特征和训练闭环，不证明效果优于流体价值。`train_sac` 使用奖励 transition、replay buffer 和 actor/critic 更新，才是强化学习；其学习率同时传给 actor、critic 和温度优化器。

## 先生成数据，再比较算法

```powershell
.\scripts\exp.ps1 dataset generate --config configs/datasets/matching-demo.json --json
.\scripts\exp.ps1 experiment validate --config configs/experiments/matching-myopic.json --json
.\scripts\exp.ps1 experiment plan --config configs/experiments/matching-rollout.json --json
.\scripts\exp.ps1 run --config configs/experiments/matching-myopic.json --json
.\scripts\exp.ps1 run --config configs/experiments/matching-fluid.json --json
.\scripts\exp.ps1 run --config configs/experiments/matching-rollout.json --json
```

Shell 对应 `bash scripts/exp.sh ...`。运行样例中的 `revision: "latest"` 只在终端解析为精确 hash，且只允许唯一未归档版本。出现多个版本必须明确指定 hash。网页始终选择精确 revision。每个运行保存 resolved 配置、实际场景 ID、policy seed 与内容 hash。

冻结目录区分 `raw.json`、`normalized.json`、`derived.json` 和 manifest。再次生成相同内容得到相同 revision；修改订单会形成新 revision。加载会验证 hash，避免运行引用的数据被悄悄修改。训练、验证、测试场景显式区分，checkpoint 评估拒绝与训练场景重叠。

数据管理页支持生成预设、选择版本、预览、校验、导出、归档和基于模板导入。API 支持 CSV/JSON/JSONL 字段映射；初版网页使用与冻结模板一致的字段。原生冻结 JSON 可用 `dataset import --config` 导入。没有行业专用 ETL 与真实地图路径；不能将抽象网络图称为实际行驶轨迹。

## 训练与独立评估

```powershell
.\scripts\exp.ps1 run --config configs/experiments/value-train.json --json
.\scripts\exp.ps1 dataset generate --config configs/datasets/legacy-demo.json --json
.\scripts\exp.ps1 run --config configs/experiments/sac-train.json --json
```

将返回的训练 `run_id` 填入相应 `value-evaluate.template.json` 或 `sac-evaluate.template.json` 的 `checkpoint_run`，另存配置并运行。模板故意不填猜测的 ID，不是可直接执行的实验。权重元数据包含模型族、特征版本、训练数据 hash、训练场景和物理参数；不匹配时拒绝加载。

## BHH 研究入口与假设

```powershell
.\scripts\exp.ps1 dataset generate --config configs/datasets/bhh-demo.json --json
.\scripts\exp.ps1 run --config configs/experiments/bhh-steady.json --json
.\scripts\exp.ps1 run --config configs/experiments/bhh-finite.json --json
.\scripts\exp.ps1 run --config configs/experiments/bhh-rolling.json --json
.\scripts\exp.ps1 run --config configs/batches/bhh-tau-demand.json --json
```

数学来源保存在 `doc/bhh_finite_v1.tex` 与 `doc/bhh_steady_v1.tex`。这一族严格限制两城市 0、1，有限时域需求表示可拆分 stops，**不能解释为 kg、托盘或不可拆分乘客订单**。

局部配送容量逐个预计算 `(车辆数 v, duration)`：`min(v*M, inverse_f(v*(duration*t0-2*rho)))`，不会替换为错误的 `v*inverse_f(duration)`。直送采用独立约束：`2*rho_origin + f_origin(q)/v + tau*t0 + f_destination(q/v)`，按原式求解可行容量。空 HV 可以通过零载荷直送弧迁移。

有限时域模型包含 collection / AV line-haul / distribution / direct 四种阶段、逐商品流守恒、释放与截止时间、累计转运因果关系、波次整数选择、共享 HV 占用及城市 AV 库存。成本默认按原文的车辆时间与未服务惩罚记账；额外有限时域等待成本通过 `finite_waiting_cost` 显式开启，默认 0。预订时域结束后允许已知订单在截止时间内完成。

滚动窗口使用真实 planning/commit/completion 参数；过去已启动弧（包括取值 0 的决策）冻结，在途车辆与货物流进入下一窗口，不能重置车队或偷改承诺。最终成本从累计承诺重建。局部窗口 gap 不是全局 gap，因此全局 gap 返回 null 和原因。

稳态实现基于均衡、对称、平稳需求和不绑定有限车队的假设，报告所需 fleet，而不把 fleet 数组当约束。输出连续 wave、普通取整与满足嵌套条件的整数解、成本分解及条件性关系标记；不无条件套用平方根关系。稳态成本是单位 load 成本，有限时域成本是该场景总成本，禁止直接排名。

| 研究维度 | 当前配置 / 产物 | 验证边界 |
|---|---|---|
| 稳态与有限时域 | `bhh-steady.json`、`bhh-finite.json` | 分别可执行；尚无 warm-up 长跑等价性验收 |
| τ × Λ 驱动 | `batches/bhh-tau-demand.json` | 9 单元真实扫描，网页热力图 |
| 整数取整 | `batches/bhh-rounding.json` | 连续、普通取整、嵌套整数结果 |
| 空间校准 | `bhh-spatial.json` | 固定盘内点/边界 hub，小 TSP 每样本最多 8 stops |
| 价值分解 | `batches/bhh-value-decomposition.json` | 原稳态 direct/hub benchmark；分解恒等式检查 |
| 非对称 | `bhh-asymmetric.json` | 分城市几何、车队与空 HV 迁移；无需求失衡稳态定理 |
| 异质货物 | 导入 `datasets/bhh-heterogeneous.json` 后运行对应实验 | 截止时间、惩罚、禁止转运标记；共享 HV |

有限模型保存 `capacity-table.json`，包含局部与直送容量表（示例 duration 1–12）。该表是研究输出，不是车型标定结果。sorting rate、同步摩擦等新扩展尚未实现，不能从讨论笔记臆造目标函数。

## 队列、结果与比较

运行状态为 QUEUED / RUNNING / COMPLETED / FAILED / CANCELLED / TIMEOUT / INTERRUPTED。每个 worker 是独立进程，默认单并发、求解器单线程；没有嵌套进程池。任务提交支持幂等键。取消先写信号，超时后只终止属于该任务的进程树；启动器停止服务前也核验 PID、创建时间及实例 token。

每次提交冻结当前源代码 ZIP、Git SHA、dirty 标记、内容 hash 与依赖锁 hash。worker 从提交时的快照运行，后续编辑不会改变排队任务。重启时未完成任务标为 INTERRUPTED，不伪装自动续跑。事实记录可重建 SQLite 索引。

指标区分运营利润、增广优化目标、松弛上界、指派/送达/未完成数量与 BHH load；缺失测量为 null 并附原因。距离字段标为 planned，实际累计距离没有可靠测量时不编造。原 SAC 适配器没有经验证的送达事件，`delivered` 为 null。

匹配 `report_pending` 保留原指派记收益口径并报告终点未完成量；`drain_committed` 停止到达并只清空已承诺行程，未指派积压仍单独报告。rollout 只在克隆上评估，共用候选间的同一组未来样本，选定计划再提交一次；首段批内条件采样仍是近似，结果中披露这一局限。

配对比较要求相同冻结数据、场景集合、物理参数、信息集与终点口径。先场景内平均 policy seeds，再按场景做配对 Student-t 区间；不足两个场景时区间为 null。失败运行不能静默删除，理论参数研究不能冒充在线策略显著性比较。

## 复现与诊断

```powershell
.\scripts\exp.ps1 runs status --run RUN_ID --json
.\scripts\exp.ps1 runs logs --run RUN_ID --json
.\scripts\exp.ps1 runs replay --run RUN_ID --period 0 --json
.\scripts\exp.ps1 export --run RUN_ID --json
.\scripts\exp.ps1 reproduce --bundle workspace/runs/RUN_ID/reproduction.zip --json
.\scripts\exp.ps1 debug step --config configs/experiments/matching-myopic.json --period 0 --output workspace/debug --json
.\scripts\exp.ps1 debug diagnose --run RUN_ID --output workspace/diagnostic.zip --redact --json
```

复现包包含真实数据、spec、依赖锁、环境、schemas、源代码快照和可取得的注册权重。导入先验证路径、重复项、尺寸、Schema 与全部 hash；不会执行包内代码或加载外来 pickle。运行复现要求当前源代码内容与记录完全相同，相关训练 checkpoint 已在本工作区注册；否则明确报错。源代码匹配不代表跨操作系统求解结果逐位相同。

“复制配置”为新实验，“重跑”为相同配置与代码检查；两者不能混淆。诊断 ZIP 默认隐藏本机根目录、用户目录与常见凭据，不包含原始行业样本。结果/复现包含原始冻结数据和源代码，**分享前需自行确认敏感性**；当前版本没有自动行业数据匿名化。

API 位于 `/api/v1`，OpenAPI 及参数 Schema 在 `schemas/generated/`。示例：datasets、experiments/validate、experiments/plan、batches、runs、runs/events、runs/artifacts、comparisons、reproduction/validate。写入必须携带 workspace token，跨来源请求与非本地 Host 被拒绝。没有任意 shell、任意文件路径下载或用户模块导入接口。

CLI 成功 stdout 是 JSON；求解日志与 `--json-stream` 进度在 stderr。退出码：0 成功，2 输入错误，3 依赖缺失，4 求解失败，5 超时，6 取消，7 内部错误。单步调试当前只支持匹配族，显示未提交的 before/plan/invariant；不是在生产运行中暂停任意求解器。

## 当前尚未覆盖的规格

本次交付是可运行的 v0.1 研究平台，并非说明文档中所有高级能力均已验收。主要缺口：原生 macOS/Apple Silicon 测试、warm-up 长跑稳态桥接、任意模型断点续跑、完整插件市场式 API、行业 ETL/真实地图、多规则敏感数据匿名化、专门的多阶段 Sankey/资源甘特交互与前端 SSE 自动重连。网页当前轮询队列/事件，API 的 SSE 与 Last-Event-ID 已实现。高级报告可直接查看 JSON 产物，不能把未实现的按钮列为能力。
