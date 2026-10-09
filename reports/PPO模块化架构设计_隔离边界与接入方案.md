# PPO 如何接入 single，而不侵入其他模块

代码架构设计建议｜2026 年 10 月 9 日

本文基于当前仓库接口作只读检查后提出设计。下列新目录、类名、组件 ID 和版本号均为拟议方案，并非已经实现；本次只新增本文，不修改运行代码、不启动训练、不改变默认模型。研究上的状态、动作与奖励定义见[第八轮方向判断与强化学习建模建议](第八轮方向判断与强化学习建模建议.md)。

## 1. 我的总体建议

**把 PPO 做成 single 模型族内部可选的战略控制模块，用适配层调用既有仿真、MILP 和账本；不要把 PPO 逻辑塞进现有业务引擎、通用调度器或旧实验脚本。**

第一阶段采用独立研究入口，让旧运行路径无需感知 PPO；第二阶段经过回归后，才在 single 的插件入口增加训练与评价组件。这样既保留公共平台，也避免为 PPO 复制一整套货运仿真器。

“不影响其他模块”应是可验证的行为约束，而不是承诺永久不改任何文件：

- 不选 PPO 时，旧算法的参数默认值、选择规则、物理轨迹与核算结果不变。
- PPO 不修改 legacy DRL/SAC 或 BHH 的模型包，不借用它们的状态、奖励和 checkpoint 语义。
- PPO 的训练状态、随机数、网络、日志和输出不进入其他 run 的内存或目录。
- 既有冻结实验不被重新运行或覆盖，旧代码快照与哈希记录不被重新“认证”为新版本。
- 若最后需要平台注册或导出扩展，使用显式、局部、向后兼容的改动，并为旧路径提供回归证据。

同一个 Python 环境、数据库和服务可以继续使用。需要隔离的是依赖方向、状态所有权、协议与产物，不是立即拆成几个后端系统。

## 2. 当前代码已经具备什么，还缺什么

| 当前模块 | 已有职责／能力 | 对 PPO 的含义 |
|---|---|---|
| `experiment_core/families.py` | 以受信任名称注册 single、legacy、bhh | PPO 属于 single，不新增第四个论文模型族 |
| `experiment_core/runner.py` | 只按模型族分发 | 不应写 PPO loss、reward 或训练循环 |
| `single/plugin.py` | family 参数、framework、训练识别、checkpoint 与执行入口 | 后续平台接入的主要薄入口 |
| `single/rolling/contracts.py` | `WindowRequest`、`WindowSolution`、`WindowPlanner` | 可以复用规划请求与完整动作协议 |
| `single/rolling/controller.py` | 观测快照、窗口控制、完整动作编译 | 观测隔离逻辑可复用，但不能假定其现有主循环已支持 PPO |
| `single/value/contracts.py` | `ValueProvider`、释放组 prefix 计价 | 适合接入求解前冻结的战略系数 |
| `single/value/route_milp.py` | 值感知的联合路线 MILP | 保持硬约束与动作提取，不在内部训练网络 |
| `single/planning.py` | `PeriodAction`、`compile_action` | 复用合法整数动作的执行边界 |
| `single/accounting.py` | `realized_ledger`、承诺冷却 | 奖励的权威核算来源，不另写 PPO 专用利润公式 |
| `single/analysis.py` | single 结果转换为公共分析协议 | 可以接入正式评价，但训练曲线必须单独处理 |

检查发现几处必须明确处理的现状：

1. `plugin.is_training()` 目前仅识别 `train_value`，它是监督拟合流程，不是 PPO。
2. 平台现有 `rolling_horizon` 分支要求 value 为 `zero`，其 planner registry 只注册普通 `route_milp`。研究用价值 MILP 的存在，不等于 UI 已自动支持它。
3. `single/runner.py` 的 `matching()` 是批量运行循环，不是可交互的 `reset/step` 环境，不能直接交给 PPO 当 Gym 环境。
4. `single/analysis.py` 当前只显式排除 `train_value`，新增训练 ID 后必须同步阻止训练 episode 被当成正式统计样本。
5. 通用复现导出目前按固定文件名打包 checkpoint 与权重，不会自动收集未来 PPO 的标准化、底座和动作配置等额外文件。
6. 平台已有 `training_seeds`、`policy_seeds`、scenario 身份与通用批量协议，但这些字段需要通过正确的 family 适配才能生效。

因此，“直接加一个 PPO 类就能全部自动接上”不成立。真正应控制的是接入改动的范围，而不是假装不需要接入。

## 3. 推荐的目录结构

下面是建议结构。先按最小实现需要建立文件，不要求一次创建所有空壳。

```text
model_families/
  single/
    engine/                    # 既有物理引擎，不加入 PPO 分支
    planning.py                # 既有动作编译，不加入网络
    accounting.py              # 既有权威账本，不改奖励含义
    rolling/                   # 既有控制与观测工具
    value/                     # 既有价格接口与 MILP
    strategic/                 # 新增：single 的战略控制接口，不依赖 PPO 算法
      contracts.py             # Observation、Signal、DecisionRecord 等协议
      observation.py           # 公开状态提取；禁止隐藏未来
      decoder.py               # Signal -> ValueProvider -> MILP -> PeriodAction
      reward.py                # 调用权威账本，形成逐步收益差
      episode.py               # 单个仿真 episode 的 reset/step/冷却
      baseline.py              # 冻结 MC 底座与只读解析
      artifacts.py             # 通用战略策略身份及兼容性校验
    learning/
      ppo/                     # 新增：PPO 特有内容
        config.py              # 训练、评价、网络与预算的参数 Schema
        networks.py            # Actor/Critic，含观测编码网络
        policy.py              # 训练采样／只读推理
        trainer.py             # rollout、advantage、更新、验证
        backend.py             # 隔离具体 PPO 实现或第三方库
        environment.py         # episode 到训练库环境协议的薄适配
        checkpoints.py         # 权重、优化器、随机数及恢复协议
        runner.py              # 训练／评价入口，不包含货运物理
        analysis.py            # 训练诊断与 PPO 附加指标
    plugin.py                  # 第二阶段才做少量显式接入
  legacy/                      # 不依赖 single PPO
  bhh/                         # 不依赖 single PPO

experiments/
  cfs_ppo_strategic_control/    # 拟议第八轮协议，不承载通用 PPO 算法
    config/
    run_experiment.py          # 数据划分、矩阵、固定预算与调用
    delivery.py                # QA、报告及冻结
    analysis.py                # 本轮科学比较，非训练逻辑

tests/
  test_single_strategic_contracts.py
  test_single_strategic_episode.py
  test_single_strategic_decoder.py
  test_single_ppo_training.py
  test_single_ppo_platform.py
```

边界有意分成 `strategic` 与 `learning/ppo`：将来改成 SAC、固定价格、简单策略搜索或候选选择，只替换策略实现，不必重写物理环境、奖励和 MILP 适配。但目前不应设计一个覆盖所有论文模型的“万能 RL 环境”；这些协议仍属于 single。

算法模块也不应依赖 `experiments/cfs_*` 的历史脚本。某个冻结研究目录中的工具恰好可用，不意味着它适合作为生产公共库。必要的小型适配可在新模块中按明确来源实现并独立核验，但不能复制一套仿真器，或修改冻结实验以便提取工具。

## 4. 依赖只能沿一个方向流动

```text
平台／本轮实验入口
          |
          v
    PPO runner / trainer -----> Actor / Critic
          |
          v
  single strategic episode
    |       |          |
    v       v          v
 observation decoder  reward adapter
              |          |
              v          v
      ValueProvider     accounting
              |
              v
       既有价值 MILP
              |
              v
      PeriodAction / compile_action
              |
              v
          既有物理引擎
```

必须禁止相反依赖：

- `engine`、`planning`、`accounting` 不 import PPO，不检查 `algorithm == "ppo"`。
- `ValueProvider` 不持有 optimizer，不反向传播，不读取训练 buffer。
- Actor 不读取 `all_orders`、数据库连接、正式实验文件或仿真器私有未来。
- 训练库不直接操作车辆、订单和 solver 变量；只能调用环境协议。
- 公共 `experiment_core` 不知道 hub 价格、GAE、PPO clip 或冷却现金的业务公式。
- 不使用 monkey patch、替换全局函数、运行时改注册表来切换算法。

依赖第三方 PPO 实现时，只有 `backend.py` 和训练侧适配理解该库的 rollout buffer、回调及 checkpoint 格式。领域模块不能返回该库私有类型，否则以后换算法或库版本又会牵动整个系统。

## 5. 核心接口：把“策略动作”和“物理动作”明确分开

建议使用不可变的数据记录和显式版本。只对 dataclass 写 `frozen=True` 不足以保护内部数组；数组需只读或复制，嵌套映射也不能共享可变引用。

| 协议／记录 | 主要内容 | 明确不包含 |
|---|---|---|
| `ObservedState` | 当前公开订单、车辆承诺、公共网络、时间、允许的历史、mask | 未来订单流、随机生成器、实况答案 |
| `EncodedObservation` | 按冻结 feature schema 编码的张量、维度与 mask | live simulator 引用 |
| `StrategicSignal` | 有界动作向量、动作 Schema、采样模式、policy 身份 | 路线变量或 solver model |
| `DecodeResult` | 完整 `PeriodAction`、状态身份、求解诊断、系数记录 | 修改后的 live simulator |
| `Transition` | 下一观测、reward、终止信息、审计引用 | 可被策略读取的隐藏未来 |
| `PolicyArtifact` | 网络与特征版本、动作尺度、底座身份、权重 hash | 任意用户模块 import 路径 |

概念调用链为：

```text
observation = observer.build(public_snapshot)
signal      = policy.act(observation, policy_state, rng)
decision    = decoder.decode(public_snapshot, signal, baseline)
transition  = episode.execute(decision)
trainer.observe(observation, signal, transition)
```

这是拟议接口示意，不是当前可运行代码。未来应允许有记忆的策略返回显式 `policy_state`，并在 episode 起点重置；不要把隐藏状态放在跨 run 共享的单例里。

## 6. MILP 接入：一次决策只冻结一份信号

沿用前一份建议：PPO 可输出每 hub 的近、远释放价格修正，再转换成释放查询的标量系数。具体动作公式与尺度属于版本化的 `ActionSchema`，而不是写死在求解器内部。

推荐 decoder 的执行顺序：

1. 对公开快照生成身份标识，检查 `period/revision`。
2. 验证动作维数、hub 顺序、有限性、范围及所用底座。
3. 建立仅绑定本次动作与快照的 `ReleaseSignalProvider`。
4. 创建本次 `ValueRouteMILPPlanner(provider, summary)`，使用既有 H=1、单期执行协议。
5. 验证求解状态，提取完整当前 `PeriodAction`。
6. 对相同状态通过 `compile_action` 编译整数动作，再由 episode 唯一地提交执行。
7. 保存输入信号、实际系数、求解状态、动作 hash、编译金额及执行账本。

provider 必须是确定性查询：同一次决策中，同一个 `(release_time, hub, q)` 多次调用应返回相同数值。**不能每次价值查询都调用随机 Actor，不能每个 q 抽样一次新策略动作。** 网络推理发生在求解之前，返回普通标量，不把 Torch tensor 或梯度图交给 Gurobi。

保持既有 prefix 约束、负价格语义、终端自由资源归零和 tie-break；不把 PPO 接入顺便变成“修正所有旧接入假设”的重构。若需要研究新的 q 语义或耦合项，应给 decoder 新版本和独立实验。

零修正应恢复同一底座和同一价值 planner 的行为。它不自动保证等于另外一个 M0 MILP 的逐动作结果，因为不同表达在多最优解下可能有不同 tie。测试应分别定义“恢复父策略”和“与 M0 目标／动作的关系”。

## 7. episode：环境拥有物理状态，训练器只能交互

`StrategicEpisode` 是唯一拥有 live simulator、未来订单调度和现金事件的对象。Actor、Critic、decoder 都不能直接改它。建议采用明确状态机：

```text
NEW -> DECISION_READY -> EXECUTING -> DECISION_READY
                                 \-> COOLDOWN -> TERMINATED
任意阶段 -> FAILED / CANCELLED（不伪装成正常终止）
```

### 7.1 reset 做什么

- 按 scenario 身份和物理参数创建新仿真状态。
- 重置策略记忆、奖励累计、trace 和运行内随机数。
- 按既有时序处理初始公开信息，生成第一份公开快照。
- 保存 scenario、参数、代码依赖、solver 设置和 episode 身份。

每次 reset 必须创建独立状态；不能复用上次已修改的订单或车辆对象来节省时间。缓存只允许保存不可变数据，且缓存键包含版本与参数。

### 7.2 step 做什么

step 接收战略动作，完成一次解码和真实执行，推进到下一决策边界。公开订单只由 scheduler 按时间释放。可以复用现有 `observed_state/observed_order` 的信息裁剪逻辑，但新观测提取仍需显式字段白名单，不能直接将 `state_snapshot` 全量喂给网络。

若训练框架提供普通 `info` 字典，不要在其中附带可被 callback 轻易拿到的隐藏未来对象；完整私有审计信息交给独立 writer。没有决策自由度的时点也应保持明确转移与奖励，不随意丢掉这段时间。

### 7.3 冷却和终止

到达日终后，严格继承当前订单与承诺处理规则，只执行已承诺运输。冷却是 episode 的自动步骤，不是新的 PPO 控制机会。最后一个决策 transition 包含剩余冷却现金，完成后才是真终止。

求解器超时、无可行解、会计差异、用户取消和达到技术时间上限是不同状态。正式研究默认应按预先固定规则记录失败／停止，不能偷偷改成 M0 并把该 episode 当 PPO 正常完成。若部署版需要 fallback，必须另设显式协议、独立标记和统计口径。

训练框架的终止和截断适配只写在 `learning/ppo/environment.py`。技术截断不能当作利润已经结清，也不能错误地将 critic bootstrap 清零；本研究要求完整 episode 时，残缺数据应按协议隔离，而不是当完整样本训练。

## 8. reward：只做账本适配，不维护第二套经济模型

`strategic/reward.py` 不应再计算一遍路线成本、运费或罚金。它调用权威 `realized_ledger`，以相同事件边界计算累计实现利润差：

\[
r_t=L_{t+1}-L_t,\qquad
r_{T-1}=J_{\mathrm{full\ cooldown}}-L_{T-1}.
\]

整条 episode 的奖励总和必须与完整冷却利润一致。MILP 的 augmented objective、战略 bonus、tie-break 和 assignment-booked profit 只作诊断，不能混入真实 reward。

最初可以通过同一账本在不同 cutoff 重新计算来保证正确。若发现逐期重复计算开销大，再实现增量缓存；缓存必须与权威账本逐步对照，不以性能优化为名改变核算。尤其不能为了节省时间，回到 `sim.J` 的旧预记收益作为训练奖励。

原始 reward 与用于训练的固定尺度 reward 分别记录。折扣、GAE 参数、归一化策略属于训练配置，不应改变领域账本。推荐有限完整 episode 的业务目标不折扣；若实现中采用其他折扣，必须显式声明目标已经变化。

## 9. PPO 训练与只读评价必须分成两条路径

### 9.1 训练路径

`trainer` 负责 on-policy rollout、旧策略 log probability、critic、advantage、更新、验证与 checkpoint。它不负责未来订单生成、不直接改 solver，也不在 update 时重新执行历史物理动作。

记录策略版本，保证一批 PPO 数据对应可追踪的采样策略。多个采样 worker 的结果只在一致的更新边界使用；不能把任意旧 checkpoint 的经验混进来，悄悄变成未定义的离策略训练。

Actor 输出的有界动作、对应 log probability 与 decoder 实际接收的动作必须一致。若使用 tanh 变换，概率计算应包含相应变换；不能训练记原始高斯动作，执行时却额外裁剪。动作 Schema、尺度与 mask 的变化也需要切断旧 rollout 的使用。

### 9.2 评价路径

评价只加载冻结 Actor、编码器、标准化、动作 Schema 和底座，关闭 dropout 与参数更新。正式评价不能更新观测均值方差、critic 或先验。确定性／随机评价模式、policy seed 需要写入配置；网络均值为零与实际抽样动作等于零不是一回事。

推理包不必加载 optimizer 或训练 buffer。这样平台运行 PPO 评价时，不会意外开启训练、占用大量训练显存或改变模型文件。

### 9.3 随机性和进程

scenario、训练初始化与采样、评价 policy、solver 的随机种子分别记录。训练内部若有数据抽样、minibatch 和 worker 子种子，应由明确规则派生，不能共用一个全局随机数流。

Windows worker 使用进程安全的入口；solver 对象在 worker 内创建，不跨进程传递，不在模块 import 时创建线程或训练环境。并行预算需同时约束“平台并行训练 run × 每个训练 run 的采样 worker × solver threads”，防止三层并行相乘。需要改变线程环境变量时，在本次 worker 启动环境中设置，不修改整台机器或其他 run。

当前依赖已包含 Torch 和 Gymnasium；首版不应为了新库无理由升级现有锁文件。若需新依赖，作为独立受审查变更固定版本，并跑既有回归。导入 single manifest 或运行非 PPO 策略时不应触发 Torch/CUDA 初始化；网络与训练 backend 使用惰性导入。

## 10. checkpoint 与输出：模型身份必须完整

只保存 `actor.pt` 不足以复现。推理结果还取决于观测字段顺序、hub 编码、归一化、动作尺度、MC 底座和 decoder。

建议 checkpoint manifest 至少记录：

- `family=single`、算法及网络版本。
- observation、action、decoder、reward、环境协议版本。
- hub 顺序、输入维数、mask 和标准化状态。
- 物理参数、求解器版本／参数、信息集合、完整冷却规则。
- MC 底座、先验和数据 split 的身份与 hash。
- 训练 seed、已完成 episode／update 数、模型选择依据。
- 代码快照、依赖锁、所有 artifact 的 hash 与相对路径。

建议区分两类包：

| 包 | 内容 | 用途 |
|---|---|---|
| 推理 checkpoint | Actor/编码器权重、标准化、动作配置、底座及兼容性元数据 | 正式评价与部署 |
| 训练恢复 checkpoint | 上述内容，加 Critic、optimizer、调度器、随机数、数据流位置与必要 buffer | 恢复同一训练过程 |

第一版优先支持**完整 rollout/update 边界**恢复；不承诺可在任意 MILP 求解中途精确续训。恢复时若参数、数据、接口或底座不匹配，应拒绝；用户选择换配置，应创建新训练 attempt，而不是覆写旧身份。硬件差异下的位级确定性也不能无条件保证。

网络加载使用受控的权重与配置格式，不执行 checkpoint 指定的任意 Python 对象或 import 路径。相对路径必须解析在本 checkpoint 目录内，禁止越界。权重格式和加载参数按最终选用库版本验证。

新输出放在新 attempt 或平台 run 目录中：训练曲线、正式评价、物理 trace、PPO transition 日志分别存放。推荐额外记录每期 observation hash、signal、实际系数、完整动作 hash、reward、累计账本与 solver 状态。不要改变旧 `single-trace/v2` 字段含义；新增训练信息可先用独立 sidecar，以共享 transition ID 关联。

## 11. 平台接入：少量明确注册，不扩散 PPO 分支

### 11.1 第一阶段：平台零改动的研究入口

先由 `experiments/cfs_ppo_strategic_control` 调用新训练和评价 runner，完成小规模端到端验证。其配置、数据 split 和产物独立；既有平台按钮、默认 framework 和旧流程完全不动。

此阶段的“零改动”是旧运行入口不变，不意味着整个仓库快照 hash 不变：新增源码当然会改变新版本的源码快照。这不是历史结果被污染，前提是原冻结文件没有改写，且历史实验继续引用其自己的快照。

### 11.2 第二阶段：单一 family 入口接入

建议增加显式新 ID，例如：

- framework：`ppo_strategic`。
- training controller：`train_ppo_strategic`。
- evaluation controller：`ppo_strategic`。
- evaluation signal component：`ppo_release_signal`，引用 `checkpoint_run`。

最后一个组件可先利用现有 `value_function` 插件引用承载，以兼容平台结构，但 UI 与数学展示必须明确它是战略信号，不是 RL Critic，也不是原有资源边际真值。训练阶段按当前平台协议使用 `value_function=zero`，意思是该字段不加载评价 checkpoint，并不表示训练 decoder 没有 MC 底座；底座由新训练配置显式引用。

必须同步处理以下入口：

| 接入位置 | 最小必要变化 | 保持不变 |
|---|---|---|
| `single/plugin.py: REGISTRY/FRAMEWORKS` | 增加 PPO 参数 Schema 与 framework，惰性分发 | 所有旧 ID、默认值与排序约定 |
| `is_training()` | 识别新训练 controller | 原监督训练含义 |
| `execute()` | 对新 ID 调用新 runner，其他继续旧 runner | 不改 `matching()` 的旧主循环 |
| `validate_selection()` | 校验新训练 split、checkpoint、信息集合、账本与 decoder | 旧组合的校验行为 |
| `checkpoint_reference()` | 返回新 signal 的 checkpoint 引用 | 旧 learned component 引用 |
| single 的数学与模板 | 增加 PPO 专用说明与明确训练依赖 | 旧页面的公式与默认模板 |
| single 分析入口 | 排除 PPO 训练 run，接入评价格式与附加诊断 | 旧指标定义与解析方式 |

新配置放在 PPO 包内，不把学习率、clip、网络层数加入通用 `MatchingParameters`。公共批量协议可扫描新组件声明的参数；物理参数与训练参数保持可区分，训练 seed 也不能变成新的“实验条件值样本”。

同样重要：新增 PPO 不意味着 single 全部页面都默认有训练。只有该 framework 声明训练依赖；myopic、MC、rolling 等继续直接运行。

### 11.3 复现导出是一个真实缺口，不能漏掉

当前 `experiment_core/reproduction.py` 主要按固定名称收集 `checkpoint.json`、`weights.json`、`weights.pt`。如果 PPO 的底座或标准化放在其他文件，只添加注册会得到不完整的复现包。

不建议在通用模块写 `if algorithm == ppo`。第二阶段可添加一个可选的 family artifact 清单接口：single 返回有版本、有 hash 的 checkpoint 依赖；公共打包器只负责校验路径、体积和打包。未实现该接口的 family 继续原行为。导入验证也要同步支持相同清单，不能只修导出。

另一种初期做法是 PPO 研究入口自己导出完整研究包，暂不宣称网页的一键复现已支持 PPO。推理复现包与大体积训练恢复包应分开；不要悄悄取消现有导入体积与安全限制。

平台当前 descriptor 的 `resume` 仍为 false。拥有内部训练 checkpoint 不等于平台已支持“恢复任务”，这一能力必须等入口、取消、调度与校验全部接通后再宣告。

## 12. 分析隔离：训练进步不是正式实验样本

训练 episode、validation checkpoint 选择与正式 rollout 属于不同数据层。训练曲线可以展示，但不能进入“多实验利润比较”的正式样本池。

评价记录应保留 scenario/instance、policy seed、training seed、checkpoint 与底座银行身份；共用测试场景的多个训练模型不能按运行行数扩大独立样本量。训练多 seed 的分层统计先放在 PPO／本轮专用分析模块，不为赶进度改变公共分析的既有聚合语义。

当前分析适配的物理比较键还包含代码快照身份。新版本 PPO 与历史冻结基准可能因此无法直接在平台成对比较，这是保护机制，不应为了出图删除源码校验。应在新第八轮独立目录中、用同一验证后的环境版本运行新的基准对照，或提供明确的经核验桥接记录；历史正式结果只作背景，不伪装成新版本同源样本。

结果正文必须区分 raw USD reward、标准化训练 reward、critic loss 与真正的评估利润。训练 loss 下降或策略熵增加，均不能替代正式利润证据。

## 13. 用什么测试证明“不影响其他模块”

建议分层验收，而不是只跑一次 PPO 能否收敛。

| 层级 | 必需检查 | 防止的问题 |
|---|---|---|
| 导入与依赖 | 不选 PPO 的 manifest/schema 不初始化训练；禁止逆向 import | 旧功能突然需要 GPU、慢启动或加载失败 |
| 信息边界 | 修改隐藏未来但保持公开状态时，Actor 输入和当前 decoder 相同 | 未来泄漏、scenario ID 捷径 |
| 对象隔离 | 策略／decoder 操作不改变 live state；多个 env 不共享可变对象 | worker 串状态、复现实验漂移 |
| 系数与动作 | 同查询恒定、负值／prefix／终端正确、输入过期拒绝 | 一次求解多次采样、重复资源计价 |
| 零信号回归 | 同底座同 planner 的目标、动作与轨迹恢复 | 加 PPO 外壳改变基准 |
| 物理与账本 | compile、容量／承诺、逐步 replay、reward telescoping | 学会利用执行或金额漏洞 |
| PPO 数值 | log probability 与变换一致、GAE 终端正确、有限梯度 | 环境正确但训练目标实现错误 |
| 推理隔离 | 评价不更新任何权重、标准化或底座 | 正式集参与训练 |
| checkpoint | 版本／字段／hub 顺序／物理参数／hash 不匹配拒绝 | 权重可加载但语义错误 |
| 恢复与失败 | update 边界恢复；取消／超时无伪造 COMPLETED | 残缺训练当成完整成果 |
| 平台兼容 | 旧 spec 解析与 schema 默认值、旧 framework 路由不变 | 注册引入跨模块回归 |
| 家族隔离 | single 的 checkpoint 不被 legacy/BHH 接受，反之亦然 | 论文模型和训练产物混用 |

单元测试可以使用假策略、假 planner 和微型状态，不必每项都启动神经训练或商业求解器。集成测试使用固定小场景；科学评价再使用独立正式场景。训练曲线达到某个利润不应成为脆弱的单元测试断言。

不要在原冻结实验目录输出新的 XML、trace 或 QA；旧数值入口也不为测试而重跑。历史回归在新临时／测试输出目录内执行，并核对旧文件 hash 未变。

完整平台测试期间应冻结源码与报告编辑，避免仓库快照一致性测试因同时新增文件而失败。失败记录保留，不能把随后通过的结果覆盖成“从未失败”。

## 14. 实施顺序与改动预算

我建议分四个交付边界，而不是同时大改平台和训练器：

1. **先实现领域适配，不训练。** 新 `strategic` 接口用固定／零信号跑完整 episode，验证观测、动作、奖励与冷却。旧文件原则上只读。
2. **再接 PPO，仍用独立入口。** 小规模检查策略采样、更新、恢复和只读评价；记录实际求解成本。不要把训练循环直接接到 UI 请求线程。
3. **完成预注册数值试验。** 冻结代码与配置，生成独立产物，执行基准与多训练 seed 比较。不得在正式结果后改奖励、动作尺度或模型选择规则。
4. **最后接入平台。** 以少量 single 插件注册、分析桥接和通用 artifact 扩展提供 UI 能力；完成全平台回归后才能称“不影响原功能”。

第一阶段不需要改 `engine/mt_prototype.py`、`planning.py`、`accounting.py`、现有 MILP 约束、legacy 或 BHH。第二阶段若触及这些文件，应先说明为何适配层不能解决，作为独立审查项，而不是边开发边顺手重构。

还有一项版本约束：部分既有实验冻结清单包含共享源码。未来一旦改动 `single/plugin.py` 等受冻结引用的文件，旧实验在当前 checkout 的来源校验可能正确地拒绝通过，即使旧算法行为未变。应先核对依赖清单，在新开发版本／工作副本集成，保存原冻结代码快照；不能通过更新旧 hash 或放宽保护来让新旧身份看起来相同。

“不影响”最终包含两种保证：旧版本可按原身份复现；新版本运行旧算法通过明确回归。两者不同，不能只用其中一个代替另一个。

## 15. 我的取舍

最需要避免的是两个极端：一个是在现有 `runner.py` 中不断添加 PPO 特判，另一个是复制整个 single 仿真器，得到以后无法同步修复的第二套业务引擎。

我推荐中间路线：

> **复用权威物理与账本，新增独立交互环境；复用值感知 MILP，新增确定性信号适配；PPO 只拥有网络和训练，平台只拥有调度与产物。**

这样，即使第八轮 PPO 没有稳定利润优势，留下的环境、decoder、奖励和审计接口仍可支持简单策略搜索、其他 RL 算法或有限候选策略；原有实验也不会因一次研究尝试而被重构或污染。这比为 PPO 单独搭一套平台更有长期价值。

### 本次核对的代码入口

- [模型族注册](../experiment_core/families.py)、[公共 runner](../experiment_core/runner.py)、[插件解析](../experiment_core/plugins.py)。
- [single 插件](../model_families/single/plugin.py)、[现有 runner](../model_families/single/runner.py)、[分析适配](../model_families/single/analysis.py)。
- [窗口协议](../model_families/single/rolling/contracts.py)、[观测及滚动控制](../model_families/single/rolling/controller.py)、[planner 注册](../model_families/single/rolling/registry.py)。
- [价值接口](../model_families/single/value/contracts.py)、[值感知 MILP](../model_families/single/value/route_milp.py)、[动作编译](../model_families/single/planning.py)、[实现账本](../model_families/single/accounting.py)。
- [批量参数协议](../experiment_core/batching.py)、[平台调度](../experiment_core/service.py)、[复现导出](../experiment_core/reproduction.py)、[依赖定义](../pyproject.toml)。

本文件没有宣称上述接入、训练恢复或 UI 支持已经完成；实际实施仍需按此边界实现和验证。
