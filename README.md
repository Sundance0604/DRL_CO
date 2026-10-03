# DRL_CO · Single-level dispatch 研究原型

当前分支：`single-level-dispatch-model`。文档更新：2026-10-03。

## 本分支的作用与边界

本分支用于单独研究单层匹配模型：将旧“RL 选择虚拟出发城市 + 下层优化”路线改成每期一次求解的 single-level MILP，包含跨枢纽接单、首段 HV 接驳、位置价值、完全信息/流体上界及在线 rollout。

本分支保留旧 `drl_co/` 和 `experiments/`，用于原研究对照与场景工具；新增研究实现集中在 `model/`。**这里的 `model/*.py` 是真正实现，不是平台分支的兼容入口。**

本分支的单层原型主要是非学习型优化/前瞻策略；规则筛选使用“训练场景”不等于实施 SAC 强化学习。旧 `drl_co/rl/` 中的 SAC 是另一条研究路线，不应把两者的假设、指标或结果混在一起。

本分支不包含 React/FastAPI 实验平台、BHH 平台模型、`model_families/` 三族独立包、统一批实验协议或 single 论文分析模块。若要网页、完整逐 t 导出、范围/间隔批实验、多运行统计与 Python 论文图，应使用 `react-experiment-platform`。

本次仅更新 README，保留该分支已有代码与审阅报告，不把平台功能合入本分支。

## 三个分支的区别

| 分支 | 用途 | 推荐场景 |
|---|---|---|
| [main](https://github.com/Sundance0604/DRL_CO/tree/main) | 旧 DRL/SAC 研究基线；调度环境、候选共享 SAC、下层匹配与历史训练评估 | 阅读或复现旧 RL 研究 |
| [single-level-dispatch-model](https://github.com/Sundance0604/DRL_CO/tree/single-level-dispatch-model) | 单层匹配模型原型、上界、在线前瞻与审阅；保留旧 DRL/SAC 作对照 | 单独研究 single 原型及与旧模型的差异 |
| [react-experiment-platform](https://github.com/Sundance0604/DRL_CO/tree/react-experiment-platform) | 当前本地 React 实验平台；三族独立模型包、统一批实验、single 记录与论文分析 | 新实验、参数扫描、逐 t 分析与论文图导出 |

分支是代码版本路线，不是模型族：平台分支内同时有 legacy、BHH、single 三族。其他分支没有平台三族拆包结构；不能照搬平台的路径或启动命令。


## 本分支目录导航

```text
仓库根目录/
├─ model/                    Single-level 真正的研究实现
│  ├─ demand.py              城内需求与首段接驳生成
│  ├─ mt_prototype.py        单层 MILP、仿真、状态与不变量
│  ├─ online_lookahead.py    流体价值、规则筛选和在线 rollout
│  ├─ bounds_check.py        完全信息/流体上界
│  ├─ penalized_bound.py     带惩罚的信息松弛上界
│  ├─ pervehicle_bound.py    按车完全信息上界
│  └─ online_lookahead_chosen.json  原型规则参数，不是 SAC checkpoint
├─ doc/main_v2.tex           单层研究的数学说明
├─ docs/
│  ├─ SINGLE_LEVEL_MODEL.md        代码包与研究入口说明
│  ├─ SINGLE_LEVEL_MODEL_REVIEW.md 审阅、差异、问题和测试证据
│  ├─ DIAGNOSIS.md                 旧 SAC 历史诊断
│  └─ EXPERIMENT_REPORT.md         旧 SAC 历史研究报告
├─ checks_on_original/      原模型虚拟出发/车辆订单 trace 核验
├─ drl_co/                  旧 DRL/SAC 实现与领域/场景工具
├─ experiments/             旧 SAC 的训练、评估、基准与分析
├─ data/                    小型旧样例
├─ results/                 Git 保存的旧研究汇总
├─ tests/                   旧模型回归 + 单层模型测试
├─ requirements.txt         本分支依赖
└─ runs/                    按运行命令生成，默认不提交
```

| 想找什么 | 文件 |
|---|---|
| 单层匹配目标、容量/路线/订单约束 | [model/mt_prototype.py](model/mt_prototype.py) |
| 流体价值与 rollout | [model/online_lookahead.py](model/online_lookahead.py) |
| 首段需求/接驳 | [model/demand.py](model/demand.py) |
| 上界实现 | [bounds_check.py](model/bounds_check.py)、[penalized_bound.py](model/penalized_bound.py)、[pervehicle_bound.py](model/pervehicle_bound.py) |
| 数学模型文本 | [doc/main_v2.tex](doc/main_v2.tex) |
| 代码差异、风险与验证 | [SINGLE_LEVEL_MODEL_REVIEW.md](docs/SINGLE_LEVEL_MODEL_REVIEW.md) |
| 原型运行说明 | [SINGLE_LEVEL_MODEL.md](docs/SINGLE_LEVEL_MODEL.md) |
| 原模型核验 | [reloc_check.py](checks_on_original/reloc_check.py)、[trace_check.py](checks_on_original/trace_check.py) |
| 旧 SAC 对照 | [candidate_sac.py](drl_co/rl/candidate_sac.py)、[dispatch.py](drl_co/environment/dispatch.py)、[lower_layer.py](drl_co/optimization/lower_layer.py) |
| 单层回归测试 | [tests/test_single_level_model.py](tests/test_single_level_model.py) |

## 入口、环境与研究限制

从本分支仓库根目录执行，不要套用原始外层代码包的额外 PYTHONPATH，也不要用平台分支不存在于此处的 `scripts/start.ps1` 或 `model_families/` 路径。

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m model.mt_prototype
```

需要兼容依赖与可用 Gurobi 许可证；原型完整基准会实际求解多个场景，并非瞬时健康检查。更详细的运行和历史环境见模型说明/审阅报告。

成本、城市几何、需求和终点记账有研究原型限制。指派收益不是实际送达收益，计划成本不是实际现金支出；oracle 与 online 策略不能假装具有相同信息集。后续平台提供独立参数与记录协议，并不意味着本分支原型已自动获得这些功能。

原型脚本可能输出终端结果或规则 JSON；不能期待它生成平台的 `workspace/runs/`、`workspace/analyses/`、统一 plotting.py 或论文 ZIP。要查历史样例/结果看 `data/`、`results/`；要查新平台实验切换到平台分支。

## 保留的研究背景与旧 DRL/SAC 记录

下面保留原 README 的单层介绍及旧 SAC 研究记录。历史性能和测试数不是平台验收，也不是本次文档更新后的重新计算结果。旧 SAC 的“快速开始”不等于 single 的训练流程。

本项目研究动态订单场景中的车辆调度：上层策略为每个订单选择“虚拟出发城市”，下层 Gurobi 模型在硬约束下完成车辆—订单联合匹配。仓库从一个无法验证收敛的本科实验，重构成了可测试、可复现的候选共享离散 SAC 基线。

当前结论不是“RL 已经击败优化算法”，而是：原模型首先因为动作没有真正进入优化器而不可能学习；修复数据流与 SAC 后能够稳定学习；进一步改成置换等变的候选共享策略后，跨图表现接近供给启发式和一步 MILP；在车少单多场景中优于短视 MILP，但尚未稳定超过结构化的 supply 基线。

### 单层匹配模型

`model/` 提供一条与现有 SAC 实验并行的研究路线：把“虚拟出发城市 + 下层匹配”改成每期一次求解的单层 MILP，显式计入车辆前往真实出发枢纽的时间和空驶成本，并加入首段 HV 成批接驳、车辆位置价值、完全信息/流体上界及在线 rollout。数学说明见 [doc/main_v2.tex](doc/main_v2.tex)，代码包说明见 [docs/SINGLE_LEVEL_MODEL.md](docs/SINGLE_LEVEL_MODEL.md)。

这套代码仍是研究原型，成本、城市几何和需求均未标定，且存在时域末端记账等已知局限。完整代码审阅、修改原因、风险与测试证据见 [docs/SINGLE_LEVEL_MODEL_REVIEW.md](docs/SINGLE_LEVEL_MODEL_REVIEW.md)。最小验证命令：

```powershell
python -m pytest -q
python -m model.mt_prototype
```

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

项目使用 Python 3.9 的 `pavane` conda 环境验证。Gurobi 需要本机可用许可证。

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
