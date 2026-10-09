# 第七轮：完整联合动作价值接入与独立 MC 诊断

本目录是独立实验包，不修改 single 的物理、默认策略或历史实验。当前状态以 `manifest.json` 为准。`NUMERICAL_COMPLETE_ANALYSIS_QA_PENDING` 仍需要核验、分析、图表和真实视觉检查。只有 `manifests/final_manifest.json` 的 COMPLETED 才表示全部完成。

## 一键数值入口（Windows PowerShell，仓库根目录）

```powershell
$env:PYTHONHASHSEED='0'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
./.venv/Scripts/python.exe -m experiments.cfs_joint_action_value_diagnostic.run_experiment --stage prepare
./.venv/Scripts/python.exe -m experiments.cfs_joint_action_value_diagnostic.run_experiment --stage all
```

共享一个 Python 环境，最多3个数值 worker，各 solver 单线程。真实 Gurobi OPTIMAL 才可进入结果，不把 time-limit incumbent 当样本。已完成数值阶段用身份 hash 缓存；不启动第二个主进程。现金/泄漏/物理故障触发 stop 时保留本 attempt，不自动修复混合结果。入口失败与受控停止分别保留。

## 样本与比较口径

主工况 CA80/3、W40/8，W80/3 为保护性对照；W20/8 只做12项历史回归的一部分。两个来源 M0 和 pi1 每个 seed 只采一个 t，按 [3,8,12,16,20]循环，不挑有利时刻。诊断20场景/来源/工况，正式40场景/来源/工况。主 family 是6项 select32−pi1-price，scenario 为独立单位，双侧 Holm6，原始 t95 与10000 bootstrap 区间未作多重区间校正。

pi1 明确是原 A 的首个 MC 改进，即 A2 的 U1，使用原 A V0。它与 B1 不是同一个表；B1 的全部 hub 加密标签设计不同。本轮 B2 控制只使用已冻结 exact-accounting attempt，不复用原失败 B 缓存。D4 使用已冻结 alpha=.25。

候选最多6个唯一完整联合动作：M0、pi1、D4、B2、最多两个确定性 H1 MILP 备选。等价动作合并保留标签。备选生成时可限制一辆车的路线，但强制评估时整个 joint action 的所有路线、assignment、等待及取消均固定，下一期才恢复同一 M0。小样本 pi1 后续策略敏感性单列。

## 独立未来与限制

每状态所有动作共用32条 selection CRN（8条是其前缀），先冻结动作选择，再用独立64条 assessment。每个 role 使用 condition/state/role/replicate 的 SHA namespace。assessment 不重选动作。价格 L_V 与真实增量 J 分开，J 不额外加入当期 g 或 value bonus。

未来分布是 **surrogate-distribution rollout**。固定 B construction 8300–8359 的经验 marks 与平均到达率，用 Poisson 未来订单数和 iid marks，保持公开订单、既有承诺及物理规则。它不是原按 OD 平衡、固定 N、不放回生成过程的严格条件分布，不读取测试场景隐藏 shipment/实际剩余 OD 或 N。结果不能推广为现实条件期望最优。未来资源交互仅注入当前真正 idle 车辆，跨时点自由注入明确未实施。

## 阶段与输出目录

- `config`：固定参数、未来先验（仅既有 construction）、预算。所有正式修改必须另建版本。
- `manifests`：源代码快照、seed 检查、状态→动作→未来→选择冻结顺序、预算、阶段与最终状态。
- `raw/states`：完整车辆/订单/承诺/事件/账本底座；已揭晓订单和源策略前缀，不向采样器暴露隐藏订单。
- `raw/actions`：全部唯一 joint action、来源、精确 booked 金额和每个历史价格下的 L_V、tie-break、release 分组。
- `raw/futures`：每条 future 的独立 role/hash/完整订单，可重构所有世界。
- `raw/worlds`：内容寻址压缩 replay，每个世界初态、动作、未来、continuation、逐 t 执行及最终账本可追溯。
- `raw/selections`：先于 assessment 的 frozen choice 与 selection path 收益。
- `raw/banks`、`tables`、`raw/deployment`：4个独立银行各60场景的一次 E(M0)/一次改进；全部银行在同40部署场景配对，不选优胜银行。q/time/hub 采样同原 A，每个银行一致。
- `processed`：任务书的12组科学 CSV/JSON；每列保留数值单位，原始世界不覆盖。缺失项写未完成原因，不能做空白图凑组。
- `qa`：定向测试、历史回归、全世界独立重放与 hash、历史/CFS SHA、图形真实视觉核验。
- `figures`：最多6个任务指定组，PDF/SVG 向量、600dpi PNG、source CSV/config，以及图册。

`semantic_audit.md` 记录已确认一致、已声明代理和未解决之处。它不会把疑点预先写成错误。

## 交付入口

```powershell
./.venv/Scripts/python.exe -m experiments.cfs_joint_action_value_diagnostic.delivery --stage analysis
./.venv/Scripts/python.exe -m experiments.cfs_joint_action_value_diagnostic.delivery --stage worlds
./.venv/Scripts/python.exe -m experiments.cfs_joint_action_value_diagnostic.delivery --stage history
```

随后仅运行本轮相关测试与必要历史控制，保存真实 XML；生成/渲染后实际审阅所有图和图册页。所有 mandatory QA 通过才 `delivery --stage freeze`。不能伪造视觉或测试通过。完成后本目录也永久只读；不重复入口、分析、绘图或冻结。

### 实际最终相关测试命令

下列命令只覆盖新实现及必要历史金额/slot/终端/物理控制，不扩大为平台全套测试。
测试期间不修改数值源码。最终 XML 必须来自实际成功运行，而不是复用预检。

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_joint_action_value_diagnostic.py tests/test_joint_action_value_delivery.py tests/test_integer_action_accounting.py tests/test_resource_value_state_bucket_exact_accounting.py tests/test_classical_value.py::test_collision_slots_are_prefix_not_repeat_or_cherry_pick tests/test_classical_value.py::test_free_resource_terminal_skips_provider_but_not_accounting tests/test_classical_value.py::test_nonfinite_coefficient_is_not_silently_zero tests/test_classical_value.py::test_zero_value_h1_matches_original_current_objective_and_is_executable tests/test_classical_value.py::test_counterfactual_removes_only_free_query_supply_and_reindexes_safely -q -p no:cacheprovider --junitxml=experiments/cfs_joint_action_value_diagnostic/qa/pytest_final.xml
./.venv/Scripts/python.exe -m experiments.cfs_joint_action_value_diagnostic.delivery --stage tests
```

图形视觉检查用 bundled runtime 运行 A2 只读通用 `pdf_qa.py --directory` 指向本轮目录，
只写本轮 `qa/pdf_render`，不能渲染写入A2。真实审阅所有最新单图和不相同atlas页后，
才允许加 `--visual-reviewed`。全部世界与历史核验可各启动一个交付进程，但不能重启数值主进程。

### 补充非数值审计

`fallback_economic_supplement` 从各完整原 B2 求解记录恢复所有潜在coefficient分母，
分别统计各selector选中prefix与相对price动作改变的prefix；使用绝对USD金额防止正负抵消，
terminal、非terminal与空分母均明确。`continuation_sensitivity_supplement` 单列前4个
diagnostic seed的pi1后续策略；这些次分析不改变冻结主family、selector或统计口径。

`figures.py` / `delivery.py` 是预声明的非数值交付层，明确排除在数值source freeze之外；
它们的最终代码SHA仍写入交付manifest。时间数据未单独识别序列化与fit aggregation，
不能在报告中虚构精确分解或把inclusive pipeline与solver/replay相加。
