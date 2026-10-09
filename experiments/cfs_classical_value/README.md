# 第三次数值实验：经典 continuation value

本目录不修改前两次实验，也不训练 DRL。六类 ValueProvider 位于
`model_families/single/value/`；统一接口接受释放时刻、hub、整数资源槽和在线摘要。
规划器在求解前计算常数系数，MILP 用有序前缀槽避免相同 q 的重复计价。

## 运行

在仓库根目录的 PowerShell：

```powershell
$env:PYTHONHASHSEED='0'
$env:PYTHONIOENCODING='utf-8'
.venv/Scripts/python.exe -B -m experiments.cfs_classical_value.run_experiment
.venv/Scripts/python.exe -B -m experiments.cfs_classical_value.analysis
```

可用 `--through inputs|targets|fit|pilot|formal|all` 分阶段检查。相同源码、
配置和模型身份才允许续跑；如改变数值实现，必须先使用新输出目录，不可覆盖旧结果。
当前 OUTPUT 常量固定为 results/cfs_classical_value；复现不同协议时先明确更改
OUTPUT 并保存新配置，不能放宽身份检查。

新增single/value文件会被旧实验“扫描整个single包”的身份检查识别。旧结果与
保护模块仍保持不变，但如需重跑旧研究，请使用它的独立源码快照和新目录；
不要改写旧manifest使旧数值假称来自当前扩展工作树。

## 模块

- common.py：源码快照、基线保护、CFS只读哈希、种子与冻结输入。
- simulation.py：原物理执行/双账本；未来记录仅在调度器或专用 oracle 中。
- build_mc_targets.py：同世界 q 与 q−1 的实际利润配对；自然状态、主动时间节点。
- build_fluid_value.py：构造集需求的分数 LP dual surface，不是真实未来。
- fit_models.py、fit_linear.py、fit_chebyshev.py、fit_spline.py：经典可解释逼近。
- run_value_policy.py：全策略 ladder 与逐 t 记录；run_terminal_value.py 共享该入口。
- analysis.py：独立预测评价、配对统计、稳定性诊断、论文图和报告。

## 解释边界

M+V 用 H1 窗口实现同一单期可行集，V0仍调用原 Myopic。RH+V 的 tail
只计每车最后进入 sink 的资源；不在每次中间到达重复领取价值。
同 hub/time 的 q 只能按1..n领取，即使估计非单调也不挑更高槽。

自然 MC 状态先去掉查询 hub 的自由 idle 车辆，保留全部承诺；再分别加
q、q−1辆。两世界复用未来、成本、seed、continuation，评价到完整 cooldown。
T时新增自由车辆不能触发新控制，故边际零；这不意味着原承诺账本为零。

统计样本是 scenario seed，不是参数、MC路径或资源槽。构造/验证/正式
互不重叠；模型在正式策略利润产生前冻结。第二条 MC 路径来自同split donor，
是经验未来重采样近似，并非正确估计了现实条件分布。

Fluid 是刻意简单的离线初始供给 surface：允许分数载荷、空调位和无锚点，
不响应当前 backlog。它是乐观结构对照，不等同于真实整数资源价值。
回归用当前可见摘要，候选释放时间可晚于当前；不更新候选动作后的 backlog，
故 RH-terminal 属于冻结摘要近似，而非完整 post-decision state value。

原始输入、双账和 trace 在 raw/；targets/保留两条路径的paired明细及解析终端
锚点；models/candidates/保留raw/anchored、ordinary/natural全部候选；
manifests/model_freeze.json 记录正式前冻结；processed/供 Python 重新作图。

## 本次结果与查看入口

完整研究报告在 `reports/cfs_classical_value/classical_value_report.md`；图在其
`figures/`，每图提供 PDF、SVG 和600dpi PNG。`results/cfs_classical_value/manifest.json`
是本次数值运行的身份与完成状态，不要用平台旧实验的manifest替代它。

Python多实验绘图优先读取 `processed/paper_policy_samples.csv`（每行一个条件下
一个scenario和一个策略），以及 `paper_value_predictions.csv`（独立价值评价）。
利润比较必须按 `condition + scenario_seed + scenario_hash` 对齐；不同参数配置
不是同一统计样本。`plotting_config.json`记录图形配置与输入哈希，
`trace_field_semantics.json`区分真实利润与窗口优化目标。

全量测试应把临时目录放在被忽略的workspace内，避免平台源码扫描读取Windows
正在占用的coordinator.lock；固定hash seed用于既有基线的确定性复现：

```powershell
$env:PYTHONHASHSEED='0'
$env:PYTHONIOENCODING='utf-8'
.venv/Scripts/python.exe -B -m pytest -q --basetemp=workspace/classical-value-check --junitxml=results/cfs_classical_value/processed/pytest-full.xml
```

本次新增provider只接入独立研究runner，尚不是网页中新建的可选framework。
本次没有训练神经网络、DRL/SAC，也没有修改BHH模型。
