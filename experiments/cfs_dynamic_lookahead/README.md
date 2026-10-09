# CFS dynamic lookahead：第二次数值仿真

范围：原网络距离rho、原CFS adapter和full-cooldown realized accounting不改；V=0，不训练，不实现RL/ValueProvider/multi-backbone。

## 本地输出

- `results/cfs_dynamic_lookahead/phase0/`：修改前baseline40次精确复现、源码快照、hash。
- `results/cfs_dynamic_lookahead/raw/datasets/`：每region/N/K/seed冻结JSON和完整observed/generated对应。
- `raw/runs/`：spec、manifest/hash、metrics、trace每t、两份ledger、状态与运行时间。
- `regime_map_results.csv`：3区域×3需求×3车队×3半径×10seed，共810次myopic。
- `sanity_results.csv`：LA–SF sanity，两个正rho标签可能相同，不当独立样本。
- `representative_regimes.json`：确定规则选择的角色、工况、pilot gain。
- `rolling_pilot_results.csv`：同10 pilot seeds的rolling运行预算校验，不进入正式CI。
- `rolling_results.csv`：独立30 seeds的正式myopic / known_supply / oracle_preview矩阵。
- `full_preview_results.csv` / `H1_diagnostic_results.csv`：小样本诊断，单独存放。
- `summary.csv` / `paired_differences.csv` / `horizon_increments.csv`：按条件、配对和H增量统计。
- `traces/western_case.*`：最大改善seed的事后解释，不能当总体或单动作因果证明。
- `reports/cfs_dynamic_lookahead/`：Markdown最终报告、7组PNG600dpi/PDF/SVG论文图。

大型raw运行记录与冻结数据保存在本地并被Git忽略，避免平台把所有数值产物打进每次研究源码快照；没有删除或上传它们。共享完整结果时应另带raw目录，不能仅拿CSV声称拥有完整trace。源快照/锁文件、CFS身份、生成配置和连续seed使本机可以重新生成。

## 运行顺序

在仓库根目录使用本项目环境（需要本机CFS SQLite和Gurobi许可）：

```powershell
$env:PYTHONHASHSEED = '0'
.venv/Scripts/python.exe -B -m experiments.cfs_dynamic_lookahead.run_regime_map
.venv/Scripts/python.exe -B -m experiments.cfs_dynamic_lookahead.run_rolling_horizon --pilot
.venv/Scripts/python.exe -B -m experiments.cfs_dynamic_lookahead.run_rolling_horizon
.venv/Scripts/python.exe -B -m experiments.cfs_dynamic_lookahead.analysis
```

本轮phase0由实际执行并核对旧frozen数据的40次run生成，不假装旧CSV等于新复现。新output需要先通过phase0 gate；可读取已保存的baseline源快照恢复当时实现，再重放原CFS frozen数据。`prepare()`拒绝不相同的source/config/运行代码；源码有改动应新建输出目录，不能覆盖旧产物。

正式运行固定`PYTHONHASHSEED=0`，避免Python集合的约束遍历次序随worker进程变化；Gurobi Seed=0、Threads=1。Pilot用于预算验证，运行时采用解释器默认hash seed，其数据不并入正式CI；manifest明确区分两者。完整同算法/同信息复制时也需要固定此环境。

验收采用相同固定环境：`$env:PYTHONHASHSEED='0'`后运行pytest。本轮107项通过。默认hash环境曾使canonical Test C失败：零等待/积压成本下立即与稍后服务同为最优，但该断言要求立即服务，因而对tie选择敏感；并非凭空出现未来车辆。为保持冻结源码身份，本轮不在仿真中途改写该fixture。`processed/qa_validation.json`保留首次失败及固定环境重测证据，未来测试版本应改为检查window最优目标/物理可行性，而非强制某个非唯一当前动作。

数值执行源码和后处理源码分别有hash/zip；修复图表/报告不应让原数值结果被冒称重新运行。报告由analysis自动生成，图的数据和绘图配置一并保存。snapshot记录工作树实际实现，不假称未提交代码已在HEAD。

## H、未来信息与暂缓的精确定义

H为平台已有的决策期数，窗口`[t,min(t+H,T))`，k固定1。H1与当前单期M_t目标在canonical regression一致（同目标存在不同tie时动作不必唯一）。RH-Supply只访问当前已booking订单及当前承诺确定的未来车辆到达，不接触未来scenario订单。RH-Preview专门以`oracle_preview`标记，额外看到`(t,t+H]`的真实未来bookings，不能在booking前分配或pickup。

由于半开窗口，恰在t+H的booking虽然可见，但没有本窗口决策slot；H1的preview不能靠提前执行未来订单制造收益。full-preview使用H=T，随时间自然clamp为剩余完整决策期。未揭示batch ready仍采用当前upper bound，不泄露未来实际值。

两种RH用相同route-macro-action MILP，承诺不撤销，未来计划后缀不写入真实状态。窗口内部不额外规划途中接单，下一期replan再考虑原single前缀兼容动作。该受限模型不是所有M_t序列的精确展开，oracle不是全球最优上界。窗口objective/gap不能与正式realized profit/全局gap混称。

deferral是执行后的审计，不反向改变目标。当前有可行锚点coalition却不分配/取消的订单标为deliberate，累计其实际暂缓期数和最终车辆；另记录no_vehicle、infeasible和cancellation。每期所有committed hub arrivals记录ETA；单订单`future_supply_available`只判断已知最终idle release可行，是保守信息，不会把受既有路线约束的中间站车辆假当空车。成功率分母为有deferral订单；没有deferral的run记missing/not-applicable，不用0代替。

## 统计与抽样

保持原log(U)/WGT_FACTOR weighted reservoir、eligible OD、方向配平、observed/generated定义。只改变需求量和确定fleet placement。车辆按完整eligible CFS outbound WGT_FACTOR mass排位（不是未来时序需求）：3辆top3，5辆一hub一车，8辆一hub一车再补top3；tie按hub ID。

Pilot seeds=1000–1009；formal为独立的1010–1039。不同N/K/rho/H是condition而非replication。配对先验证相同seed与scenario hash，再作差，分别给mean、sample SD、median、Student-t95%CI；明显偏态增加10000次paired bootstrap，rng seed固定。全矩阵失败保留；有失败时停止推断，不把成功样本偷偷挑出来。失败率、gap和耗时另报告。

本任务没有修改或训练legacy DRL/BHH，没有改变network/accounting/adapter数学语义。CI仅是仿真场景replication uncertainty，不是CFS survey uncertainty，不是实际经营收益。
