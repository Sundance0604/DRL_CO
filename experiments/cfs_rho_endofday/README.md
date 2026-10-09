# CFS / rho / end-of-day study

本实验只实施物理网络 pickup radius 与日终真实核算。使用 single 的现有 `experiment_core.runner.execute`；不新建平台或修改其他模型族算法。正式实验使用 myopic / V=0，不训练。

## PowerShell 复现

在本仓库根目录（含 `.venv`）：

```powershell
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=workspace/cfs-task-qa/reproduce-unique
.venv/Scripts/python.exe -B -m experiments.cfs_rho_endofday.run_experiment
.venv/Scripts/python.exe -B -m experiments.cfs_rho_endofday.analysis
```

正式配置为 [main.json](configs/main.json)。默认10个连续场景种子1000–1009，每场景40个真实CFS抽样shipment，4区域；参数设置不是统计样本。policy_seed=0，training_seed不存在。原始本地SQLite需在指定路径可读；否则报错，不使用替代/合成需求。Gurobi许可也必须可用。

快速单区域真实数据 smoke（明确不是正式10-seed统计）：

```powershell
.venv/Scripts/python.exe -B -m experiments.cfs_rho_endofday.run_experiment --limit-seeds 1 --region la_sf --output workspace/cfs-smoke
```

已有正式输出可同配置、同source metadata/hash、同实现hash续跑；改源码/配置必须指定新 `--output`，旧结果不会被静默覆盖。改数据位置时用新的配置文件并显式 `--config`。所有输出位于仓库内 `results/cfs_rho_endofday`；不会写原始CFS或Intercity仓库。

## 数据与核算含义

- CFS观测：OD、磅重、货值、大圆距离、mode、SCTG、survey weight及源ID。读mode111/112、domestic、50mile以上、不超过40,000lbs、每方向至少10条的metro OD。
- 参考/适配Intercity `agent/paper-solution-approaches` 的area code、indexed SQLite、双向筛选、weighted reservoir及追溯机制，不复制其业务模型。
- 同一pair方向配平，然后按log(U)/WGT_FACTOR加权无放回抽样；不是全国自然OD份额或总体估计。
- **生成** booking、ready、deadline、fleet、负载单位、tariff/cost及geometry-informed metro network。lat/lon是公开配置的近似代表城市位置假设，不是CFS centroid；geometry stretch不是观测道路里程。
- 运费按配置的network miles与weight生成，货值只保留为observed字段、不充当fare。
- `pickup_radius` 单位network miles，rho0仅own hub；`pickup_rule=legacy_adjacency`仅作旧规则诊断。既有backbone、在途扩展、delay规则保留。
- decision objective/booked账不改变。事件段记录实际载重及里程；仅delivery计fare，T后只执行commitment，直到完成。候选数为过滤后的可服务/续行路线数之和。
- cutoff使用[0,T)，恰在T到达属于冷却期；冷却期车完成后不继续收idle waiting。
- 实际订单等待报告booking→pickup或cancel/T，已有backlog费用沿用；不额外发明已分配订单等待费。
- completed+cancelled+unfinished_orders=arrivals；unfinished_orders包含unfinished_committed和pending_unassigned。冷却只保证已承诺全部完成，不抹掉未分配订单。
- 对`single-realized-v1`必须设置`terminal_policy=cooldown_realized`，`operating_profit`代表实际利润。旧`legacy-assignment-v1/report_pending`仍为booked，禁止混合统计。

## 文件索引

| 文件 | 作用 |
|---|---|
| `model_families/single/network.py` | 几何图、确定性物理Dijkstra、rho资格 |
| `model_families/single/accounting.py` | 实际事件账、自动完成上界、纯执行冷却 |
| `model_families/single/cfs_adapter.py` | 只读indexed CFS抽样/模型字段适配 |
| `run_experiment.py` | 对冻结场景执行全部paired setting，调用共享runner |
| `analysis.py` | 条件/配对统计、7张图、自动生成最终报告 |
| `results/cfs_rho_endofday/manifest.json` | source SHA/metadata、版本、源码hash、完整配置、seed、完成状态 |
| `raw/*.source.json` | 抽中observed完整字段与generated订单一一对应 |
| `raw/*.dataset.json` | 冻结平台dataset/v1场景 |
| `raw/runs/<run>/` | resolved spec、scenario、metrics、trace每t、双ledger、state |
| `processed/runs.csv` | 每run所有标量指标，scenario_hash证明配对一致 |
| `summary.csv` | 每条件/指标n、mean、SD、median、t95%CI |
| `processed/paired_differences.csv` | 同seed对rho0差值，偏态时额外bootstrapCI |
| `processed/details.json` | 日终截止与0/5/10/20/full冷却扫描 |
| `processed/horizon_probe.json` | 单个真实来源订单的booking时间平移诊断，不当作额外独立场景 |
| `processed/plotting_config.json` | 字体、尺寸、色/符号、输入hash、绘图CI口径 |
| `reports/cfs_rho_endofday/` | 事前audit、最终报告、测试记录、7张PNG/PDF/SVG |

论文图为7.2英寸双栏、400dpi PNG及PDF/SVG矢量，色盲友好色、不同marker/linestyle支持黑白阅读。图表数据另存，不依赖网页截图。

本任务不承诺rho单调提升利润，不将模拟收入解释成实际经营收入。真实道路/时间/运费校准、其他算法比较不在本次范围。
