# Implementation audit（修改前）

审计日期：2026-10-04。基线：`react-experiment-platform`，commit `7b861b28a2d927e8919c1ba79fd3e3c6163a75a2`。本文件先于本任务的实现修改保存。既有未提交的 README 与 single 平台指南不属于本任务，保留。

## 网络逐项审计

| 项目 | 修改前事实 / 定位 |
|---|---|
| hub | `model_families/single/scenarios.py` 调用 `scenario_support/city_graph.py`；随机城市节点、随机车辆初始 hub |
| 坐标 | 随机场景没有地理坐标；指定 CFS 数据和 Intercity data 模块没有公路拓扑或道路距离矩阵 |
| edge | 随机打乱节点形成连通链，再按概率补边；无向图 |
| 边长 | 随机整数 10–30；没有几何约束，不能解释为真实公路英里 |
| travel time | `engine/mt_prototype.py:Net.lt` 每条边 `max(1,ceil(weight/speed))`，路径时间为各边时间之和 |
| shortest path | `Net` 调用 NetworkX 单源 Dijkstra；使用 `weight + seed 随机微扰` |
| distance | `Net.d[(u,v)]` 保存所选路径原始边权之和；`path`、`tau` 保存路径和时间 |
| pickup eligibility | `build_and_solve` 使用 `[own hub] + list(net.H.neighbors(c))`，是直接邻接，不是 hops 上限、距离或时间阈值 |
| candidate routes | 闲置车辆的 pickup 候选隐式依赖 neighbors；既有在途扩展要求固定路径前缀一致，此规则不在本任务修改范围 |
| tie break | 同 seed / 同插边顺序通常稳定；微扰可能改变很接近的物理最短路径，不能保证与插边顺序无关 |

## 日终逐项审计

定位：`engine/mt_prototype.py:build_and_solve / Sim.commit_plan / Sim.begin_period`；`runner.py:matching`。

| 项目 | 修改前事实 |
|---|---|
| fare | assignment 当期全额计入优化 profit，`Sim.J += profit`；delivery 只增加 delivered 计数 |
| route cost | 闲置新路线一次性按计划空驶 + 全 backbone 计费；在途扩展计增量计划距离；不是按已行驶里程计费 |
| waiting | 当前未分配订单每期扣 penalty；idle 扣 waiting_cost，选路线延期提前扣 delay×waiting_cost |
| cancellation | 当期不可再等一期的 unassigned 订单扣 expiry_cost，commit 删除并增加 cancelled |
| T 后执行 | 默认 report_pending 不执行；旧 drain_committed 在 T 后继续调用 solve，仍可能战略优化/调位，不是纯执行 |
| 截断 | drain 的上界由订单 end_time 推导，超上界仍可能静默保留 commitment |
| T 未完成 | pending_unassigned / pending_committed 保留计数，没有真正的 realized ledger |
| summary profit | operating_profit=Sim.J，属于 assignment-time booked profit；actual_distance=None |
| 已分配、已计 fare、未交付 | 可以同时发生；最后一期接受长路线尤其明显 |
| 公平性 | 不同策略可能具有不同期末 commitment / 提前扣费，booked 和 cutoff-realized 均不能替代 full-completion realized；偏差方向未必永远正 |

到决策循环结束时 clock=T，但尚未处理恰在 T 到达的事件。实现应明确事件截止口径，不能混用不同边界。

## 数据审计及复用边界

唯一正式源：`D:/mycodelife/workshop/Intercity_Delivery_System_Design/data/cfs_2022_pums.sqlite`。存在原始 CSV、ZIP 和 2022 CFS users guide。SQLite metadata 记录 37,576,546 行，schema_version=1，OD / OD-mode 索引；MODE 111/112 的 LA→SF 与反向均有实际记录。只读使用，不重建/下载/写源库。

本地 Intercity clone 位于 `D:/mycodelife/workshop/Intercity_Delivery_System_Design`，分支 `agent/paper-solution-approaches`，commit `a0076597899950722b114a8e5127f36992f498a8`。已阅读 `cfs_catalog.py`、`cfs_processor.py`、`sqlite_store.py`、`loader.py`；复用/适配 area code 与名称、双向 OD、索引过滤、log(U)/WGT_FACTOR 加权无放回 reservoir、固定 seed、observed/generated 分栏。**不复制其 quantity clipping、deadline 截到 T 或业务求解模型**。

用户指南第 2、3、5、8、9 页确认：value 是货值，不是运费；weight 是磅，distance 是大圆英里，不是道路里程；MODE 111 为 for-hire truck，112 为 company-owned truck；WGT_FACTOR 是扩展权重，数据有抽样/噪声/截顶，不应把实验 CI 解释为 CFS 全国估计误差。

未发现可可靠复用的 centroid / highway 数据。因此使用明确公开配置的**近似代表城市位置（建模假设，不是 CFS area centroid）**与 kNN+MST、stretch≥1 的 geometry-informed metro network。真实性仅限所抽取的 CFS shipment fields。生成 booking / ready / deadline、fleet、fare、cost 不称为观测或实证校准。

## 限范围实施与验收计划

1. 新增物理距离 rho 规则；直接邻接只以显式 legacy diagnostic baseline 保留。固定路径以精确边权加确定性次级排序，消除随机微扰。
2. 优化 objective 和既有 dispatch / backbone / value 逻辑不变。增加事件账簿和纯执行 cooldown，自动完成上限，超限错误。保留原 booked 指标；论文比较显式使用 realized_profit。
3. 使用 single 现有 runner，增加本任务数据 adapter 和实验/统计脚本；不复制平台或扩展其他模型族。
4. 先通过网络与日终回归，再运行连续、不挑选的 10–20 seeds；区域由真实可用的双向 OD 过滤决定。保存全部原始记录、配置、hash、逐 t 轨迹、图与报告。

本次不实现 delayed matching、future supply、ValueProvider、RL/neural、multi-backbone 或动态方向。
