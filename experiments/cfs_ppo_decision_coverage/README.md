# PPO 控制通道的全时刻覆盖诊断

承接已冻结的 signal_action_v1，不改历史目录或平台默认值。本批只问：六个固定时刻是否遗漏能被既有轴向控制改变的整数决策？工况 W80/3，新开发场景 23004/23005，MC 自然轨迹 t=0…23 全覆盖。每状态为零信号加同一组 10 轴的 ±1 边界控制，不训练、不加载 actor 做候选搜索、不选择利润赢家。最多 1,200 求解、30 分钟、2 GiB，串行 CPU/solver 各一线程，预计 1,056 求解。

主指标为至少一个轴向探针改变完整整数动作的状态数；同批分开计算旧六时刻与其余 18 时刻。主动作身份将顶层 assignment/route/reposition 列表排序，忽略仅列表排列变化，但保留路线内部次序；另存原始 decoder hash 供核验。完整轨迹保留所有状态，包括无订单、无空闲车和终端 gate。资源活动为 pending>=1 且 idle>=1，不是替代可行动作存在的证明。系数接入、公开车辆既定释放预测和终端系数门分开记录；候选 route 数不是可行联合动作数，系数变化不是利润改善。

prepare 校验冻结父实验及上一批全交付 SHA、原始准备场景 SHA、内容/seed 隔离、物理及配置身份，冻结本包源码、依赖的只读探针工具和分析源码。run 独立编译每个探针、重算现金与系数、检查 live 状态不变，实际零信号提交与探针一致，两个完整 MC 日 full cooldown 后独立重放。analysis 独立重算所有身份、系数、计数和覆盖，生成 3 个 CSV、一组 PNG/SVG 与中文报告。最终相关测试实际 XML 和真实视觉记录通过才 freeze。

## 执行入口

在仓库根目录，以仓库 `.venv/Scripts/python.exe` 执行 `-m experiments.cfs_ppo_decision_coverage.run prepare|run|freeze --attempt experiments/cfs_ppo_decision_coverage/runs/full_period_v1`。环境固定 PYTHONHASHSEED=0、PYTHONIOENCODING=utf-8、PYTHONDONTWRITEBYTECODE=1、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。分析 `-m experiments.cfs_ppo_decision_coverage.analyse --attempt ...`，首次 CSV 前成功执行一次技能 marker（3 CSV）。相关测试包括本包、前批、strategic PPO、classical value、single rolling，使用新工作区 basetemp 保存 XML。

若遗漏时刻存在响应，只能说明本批六时刻方案漏检；不能直接推广到旧场景或推出利润收益。若全时刻响应仍稀少，下一步应检查候选/承诺/终端限制和价格接入，而非直接长训练。若响应较广而既有 actor 少变，后续再登记短训练预算对照。本批结束只建议一个后续问题，不自动执行下一研究阶段。
