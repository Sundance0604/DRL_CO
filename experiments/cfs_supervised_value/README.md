# AutoGluon 监督资源价值轻量试验

本模块读取第六轮精确核算的单一M0配对资源标签；保留原始资源替换定义，以同一自然M0轨迹较早公开快照构造(t,τ)输入，原始反事实现金标签只使用一次。所有原档案只读，不追加MC标签、强化学习或修改原MILP。

执行顺序：仓库Python `-m experiments.cfs_supervised_value.prepare`；预检通过后 `-m experiments.cfs_supervised_value.run`。AutoGluon使用现有pavane环境，原Gurobi运行使用仓库环境，批量预测经本机子进程通信。两个fit各软预算150秒、实际最大180秒，主评价24日576求解，数值阶段900秒、产物2GiB。

结果目录：`reports/cfs_supervised_value/autogluon_pilot_v3`。v1/v2预检记录保留；v3在任何fit前修正浮点诊断核验、保守seed扫描及实际状态字段。数据、权重、原始轨迹和CSV留本地；报告说明时间、q语义、监督划分、MC噪声与开发评价的限制。终态目录不能重跑或覆盖。
