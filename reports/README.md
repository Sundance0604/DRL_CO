# 数值实验报告索引

本次同步包含实验源码、运行配置、测试、方法说明、Markdown 汇总报告与报告用 PNG／SVG 图。实际数据集、原始逐步轨迹、训练权重、MC／拟合价值表、完整结果 JSON、源码快照 ZIP 和运行目录保留本地。较早批次的 CSV／JSON仍保留本地；2026-10-13新增Direct批次的精简分析CSV、冻结协议及小型QA记录随仓库发布。报告中的原始轨迹／权重路径仅供本地核验，不代表GitHub提供完整运行包。

## 已归档的 PPO 结果

| 实验 | 可读报告 | 状态与范围 |
|---|---|---|
| 完整方案准备、工程预检与 pilot | [预检摘要](cfs_ppo_strategic_control/cfs_first_ppo_v1/preformal_summary.md) | pilot 通过；大规模正式预算未启动 |
| 笔记本轻量 PPO | [实验报告](cfs_ppo_strategic_control/cfs_laptop_ppo_v1/laptop_experiment_report.md) | 完成；3 工况×3 模型，每模型32个训练日 |
| 动态／每枢纽固定信号 | [机制报告](cfs_ppo_strategic_control/cfs_dynamic_vs_constant_v2/mechanism_experiment_report.md) | 完成；原分析导出失败与恢复经过在报告中披露 |
| 信号到整数动作诊断 | [控制通道报告](cfs_ppo_signal_action_diagnostic/signal_action_v1/signal_action_diagnostic_report.md) | 完成；机制诊断，不追加训练 |
| 全时刻决策覆盖 | [覆盖报告](cfs_ppo_decision_coverage/full_period_v1/decision_coverage_report.md) | 完成；W80 两个开发场景的全24时刻 |
| 冻结 PPO 已学方向 | [方向报告](cfs_ppo_learned_direction_v2/direction_v2/learned_direction_report.md) | v2完成；原v1 INVALID及修正经过在README披露 |
| 无MC Direct PPO | [正式比较报告](cfs_ppo_strategic_control/direct_ppo_no_mc_v3/direct_ppo_vs_mc_report.md) | v3完成；9模型、2064完整episode、64相关测试；三个工况均劣于Pure MC |
| Direct PPO失败机制诊断 | [诊断报告](cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v2/direct_ppo_failure_diagnosis_report.md)、[M0比较](cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v2/m0_comparison_report.md) | 恢复v2完成；72M0＋72zero，54状态810正式探针，7反事实案例，81相关测试 |
| 带q的无MC方案B | [技术设计](cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v2/scheme_b_architecture_proposal.md) | 设计与可行性评估；没有正式训练 |

报告同目录的 `figures/` 保存已生成的汇总图。原本地运行目录未搬迁或修改；本目录中的运行报告是额外归档副本。

## 架构、任务书和研究判断

- [PPO强化学习训练架构与现状说明](PPO强化学习训练架构与现状说明_20261010.md)
- [PPO模块化架构设计](PPO模块化架构设计_隔离边界与接入方案.md)
- [PPO首次数值实验任务书](PPO首次数值实验任务书_第八轮.md)
- [第八轮方向判断与强化学习建模建议](第八轮方向判断与强化学习建模建议.md)
- [七轮数值实验研究汇总](七轮数值实验研究汇总_导师交流稿.md)
- [笔记本数值实验规范](笔记本数值实验规范_假设驱动与逐级验证.md)

## 较早 CFS 数值实验

| 方向 | 报告／入口 |
|---|---|
| 接单半径与日终实现账本 | [报告](cfs_rho_endofday/simulation_report.md)、[入口](../experiments/cfs_rho_endofday/README.md) |
| 动态前瞻与滚动规划 | [报告](cfs_dynamic_lookahead/dynamic_lookahead_report.md)、[入口](../experiments/cfs_dynamic_lookahead/README.md) |
| 经典价值函数 | [报告](cfs_classical_value/classical_value_report.md)、[入口](../experiments/cfs_classical_value/README.md) |
| 局部后验价值 | [报告](../experiments/cfs_local_post_bayesian_value/local_post_bayesian_value_report.md) |
| rollout目标与部署对齐 | [报告](../experiments/cfs_rollout_target_alignment/deployment_aligned_rollout_target_report.md) |
| 资源价值策略迭代 | [报告](../experiments/cfs_resource_policy_iteration/resource_policy_iteration_report.md) |
| 阻尼策略迭代 | [报告](../experiments/cfs_resource_policy_iteration_damping/damping_policy_iteration_report.md) |
| 状态分桶 | [入口与停止诊断](../experiments/cfs_resource_value_state_bucket/README.md) |
| 状态分桶精确核算 | [报告](../experiments/cfs_resource_value_state_bucket_exact_accounting/resource_value_state_bucket_report.md)、[核算修订](../experiments/cfs_resource_value_state_bucket_exact_accounting/ACCOUNTING_AMENDMENT.md) |
| 联合动作／价值诊断 | [报告](../experiments/cfs_joint_action_value_diagnostic/joint_action_value_diagnostic_report.md) |

## 本地复现边界

配置文件保留超参数、场景生成规则和随机种子，不包含采样订单或权重。CFS 数据来源路径需要在使用者自己的机器上配置；价值表、冻结场景和检查点需在本地准备。PPO快速示例引用的 `smoke_dataset.json`、`smoke_mc_table.json` 也留在本地，未作为数据文件上传。测试中通过代码构造的合成工程场景随测试源码保留。

GitHub 的报告展示已执行实验的汇总结论，不能代替本地完整轨迹、统计数据和冻结身份审计。各实验是否完成、是否触发停止条件，以相应报告的明确状态为准。

新增批次的文件分工、QA与本地复现前提见[同步说明](../docs/DIRECT_PPO_RESEARCH_SYNC.md)。本地完整COMPLETED目录不可重跑；GitHub只保存其精简发布部分，不应把缺少原始文件的clone当作已经通过全轨迹SHA验证的完整attempt。
