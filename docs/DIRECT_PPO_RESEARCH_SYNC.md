# Direct PPO 与失败机制诊断：GitHub同步说明

2026-10-13。发布分支为 `react-experiment-platform`，不合并到main或single-level-dispatch-model，不修改平台默认研究模型。

## 先读哪份报告

| 问题 | 文件 |
|---|---|
| 无MC PPO是否胜过Pure MC？ | [Direct正式报告](../reports/cfs_ppo_strategic_control/direct_ppo_no_mc_v3/direct_ppo_vs_mc_report.md) |
| 是否胜过M0、Actor/Critic学到了什么、MILP能否响应？ | [失败机制诊断](../reports/cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v2/direct_ppo_failure_diagnosis_report.md) |
| 只看新增短视基准与业务对比 | [M0比较报告](../reports/cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v2/m0_comparison_report.md) |
| 未来带q模型怎样隔离和接入？ | [方案B设计](../reports/cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v2/scheme_b_architecture_proposal.md) |

Direct正式实验三个工况均弱于Pure MC。新增M0诊断不支持“三工况均破坏短视策略”：CA点估计为正但事后family调整区间含零，W40/W80差异不确定。Actor状态变化微弱、Critic拟合薄弱是实测现象，不能写成已证明共享梯度或缺q就是根因。完整限制、数值与区间以报告为准。

## 模块在哪里

| 路径 | 作用 |
|---|---|
| `experiments/cfs_direct_ppo/` | 首版实现／预算受限历史变体源码，不作为新正式结果 |
| `experiments/cfs_direct_ppo_v3/` | 冻结Direct版本：独立观测、10维价格、网络、训练、隔离／回放、阶段入口、分析 |
| `experiments/cfs_direct_ppo_diagnosis/` | 原诊断源码：M0／zero、公开状态恢复、车辆等价、Actor/Critic、固定探针／反事实 |
| `experiments/cfs_direct_ppo_diagnosis_recovery/` | 完整性检查位置修复与只读复用适配、导出、独立QA和freeze |
| `scripts/research/direct_ppo_delivery_qa.py` | 本地workspace交付QA工具的逐字节镜像；不修改历史实验源码 |
| `tests/test_direct_ppo*.py` | Direct隔离、动作、GAE、状态恢复、规范化、恢复边界和统计工程测试 |

原冻结README引用 `workspace.direct_ppo_delivery_qa`。workspace默认不入Git，本次将同一源码镜像到scripts，可通过 `python -m scripts.research.direct_ppo_delivery_qa --help`查看参数；旧命令保留为历史执行记录。镜像不是新数值模型，不用于重新签署已完成attempt。

## GitHub保存什么

- 源码、README、测试与任务书。
- Direct v3四份分析CSV：配对利润、业务指标、学习曲线、统计摘要。
- 诊断v2六份CSV：M0配对、Actor/Critic更新、动作敏感性、业务指标、反事实案例、比较摘要。
- 三组Direct图和四组诊断图：SVG、600dpi PNG及plotting config。
- 小型冻结协议、统计JSON和QA摘要、真实最终pytest XML。工程失败及预算停止的小型证据保留，不掩盖失败历史。

完整数据订单、SQLite、MC表、权重/recovery.pt、逐日逐t原始轨迹、完整hash索引、PDF渲染预览和虚拟环境不上传。未发布完整科学manifest，避免把依赖未发布轨迹的身份记录误当可独立重算运行包。

## 本地终态身份

| 本地attempt | 状态 | 完整final_manifest SHA256 |
|---|---|---|
| Direct `direct_ppo_no_mc_v3` | COMPLETED，2573文件，64相关测试 | `3e9c43e2fdf2978d6c45a970bd3dd384a47fd7db7a7728e92aa5013567c5fea7` |
| 诊断 `direct_ppo_failure_diagnosis_v2` | COMPLETED，131文件，81相关测试 | `420e24e98ecb0ce2d2970e6ade7614315855ca6ffbf65f06c6b29f4c30efe304` |

这些hash标识本地完整目录的科学终态，不是Git提交SHA；精简发布不改变本地源文件或既有hash。本地COMPLETED及失败目录都保持只读。

诊断v1在全部72M0+72zero通过之后，因完整性哈希读取放在no-sealed保护区内停止，离线pilot尚未产生结果。v2把检查移到保护区外并保留guard，复用144个成功日；没有改模型、样本或统计。Direct早期准备路径失败和v2预算门停止也保留披露。

## 复现前提与运行风险

查看报告、绘图和重新分析精简配对CSV不需要CFS数据库或权重。运行完整训练／诊断需要另外准备原冻结本地场景、MC对照表、checkpoint和历史身份文件，以及适配本机的数据路径；原源码含历史Windows任务路径，不是fresh clone的一键复现包。部分状态恢复测试直接读取本地冻结验证轨迹，因此没有这些输入时不能声称所有相关测试可独立通过。

研究环境使用仓库Python、Torch、numpy与Gurobi许可证，数值CPU串行、BLAS/Torch/solver各1线程；科学导出可使用bundled numpy，图形使用既有Matplotlib。报告明确区分工程测试、统计与完整日现金核验。

不要在已冻结目录运行prepare/train/analysis/freeze来填补GitHub未发布文件。新的数值运行应使用新的独立协议、输出目录和适配后的输入路径；方案B在本次只有设计，没有正式训练。
