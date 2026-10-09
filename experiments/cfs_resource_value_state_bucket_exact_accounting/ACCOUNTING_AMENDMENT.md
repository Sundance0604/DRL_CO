# B 独立重跑的整数执行金额修正

## 授权、原故障与留存

用户已同意：保留原失败 B attempt，另建仅修复实际整数动作金额重算的新 attempt 后重跑全部 B。原目录、A2、以前实验及 CFS 均只读。原停止证据是 `../cfs_resource_value_state_bucket/qa/stop_condition_failure.json`。

CA80/3 seed8405，t=7，hub=06-348，q=3。原 solver 状态 OPTIMAL，near-binary 数值在默认容差内，但浮点目标与 X>0.5 提取后的整数动作金额相差 -0.0002744013454503147 USD，超过原 1e-5 USD 检查。失败世界未接受，formal 未开始。

## 唯一修正

先进行原 solve_plan，再将原选中整数路线、订单分配和重定位交给已有独立 compile_action 验证可行性并计算金额。规范动作（包括路线金额）必须完全相同，输入状态必须不变。执行仍使用原 sol/routes/expired 对象和原列表顺序。没有第二次优化，没有调整 solver 容差、截止时间、目标或物理动作。

只替换 DecisionPlan 的执行 profit 和整数动作 objective。原浮点 solver profit、objective、accounting 与差额完整保留在每步诊断中。累计 booked J 与独立整数执行回放采用同一合法动作核算。完整 realized ledger 不变，严格金额检查与 replay 门槛不变。

修正不改变当前动作。后续若某代码把累计 booked J 用于决策，该修正可能成为额外行为变量；本轮 B 的 M0/slot价格/pressure 不使用累计 J 决策。历史回归必须核对所有物理字段和全部动作完全一致，只将 before/after.profit 单列为修正诊断，完整日 realized profit 仍要求 1e-6 USD 一致。

## 复现和科学边界

独立重跑全部 construction、diagnostic、pilot 和 formal，不复用原 B 数值缓存。相同 CFS、条件、原始种子、采样时间/hub/q、阈值规则、支持回退与统计方案。新数据与旧 attempt 已存在场景逐条相等检查。Base/bucket 共用同一个修正版 M0 父策略和相同标签，B2-B1 仍是严格内部状态扩展对照。

原 A2 不重新核算、不覆盖。跨 A2/B 的机制解释须披露本次执行金额修正；不同 attempt 不混合当作同源样本。本轮不自动启动 B-PI 或 RL。

## 验证

原失败状态 seed8405/q3 必须在原参数下重现浮点差额，然后通过新整数核算、完全相同动作与完整冷却 replay。玩具近整数金额、nonoptimal、陈旧计划、重复订单反例必须通过。实际结果保存在本目录 qa 下的真实 pytest XML；不是事先填好的通过声明。

后续全部 72,800 资源世界与 1,488 闭环 run 执行原完整 replay 与 lineage 审计，并核对整数动作哈希、逐步收入成本与保存金额；最终全测试、历史 SHA、论文图逐页视觉核验后才冻结。

## 首次视觉审阅的非数值排版修正

首次审阅发现桶占比图图例遮挡柱形，因此仅将该图图例移至图外。热图补充每个 hub block 的 q 节点标签；导出时固定已经求解的布局，避免单图与图册在重复绘制时重新排版。没有更改数据、选择展示格的规则、坐标数值、证据阈值或统计源码。首次数值预检的 `unchanged_sources` 保留当时图形源码哈希；最终图形源码及重渲染产物哈希在最终交付清单中记录，不把排版改变伪装成与原 B 字节相同。
