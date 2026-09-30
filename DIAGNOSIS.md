# 不收敛根因审计

## 主因：策略动作没有影响环境

`multi_test.ipynb` 最后一版 SAC 代码调用了 `take_action_vehicle` 产生 `actions`，但没有调用 `env.test_step`，也没有在构造 `Lower_Layer` 前更新 `order.virtual_departure`。即使旧版调用了 `test_step`，已经构造好的 `city.virtual_departure` 桶也没有刷新。

因果链因此断开：

`policy -> action -X-> Gurobi model -> objective/reward`

当 reward 与 action 无关时，critic loss 仍然可以下降（拟合常数即可），但策略不可能学到调度规则。审计旧 `training_curves.npz` 时发现：210 万次 value update 后 loss 接近 0，但这不是策略收敛的证据。该 10 MB 中间文件已在仓库清理中删除，结论保留于本文。

修复：`DispatchEnv.apply_actions` 同时修改订单并刷新 city buckets，训练入口在每次 Gurobi solve 之前强制经过这个边界。

## 其他确认的实现错误

1. 旧 `multiagent.py` 在每次动作采样前执行 `torch.manual_seed(114514)`，导致探索序列被不断重置。
2. 多个 replay actor-critic 版本把 `log_prob` 转为 list/tensor 后入池，更新时已脱离当前 actor 的计算图，不能产生正确的 policy gradient。
3. 最新 SAC 按数组位置对齐 `order_states[i]` 和 `next_order_states[i]`。订单被匹配或取消后，动态集合会变长并重排，导致错误的 Bellman transition。
4. SAC 的 alpha loss 符号使 alpha 倾向单调增大，而不是追踪目标熵。
5. target critic 使用 dropout 且处于 train mode，使 Bellman target 自身带随机噪声。
6. 动作 mask 可以整行为 0，然后旧代码默默退化为动作 0。现在把“保留真实出发地”定义为始终可用的 no-op，并对空 mask fail fast。
7. `Lower_Layer.constrain_3` 的互斥约束把 `order1` 写了两次，结果是强制 `order1=0`，而非限制 `order1 + order2 <= 1`。
8. 下层收益使用了冗余的 `X_Vehicle * X_Order` 二次项，把本可线性的模型变成二次模型。
9. 最新状态只给 actor 输入可行性 mask，订单的人数、收益、截止时间和电量需求全部丢失，策略无法区分同 mask 的不同订单。
10. `X_Order[order, vehicle]` 没有全局绑定到车辆的实际城市和 dispatch 状态；不在订单虚拟出发地的车辆也能在目标函数中制造虚假匹配。
11. 修复后的第一版仍把城市 ID 作为固定 one-hot 输入，并为每个城市保留独立输出神经元。它能记住训练图，却不满足节点置换等变性；held-out greedy 目标值仅 236,403。候选共享打分器去掉绝对 ID 后，在相同 held-out 上达到 303,236；再配合多图训练达到 307,622。

## 新训练流

1. 按时间激活订单，刷新 city snapshot。
2. 生成带尺度缩放的车队与订单特征。
3. SAC 从 mask 内采样，`apply_actions` 把动作写入订单和 city snapshot。
4. Gurobi 求解匹配并推进环境。
5. 以订单匹配/取消结果构造 action-dependent reward。
6. 用稳定 `order_id` 匹配动态 `s'`；消失的订单标记为 terminal。
7. 记录 objective、match/cancel、entropy、alpha、Q/target 和梯度范数。
