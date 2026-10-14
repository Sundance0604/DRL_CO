# 在线 q 的语义与实际覆盖

q 是相同 (释放时刻,枢纽) sink-flow 的数量前缀，不是车辆编号。原 add_release_slots 对每组构建 K 个 binary z，sum z=sum path-flow，z_q<=z_(q-1)，目标加入 sum V(q)z_q。route/wait/locked-arrival 的流变量通过等式连接 z，订单assignment x 与 route y 原约束不变。

异质车辆的当前承诺/位置/路线决定其可行流，slot本身不绑定车辆身份。车辆不能任意置换；canonical只合并完整当前状态相同车辆。releases按(time,hub)排序，q从1开始。候选由observed_state及当前订单生成，不读未来订单；承诺车辆只投影锁定移动。tau>=T在建模前排除战略项。

所有组都会查询q=1..K，即便某组不可达。因此query coverage不是occupied coverage，更不是q有效控制频率。本轮分别保存查询、选中z、所有固定扰动中选中q>=2及规范化动作差。也没有把构建的 route候选数当合法联合动作数。

结构秩扣除真实MILP等式及固定变量行空间，容差1e-8。它只描述仿射松弛，未完全扣除不等式导致的不可达方向，不能称精确integer hull表达维数。真实canonical switch且occupiedq>=2是更直接的有限控制证据。MC的q来自资源注入，不等于这个group prefix；新PPO不读取MC标签，但可加性仍存在。
