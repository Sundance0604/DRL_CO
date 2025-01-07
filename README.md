# Reinforcement Learning and Combinatorial Optimization  

**11.29**
加入约束函数，在Lower_Layer中<br>
**11.30**
整理类，完成了除时间约束外的约束条件<br>
**12.01**
完成了下层的基本框架，有些约束尚未实现<br>
**12.03**
大致完善，部分函数未完成<br>
**01.03**
infeasible<br>
**01.04**
solution found<br>
**01.05**
可循环，路径重合识别<br>
**01.07**
1. 不再任意分配virtual departure，而是只选择前驱的邻接点<br>
2. 对于vehicle是否进入城市，应为其优先计算，如果其进入城市后有可匹配的订单，为之匹配，否则直接经过<br>
3. 电不够不可充电，电满了也不可充电<br>
4. 订单的生成遵循VRP的结果<br>
5. 优先编写强化学习<br>
