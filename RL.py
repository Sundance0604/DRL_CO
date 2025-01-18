import csv
import numpy as np
from gurobipy import *
from CITY_GRAPH import *
from CITY_NODE import *
from ORDER import *
from VEHICLE import *
from tool_func import *
from Lower_Layer import *
import SETTING
import RL
import importlib
import tool_func
from update import *
import os
import time as tm
import copy
from my_env import *
import torch
from actor_critic import *

"""这里是非强化学习部分"""
# 初始化
num_vehicle = 20
num_order = 5
num_city = 5
TIME = 1440
CAPACITY = 7
row = [10, 1, 3, 10]
Vehicles = {}
speed = 20
cancel_penalty = 300
battery_consume = 10
battery_add = 300

matrix = np.tile(row, (num_vehicle, 1))

# 初始化
Vehicles = vehicle_generator(num_vehicle, num_city)
orders_unmatched = {}
G = CityGraph(num_city, 0.3, (10, 30))
name = "navie"
cancel_penalty = 300
order_canceled = 0
Total_order = {}
objval = 0
invalid_time = 0
# 设置s_0
for time in range(TIME):
    Orders = order_generator(num_order, time, num_city-1, CAPACITY, G ,speed)
    for order in Orders.values():
        Total_order[order.id] = order

# 深复制最初的订单与车辆
prim_order = copy.deepcopy(Total_order)
prim_vehicle = copy.deepcopy(Vehicles)

"""这里是强化学习部分"""
# 超参数
STATE_DIM_VEHICLE = 11   # 车辆状态的特征维度
STATE_DIM_ORDER = 12     # 订单状态的特征维度
HIDDEN_DIM = 128         # 隐藏层维度
ACTION_DIM = 10          # 动作空间维度
ACTOR_LR = 1e-4          # Actor 学习率
CRITIC_LR = 1e-3         # Critic 学习率
GAMMA = 0.99             # 折扣因子
NUM_EPISODES = 500       # 总训练轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = ActorCritic(
    vehicle_dim=STATE_DIM_VEHICLE,
    order_dim=STATE_DIM_ORDER,
    hidden_dim=HIDDEN_DIM,
    action_dim=ACTION_DIM,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    gamma=GAMMA,
    device=DEVICE
)

# 开始计时
start_time = tm.time()
episode_reward = 0
for time in range(TIME):
    group = [[], []]
    
    
    # 按时间给出订单
    for order in Total_order.values():
        if order.start_time == time:
            orders_unmatched[order.id] = order 
    if time != 0:
        next_vehicle_states = vectorization_vehicle(Vehicles)
        next_order_states = vectorization_order(orders_unmatched)
        agent.update(vehicle_states, order_states, reward, next_vehicle_states, next_order_states, True ,action)
    if time == 0:
        orders_virtual = orders_unmatched
        city_node = city_node_generator(G, orders_virtual, Vehicles, orders_unmatched)
        env = DispatchEnv(
            time=time,
            G=G,
            vehicles=Vehicles,
            orders=Total_order,
            cities=city_node,
            capacity=CAPACITY
        )
    else:
        city_node = city_update_without_drl(city_node , Vehicles, orders_unmatched)
    vehicle_states = vectorization_vehicle(Vehicles)
    order_states = vectorization_order(orders_unmatched)
    action = agent.take_action(vehicle_states, order_states)
    reward = env.step(action)
    city_update_base_drl(city_node, orders_unmatched)
    
    for vehicle in Vehicles.values():
        if vehicle.whether_city:
            group[0].append(vehicle.id)
        else:
            group[1].append(vehicle.id)

    if len(group[0]) != 0:
        temp_Lower_Layer = Lower_Layer(G, city_node, Vehicles, orders_unmatched, name, group, time)
        temp_Lower_Layer.get_decision()
        temp_Lower_Layer.constrain_1()
        temp_Lower_Layer.constrain_2()
        temp_Lower_Layer.constrain_3()
        temp_Lower_Layer.constrain_4()
        temp_Lower_Layer.constrain_5()
        temp_Lower_Layer.model.setParam('OutputFlag', 0)
        total_penalty = cancel_penalty * order_canceled
        temp_Lower_Layer.set_objective(matrix)
       
        temp_Lower_Layer.model.optimize()

        if temp_Lower_Layer.model.status == GRB.OPTIMAL:
            save_results(temp_Lower_Layer,time)
            print("Objective value:", temp_Lower_Layer.model.objVal)
            objval = temp_Lower_Layer.model.objVal 
        else:
            temp_Lower_Layer.model.computeIIS()
            temp_Lower_Layer.model.write('iis.ilp')  # 保存不可行约束
            print(f"{time}次，No optimal solution found.")
            self_update(Vehicles, G)
            objval = basic_cost(Vehicles, orders_unmatched)
            
        
        _, var_order = temp_Lower_Layer.get_decision()
        update_var(temp_Lower_Layer, Vehicles, orders_unmatched)
        vehicle_in_city = update_vehicle(Vehicles, battery_consume, battery_add, speed, G)
        order_canceled = order_canceled + update_order(orders_unmatched, time, speed)
        
    else:
        self_update(Vehicles, G)
        print(f"{time}次，{len(group[1])}辆车不在城市")
        order_canceled = order_canceled + update_order(orders_unmatched, time, speed)
        objval = basic_cost(Vehicles, orders_unmatched)
        # 利润（如果有）减去新增的取消订单
        
        invalid_time += 1
    objval = objval - update_order(orders_unmatched, time, speed) * cancel_penalty 
    reward += objval
    episode_reward += reward
    print(f"{len(orders_unmatched)}订单未被匹配,{order_canceled}订单超时,总利润为{objval},强化学习利润为{reward}")
    

end_time = tm.time()
execution_time = end_time - start_time
print(f"执行时间: {execution_time} 秒,{invalid_time}次未求解")
