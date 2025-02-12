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
import importlib
import tool_func
from update import *
import os
import time as tm
import copy
from my_env import *
import torch
import multiagent as magent
import matplotlib.pyplot as plt

"""这里是非强化学习部分"""
# 初始化
num_vehicle = 20
num_order = 6
num_city = 8
TIME = 144  # 
CAPACITY = 7
row = [10, 1, 3, 10]
Vehicles = {}
speed = 20 # 之前是20
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
ACTION_DIM = num_city          # 动作空间维度
ACTOR_LR = 1e-2          # Actor 学习率
CRITIC_LR = 1e-2         # Critic 学习率
GAMMA = 0.99             # 折扣因子
NUM_EPISODES = 40     # 总训练轮数
# 这里也改了
STATE_DIM = 2 *HIDDEN_DIM       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = magent.MultiAgentAC(
    device = DEVICE,
    VEHICLE_STATE_DIM = STATE_DIM_VEHICLE,
    ORDER_STATE_DIM = STATE_DIM_ORDER, 
    NUM_CITIES = num_city, 
    HIDDEN_DIM = HIDDEN_DIM, 
    STATE_DIM = STATE_DIM
)
print(type(agent))
grid_rewards = []
# 开始计时
start_time = tm.time()

episode_reward = 0
ACTIONS = []
base_revenue = []
first_revnue = []
base_vehicle = []
base_vehicle_class = []
base_order_class = []
base_city_node = []
train_rewards = []
burn_in = 20
for episode in range(NUM_EPISODES):
    print(f"第{episode}次训练")
    Total_order = copy.deepcopy(prim_order)
    Vehicles = copy.deepcopy(prim_vehicle)
    objval = 0
    total_objval = 0
    reward = 0
    episode_reward = 0
    invalid_time =  0
    orders_unmatched = {} # 忘记加这个了
    orders_virtual = {}
    
    if episode > 0:
        # agent.load_model(load_path)
        if episode > 1:
            env.time = 0
        
        if episode < burn_in:
            greedy_epsilon = 0.6
            explore = True
        else:
            greedy_epsilon = 0.01
            explore = False
            if episode == burn_in:
                best_model = train_rewards.index(max(train_rewards))
                agent = torch.load(f'model_checkpoint_{best_model+1}.pth',map_location="cpu")
                print(f"最优模型为{best_model}")
                agent.eval()
            if episode > burn_in:
                if train_rewards[best_model] > train_rewards[-1]:
                    agent = torch.load(f'model_checkpoint_{best_model+1}.pth',map_location="cpu")
                    print(f"最优模型为{best_model}")
                    agent.eval()
                else:
                    best_model = train_rewards.index(train_rewards[-1]) 
                    agent = torch.load(f'model_checkpoint_{best_model+1}.pth',map_location="cpu")
                    print(f"最优模型为{best_model}")
                    agent.eval()
    if_end = False
    for time in range(TIME):
    
        group = [[], []]
        if time == TIME -1:
            if_end = True
        # 按时间给出订单
        for order in Total_order.values():
            if order.start_time == time:
                orders_unmatched[order.id] = order
            # 加上这个代码后会导致性能降低
            """
            if order.matched is False:
                order.virtual_departure = order.departure 
            """
        if time != 0 and episode != 0:
            next_vehicle_states = vectorization_vehicle(Vehicles)
            # 改了，不再是total_order
            next_order_states = vectorization_order(orders_unmatched)
            # 这里防止梯度爆炸缩小了reward
            agent.update(vehicle_states, order_states, action,
                         grid_reward, next_vehicle_states, next_order_states , if_end)
            env.time = time
        if time == 0 :
            orders_virtual = orders_unmatched
           
            city_node = city_node_generator(G, orders_virtual, Vehicles, orders_unmatched)
            if episode == 1 :
                env = DispatchEnv(
                    G=G,
                    vehicles=Vehicles,
                    orders=Total_order,
                    cities=city_node,
                    capacity=CAPACITY
                )
            elif episode > 1:
                env.cities = city_node
           
            
        else:
            if episode == 0:
                city_update_without_drl(city_node , Vehicles, orders_unmatched ,time)
            else:
                city_update_without_drl(env.cities , Vehicles, orders_unmatched, time)
            
        if episode != 0:
            
            vehicle_states = vectorization_vehicle(Vehicles)
            # 这里也改了
            order_states = vectorization_order(orders_unmatched)
            
            greedy = random.randint(0, 1)
            if greedy > greedy_epsilon:
                greedy = True
            action = agent.take_action(vehicle_states, order_states, explore, greedy)
            reward = env.test_step(orders_unmatched,action)
            # 一个循环代码让我达到最优
            # 屁股后面的代码是为了让我达到最优
            # 这里是为了让我达到最优
            COUNT = 1000
            max_reward = -999999
            max_action = action
            
           
            while reward != 1000 and COUNT > 0:
                greedy = random.randint(0, 1)
                if greedy > greedy_epsilon:
                    greedy = True
                action = agent.take_action(vehicle_states, order_states, explore, greedy)
                reward = env.test_step(orders_unmatched,action)
                COUNT -= 1
                if reward > max_reward:
                    max_reward = reward
                    max_action = action
            if COUNT == 0:
                reward = env.test_step(orders_unmatched, max_action)
           
            
            ACTIONS.append(action) 
            """
            while reward != 0 :
                action = agent.take_action(vehicle_states, order_states)
                reward = env.test_step(orders_unmatched,action)
            
            """
            

        for vehicle in Vehicles.values():
            if vehicle.whether_city:
                group[0].append(vehicle.id)
            else:
                group[1].append(vehicle.id)

        if len(group[0]) != 0:
            if episode == 0:
                temp_Lower_Layer = Lower_Layer(G, city_node, Vehicles, orders_unmatched, name, group, time)
            else:
                temp_Lower_Layer = Lower_Layer(G, env.cities, Vehicles, orders_unmatched, name, group, time)
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
                # save_results(temp_Lower_Layer,time)
                # print("Objective value:", temp_Lower_Layer.model.objVal)
                objval = temp_Lower_Layer.model.objVal 
            else:
                temp_Lower_Layer.model.computeIIS()
                temp_Lower_Layer.model.write('iis.ilp')  # 保存不可行约束
                # print(f"{time}次，No optimal solution found.")
                self_update(Vehicles, G)
                objval = basic_cost(Vehicles, orders_unmatched)
                
            
            _, var_order = temp_Lower_Layer.get_decision()
            update_var(temp_Lower_Layer, Vehicles, orders_unmatched)
            vehicle_in_city = update_vehicle(Vehicles, battery_consume, battery_add, speed, G)
            order_canceled = order_canceled + update_order(orders_unmatched, time, speed)
            
        else:
            
            self_update(Vehicles, G)
            # print(f"{episode}轮，{time}次，{len(group[1])}辆车不在城市")
            order_canceled = order_canceled + update_order(orders_unmatched, time, speed)
            objval = basic_cost(Vehicles, orders_unmatched)
            # 利润（如果有）减去新增的取消订单
            
            invalid_time += 1
        objval = objval - update_order(orders_unmatched, time, speed) * cancel_penalty
        if episode != 0: 
            
            # 防止梯度爆炸
            grid_reward =   reward
            grid_rewards.append(reward)
            # print(grid_reward)
            episode_reward += reward + objval
        total_objval += objval

        if episode == 0:
            base_revenue.append(objval)
            base_vehicle.append(copy.deepcopy(group[0]))
            # base_vehicle_class.append(copy.deepcopy(Vehicles))
            # base_order_class.append(copy.deepcopy(Total_order))
            
            base_city_node.append(copy.deepcopy(city_node))
            """
            if episode == 1:
                first_revnue.append(objval)
            """
        else:
            # if base_revenue[time] != objval:
            #    print("base_revenue",time, objval,base_revenue[time])
            # if first_revnue[time] != objval:
            #    print("first_revenue",time, first_revnue[time], objval)
            """ 
            if base_vehicle[time] != group[0]:
                print("vehicle is different", len(base_vehicle[time]), len(group[0]))
            if base_city_node[time] != env.cities:
                print(time, base_city_node[time],"\n", env.cities)
            """
            
        # print(f"{len(orders_unmatched)}订单未被匹配,{order_canceled}订单超时,总利润为{objval},强化学习利润为{reward}")
    end_time = tm.time()
    execution_time = end_time - start_time
    if episode != 0:
        print(f"执行时间: {execution_time} 秒,{invalid_time}次未求解，当前强化学习值为{episode_reward},利润为{total_objval}")
        # torch.save(agent.state_dict(), 'model_checkpoint.pth')
        torch.save(agent, f"model_checkpoint_{episode}.pth")
        train_rewards.append(total_objval)
    else:
        print(f"未加强化学习利润为{total_objval},{invalid_time}次未求解")
    # grid_rewards.append(0)
    # save_path = f"actor_critic_model{episode}.pth"
    # load_path = f"actor_critic_model{episode}.pth"
    
plt.plot(grid_rewards, label='Grid Reward Curve')
plt.xlabel('Iteration')
plt.ylabel('Grid Reward')
plt.title('Reward Curve')
plt.legend()
plt.grid(True)
plt.show()
# print(find_duplicates_with_positions(ACTIONS))