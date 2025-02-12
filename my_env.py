import gym
from gym import spaces
import numpy as np
from CITY_NODE import *
from VEHICLE import *
from tool_func import *
from Lower_Layer import *
from CITY_GRAPH import *
import copy

class DispatchEnv(gym.Env):
    def __init__(self, G :CityGraph, 
                 vehicles: Dict, orders: Dict, cities:Dict,capacity):
        super(DispatchEnv, self).__init__()
        
        self.time = 0
        # 必须深复制
        self.G = G
        self.vehicles = vehicles
        self.orders = orders
        self.cities = cities 
        self.capacity = capacity
        """
        车辆状态空间：
        车辆id即为行号，其decision，capacity、电量状态（建议更替为百分比）、intercity与intocity
        """
        self.vehicle_state = spaces.Discrete(
           len(self.vehicles)*11
        )
        #  期数，是否被匹配，载客数，在哪个城市
        self.order_state = spaces.Discrete(
            len(self.orders)*12
        )
        # 动作空间：为每个订单分配虚拟出发地点（连续或离散）
        # 动作空间：矩阵形式，每个订单可以从多个城市选择一个出发地点
        self.action_space = spaces.MultiDiscrete([len(self.cities)] * len(self.orders))
    """
    def count_num(self):
        num_vehicles, num_orders ,num_cities= 0
        for city in self.Cities.values():
            num_vehilces = len(city.vehicles.value()) + num_vehilces
            num_orders = len(city.orders.value()) + num_orders
        return num_vehicles ,num_orders , len(self.Cities.values())
    """
    def update(self, vehicles_matrix: np, orders_matrix: np, cities:dict):
        """更新环境"""
        self.vehicle_state = vehicles_matrix
        self.observation_space = orders_matrix
        self.cities = cities
        self.time += 1

    def reset(self, vehicles:Dict, orders:Dict):
        """重置环境"""
        self.time = 0
        self.orders = orders
        self.vehicles = vehicles
        # self.cities = cities
    def get_state(self):
        print(self.vehicles)
        print(self.orders)
        print(self.time)

    def _multiagent_step(self, action, orders_unmatched:dict):
        """执行动作并返回结果"""
        """action是一个len(self.orders) * len(self.cities)维度的数组"""
        """一刀-wrong_punlishment"""
        correct_combinations = []
        wrong_punlishment = 10
        reward = 0
        for i in range(len(orders_unmatched)):
            
            matched_amount = sum(action[i])
            # if orders_unmatched[i].matched:
            
            #    continue
            if matched_amount > 1:
                reward += -wrong_punlishment
                continue
            # if matched_amount == 1:
                if orders_unmatched[i].start_time > self.time:
                    # correct_combinations.extend([(i, j) for j in range(len(self.cities))])
                    # reward += -wrong_punlishment
                    # 貌似无需惩罚
                    continue
            # if matched_amount == 0:
                # correct_combinations.extend([(i, j) for j in range(len(self.cities))])
                # 我的想法是无需添加
                # correct_combinations.append((i, j for j in range(len(self.cities))))
                continue
            for j in range(len(self.cities)):
                # 已匹配者不可匹配
                if orders_unmatched[i].destination == j and action[i][j] == 1:
                    continue
                if orders_unmatched[i].matched and action[i][j] == 1:
                    continue  # 继续下一个组合，跳过惩罚

                # if self.orders[i].matched is False:
                _, path_order = self.G.get_intercity_path(*orders_unmatched[i].route())
                
                # 前驱不可行
                if j == path_order[1] and action[i][j] == 1:
                    continue  # 继续下一个组合，跳过惩罚
                # 需在邻接城市里
                if j not in self.G.get_neighbors(orders_unmatched[i].departure) and action[i][j] == 1:
                    continue  # 继续下一个组合，跳过惩罚

                # 邻接城市需要有合适的车
                vehicle_found = False
                # 起码要有车
                if orders_unmatched[j].vehicle_available.values():

                    for vehicle in self.cities[j].vehicle_available.values():
                        if len(vehicle.longest_path) > 0:
                            if vehicle.longest_path[0] == path_order[1] and action[i][j] == 1:
                                if self.capacity - vehicle.get_capacity > orders_unmatched[i].values().passenger:
                                    vehicle_found = True
                                    break
                    
                    if not vehicle_found:
                        continue  # 继续下一个组合，跳过惩罚
                else:
                    continue
                # 如果没有触发任何惩罚条件，则是正确的组合
                correct_combinations.append((i, j))

        for i in range(len(self.orders)):
            for j in range(len(self.cities)):
                if (i,j) not in correct_combinations:
                    reward = reward - orders_unmatched[i].revenue
                else:
                   
                    if self.time == orders_unmatched[i].start_time:
                        orders_unmatched[i].virtual_departure = j
                    
                    
        return reward  # 记住还需调用gurobi求解合理匹配下的值
    
    def step(self, action):
        """执行动作并返回结果"""
        """action是一个len(self.orders) * len(self.cities)维度的数组"""
        """一刀-wrong_punlishment"""
        correct_combinations = []
        wrong_punlishment = 10
        reward = 0
        for i in range(len(self.orders)):
            
            matched_amount = sum(action[i])
           
            if self.orders[i].matched:
                # correct_combinations.extend([(i, j) for j in range(len(self.cities))])
                continue
            if matched_amount > 1:
                reward += -wrong_punlishment
                continue
            if matched_amount == 1:
                if self.orders[i].start_time > self.time:
                    # correct_combinations.extend([(i, j) for j in range(len(self.cities))])
                    # reward += -wrong_punlishment
                    # 貌似无需惩罚
                    continue
            if matched_amount == 0:
                # correct_combinations.extend([(i, j) for j in range(len(self.cities))])
                # 我的想法是无需添加
                # correct_combinations.append((i, j for j in range(len(self.cities))))
                continue
            for j in range(len(self.cities)):
                # 已匹配者不可匹配
                if self.orders[i].destination == j and action[i][j] == 1:
                    continue
                if self.orders[i].matched and action[i][j] == 1:
                    continue  # 继续下一个组合，跳过惩罚

                if self.orders[i].matched is False:
                    _, path_order = self.G.get_intercity_path(*self.orders[i].route())
                    
                    # 前驱不可行
                    if j == path_order[1] and action[i][j] == 1:
                        continue  # 继续下一个组合，跳过惩罚
                    # 需在邻接城市里
                    if j not in self.G.get_neighbors(self.orders[i].departure) and action[i][j] == 1:
                        continue  # 继续下一个组合，跳过惩罚

                    # 邻接城市需要有合适的车
                    vehicle_found = False
                    # 起码要有车
                    if self.cities[j].vehicle_available.values():

                        for vehicle in self.cities[j].vehicle_available.values():
                            if len(vehicle.longest_path) > 0:
                                if vehicle.longest_path[0] == path_order[1] and action[i][j] == 1:
                                    if self.capacity - vehicle.get_capacity > self.orders[i].values().passenger:
                                        vehicle_found = True
                                        break
                        
                        if not vehicle_found:
                            continue  # 继续下一个组合，跳过惩罚
                    else:
                        continue
                # 如果没有触发任何惩罚条件，则是正确的组合
                correct_combinations.append((i, j))

        for i in range(len(self.orders)):
            for j in range(len(self.cities)):
                if (i,j) not in correct_combinations:
                    reward = reward - self.orders[i].revenue
                else:
                   
                    if self.time == self.orders[i].start_time:
                        self.orders[i].virtual_departure = j
                    
                    
        return reward  # 记住还需调用gurobi求解合理匹配下的值
    
    def test_step(self, orders_unmatched, actions):
        reward = 0
        i= 0 
        for order in orders_unmatched.values():
            _, path_order = self.G.get_intercity_path(*order.route())
            # 表示不可目的地
            if order.destination == actions[i]:
                reward += -100
                order.virtual_departure =  order.departure
            # 表示不可非邻接
            elif actions[i] not in self.G.get_neighbors(order.departure):
                reward += -100
                order.virtual_departure =  order.departure
            # 表示不可在前驱
            elif actions[i] == path_order[1]:
                reward += -100
                order.virtual_departure =  order.departure
            else:
                order.virtual_departure = actions[i]
            i += 1
        return reward
    
    def dynamic_step(self, total_orders, actions, mask):
        reward = 1000
        i = 0
        for i, order in enumerate(total_orders.values()):
            if mask[0,1] == True:
                _, path_order = self.G.get_intercity_path(*order.route())
                if order.destination == actions[i]:
                    reward += -100
                    order.virtual_departure =  order.departure
                elif actions[i] not in self.G.get_neighbors(order.departure):
                    reward += -100
                    order.virtual_departure =  order.departure
                elif actions[i] == path_order[1]:
                    reward += -100
                    order.virtual_departure =  order.departure
                else:
                    order.virtual_departure = actions[i]
                i += 1
        return reward
    
    def get_mask(self, orders_unmatched):
        
        i= 0 
        mask = np.ones((len(orders_unmatched), len(self.cities)))
        for order in orders_unmatched.values():
            _, path_order = self.G.get_intercity_path(*order.route())
            for j in range(len(self.cities)):
                if order.destination == j:
                    mask[i][j] = 0
                elif j not in self.G.get_neighbors(order.departure):
                    mask[i][j] = 0
                elif j == path_order[1]:
                    mask[i][j] = 0
            i += 1
        return mask
        
        
        
                    
                    
    
