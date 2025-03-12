from ORDER import *
from CITY_GRAPH import *
from VEHICLE import *
import numpy as np
from CITY_NODE import * 
import os
import Lower_Layer
from collections import defaultdict
import torch
import copy

per_distance_battery = 10
# 目前这个函数没有用了
def route_combine(lst, list1, list2):
    # 遍历所有可能的分割位置
    for i in range(1, len(lst)):  # i 是分割点
        # 分割列表为两部分
        left = lst[:i]
        right = lst[i:]
        # 判断左边部分是否与list1相等，右边部分是否与list2相等
        if left == list1 and right == list2:
            return True
    return False



def vehicle_generator(num_vehicle:int, num_city:int):
    Vehicles = {}
    for i in range(0,num_vehicle):
        vehicle_id = i          # 车辆编号
        into_city = 0        #默认城市内
        intercity = random.randint(0, num_city-1)    # 当前所在城市编号 
        decision = 2  # 当前决策
        battery = random.uniform(0, 100)*200         # 电池剩余电量（百分比）
        orders = {}  # 无订单

        # 创建 Vehicle 对象
        vehicle = Vehicle(vehicle_id, 0, into_city, intercity, decision, battery, orders)
        Vehicles[vehicle.id] = vehicle
    return Vehicles

def order_generator(num_order:int, time: int, num_city:int,CAPACITY,G:CityGraph,speed):
    Orders = {}
    for i in range(0,num_order):
        id = i + time*num_order          # 订单 ID
        passenger = random.randint(1,CAPACITY)     # 乘客数
        departure = random.randint(0, num_city-1)           # 出发地
        destination = random.randint(0, num_city-1) # 目的地
        while destination == departure:           
            destination = random.randint(0, num_city)
        
        virtual_departure = departure    # 预设为虚拟出发地
        try:
            distance,_ =G.get_intercity_path(departure, destination)
        except:
            print(departure,destination)
        start_time = time                   # 起始时间
        end_time = start_time+ distance/speed + random.randint(10,20)
        battery =random.uniform(0, 10)+distance*per_distance_battery             # 需要的电量
        revenue = distance * 100 + passenger * 50 # 随便编
        penalty = passenger * 5 # 随便编
        least_time_consume = distance/speed
        
        # 创建 Order 对象
        order = Order(id, passenger, departure, destination, start_time, 
                      end_time, virtual_departure, battery,distance,revenue,penalty,
                      least_time_consume)
        # 创建路径字符串
        order.path_key = list_str(order_feasible_action(order, num_city, G))
        
        Orders[id] = order
    return Orders 

def city_node_generator(G:CityGraph,
                        order_virtual:Dict,
                        Vehicles:Dict,
                        order_unmatched:Dict
                        ):
    
    Cities = {}
    for i in range(0,G.num_nodes):
        city_id = i
        neighbor = G.get_neighbors(city_id)
        vehicle_available = {}
        real_departure = {}
        virtual_departure = {}

        
        vehicle_available = {}
        # 根据现有的车队序列和订单序列为每个城市添加信息
        for vehicle in Vehicles.values():
            if vehicle.whether_city and vehicle.intercity == city_id:
                vehicle_available[vehicle.id]=vehicle
        for order in order_virtual.values():
            if order.virtual_departure == city_id:
                virtual_departure[order.id]=order
        for order in order_unmatched.values():
            if order.departure == city_id:
                real_departure[order.id]=order
        #生成城市
        city = City(city_id, neighbor, 
                 vehicle_available,
                 10, 
                 real_departure, 
                 virtual_departure,
                 )
        Cities[city_id] = city
    return Cities

def city_update_without_drl(cities:dict, vehicles:dict, order_unmatched:dict, time):
    for city in cities.values():
        city.clean_all()
        for vehicle in vehicles.values():
            if vehicle.whether_city and vehicle.intercity == city.city_id:
                city.add_available_vehicle(vehicle.id, vehicle)
                
        for order in order_unmatched.values():
            if order.departure == city.city_id and order.matched is False:
                if order.start_time <= time :
                    city.add_real_departure(order.id, order)
                
        for order in order_unmatched.values():
            if order.virtual_departure == city.city_id and order.matched is False:
                if order.start_time <= time :
                    city.add_virtual_departure(order.id, order)
                

def city_update_base_drl(cities:dict, order_virtual:dict ,time:int):
    for city in cities.values():
        city.virtual_departure = {}
    # 忘记缩进了 >_<
        for order in order_virtual.values():
            if order.virtual_departure == city.city_id and order.matched is False:
                if order.start_time <= time :
                    city.add_virtual_departure(order.id, order)


def save_results(temp_Lower_Layer, time):
    # 创建保存结果的文件夹
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在，则创建
    
    # 构造文件路径
    file_path = os.path.join(output_dir, f"output_{time}.txt")
    
    # 打开文件并写入优化结果
    with open(file_path, "w") as file:
        for v in temp_Lower_Layer.model.getVars():
            if v.x == 1:  # 如果变量值为 1，则写入文件
                file.write(f"{v.varName} = {v.x}\n")

def vectorization_vehicle(vehicles:Dict):
    
    return np.vstack([
        np.array([  vehicle.time, 
                    vehicle.into_city, 
                    vehicle.intercity, 
                    int(vehicle.whether_city), 
                    vehicle.battery,
                    vehicle.decision,
                    vehicle.get_capacity(),
                    vehicle.time_into_city,
                    len(vehicle.get_orders()),
                    # 是否凑成12个？
                    vehicle.last_decision,
                    vehicle.longest_path[0] if vehicle.longest_path else 0
                    ],
                    dtype=np.int32)
    for vehicle in vehicles.values()])
def vectorization_order(orders):
    if isinstance(orders, dict):
        return np.vstack([
        np.array([
            order.passenger,
            order.departure,
            order.destination,
            order.start_time,
            order.end_time,
            order.virtual_departure,
            int(order.matched),  # 布尔值转为整数
            order.battery,
            order.distance,
            order.revenue,
            order.penalty,
            order.least_time_consume
        ], dtype=np.int32)  # 使用合适的 dtype
        for order in orders.values()  # 假设 `orders` 是包含所有订单的字典
        ])
    if isinstance(orders, list):
        return np.vstack([
        np.array([
            order.passenger,
            order.departure,
            order.destination,
            order.start_time,
            order.end_time,
            order.virtual_departure,
            int(order.matched),  # 布尔值转为整数
            order.battery,
            order.distance,
            order.revenue,
            order.penalty,
            order.least_time_consume
        ], dtype=np.int32)  # 使用合适的 dtype
        for order in orders  # 假设 `orders` 是包含所有订单的列表
        ])

def vectorization_order_mask(orders, G:CityGraph, num_city):
    
    return np.vstack([

    
    feasible_action_binary(order, num_city, G)

    for order in orders.values()  
    ])
        
def basic_cost(vehicles:dict, orders_unmatched:dict):
    vehicle_cost = 0
    order_cost = 0
    for vehicle in vehicles.values():
        if vehicle.decision == 0 or vehicle.decision == 3:
            vehicle_cost += 10
        if vehicle.decision == 1:
            vehicle_cost += 1
        if vehicle.decision == 0:
            vehicle_cost += 0
    for order in orders_unmatched.values():
        order_cost += order.penalty
    return vehicle_cost + order_cost

def find_duplicates_with_positions(lst):
    # 创建一个字典，用于存储元素及其索引位置
    element_positions = defaultdict(list)

    # 遍历列表，记录每个元素的索引
    for index, value in enumerate(lst):
        element_positions[value].append(index)

    # 筛选出重复的元素及其索引位置
    duplicates = {key: positions for key, positions in element_positions.items() if len(positions) > 1}

    return duplicates

def compare_model(path_before, path_after):
        """比较两次学习后的模型参数是否一致"""
        # 加载前后参数
        params_before = torch.load(path_before)
        params_after = torch.load(path_after)

        # 比较每部分参数
        for key in params_before.keys():
            if key not in params_after:
                print(f"{key} 不存在于 {path_after}")
                continue

            # 获取状态字典
            state_before = params_before[key]
            state_after = params_after[key]

            # 遍历每个层的参数
            for param_name in state_before:
                if param_name not in state_after:
                    print(f"参数 {param_name} 不存在于 {key} 的 {path_after}")
                    continue

                # 比较参数
                param_diff = torch.isclose(
                    state_before[param_name], state_after[param_name], atol=1e-6
                )

                if not param_diff.all():
                    print(f"参数 {param_name} 在 {key} 中发生了变化")
                    return False  # 参数不同

        print("两次模型参数完全一致")
        return True  # 参数一致

def order_same_action(Total_order:dict, num_city, G:CityGraph):
    order_with_same_action = {}
    for order in Total_order.values():
        
        if order_with_same_action:
            if order.path_key in order_with_same_action:
                order_with_same_action[order.path_key].append(order)
            else:
                order_with_same_action[order.path_key] = [order]
        else:
            order_with_same_action[order.path_key] = [order]
    return order_with_same_action

def order_feasible_action(order, num_city, G:CityGraph):
    feasible_action = []
    j = 0
    _, path_order = G.get_intercity_path(*order.route())
    for j in range(num_city):
        if order.destination == j:
            continue
        elif j not in G.get_neighbors(order.departure):
            continue
        elif j == path_order[1]:
            continue
        feasible_action.append(j)
        j = j + 1
    return feasible_action

def list_str(my_list):
    return str("".join(map(str, my_list)))

# 这个的作用是，判断是否有可行的订单，如果没有该agent当期不活跃


def active_test(action_type, agent, orders_unmatched):
    agent.active = False
    agent.last_order = copy.deepcopy(agent.current_order)
    agent.current_order = []
    for order in orders_unmatched.values():
        if order.path_key == action_type:
            agent.active = True
            agent.current_order.append(order)

def get_multi_reward(agent):
    agent.reward = 0
    for order in agent.last_order:
        if order.matched:
            agent.reward += order.revenue + random.randint(0, 100)
# 直接把掩码打进向量    
def feasible_action_binary(order:dict, num_city, G:CityGraph):
    feasible_action = [0]*num_city
    j = 0
    _, path_order = G.get_intercity_path(*order.route())
    for j in range(num_city):
        if order.destination == j:
            feasible_action[j] = 0
        elif j not in G.get_neighbors(order.departure):
            feasible_action[j] = 0
        elif j == path_order[1]:
            feasible_action[j] = 0
        else:
            feasible_action[j] = 1    
        j = j + 1
    return feasible_action
        
# 每个城市当前可获得的车辆
def seat_count(capacity,city_node):
    seat_city = []
    for city in city_node.values():
        seat_city.append(city.city_seat_count(capacity))
    return seat_city

    