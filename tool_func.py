from ORDER import *
from CITY_GRAPH import *
from VEHICLE import *
import numpy as np
import SETTING
from CITY_NODE import * 

per_distance_battery = 10
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

def time_consume(order:Order,speed=1000):
    return order.distance/speed
def time_cost(u:int , v:int,G:CityGraph):
    pass

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

def order_generator(num_order:int, time: int, num_city:int,CAPACITY,G:CityGraph):
    Orders = {}
    revenue_vector=[0]*num_order
    penalty_vector=[0]*num_order
    G.plot_graph()
    for i in range(0,num_order):
        id = i + time*num_order          # 订单 ID
        passenger = random.randint(1,CAPACITY)     # 乘客数
        departure = random.randint(0, num_city-1)           # 出发地
        destination = random.randint(0, num_city-1) # 目的地
        while destination == departure:           
            destination = random.randint(0, num_city)
        start_time = time                   # 起始时间
        end_time = start_time+1000 #+ random.randint(0, TIME)  #+ time_consume(departure,destination) # 截止时间
        virtual_departure = departure    # 预设为虚拟出发地
        try:
            distance,_ =G.get_intercity_path(departure, destination)
        except:
            print(departure,destination)
        battery =random.uniform(0, 10)+distance*per_distance_battery             # 需要的电量
        revenue_vector[i] = distance * 100 + passenger * 50 # 随便编
        penalty_vector[i] = passenger * 5 # 随便编
        # 创建 Order 对象
        order = Order(id, passenger, departure, destination, start_time, end_time, virtual_departure, battery,distance)
        
        Orders[id] = order
    return Orders , revenue_vector, penalty_vector

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
            if vehicle.which_city() == city_id:
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
                 virtual_departure)
        Cities[city_id] = city
    return Cities
def update(X:Dict):
    pass

