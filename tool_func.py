from ORDER import *
from CITY_GRAPH import *
from VEHICLE import *
import numpy as np


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

def time_consume(order:Order):
    pass
def time_cost(u:int , v:int):
    pass

def vehicle_generator(num_vehicle:int, num_city:int):
    Vehicles = {}
    for i in range(0,num_vehicle):
        vehicle_id = i          # 车辆编号
        into_city = 0        #默认城市内
        intercity = random.randint(0, num_city)    # 当前所在城市编号 
        decision = Decision.IDLE  # 当前决策
        battery = random.uniform(0, 100)          # 电池剩余电量（百分比）
        orders = {}  # 无订单

        # 创建 Vehicle 对象
        vehicle = Vehicle(vehicle_id, 0, into_city, intercity, decision, battery, orders)
        Vehicles[vehicle] = vehicle
    return Vehicles

def order_generator(num_order:int, time: int, num_city:int,CAPACITY,TIME):
    Orders = {}
    for i in range(0,num_order):
        order_id = i             # 订单 ID
        passenger_count = random.randint(1,CAPACITY)     # 乘客数
        departure = random.randint(0, num_city)           # 出发地
        destination = random.randint(0, num_city)
        while destination == departure:           # 目的地
            destination = random.randint(0, num_city)
        start_time = time                   # 起始时间
        end_time = start_time + random.randint(0, TIME) + time_consume(departure,destination) # 截止时间
        virtual_departure = departure    # 虚拟出发地
        battery =random.uniform(0, 100)               # 需要的电量
        # 创建 Vehicle 对象
        order = Order(order_id, passenger_count, departure, destination, start_time, end_time, virtual_departure, battery)
        Orders[order_id] = order
    return Orders

