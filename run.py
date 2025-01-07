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

num_vehicle = 20
num_order = 10
num_city = 8
TIME = 100
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
G = CityGraph(num_city, 0.3, (20, 100))
# 获取图的邻接信息
adj_data = G.G.adjacency()

# 将邻接信息保存为CSV文件
with open('adjacency.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Node', 'Adjacent Nodes'])  # CSV表头
    for node, neighbors in adj_data:
        writer.writerow([node, list(neighbors.keys())])  # 每一行包含节点和它的邻居

name = "navie"
cancel_penalty = 300
order_canceled = 0
Total_order = {}
objval = 0.114514
# 记录所有时间点车辆信息
all_vehicle_records = []

for time in range(TIME):
    group = [[], []]
    
    # 记录每个时刻的车辆信息
    vehicle_records = []
    # 生成订单
    Orders = order_generator(num_order, time, num_city-1, CAPACITY, G)
    for order in Orders.values():
        orders_unmatched[order.id] = order
        Total_order[order.id] = order

    orders_virtual = orders_unmatched
    if time == 0:
        city_node = city_node_generator(G, orders_virtual, Vehicles, orders_unmatched)

    for vehicle in Vehicles.values():
        # 将车辆信息存入字典
        vehicle_records.append({
            "id": vehicle.id,
            "time": vehicle.time,
            "into_city": vehicle.into_city,
            "intercity": vehicle.intercity,
            "passenger": vehicle.get_capacity(),
            "decision": vehicle.decision,
            "battery": vehicle.battery,
            "whether_city":vehicle.whether_city,
            "matched_order": vehicle.orders,
            "num_matched": len(vehicle.get_orders()),
            "orders_unmatched" :len(orders_unmatched),
            "object_value": objval
        })

        # 通过条件分组
        if vehicle.whether_city:
            group[0].append(vehicle.id)
        else:
            group[1].append(vehicle.id)

    # 将当前时刻的车辆记录添加到所有记录中
    all_vehicle_records.append(vehicle_records)

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
        temp_Lower_Layer.set_objective(matrix, total_penalty)
        temp_Lower_Layer.model.optimize()

        if temp_Lower_Layer.model.status == GRB.OPTIMAL:
            # 打印最优解
            print(f"{time}次，Optimal solution:")
            i = 0
            with open(f"output_{time}.txt", "w") as file:
                for v in temp_Lower_Layer.model.getVars():
                    if v.x == 1:
                        file.write(f"{v.varName} = {v.x}\n")  # 将结果写入文件
            print("Objective value:", temp_Lower_Layer.model.objVal)
            objval = temp_Lower_Layer.model.objVal
        else:
            temp_Lower_Layer.model.computeIIS()
            temp_Lower_Layer.model.write('iis.ilp')  # 保存不可行约束
            print(f"{time}次，No optimal solution found.")
            self_update(Vehicles, G)
            objval = 0.114514
        
        _, var_order = temp_Lower_Layer.get_decision()
        update_var(temp_Lower_Layer, Vehicles, orders_unmatched)
        vehicle_in_city = update_vehicle(Vehicles, battery_consume, battery_add, speed, G)
        order_canceled = update_order(orders_unmatched, time, speed)
       
    else:
        self_update(Vehicles, G)
        print(f"{time}次，没有车")
        vehicle_in_city = update_vehicle(Vehicles, battery_consume, battery_add, speed, G)
        order_canceled = update_order(orders_unmatched, time, speed)
        objval = 0.114514
    print(f"{len(orders_unmatched)}订单未被匹配")
# 将记录保存为 CSV 文件
csv_file = "vehicle_records.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    # 写入CSV文件的列名
    fieldnames = ["id", "time", "into_city", "intercity", "passenger", "decision"
                  ,"battery", "whether_city","matched_order","num_matched"
                  ,"orders_unmatched","object_value"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    
    # 遍历所有时刻的车辆记录
    for vehicle_records in all_vehicle_records:
        writer.writerows(vehicle_records)

print(f"Vehicle information saved to {csv_file}")
