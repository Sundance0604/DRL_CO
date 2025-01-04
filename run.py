from gurobipy import *
from typing import Dict
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
num_order = 30
num_city = 20
TIME = 24
CAPACITY = 7
row = [10, 3, 1, 10]
Vehicles ={}
speed = 20
cancel_penalty = 300
revenue_vector=[]
penalty_vector=[]
matrix = np.tile(row, (num_vehicle, 1))

# 初始化
Vehicles = vehicle_generator(num_vehicle, 20)
orders_unmatched = {}
G = CityGraph(num_city,0.3, (20,100))
name = "navie"
cancel_penalty = 300
order_canceled = 0
for time in range(TIME):
    group = [[],[]]
    for vehicle in Vehicles.values():
        if vehicle.decision == 3:
            group[1].append(vehicle.id)
        else:
            group[0].append(vehicle.id)
    if len(group[0]) == 0:
        continue
    Orders, revenue_vector, penalty_vector = order_generator(num_order, time, num_city,CAPACITY,G)
    for order in Orders.values():
        orders_unmatched[order.id] = order

    orders_virtual = orders_unmatched
    city_node = city_node_generator(G,orders_virtual,Vehicles,orders_unmatched)
    # order_virtual = RL.allocate(orders_unmatched) #此处强化学习

    temp_Lower_Layer = Lower_Layer(num_vehicle, num_order, G, 
                                city_node ,Vehicles, orders_unmatched ,name,group)
    temp_Lower_Layer.get_decision()
    temp_Lower_Layer.constrain_1()
    temp_Lower_Layer.constrain_2()
    temp_Lower_Layer.constrain_3()
    temp_Lower_Layer.constrain_4()
    temp_Lower_Layer.constrain_5()

    temp_Lower_Layer.set_objective(matrix, revenue_vector, cancel_penalty*order_canceled)
    temp_Lower_Layer.model.optimize()
    if temp_Lower_Layer.model.status == GRB.OPTIMAL:
        # 打印变量的最优值
        print("Optimal solution:")
        i = 0
        with open(f"output_{time}.txt", "w") as file:
            for v in temp_Lower_Layer.model.getVars():
                if v.x == 1:
                    file.write(f"{v.varName} = {v.x}\n")  # 将结果写入文件
            # 打印目标函数值
        print("Objective value:", temp_Lower_Layer.model.objVal)
        
        
    else:
        print("No optimal solution found.")
    _,var_order = temp_Lower_Layer.get_decision()
    update_var(temp_Lower_Layer, Vehicles,Orders,orders_unmatched)
    update_vehicle(Vehicles)
    order_canceled = update_order(orders_unmatched)
    