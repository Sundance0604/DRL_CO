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

Vehicles = vehicle_generator(SETTING.num_vehicle, SETTING.num_city)
orders_unmatched = {}
G = CityGraph(SETTING.num_nodes,SETTING.edge_prob,SETTING.weight_range)
city_node = city_node_generator(G)
name = "navie"

for turn in range(0, SETTING.train):
    for t in range(0, set.total_time):

        for order in order_generator(SETTING.num_order, SETTING.num_city):
            orders_unmatched[order.id] = order
        
        order_virtual = RL.allocate(orders_unmatched) #此处强化学习
        temp_Lower_Layer = Lower_Layer(SETTING.num_vehicle, SETTING.num_order, G, 
                                    city_node ,Vehicles, order_virtual ,name)
        temp_Lower_Layer.get_decision()
        temp_Lower_Layer.constrain_1()
        temp_Lower_Layer.constrain_2()
        temp_Lower_Layer.constrain_3()
        temp_Lower_Layer.constrain_4()
        temp_Lower_Layer.constrain_5()
        temp_Lower_Layer.set_objective()
        temp_Lower_Layer.model.optimize()
        if temp_Lower_Layer.model.status == GRB.OPTIMAL:
            # 打印变量的最优值
            print("Optimal solution:")
            for v in temp_Lower_Layer.model.getVars():
                print(f"{v.varName} = {v.x}")
            # 打印目标函数值
            print("Objective value:", temp_Lower_Layer.model.objVal)
            RL.reward(temp_Lower_Layer.model.objVal)#此处返回给强化学习
            update(temp_Lower_Layer.model.getVars(),t)
        else:
            print("No optimal solution found.")
            break
    RL.update()
    


    