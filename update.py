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

def update_var(temp_Lower_Layer:Lower_Layer, Vehicles:Dict,Orders:Dict,orders_unmatched:Dict):
    i = 0
    real_id = temp_Lower_Layer.get_real_id()
    changed_vehicle = []
    delet_list = []
    num_vehicle = temp_Lower_Layer.num_vehicle
    for v in temp_Lower_Layer.model.getVars():
       
        # 按下标获取车辆
        if i < 4 * num_vehicle:
            
            # 对于dispatching
            if i%4 == 0:
                Vehicles[i//4].update_time()
                if v.x == 1.0:
                    Vehicles[i//4].update_state(0) 
                    # 前往的城市在订单的迭代中修改
            # 对于charging
            if i%4 == 1:
                if v.x == 1.0:
                    Vehicles[i//4].update_state(1)
            # 对于idle
            if i%4 == 2:
                if v.x == 1.0:
                    Vehicles[i//4].update_state(2)
        else:
            
            # 如果该订单被匹配
            if v.x == 1.0:
                order_temp = Orders[(i - 4 * num_vehicle) // num_vehicle] 
                vehicle_temp = Vehicles[(i - 4 * num_vehicle) % num_vehicle]
                
                # 添加此订单并修改目标城市
                delet_list.append(order_temp.id)
                # 在此添加车辆ID

                # 添加订单和车辆匹配
                vehicle_temp.add_order(order_temp)
                order_temp.match_vehicle(vehicle_temp.id)
                # 只有当车辆ID不在changed_vehicle中时，才会执行以下操作
                if vehicle_temp.id not in changed_vehicle:
                    
                    if order_temp.virtual_departure != order_temp.departure:
                        vehicle_temp.move_to_city(order_temp.departure)
                    else:
                        _, path = temp_Lower_Layer.city_graph.get_intercity_path(*order_temp.route())
                        
                        vehicle_temp.move_to_city(path[1])
                        
                changed_vehicle.append(vehicle_temp.id)
        i+=1
    j = 0
    for order in orders_unmatched.values():
        order.id = real_id[j]
        j+= 1
    print(delet_list)
    print(real_id)
    for order_id in delet_list:
        del orders_unmatched[order_id]

def update_vehicle(Vehicles:Dict,battery_consume:int,battery_add:int,speed:int,G:CityGraph):
    for vehicle in Vehicles.values():
        vehicle.update_time()

        if vehicle.decision == 3:
            vehicle.battery -= battery_consume
            distance,_ = G.get_intercity_path(vehicle.into_city,vehicle.intercity)
            if distance/speed < vehicle.time - vehicle.into_city_time:
                vehicle.move_into_city()
            else:
                if vehicle.last_decison == 0:
                    vehicle.last_decision = 3
        if vehicle.decision == 1:
            vehicle.battery += battery_add
            if vehicle.battery >= 100:
                vehicle.battery = 100

def update_order(order_unmatched:Dict,time:int,speed:int):
    to_delete = []
    order_canceled = 0  
    for order_id, order in order_unmatched.items():
        if order.end_time - time < order.distance / speed:
            order_canceled += 1  
            to_delete.append(order_id)
    
   
    for order_id in to_delete:
        del order_unmatched[order_id]
    
    return order_canceled
 
    



        