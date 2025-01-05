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

def update_var(temp_Lower_Layer:Lower_Layer, Vehicles:Dict,orders_unmatched:Dict):
    i = 0
    real_id = temp_Lower_Layer.get_real_id()
    
    delet_list = []
    num_vehicle = temp_Lower_Layer.num_vehicle
    j = 0
    for order in orders_unmatched.values():
        order.id = real_id[j]
        j+= 1
    for v in temp_Lower_Layer.model.getVars():
        if temp_Lower_Layer.model.status != GRB.OPTIMAL:
            break
        # 按下标获取车辆
        if i < 4 * num_vehicle:
            
            # 对于dispatching
            if i%4 == 0:
                # Vehicles[i//4].update_time()
                if v.x == 1.0:
                    Vehicles[i//4].decision = 0
                    # 前往的城市在订单的迭代中修改
            # 对于charging
            if i%4 == 1:
                if v.x == 1.0 :
                    Vehicles[i//4].decision = 1
                
            # 对于idle
            if i%4 == 2:
                if v.x == 1.0 :
                    Vehicles[i//4].decision = 2
                    
            if i%4 == 3:
                if v.x == 1.0 :
                    Vehicles[i//4].decision = 3
                    # 整一个
        else:
            
            # 如果该订单被匹配
            if v.x == 1.0:
                try:
                    order_temp = orders_unmatched[real_id[(i - 4 * num_vehicle) // num_vehicle]] 
                except:
                    for order in orders_unmatched.values():
                        print(order)
                    print(real_id[(i - 4 * num_vehicle) // num_vehicle])
                    
                vehicle_temp = Vehicles[(i - 4 * num_vehicle) % num_vehicle]
                
                # 添加此订单并修改目标城市
                delet_list.append(order_temp.id)
                # 在此添加车辆ID

                # 添加订单和车辆匹配
                vehicle_temp.add_order(order_temp)
                
                order_temp.match_vehicle(vehicle_temp.id)
                # 只有当车辆ID不在changed_vehicle中时，才会执行以下操作
                """
                if vehicle_temp.id not in changed_vehicle:
                    
                    if order_temp.virtual_departure != order_temp.departure:
                        vehicle_temp.move_to_city(order_temp.departure)
                    else:
                        _, path = temp_Lower_Layer.city_graph.get_intercity_path(*order_temp.route())
                        
                        vehicle_temp.move_to_city(path[1])
                        
                changed_vehicle.append(vehicle_temp.id)
                """
        i+=1
    if delet_list:
        for order_id in delet_list:
            del orders_unmatched[order_id]
   
                
            
   
    
    

def update_vehicle(Vehicles:Dict,battery_consume:int,battery_add:int,speed:int,G:CityGraph):
    vehicle_intercity = 0

    for vehicle in Vehicles.values():
        
        if vehicle.decision == 3:

            vehicle_intercity += 1
            vehicle.battery -= battery_consume
            
            distance,_ = G.get_intercity_path(vehicle.into_city,vehicle.intercity)
            if distance/speed < vehicle.time - vehicle.time_into_city:
                vehicle.move_into_city()
                vehicle.decision = 0
                delete_list = []
                for order in vehicle.get_orders():
                    if order.destination == vehicle.intercity and vehicle.whether_city:
                        delete_list.append(order.id)
                vehicle.delete_orders(delete_list)
                
            else:
                vehicle.decision = 3

        if vehicle.decision == 1:
            
            vehicle.battery += battery_add
            if vehicle.battery >= 100:
                vehicle.battery = 100

        if vehicle.last_decision == 0 :
            if vehicle.decision != 3:
                
                order = vehicle.get_orders()[0]
                if vehicle.intercity != order.departure:
                    vehicle.move_to_city(order.departure)
                else:
                    _, path = G.get_intercity_path(*order.route())
                    vehicle.move_to_city(path[1])
                vehicle.decision = 3
            
        

            # 强制驱逐
            """
            if vehicle.last_decison == 0 and vehicle.decision == 0:
                order = vehicle.get_orders()[0]
                print(order)
                if vehicle.intercity != order.departure:
                    vehicle.move_to_city(order.departure)
                else:
                    _, path = G.get_intercity_path(*order.route())
                    vehicle.move_to_city(path[1])
                vehicle.update_state(3)
            vehicle.update_state(0)
            """
        vehicle.update_time()
            
    return len(Vehicles)-vehicle_intercity

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
 
def self_update(Vehicles:Dict,G:CityGraph):
    for vehicle in Vehicles.values():
        if vehicle.get_capacity() > 3 and vehicle.decision == 0 :
            vehicle.decision = 3
            order = vehicle.get_orders()[0]
            if vehicle.intercity != order.departure:
                vehicle.move_to_city(order.departure)
            else:
                _, path = G.get_intercity_path(*order.route())
                vehicle.move_to_city(path[1])
        elif vehicle.decision == 3:
            vehicle.decision = 3
        elif vehicle.decision == 0:
            vehicle.decision = 0
        else:
            vehicle.decision = 2
    



        