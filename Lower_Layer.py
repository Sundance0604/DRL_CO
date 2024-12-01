from typing import Dict
from gurobipy import *
from CITY_GRAPH import *
from CITY_NODE import *
from ORDER import *
from VEHICLE import *
from tool_func import *
import SETTING

SET = SETTING()
class Lower_Layer:
    """代码错了，约束越多越反动"""
    def __init__(self, num_vehicle:int, num_order:int, city_graph: CityGraph, 
                 city_node: Dict[int, 'City'], 
                 Vehicle:Dict[int, 'Vehicle'], Order: Dict[int, 'Order'], name, group):
        """初始化参数和模型"""
        self.num_vehicle = num_vehicle
        self.num_order = num_order
        
        self.city_graph = city_graph  # City graph object
        self.city_node = city_node    # 字典
        self.Vehicle = Vehicle        # 字典
        self.Order = Order            # 字典

        self.model = Model(name)      # Gurobi
        
        self.group = group            # 用于划分的下标索引

        # 初始化决策变量
        self._add_variable_vehicle()
        self._add_variable_order()
    
    def _add_variable_vehicle(self):
        """初始化车辆的决策变量，4种状态：dispatching, charging, idling, intercity"""
        self.X_Vehicle = self.model.addVars(self.num_vehicle, 4, 
                                            vtype=GRB.BINARY, name="var_vehicle")

    def _add_variable_order(self):
        """初始化订单的决策变量，表示是否被某车辆匹配"""
        self.X_Order = self.model.addVars(self.num_order, self.num_vehicle, 
                                          vtype=GRB.BINARY, name="var_order")

    def get_decision(self):
        """返回车辆和订单的决策变量"""
        return self.X_Vehicle, self.X_Order
    
    def constrain_1(self):
        """先划成分，再扣帽子（指0和1）"""
        """对于在城市中的，能批添加则批添加"""
        constraints_1_1 = [
            (sum(self.X_Vehicle[v, c] for c in range(3)) == 1) for v in range(self.group[0])
        ]
        constraints_1_2 = [
            (self.X_Vehicle[v, 3] == 0) for v in range(self.group[0])
        ]

        self.model.addConstrs(constraints_1_1, name="constraints_1_1")
        self.model.addConstrs(constraints_1_2, name="constraints_1_2")

    def constrain_2(self):
        """对于在城市间的"""
        constraints_2_1 = [
            (sum(self.X_Vehicle[v, c] for c in range(3)) == 0) for v in range(self.group[0])
        ]
        constraints_2_2 = [
            (self.X_Vehicle[v, 3] == 1) for v in range(self.group[0])
        ]

        self.model.addConstrs(constraints_2_1, name="constraints_2_1")
        self.model.addConstrs(constraints_2_2, name="constraints_2_2")

    def constrain_3(self):
        """电池约束：防止电池不足时的错误匹配"""
        """足电，足车，客之信矣"""
        battery_demand = {}
        constrain_3_1 = []
        constrain_3_2 = []
        constrain_3_3 = []
        constrain_3_4 = []
        constrain_3_5 = []
        
        for city in self.city_node.values():
            # 获取每个城市的最低电池需求
            least_battery_demand = min(
                order.battery_demand for order in city.get_virtual_departure().values())
            for vehicle in city.available_vehicles.values():
                
                if vehicle.battery < least_battery_demand:
                        # 禁止订单匹配,是列哦
                    constrain_3_1.append(self.X_Order[:,vehicle.id] == 0)
                    # 车辆禁止执行出发操作
                    constrain_3_2.append(self.X_Vehicle[vehicle.id, 0] == 0)  
    
                else:
                    # 车辆电量足够，但如果电量小于订单需求，禁止匹配
                    for order in city.get_virtual_departure().values():
                        if vehicle.battery <= order.battery_demand:
                            constrain_3_1.append(self.X_Order[order.id, vehicle.id] == 0)
                        # 对于电量足够匹配的订单，不可违背路径：
                        else:
                            furthest = self.city_graph.passby_most(vehicle.get_orders())
                            _, current = self.city_graph.get_dijkstra_results(order.virtual_departure).values()
                            _, to_furthest = self.city_graph.get_dijkstra_results(order.destination,furthest[-1]).values()
                            if route_combine(furthest, current, to_furthest) == False:
                                constrain_3_1.append(self.X_Order[order.id, vehicle.id] == 0)
                            #如果时间不足，同样不可匹配
                            _, deadline = order.timewindow
                            if time_consume(order) > deadline - vehicle.time:
                                constrain_3_4.append(self.X_Order[order.id, vehicle.id] == 0)
                            # 如果接单导致载客过多，同样不可匹配
                            if order.passenger_count + vehicle.get_capacity > SET.capacity:
                                constrain_3_5.append(self.X_Order[order.id, vehicle.id] == 0)

            else:
                # 对于不在当前城市的车辆，禁止匹配
                for order in city.get_virtual_departure().values():
                    constrain_3_3.append(self.X_Order[order.id, vehicle.id] == 0)
        
        self.model.addConstrs(constrain_3_1, name="constrain_3_1")
        self.model.addConstrs(constrain_3_2, name="constrain_3_2")
        self.model.addConstrs(constrain_3_3, name="constrain_3_3")
        self.model.addConstrs(constrain_3_4, name="constrain_3_4")
        self.model.addConstrs(constrain_3_4, name="constrain_3_5")

    def constrain_4(self):
        """充电站有限的"""
        constrian_4 = []
        for city in self.city_node.values():
            # y^k_u之和不得大于容量
            constrian_4.append(sum(self.X_Vehicle[vehicle.id, 1]<=city.charging_capacity) 
                               for vehicle in city.available_vehicles.values())
        self.model.addConstrs(constrian_4, name="constrain_4")

    def constrain_5(self):
        """有单不得充电、闲置，只能dispatching，不需要预传值"""
        constrain_5 = [ self.X_Vehicle[key, 1] == 1 
                       for key,vehicle in self.Vehicle.items() 
                       if len(vehicle.get_orders()) > 0]
        self.model.addConstrs(constrain_5, name="constrain_5")
    
    def set_objective(self, cost_matrix, revenue_vector, penalty_vector):
        # 好像gurobi不能进行矩阵计算
        """目前无法实现“动一动”功能。
            办法一：限制连续dispatching次数，可以在vehicle中增加记录功能
            一种办法：约束函数仅对每个城市构建，而非全局
        """
        order_revenue = quicksum(self.X_Order[o, v] * revenue_vector[o] for o in range(self.num_orders) for v in range(self.num_vehicles))
        vehicle_cost = quicksum(self.X_Vehicle[v, c] * cost_matrix[c][v] for v in range(self.num_vehicles) for c in range(4))
        order_penalty = quicksum((1 - quicksum(self.X_Order[o, v] for v in range(self.num_vehicles))) * penalty_vector[o] for o in range(self.num_orders))

        self.model.setObjective(order_revenue - vehicle_cost - order_penalty, GRB.MAXIMIZE)
    
    

        
            
                
    
            
        
        
        
    

