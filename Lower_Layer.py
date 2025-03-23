from typing import Dict
from gurobipy import *
from CITY_GRAPH import *
from CITY_NODE import *
from ORDER import *
from VEHICLE import *
from tool_func import *
from itertools import combinations

class Lower_Layer:
    """代码错了，约束越多越反动"""
    def __init__(self, city_graph: CityGraph, 
                 city_node: Dict[int, 'City'], 
                 Vehicle:Dict[int, 'Vehicle'], 
                 Order: Dict[int, 'Order'], 
                 name, group,time):
        """初始化参数和模型"""
        self.num_vehicle = len(Vehicle)
        self.num_order = len(Order)
        
        self.city_graph = city_graph  # City graph object
        self.city_node = city_node    # 字典
        self.Vehicle = Vehicle        # 字典
        self.Order = Order            # 字典

        self.model = Model(name)      # Gurobi
        
        self.group = group            # 用于划分的下标索引
        self.time = time
        self.real_id =[0]*self.num_order
        # 初始化决策变量
        self._add_variable_vehicle()
        self._add_variable_order()
        self._change_index()
    def _change_index(self):
        i = 0
        for order in self.Order.values():
            self.real_id[i] = order.id
            order.id = i
            i += 1
    def _add_variable_vehicle(self):
        """初始化车辆的决策变量，4种状态：dispatching, charging, idling, intercity"""
        self.X_Vehicle = self.model.addVars(self.num_vehicle, 4, 
                                            vtype=GRB.BINARY, name="var_vehicle")

    def _add_variable_order(self):
        """初始化订单的决策变量，表示是否被某车辆匹配"""
        self.X_Order = self.model.addVars(self.num_order, self.num_vehicle, 
                                          vtype=GRB.BINARY, name="var_order")

    def list_str(slef,my_list):
        return str("".join(map(str, my_list)))
    
    def get_decision(self):
        """返回车辆和订单的决策变量"""
        return self.X_Vehicle, self.X_Order
    
    def constrain_1(self):
        """先划成分，再扣帽子（指0和1）"""
        """对于在城市中的，能批添加则批添加"""
        self.model.addConstrs(
    (sum(self.X_Vehicle[v, c] for c in range(3)) == 1 for v in self.group[0]),
    name="constraints_1_1"
        )

        # constraints_1_2
        self.model.addConstrs(
            (self.X_Vehicle[v, 3] == 0 for v in self.group[0]),
            name="constraints_1_2"
        )

    def constrain_2(self):
        """对于在城市间的"""
        self.model.addConstrs(
            (sum(self.X_Vehicle[v, c] for c in range(3)) == 0 for v in self.group[1]),
            name="constraints_2_1"
        )

        self.model.addConstrs(
            (self.X_Vehicle[v, 3] == 1 for v in self.group[1]),
            name="constraints_2_2"
        )


    def constrain_3(self):
        """电池约束：防止电池不足时的错误匹配"""
        """足电，足车，客之信矣"""

        # 最多只能被一辆车匹配
        self.model.addConstrs(
            (sum(self.X_Order[order.id, vehicle_id] for vehicle_id in range(self.num_vehicle)) <= 1 
            for order in self.Order.values()),
            name="constrain_3_0"
            
        )


        for city in self.city_node.values():
            if not city.available_vehicles:
                # print(city.id)
                continue
            if not city.virtual_departure:
                continue
            # 获取每个城市的最低电池需求
            least_battery_demand = min(
                order.battery for order in city.virtual_departure.values()
            )
            # 按照路径划分订单,如果两者未重合或包含，那么不可同时匹配
            """
            ABCD
            EABCD 
            FEABCD
            FEABCDG
            HABCD
            """
            for order1, order2 in combinations(city.virtual_departure.values(), 2):
                _, path_order1 = self.city_graph.get_intercity_path(*order1.virtual_route())
                _, path_order2 = self.city_graph.get_intercity_path(*order2.virtual_route())
                if  list_str(path_order1) not in list_str(path_order2) and list_str(path_order2) not in list_str(path_order1):
                    self.model.addConstrs(
                        (self.X_Order[order1.id, vehicle.id] + self.X_Order[order1.id, vehicle.id] <= 1
                            for vehicle in city.vehicle_available.values()),
                        name=f"constrain_3_1_order_contain_1"
                    )
                
                

            for vehicle in city.available_vehicles.values():
                # 禁止电量不足的车辆匹配订单
                if vehicle.battery < least_battery_demand:
                    #print(vehicle.id)
                    self.model.addConstrs(
                        (self.X_Order[order.id, vehicle.id] == 0 for order in city.virtual_departure.values()),
                        name=f"constrain_3_2_0_vehicle_{vehicle.id}"
                    )
                    # 强制充电
                    if vehicle.id in self.group[0]:
                        self.model.addConstr(
                                (self.X_Vehicle[vehicle.id, 1] == 1),
                                name=f"constrain_3_2_2_vehicle_{vehicle.id}"
                            )
                    if len(vehicle.get_orders()) == 0:
                        self.model.addConstr(
                            (self.X_Vehicle[vehicle.id, 0] == 0),
                            name=f"constrain_3_2_1_vehicle_{vehicle.id}"
                        )
                
                for order in city.virtual_departure.values():
                    """确保闲置充电者无订单"""
                    self.model.addConstr(
                        self.X_Order[order.id, vehicle.id] + 
                        (self.X_Vehicle[vehicle.id, 1] + self.X_Vehicle[vehicle.id, 2]) <= 1,
                        name=f"constrain_3_3_order_{order.id}_vehicle_{vehicle.id}"
                    )

                    # 禁止不在当前城市的车辆匹配
                    if vehicle.intercity != order.virtual_departure:
                        self.model.addConstr(
                            (self.X_Order[order.id, vehicle.id] == 0),
                            name=f"constrain_3_4_order_{order.id}_vehicle_{vehicle.id}"
                        )
                    # 电量不足时禁止匹配
                    if vehicle.battery <= order.battery:
                        # print(vehicle.id)
                        self.model.addConstr(
                            (self.X_Order[order.id, vehicle.id] == 0),
                            name=f"constrain_3_5_order_{order.id}_vehicle_{vehicle.id}"
                        )
                    # 时间窗口约束
                    if order.end_time < order.least_time_consume  + vehicle.time:
                        self.model.addConstr(
                            (self.X_Order[order.id, vehicle.id] == 0),
                            name=f"constrain_3_6_order_{order.id}_vehicle_{vehicle.id}"
                        )
                    if vehicle.get_orders():
                        # 含有订单者
                        _,path_order = self.city_graph.get_intercity_path(*order.virtual_route())
                        if str(vehicle.longest_path) not in list_str(path_order) and list_str(path_order) not in str(vehicle.longest_path):
                            self.model.addConstr(
                                self.X_Order[order.id ,vehicle.id]==0,
                                name = f"constrain_3_7_order_{order.id}_vehicle_{vehicle.id}"
                            )
                    else:
                        continue
                        # 暂时跳过
                        # 路径约束：车辆必须遵守路径规则
                        # 需要详细的分情况解决
                        furthest = self.city_graph.passby_most(vehicle.get_orders())
                        _,past = self.city_graph.passby_most(furthest[0],order.departure)
                        current = str(furthest)[len(str(past)):]
                        _,path_order3 = self.city_graph.get_intercity_path(*order.route())   
                        if str(current)[1:-1] not in str(path_order3) or str(path_order3)[1:-1] not in str(current):
                            self.model.addConstr(
                                (self.X_Order[order.id, vehicle.id] == 0),
                                name=f"constrain_3_7_invalid_route_order_{order.id}_vehicle_{vehicle.id}"
                            )
    def constrain_4(self):
        """充电站有限的"""
        for city in self.city_node.values():
            # 计算每个城市充电站的充电需求和约束
            charging_demand = 0
            for vehicle in city.available_vehicles.values():
                charging_demand += self.X_Vehicle[vehicle.id, 1]
                # 禁止充电
                if vehicle.battery == 100:
                    self.model.addConstr(
                        self.X_Vehicle[vehicle.id,1] == 0,
                        name=f"constrain_4_1{city.id}"
                    )
            # 添加约束：充电需求不得超过城市的充电站容量
            self.model.addConstr(charging_demand <= city.charging_capacity,
                                 name=f"constrain_4_2{city.id}")

    def constrain_5(self):
        """约束：至少有一个订单的车辆，其充电状态为0"""
       
        for vehicle in self.Vehicle.values():
            if vehicle.id in self.group[0]:
            # 如果车辆有订单,这个并不能应对初始情况
                if len(vehicle.get_orders()) > 0:
                    # 为车辆添加约束：其充电状态为0,闲置同理
                    self.model.addConstr(self.X_Vehicle[vehicle.id, 0] == 1,
                                        name=f"constrian_5_0_{vehicle.id}")
                else:
                    # 是否有订单，这个确保了不会出现无订单却dispatching
                    self.model.addConstr(
                        sum(self.X_Order[order.id, vehicle.id] for order in self.Order.values()) 
                        >= self.X_Vehicle[vehicle.id, 0],
                        name=f"constrain_5_{vehicle.id}"
                    )
                # 容量约束
                self.model.addConstr(
                    sum(self.X_Order[order.id, vehicle.id]* order.passenger 
                        for order in self.Order.values())<= 7-vehicle.get_capacity(), #vehicle.get_capacity()
                    name=f"constrain_5_1_{vehicle.id}"
                )
  
    def set_objective(self, cost_matrix):
        # 好像gurobi不能进行矩阵计算
        """目前无法实现“动一动”功能。
            办法一：限制连续dispatching次数，可以在vehicle中增加记录功能
            一种办法：约束函数仅对每个城市构建，而非全局
        """
        try:
            order_revenue = quicksum(self.X_Vehicle[v, 0]*self.X_Order[order.id, v] * order.revenue for order in self.Order.values() for v in self.group[0])
        except:
            for order in self.Order.values():
                for v in self.group[0]:
                    try:
                        # 打印当前下标
                        print(f"o: {order.id}, v: {v}")
                        # 访问目标值，模拟表达式
                        _ = self.X_Vehicle[v, 0] * self.X_Order[order.id, v] * order.revenue
                    except IndexError as e:
                        print(f"IndexError with o={order.id}, v={v}: {e}")
                        raise
        vehicle_cost = quicksum(self.X_Vehicle[v, c] * cost_matrix[v][c] for v in range(self.num_vehicle) for c in range(0,4))
        order_penalty = quicksum(1 - quicksum(self.X_Order[order.id, v] for v in self.group[0]) * order.penalty for order in self.Order.values())

        self.model.setObjective(order_revenue - vehicle_cost - order_penalty, GRB.MAXIMIZE)
    def get_real_id(self):
        return self.real_id
    
    
    
    

        
            
                
    
            
        
        
        
    

