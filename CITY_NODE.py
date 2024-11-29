import logging
from typing import List
from VEHICLE import *
from ORDER import *
# 设置日志记录
logging.basicConfig(level=logging.INFO)

class City:
    __slots__ = ['city_id', 'neighbor', 'vehicle_available', 'charging_capacity', 
                 'real_departure', 'virtual_departure']

    def __init__(self, city_id: int, neighbor: List[int], 
                 vehicle_available: Dict[int,'Vehicle'], 
                 charging_capacity: int, 
                 real_departure: Dict[int, 'Order'], 
                 virtual_departure: Dict[int, 'Order']):


        self.city_id = city_id
        self.neighbor = neighbor
        self.vehicle_available = vehicle_available
        self.charging_capacity = charging_capacity
        self.real_departure = real_departure
        self.virtual_departure = virtual_departure

    def __repr__(self):
        """返回城市的简洁信息"""
        return (f"City(id={self.city_id}, "
                f"neighbors={self.neighbor}, "
                f"available_vehicles={len(self.vehicle_available)}, "
                f"charging_capacity={self.charging_capacity})")

    # --- 属性访问方法 ---
    @property
    def id(self) -> int:
        return self.city_id
    
    @property
    def available_vehicles(self) -> Dict[int,'Vehicle']:
        return self.vehicle_available
    
    @property
    def neighbors(self) -> List[int]:
        return self.neighbor
    
    @property
    def real_departures(self) -> Dict[int, 'Order']:
        return self.real_departure
    
    @property
    def virtual_departures(self) -> Dict[int, 'Order']:
        return self.virtual_departure

    # --- 方法 ---
    def add_available_vehicle(self, vehicle_id: int, vehicle: Vehicle):
        """增加可调度的车辆"""
        if vehicle_id not in self.vehicle_available.keys():
            self.vehicle_available[vehicle_id] = vehicle
            
    def remove_available_vehicle(self, vehicle_id: int):
        """删除不可调度的车辆"""
        if vehicle_id in self.vehicle_available.keys():
            del self.vehicle_available[vehicle_id]
            

    def add_real_departure(self, order_id: int, order: Order):
        """增加实际出发点订单"""
        if order_id not in self.real_departure.keys():
            self.real_departure[order_id] = order
            
    def remove_real_departure(self, order_id: int):
        """删除实际出发点订单"""
        if order_id in self.real_departure:
            del self.real_departure[order_id]
            
    def add_virtual_departure(self, order_id: int, order: Order):
        """增加虚拟出发点订单"""
        if order_id not in self.virtual_departure.keys():
            self.virtual_departure[order_id] = order
            

    def remove_virtual_departure(self, order_id: int):
        """删除虚拟出发点订单"""
        if order_id in self.virtual_departure:
            del self.virtual_departure[order_id]
            

    def update_virtual_departure(self, new_virtual_departure: Dict[int, 'Order']):
        """更新虚拟出发点订单列表"""
        self.virtual_departure = new_virtual_departure
        
    def get_charging_capacity(self) -> int:
        """获取充电站容量"""
        return self.charging_capacity

    def get_available_vehicle_count(self) -> int:
        """获取可调度的车辆数量"""
        return len(self.vehicle_available)

    def get_real_departure_count(self) -> int:
        """获取实际出发点订单数量"""
        return len(self.real_departure)

    def get_virtual_departure_count(self) -> int:
        """获取虚拟出发点订单数量"""
        return len(self.virtual_departure)
    
    

