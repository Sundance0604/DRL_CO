import logging
from typing import List
from VEHICLE import *
from ORDER import *
# 设置日志记录
logging.basicConfig(level=logging.INFO)

class City:
    __slots__ = ['id', 'neighbor', 'vehicle_available', 'charging_capacity', 
                 'real_departure', 'virtual_departure']

    def __init__(self, city_id: int, neighbor: List[int], 
                 vehicle_available: Dict[int,'Vehicle'], 
                 charging_capacity: int, 
                 real_departure: Dict[int, 'Order'], 
                 virtual_departure: Dict[int, 'Order']):


        self.id = city_id
        self.neighbor = neighbor
        self.vehicle_available = vehicle_available
        self.charging_capacity = charging_capacity
        self.real_departure = real_departure
        self.virtual_departure = virtual_departure

    def __repr__(self):
        """返回城市的简洁信息"""
        return (f"City(id={self.id}, "
                f"neighbors={self.neighbor}, "
                f"available_vehicles={len(self.vehicle_available)}, "
                f"charging_capacity={self.charging_capacity}, "
                f"real_departure={len(self.real_departure)}, "
                f"virtual_departure={len(self.virtual_departure)})")
   
    

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
    
    def get_order_list(self):
        return list(self.virtual_departure.values())    
    
    def clean_all(self):
        self.vehicle_available = {}
        self.virtual_departure = {}
        self.real_departure = {}
    
    def city_seat_count(self,capacity):
        return sum([capacity - vehicle.get_capacity() for vehicle in self.vehicle_available.values()])
