import numpy as np
from typing import Dict
from enum import Enum
import logging
from ORDER import *

# 设置日志记录
logging.basicConfig(level=logging.INFO)

# 定义车辆决策枚举
class Decision(Enum):
    DISPATCHING = 0
    CHARGING = 1
    IDLE = 2

# 定义车辆类
class Vehicle:
    __slots__ = ['data', 'orders']  # 限制属性，节省内存
    
    def __init__(self, vehicle_id: int, time: int, into_city: int, 
                 intercity: int, decision: Decision, battery: float, 
                 orders: Dict[int, 'Order'] = None):

        self.data = np.array([vehicle_id, time, into_city, intercity, decision.value, battery], dtype=object)
        self.orders = orders if orders else {}  # 初始化订单字典
    
    def __repr__(self):
        return (f"Vehicle(id={self.data[0]}, time={self.data[1]}, "
                f"into_city={self.data[2]}, intercity={self.data[3]}, "
                f"state={self.data[4]}, battery={self.data[5]})")
    
    @classmethod
    def from_dict(cls, vehicle_dict):
        """从字典创建车辆实例"""
        return cls(
            vehicle_id=vehicle_dict["vehicle_id"],
            time=vehicle_dict["time"],
            into_city=vehicle_dict["into_city"],
            intercity=vehicle_dict["intercity"],
            decision=Decision(vehicle_dict["decision"]),
            battery=vehicle_dict["battery"],
            orders=vehicle_dict.get("orders", {})
        )
    
    @staticmethod
    def compute_battery_cost(decision: Decision, cost_battery: Dict[int, float]):
        """计算电量消耗"""
        return cost_battery.get(decision.value, 0)
    @property
    def id(self)->int:
        return self.data[0]
    @property
    def battery(self):
        """获取电量"""
        return self.data[5]
    @property
    def time(self):
        return self.data[1]
    @battery.setter
    def battery(self, value):
        if value < 0:
            raise ValueError("Battery level cannot be negative!")
        self.data[5] = value
    
    def update_time(self):
        """更新时间到 t+1"""
        self.data[1] += 1

    def if_in_city(self):
        """获取当前位置，城市内为 False"""
        return self.data[2] != 0
    
    def which_city(self):
        """返回当前所在城市编号"""
        return self.data[3] if not self.if_in_city() else None
    
    def into_city(self):
        """进入城市"""
        self.data[2] = 0

    def intercity(self, city_id: int):
        """离开当前城市，前往指定城市"""
        self.data[2] = self.data[3]
        self.data[3] = city_id
    
    def update_state(self, decision: Decision):
        """更新车辆决策"""
        self.data[4] = decision.value
    
    def get_state(self):
        """获取当前决策"""
        return Decision(self.data[4])
    
    def update_battery(self, cost_battery: Dict[int, float]):
        """更新电量"""
        self.battery -= self.compute_battery_cost(Decision(self.data[4]), cost_battery)
    
    def add_order(self, order):
        """添加订单"""
        if order.get_id() in self.orders:
            logging.warning(f"Order {order.get_id()} already exists!")
        else:
            self.orders[order.get_id()] = order
            logging.info(f"Added order {order.get_id()}")
    
    def delete_order(self, order_id: int):
        """删除订单"""
        if order_id in self.orders:
            del self.orders[order_id]
            logging.info(f"Deleted order {order_id}")
        else:
            logging.warning(f"Order {order_id} does not exist!")
    
    def add_orders(self, *orders):
        """批量添加订单"""
        for order in orders:
            self.add_order(order)
    
    def delete_orders(self, *order_ids):
        """批量删除订单"""
        for order_id in order_ids:
            self.delete_order(order_id)
    
    def get_capacity(self):
        """获取当前载客总人数"""
        return sum(order.get_passenger() for order in self.orders.values())

    def get_orders(self):
        """获取所有订单对象"""
        return list(self.orders.values())
