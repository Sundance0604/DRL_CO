import numpy as np
from typing import Dict
from enum import Enum
import logging
from ORDER import *

# 设置日志记录
logging.basicConfig(level=logging.INFO)

# 定义车辆类
class Vehicle:
    def __init__(self, id: int, time: int, into_city: int, 
                 intercity: int, decision: int, battery: float, 
                 orders: Dict[int, 'Order'] = None):
        self._decisions = [2]
        self.id = id
        self.time = time
        self.into_city = into_city
        self.intercity = intercity
        self.whether_city = True
        self._decisions.append(decision)
        self.last_decision = self._decisions[-2]
        self.battery = battery
        self.time_into_city = time
        self.orders = orders if orders else {}  # 初始化订单字典
        self.history_orders = []
        self.longest_path = []

    def __repr__(self):
        return (f"Vehicle(id={self.id}, time={self.time}, "
                f"into_city={self.into_city}, intercity={self.intercity}, passenger={self.get_capacity()}, "
                f"decision={self.decision}, battery={self.battery}, matched_order={len(self.orders)})")

    @classmethod
    def from_dict(cls, vehicle_dict):
        """从字典创建车辆实例"""
        return cls(
            id=vehicle_dict["id"],
            time=vehicle_dict["time"],
            into_city=vehicle_dict["into_city"],
            intercity=vehicle_dict["intercity"],
            decision=vehicle_dict["decision"],
            battery=vehicle_dict["battery"],
            orders=vehicle_dict.get("orders", {})
        )

    @staticmethod
    def compute_battery_cost(decision: int, cost_battery: Dict[int, float]):
        """计算电量消耗"""
        return cost_battery.get(decision, 0)
    
    @property
    def decision(self):
        """获取最近一次的决策"""
        return self._decisions[-1] if self._decisions else None

    @decision.setter
    def decision(self, decision):
        """添加新的决策"""
        self._decisions.append(decision)
    def replace_decision(self,decision):
        self._decisions[-1] = decision
    def update_time(self):
        self.time += 1
    def get_history_decisions(self):
        return self._decisions

    def move_into_city(self):
        """进入城市"""
        self.whether_city = True

    def move_to_city(self, city_id: int):
        """离开当前城市，前往指定城市"""
        self.into_city = self.intercity
        self.intercity = city_id
        self.whether_city = False
        self.time_into_city = self.time

    def update_battery(self, cost_battery: Dict[int, float]):
        """更新电量"""
        self.battery -= self.compute_battery_cost(self.decision, cost_battery)
        if self.battery < 0:
            raise ValueError("Battery level cannot be negative!")

    def add_order(self, order):
        """添加订单"""
        if order.id not in self.orders:
            self.orders[order.id] = order

    def delete_order(self, order_id: int):
        """删除订单"""
        if order_id in self.orders:
            del self.orders[order_id]
            
    def add_orders(self, *orders):
        """批量添加订单"""
        for order in orders:
            self.add_order(order)

    def delete_orders(self, order_ids: list):
        """批量删除订单"""
        for order_id in order_ids:
            self.delete_order(order_id)

    def get_capacity(self):
        """获取当前载客总人数"""
        return sum(order.passenger for order in self.orders.values())

    def get_orders(self):
        """获取所有订单对象"""
        return list(self.orders.values())
    
    
