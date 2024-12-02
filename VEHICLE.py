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
    def __init__(self, vehicle_id: int, time: int, into_city: int, 
                 intercity: int, decision: Decision, battery: float, 
                 orders: Dict[int, 'Order'] = None):
        self.vehicle_id = vehicle_id
        self.time = time
        self.into_city = into_city
        self.intercity = intercity
        self.decision = decision
        self.battery = battery
        self.orders = orders if orders else {}  # 初始化订单字典

    def __repr__(self):
        return (f"Vehicle(id={self.vehicle_id}, time={self.time}, "
                f"into_city={self.into_city}, intercity={self.intercity}, "
                f"decision={self.decision.name}, battery={self.battery})")

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

    def update_time(self):
        """更新时间到 t+1"""
        self.time += 1

    def if_in_city(self):
        """判断是否在城市内"""
        return self.into_city == 0

    def which_city(self):
        """返回当前所在城市编号"""
        return self.intercity if self.if_in_city() else False

    def move_into_city(self):
        """进入城市"""
        self.into_city = 0

    def move_to_city(self, city_id: int):
        """离开当前城市，前往指定城市"""
        self.into_city = self.intercity
        self.intercity = city_id

    def update_state(self, decision: Decision):
        """更新车辆决策"""
        self.decision = decision

    def get_state(self):
        """获取当前决策"""
        return self.decision

    def update_battery(self, cost_battery: Dict[int, float]):
        """更新电量"""
        self.battery -= self.compute_battery_cost(self.decision, cost_battery)
        if self.battery < 0:
            raise ValueError("Battery level cannot be negative!")

    def add_order(self, order):
        """添加订单"""
        if order.get_id() not in self.orders:
            self.orders[order.id] = order
           

    def delete_order(self, order_id: int):
        """删除订单"""
        if order_id in self.orders:
            del self.orders[order_id]
            
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
        return sum(order.passengers for order in self.orders.values())

    def get_orders(self):
        """获取所有订单对象"""
        return list(self.orders.values())
