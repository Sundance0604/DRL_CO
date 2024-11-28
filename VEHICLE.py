import numpy as np
from ORDER import *

# 定义车辆类
class Vehicle:
    def __init__(self, vehicle_id, time, into_city, intercity, decision, battery, orders):
        """
        初始化车辆信息
        data: 一个长度为10的 numpy 数组
        orders: 一个字典，键为订单主码，值为 Order 实例
        """
        self.data = np.array([
            vehicle_id,
            time,
            into_city,
            intercity,
            decision,
            battery
        ],dtype = object)
        self.orders = orders  # 直接管理订单对象, 是一个字典
    
    def get_id(self):
        """获取车辆编号"""
        return self.data[0]
    
    def get_time(self):
        """获取当前时间"""
        return self.data[1]
    
    def update_time(self):
        """更新时间到 t+1"""
        self.data[1] += 1
    
    def get_location(self):
        """获取当前位置，城市内为 False"""
        return self.data[2] != 0
    
    def into_city(self):
        """进入城市"""
        self.data[2] = 0
    
    def intercity(self, v):
        """离开城市，前往城市 v"""
        self.data[2] = self.data[3]
        self.data[3] = v
    
    def update_state(self, decision):
        """更新决策,
        dispacthing: 0
        charging: 1
        idle: 2     """
        self.data[4] = decision
    
    def get_state(self):
        """获取当前决策"""
        return self.data[4]
    
    def update_battery(self):
        """更新电量"""
        self.data[5] -= cost_battery[self.data[4]]
    
    def get_battery(self):
        """获取电量"""
        return self.data[5]
    
    def get_orders(self):
        """获取所有订单对象"""
        return list(self.orders.values())
    
    def delete_order(self, order_id):
        """删除订单"""
        if order_id in self.orders:
            del self.orders[order_id]
    
    def add_order(self,order):
        """增加订单"""
        self.orders[order.get_id()] = order
    
    def get_capacity(self):
        """计算总载客数"""
        return sum(order.get_passenger() for order in self.orders.values())
