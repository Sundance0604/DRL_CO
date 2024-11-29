import numpy as np
from typing import Tuple
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)

class Order:
    __slots__ = ['data']  # 限制属性，节省内存
    
    def __init__(self, order_id: int, passenger_count: int, departure: int, destination: int, 
                 start_time: int, end_time: int, virtual_departure: int, battery: float):
    
        self.data = np.array([
            order_id,             # 订单 ID
            passenger_count,      # 乘客数
            departure,            # 出发地
            destination,          # 目的地
            start_time,           # 起始时间
            end_time,             # 截止时间
            virtual_departure,    # 虚拟出发地
            [0, None],            # 匹配状态与匹配车辆 ID [是否匹配, 车辆 ID]
            battery               # 需要的电量
        ], dtype=object)

    def __repr__(self):
        """打印订单的简洁信息"""
        matched_status = "Matched" if self.data[7][0] else "Unmatched"
        return (f"Order(id={self.data[0]}, passengers={self.data[1]}, "
                f"departure={self.data[2]}, destination={self.data[3]}, "
                f"battery={self.data[8]}, status={matched_status})")

    # --- 属性访问方法 ---
    @property
    def id(self) -> int:
        return self.data[0]
    
    @property
    def passengers(self) -> int:
        return self.data[1]
    
    @property
    def route(self) -> Tuple[int, int]:
        """返回真实出发地与目的地"""
        return self.data[2], self.data[3]
    
    @property
    def virtual_route(self) -> Tuple[int, int]:
        """返回虚拟出发地与目的地"""
        return self.data[6], self.data[3]
    @property
    def destination(self) ->int:
        """返回目的地"""
        return self.data[3]    
    @property
    def time_window(self) -> Tuple[int, int]:
        """返回起始与截止时间"""
        return self.data[4], self.data[5]
    
    @property
    def battery_demand(self) -> float:
        """返回电量需求"""
        return self.data[8]

    @property
    def matched_vehicle(self) -> int:
        """返回匹配的车辆 ID，如果未匹配则返回 None"""
        return self.data[7][1] if self.data[7][0] else None

    # --- 方法 ---
    def update_virtual_departure(self, virtual_departure: int):
        """更新虚拟出发地"""
        self.data[6] = virtual_departure
        logging.info(f"Order {self.id}: Virtual departure updated to {virtual_departure}")

    def match_vehicle(self, vehicle_id: int):
        """匹配车辆"""
        self.data[7] = [1, vehicle_id]
        logging.info(f"Order {self.id} matched to Vehicle {vehicle_id}")

    def unmatch_vehicle(self):
        """取消匹配"""
        self.data[7] = [0, None]
        logging.info(f"Order {self.id} unmatched")

    def is_matched(self) -> bool:
        """检查是否匹配"""
        return self.data[7][0] == 1

    @staticmethod
    def compute_battery_efficiency(distance: float, efficiency: float) -> float:
        """
        静态方法，计算订单所需电量
        :param distance: 距离
        :param efficiency: 电耗效率 (单位: 每公里电量)
        :return: 所需电量
        """
        return distance * efficiency
    
    @classmethod
    def from_dict(cls, order_dict: dict):
        """
        从字典初始化订单
        """
        return cls(
            order_id=order_dict['order_id'],
            passenger_count=order_dict['passenger_count'],
            departure=order_dict['departure'],
            destination=order_dict['destination'],
            start_time=order_dict['start_time'],
            end_time=order_dict['end_time'],
            virtual_departure=order_dict['virtual_departure'],
            battery=order_dict['battery']
        )

    def overlaps_with(self, other_order: 'Order') -> bool:
        """
        检查时间窗口是否与另一个订单重叠
        """
        return not (self.time_window[1] <= other_order.time_window[0] or 
                    other_order.time_window[1] <= self.time_window[0])
