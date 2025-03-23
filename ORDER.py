import logging
from typing import Tuple
import CITY_GRAPH as G

# 设置日志记录
logging.basicConfig(level=logging.INFO)

class Order:
    def __init__(self, id: int, passenger: int, departure: int, destination: int, 
                 start_time: int, end_time: int, virtual_departure: int, 
                 battery: float,distance:int, revenue:int, penalty:int,
                 least_time_consume:int):
        self.id = id             # 订单 ID
        self.passenger = passenger  # 乘客数
        self.departure = departure          # 出发地
        self.destination = destination      # 目的地
        self.start_time = start_time        # 起始时间
        self.end_time = end_time            # 截止时间
        self.virtual_departure = virtual_departure  # 虚拟出发地
        self.matched = False                # 匹配状态
        self.matched_vehicle_id = None      # 匹配的车辆 ID
        self.battery = battery              # 电量需求
        self.distance = distance
        self.revenue = revenue
        self.penalty = penalty
        self.least_time_consume = least_time_consume
    def __repr__(self):
        """打印订单的简洁信息"""
        matched_status = "Matched" if self.matched else "Unmatched"
        return (f"Order(id={self.id}, passengers={self.passenger}, "
                f"departure={self.departure}, destination={self.destination},virtual_departure={self.virtual_departure}, "
                f"matched_vehicle_id={self.matched_vehicle_id},distance={self.distance},"
                f"battery={self.battery}, status={matched_status})")

    # --- 属性访问方法 ---
    
    
    
    def route(self) -> Tuple[int, int]:
        """返回真实出发地与目的地"""
        return self.departure, self.destination
    

    def virtual_route(self) -> Tuple[int, int]:
        """返回虚拟出发地与目的地"""
        return self.virtual_departure, self.destination
    

    def time_window(self) -> Tuple[int, int]:
        """返回时间窗口"""
        return self.start_time, self.end_time
    


    def matched_vehicle(self) -> int:
        """返回匹配的车辆 ID，如果未匹配则返回 None"""
        return self.matched_vehicle_id if self.matched else None

    # --- 方法 ---
    def update_virtual_departure(self, virtual_departure: int):
        """更新虚拟出发地"""
        self.virtual_departure = virtual_departure
        logging.info(f"Order {self.id}: Virtual departure updated to {virtual_departure}")

    def match_vehicle(self, vehicle_id: int):
        """匹配车辆"""
        self.matched = True
        self.matched_vehicle_id = vehicle_id
        #logging.info(f"Order {self.id} matched to Vehicle {vehicle_id}")

    def unmatch_vehicle(self):
        """取消匹配"""
        self.matched = False
        self.matched_vehicle_id = None
        logging.info(f"Order {self.id} unmatched")

    def is_matched(self) -> bool:
        """检查是否匹配"""
        return self.matched

    @classmethod
    def from_dict(cls, order_dict: dict):
        """
        从字典初始化订单
        """
        return cls(
            order_id=order_dict['id'],
            passenger=order_dict['passenger'],
            departure=order_dict['departure'],
            destination=order_dict['destination'],
            start_time=order_dict['start_time'],
            end_time=order_dict['end_time'],
            virtual_departure=order_dict['virtual_departure'],
            battery=order_dict['battery']
        )

    
