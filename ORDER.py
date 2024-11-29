import numpy as np

class Order:
    def __init__(self, order_id, passenger_count, departure, destination, 
                 start_time, end_time, virtual_departure, battery):
        # 使用 NumPy 数组存储属性
        self.data = np.array([
            order_id,             # ID
            passenger_count,      # 乘客数
            departure,            # 出发地
            destination,          # 目的地
            start_time,           # 起始时间
            end_time,             # 截止时间
            virtual_departure,    # 虚拟出发点
            [0,0],                     # 是否匹配 (0 表示 False, 1 表示 True)
            battery               # 需要的电量
        ], dtype=object)

    def get_id(self):
        """返回订单 ID"""
        return self.data[0]

    def get_passenger(self):
        """返回订单乘客数"""
        return self.data[1]

    def get_route(self):
        """返回出发地和目的地"""
        return self.data[2], self.data[3]

    def get_time(self):
        """获取起始与截止时间"""
        return self.data[4], self.data[5]

    def change_virtual_departure(self, virtual_departure):
        """改变虚拟出发点"""
        self.data[6] = virtual_departure

    def be_matched(self, vehicle_id):
        """被匹配"""
        self.data[7][0] = 1 
        self.data[7][1] = vehicle_id

    def battery_demand(self):
        return self.data[8]