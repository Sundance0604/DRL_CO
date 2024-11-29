class City:
    
    def __init__(self, city_id, neighbor, vehicle_available, charging_capacity, real_departure, virtual_departure):
        self.city_id = city_id
        self.neighbor = neighbor                            # 邻接城市的索引或 ID 列表
        self.vehicle_available = vehicle_available          # 可调度车辆的索引列表
        self.charging_capacity = charging_capacity          # 充电站容量
        self.real_departure = real_departure                # 实际出发点订单的索引列表
        self.virtual_departure = virtual_departure          # 虚拟出发点订单的索引列表

    def get_neighbor(self):
        """获取城市的邻接城市"""
        return self.neighbor

    def get_vehicle_available(self):
        """获取可调度的车辆"""
        return self.vehicle_available

    def add_available_vehicle(self, vehicle_id):
        """增加可调度的车辆"""
        if vehicle_id not in self.vehicle_available:
            self.vehicle_available.append(vehicle_id)

    def delete_available_vehicle(self, vehicle_id):
        """删去不可调度的车辆"""
        if vehicle_id in self.vehicle_available:
            self.vehicle_available.remove(vehicle_id)

    def get_charging_capacity(self):
        """获取充电站容量"""
        return self.charging_capacity

    def get_real_departure(self):
        """获取实际出发点订单"""
        return self.real_departure

    def get_virtual_departure(self):
        """获取虚拟出发城市"""
        return self.virtual_departure

    def update_virtual_departure(self, new_virtual):
        """更新虚拟出发城市"""
        self.virtual_departure = new_virtual
