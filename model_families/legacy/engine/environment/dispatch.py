try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # compatibility with the original environment
    import gym
    from gym import spaces
import numpy as np
from model_families.legacy.engine.domain.city import *
from model_families.legacy.engine.domain.vehicle import *
from model_families.legacy.engine.simulation.tools import *
from model_families.legacy.engine.domain.city_graph import *
import copy

class DispatchEnv(gym.Env):
    def __init__(self, G :CityGraph,
                 vehicles: Dict, orders: Dict, cities:Dict,capacity):
        super(DispatchEnv, self).__init__()

        self.time = 0
        # 必须深复制
        self.G = G
        self.vehicles = vehicles
        self.orders = orders
        self.cities = cities
        self.capacity = capacity
        """
        车辆状态空间：
        车辆id即为行号，其decision，capacity、电量状态（建议更替为百分比）、intercity与intocity
        """
        self.vehicle_state = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.vehicles), 11), dtype=np.float32
        )
        #  期数，是否被匹配，载客数，在哪个城市
        self.order_state = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.orders), 12), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "vehicles": self.vehicle_state,
            "orders": self.order_state,
        })
        # 动作空间：为每个订单分配虚拟出发地点（连续或离散）
        # 动作空间：矩阵形式，每个订单可以从多个城市选择一个出发地点
        self.action_space = spaces.MultiDiscrete([len(self.cities)] * len(self.orders))
    """
    def count_num(self):
        num_vehicles, num_orders ,num_cities= 0
        for city in self.Cities.values():
            num_vehilces = len(city.vehicles.value()) + num_vehilces
            num_orders = len(city.orders.value()) + num_orders
        return num_vehicles ,num_orders , len(self.Cities.values())
    """
    def update(self, vehicles_matrix: np, orders_matrix: np, cities:dict):
        """更新环境"""
        self.vehicle_state = vehicles_matrix
        self.observation_space = orders_matrix
        self.cities = cities
        self.time += 1

    def reset(self, vehicles:Dict, orders:Dict):
        """重置环境"""
        self.time = 0
        self.orders = orders
        self.vehicles = vehicles
        # self.cities = cities
    def get_state(self):
        print(self.vehicles)
        print(self.orders)
        print(self.time)

    def _multiagent_step(self, action, orders_unmatched:dict):
        """执行动作并返回结果"""
        """action是一个len(self.orders) * len(self.cities)维度的数组"""
        """一刀-wrong_punlishment"""
        correct_combinations = []
        wrong_punlishment = 10
        reward = 0
        for i in range(len(orders_unmatched)):

            matched_amount = sum(action[i])
            # if orders_unmatched[i].matched:

            #    continue
            if matched_amount > 1:
                reward += -wrong_punlishment
                continue
            # if matched_amount == 1:
                if orders_unmatched[i].start_time > self.time:
                    # correct_combinations.extend([(i, j) for j in range(len(self.cities))])
                    # reward += -wrong_punlishment
                    # 貌似无需惩罚
                    continue
            # if matched_amount == 0:
                # correct_combinations.extend([(i, j) for j in range(len(self.cities))])
                # 我的想法是无需添加
                # correct_combinations.append((i, j for j in range(len(self.cities))))
                continue
            for j in range(len(self.cities)):
                # 已匹配者不可匹配
                if orders_unmatched[i].destination == j and action[i][j] == 1:
                    continue
                if orders_unmatched[i].matched and action[i][j] == 1:
                    continue  # 继续下一个组合，跳过惩罚

                # if self.orders[i].matched is False:
                _, path_order = self.G.get_intercity_path(*orders_unmatched[i].route())

                # 前驱不可行
                if j == path_order[1] and action[i][j] == 1:
                    continue  # 继续下一个组合，跳过惩罚
                # 需在邻接城市里
                if j not in self.G.get_neighbors(orders_unmatched[i].departure) and action[i][j] == 1:
                    continue  # 继续下一个组合，跳过惩罚

                # 邻接城市需要有合适的车
                vehicle_found = False
                # 起码要有车
                if orders_unmatched[j].vehicle_available.values():

                    for vehicle in self.cities[j].vehicle_available.values():
                        if len(vehicle.longest_path) > 0:
                            if vehicle.longest_path[0] == path_order[1] and action[i][j] == 1:
                                if self.capacity - vehicle.get_capacity > orders_unmatched[i].values().passenger:
                                    vehicle_found = True
                                    break

                    if not vehicle_found:
                        continue  # 继续下一个组合，跳过惩罚
                else:
                    continue
                # 如果没有触发任何惩罚条件，则是正确的组合
                correct_combinations.append((i, j))

        for i in range(len(self.orders)):
            for j in range(len(self.cities)):
                if (i,j) not in correct_combinations:
                    reward = reward - orders_unmatched[i].revenue
                else:

                    if self.time == orders_unmatched[i].start_time:
                        orders_unmatched[i].virtual_departure = j


        return reward  # 记住还需调用gurobi求解合理匹配下的值

    def step(self, action):
        """执行动作并返回结果"""
        """action是一个len(self.orders) * len(self.cities)维度的数组"""
        """一刀-wrong_punlishment"""
        correct_combinations = []
        wrong_punlishment = 10
        reward = 0
        for i in range(len(self.orders)):

            matched_amount = sum(action[i])

            if self.orders[i].matched:
                # correct_combinations.extend([(i, j) for j in range(len(self.cities))])
                continue
            if matched_amount > 1:
                reward += -wrong_punlishment
                continue
            if matched_amount == 1:
                if self.orders[i].start_time > self.time:
                    # correct_combinations.extend([(i, j) for j in range(len(self.cities))])
                    # reward += -wrong_punlishment
                    # 貌似无需惩罚
                    continue
            if matched_amount == 0:
                # correct_combinations.extend([(i, j) for j in range(len(self.cities))])
                # 我的想法是无需添加
                # correct_combinations.append((i, j for j in range(len(self.cities))))
                continue
            for j in range(len(self.cities)):
                # 已匹配者不可匹配
                if self.orders[i].destination == j and action[i][j] == 1:
                    continue
                if self.orders[i].matched and action[i][j] == 1:
                    continue  # 继续下一个组合，跳过惩罚

                if self.orders[i].matched is False:
                    _, path_order = self.G.get_intercity_path(*self.orders[i].route())

                    # 前驱不可行
                    if j == path_order[1] and action[i][j] == 1:
                        continue  # 继续下一个组合，跳过惩罚
                    # 需在邻接城市里
                    if j not in self.G.get_neighbors(self.orders[i].departure) and action[i][j] == 1:
                        continue  # 继续下一个组合，跳过惩罚

                    # 邻接城市需要有合适的车
                    vehicle_found = False
                    # 起码要有车
                    if self.cities[j].vehicle_available.values():

                        for vehicle in self.cities[j].vehicle_available.values():
                            if len(vehicle.longest_path) > 0:
                                if vehicle.longest_path[0] == path_order[1] and action[i][j] == 1:
                                    if self.capacity - vehicle.get_capacity > self.orders[i].values().passenger:
                                        vehicle_found = True
                                        break

                        if not vehicle_found:
                            continue  # 继续下一个组合，跳过惩罚
                    else:
                        continue
                # 如果没有触发任何惩罚条件，则是正确的组合
                correct_combinations.append((i, j))

        for i in range(len(self.orders)):
            for j in range(len(self.cities)):
                if (i,j) not in correct_combinations:
                    reward = reward - self.orders[i].revenue
                else:

                    if self.time == self.orders[i].start_time:
                        self.orders[i].virtual_departure = j


        return reward  # 记住还需调用gurobi求解合理匹配下的值

    def apply_actions(self, orders_unmatched, actions, strict=True):
        """Apply policy actions and refresh the city view used by Gurobi.

        The historical training loop changed ``order.virtual_departure`` after
        building ``self.cities`` (and the final SAC notebook did not apply the
        actions at all).  Consequently the lower-layer model could not observe
        the policy decision.  This method is now the only action boundary.

        Returns one validation reward per order (0 for valid, -1 for a fallback).
        The actual learning reward should be computed after the lower-layer
        solve, when matching success and profit are known.
        """
        orders = list(orders_unmatched.values())
        if len(actions) != len(orders):
            raise ValueError(f"received {len(actions)} actions for {len(orders)} active orders")

        mask = self.get_mask(orders_unmatched)
        validation_rewards = []
        for index, (order, action) in enumerate(zip(orders, actions)):
            action = int(action)
            valid = 0 <= action < mask.shape[1] and bool(mask[index, action])
            if not valid and strict:
                raise ValueError(f"invalid action {action} for order {order.id}")
            if not valid:
                action = order.departure
                validation_rewards.append(-1.0)
            else:
                validation_rewards.append(0.0)
            order.virtual_departure = action

        # Lower_Layer reads the city buckets, not the Order objects directly.
        city_update_without_drl(self.cities, self.vehicles, orders_unmatched, self.time)
        return validation_rewards

    def test_step(self, orders_unmatched, actions):
        """Backward-compatible action application used by old notebooks."""
        return sum(self.apply_actions(orders_unmatched, actions, strict=False))

    def dynamic_step(self, total_orders, actions, mask=None):
        penalties = self.apply_actions(total_orders, actions, strict=False)
        return 1000.0 + 100.0 * sum(penalties)

    def get_mask(self, orders_unmatched):
        """Return valid virtual-departure actions in stable dictionary order.

        Keeping the real departure is always available and represents the
        no-relocation baseline.  This also guarantees that every row contains
        an action, which a masked categorical policy requires.
        """
        mask = np.zeros((len(orders_unmatched), len(self.cities)), dtype=np.bool_)
        for index, order in enumerate(orders_unmatched.values()):
            mask[index, order.departure] = True
            path_result = self.G.get_intercity_path(*order.route())
            next_on_shortest_path = path_result[1][1] if path_result and len(path_result[1]) > 1 else None
            for city_id in self.G.get_neighbors(order.departure):
                if city_id != order.destination and city_id != next_on_shortest_path:
                    mask[index, city_id] = True
        return mask

    def cities_reload(self, cities):
        self.cities = {}
        self.cities = cities
