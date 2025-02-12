import numpy as np
from alns import ALNS, State
from alns.accept import SimulatedAnnealing
from alns.stop import MaxIterations
from alns.select import RouletteWheel
import random

# Problem Data
np.random.seed(42)
num_customers = 10
num_vehicles = 3
capacity = 100

# 随机生成坐标和需求
depot = np.array([50, 50])
nodes = np.random.randint(0, 100, (num_customers, 2))
demands = np.random.randint(5, 20, num_customers)

# 计算欧几里得距离
def distance(a, b):
    return np.linalg.norm(a - b)

distance_matrix = np.zeros((num_customers + 1, num_customers + 1))
nodes_full = np.vstack([depot, nodes])  # 加入仓库
for i in range(len(nodes_full)):
    for j in range(len(nodes_full)):
        distance_matrix[i, j] = distance(nodes_full[i], nodes_full[j])

# 初始状态
class VRPState(State):
    def __init__(self, routes):
        self.routes = routes
    
    def objective(self):
        total_cost = sum(
            distance_matrix[route[i], route[i+1]]
            for route in self.routes for i in range(len(route)-1)
        )
        return total_cost
    
    def copy(self):
        return VRPState([route[:] for route in self.routes])

# 破坏算子
def random_removal(state, num_remove=2):
    new_state = state.copy()
    for _ in range(num_remove):
        if any(new_state.routes):
            route = random.choice(new_state.routes)
            if len(route) > 2:
                route.pop(random.randint(1, len(route) - 2))
    return new_state

# 修复算子
def greedy_insert(state):
    new_state = state.copy()
    unassigned = [i for i in range(1, num_customers + 1) if not any(i in r for r in new_state.routes)]
    for i in unassigned:
        best_cost = float('inf')
        best_route = None
        best_position = None
        for route in new_state.routes:
            for pos in range(1, len(route)):
                temp_route = route[:pos] + [i] + route[pos:]
                cost = sum(distance_matrix[temp_route[j], temp_route[j+1]] for j in range(len(temp_route)-1))
                if cost < best_cost:
                    best_cost, best_route, best_position = cost, route, pos
        if best_route is not None:
            best_route.insert(best_position, i)
    return new_state

# ALNS 运行
initial_routes = [[0, i, 0] for i in range(1, num_customers + 1)]  # 每个客户单独一辆车
initial_state = VRPState(initial_routes)
alns = ALNS()
alns.add_destroy_operator(random_removal)
alns.add_repair_operator(greedy_insert)

# 设定接受准则（模拟退火）
accept = SimulatedAnnealing(1000, 0.95, 500)
select = RouletteWheel()
stop = MaxIterations(1000)

result = alns.iterate(initial_state, select, accept, stop)

# 输出最优解
best_state = result.best_state
print("Best routes:", best_state.routes)
print("Best cost:", best_state.objective())
