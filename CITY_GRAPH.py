import networkx as nx
import random

class city_graph():

    def __init__(self,num_nodes, edge_prob, weight_range=(1, 10)):
        """
        创建一个随机有向图并生成随机边的权重。
        
        参数:
            num_nodes: 节点数量
            edge_prob: 每两个节点之间存在边的概率（0 到 1）
            weight_range: 权重范围，默认为 (1, 10)
        """
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.weight_range = weight_range
        self.G = nx.Graph()
        
    def _generate_graph(self):
        self.G.add_nodes_from(range(self.num_nodes))
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if u != v and random.random() < self.edge_prob:
                    weight = random.randint(*self.weight_range)
                    self.G.add_edge(u, v, weight=weight)

    def get_graph(self):
        """
        返回绘制的图
        """
        return self.G
    
    def _graph_info(self):
        """
        提取图的节点、边及权重信息。
        
        参数:
            G: 有向图 (DiGraph)
        
        返回:
            nodes: 节点列表
            edges: 边列表
            weights: 边权重字典
        """
        self.nodes = list(self.G.nodes())
        self.edges = list(self.G.edges())
        self.weights = nx.get_edge_attributes(self.G, 'weight')
    
    def get_graph_info(self):
        return self.nodes, self.edges, self.weights
    
    def _dijkstra_calculate(self):
        
        """
        计算图 G 中任意两节点间的最短距离和路径，返回结果。
        
        Args:
        - G (nx.Graph): 无向图。
        
        Returns:
        - results (list): 每对节点的最短路径信息，包含起点、终点、距离和经过的节点。
        """
        dijkstra_results = []
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):  # 确保无重复的城市对
                source = self.nodes[i]
                target = self.nodes[j]
                try:
                    # 使用 Dijkstra 算法计算最短路径和距离
                    length = nx.dijkstra_path_length(self.G, source, target)
                    path = nx.dijkstra_path(self.G, source, target)
                    dijkstra_results.append({"source": source, "target": target, "distance": length, "path": path})
                except nx.NetworkXNoPath:
                    # 如果没有路径，跳过
                    continue
        self.dijkstra_results = dijkstra_results

    def get_dijkstra_results(self):
        return self.dijkstra_results
    
    def get_intercity_path(self, source, target):
        """
        从给定的 result 列表中提取指定城市对的最短路径和距离。

        Args:
        - result (list): 预先计算的所有城市对的最短路径信息。
                        每个元素是一个字典，格式为：
                        {'source': s, 'target': t, 'distance': d, 'path': [path]}。
        - source (int): 起始城市的编号。
        - target (int): 终点城市的编号。

        Returns:
        - dict: 包含最短路径的距离和节点顺序。
                如果 source 和 target 顺序相反，则调整顺序。
                如果无路径，返回 None。
        """
        for entry in self.dijkstra_results:
            if entry['source'] == source and entry['target'] == target:
                return {'distance': entry['distance'], 'path': entry['path']}
            elif entry['source'] == target and entry['target'] == source:
                # 反向路径，调整顺序
                return {'distance': entry['distance'], 'path': list(reversed(entry['path']))}
        return None
    
    def get_neighbors(self, city_id):
        """
        获取指定城市的邻接城市。
        
        参数:
            G: 图 (无向图)
            city_id: 城市编号（节点编号）
        
        返回:
            neighbors: 邻接城市列表
        """
        if city_id not in self.G:
            raise ValueError(f"城市 {city_id} 不在图中")
        return list(self.G.neighbors(city_id))