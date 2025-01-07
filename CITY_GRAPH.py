import networkx as nx
import random
from ORDER import *

class CityGraph:

    def __init__(self, num_nodes, edge_prob, weight_range=(1, 10)):
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
        self.G = nx.Graph()  # 创建无向图

        # 调用图生成方法
        self._generate_graph()
        self._calculate_shortest_paths()

    def _generate_graph(self):
        """生成无孤立点的随机图（连通图）"""
        self.G.add_nodes_from(range(self.num_nodes))

        # 首先生成一个随机生成树，保证连通性
        nodes = list(range(self.num_nodes))
        random.shuffle(nodes)
        for i in range(1, len(nodes)):
            u = nodes[i - 1]
            v = nodes[i]
            weight = random.randint(*self.weight_range)
            self.G.add_edge(u, v, weight=weight)

        # 添加额外的随机边
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if u != v and random.random() < self.edge_prob and not self.G.has_edge(u, v):
                    weight = random.randint(*self.weight_range)
                    self.G.add_edge(u, v, weight=weight)


    def get_graph(self):
        """返回图对象"""
        return self.G

    def get_graph_info(self):
        """
        提取图的节点、边及权重信息。
        
        返回:
            nodes: 节点列表
            edges: 边列表
            weights: 边权重字典
        """
        nodes = list(self.G.nodes())
        edges = list(self.G.edges())
        weights = nx.get_edge_attributes(self.G, 'weight')
        return nodes, edges, weights

    def _calculate_shortest_paths(self):
        """
        计算图 G 中任意两节点间的最短距离和路径，返回结果。
        """
        dijkstra_results = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):  # 确保无重复的城市对
                source = i
                target = j
                try:
                    # 使用 Dijkstra 算法计算最短路径和距离
                    length = nx.dijkstra_path_length(self.G, source, target)
                    path = nx.dijkstra_path(self.G, source, target)
                    dijkstra_results.append({"source": source, "target": target, "distance": length, "path": path})
                except nx.NetworkXNoPath:
                    continue
        self.dijkstra_results = dijkstra_results
        
    
    def get_dijkstra_results(self):
        """获取所有计算的最短路径结果"""
        return self.dijkstra_results
    
    def get_intercity_path(self, source, target):
        """
        从给定的 result 列表中提取指定城市对的最短路径和距离。

        Args:
        - source (int): 起始城市的编号。
        - target (int): 终点城市的编号。

        Returns:
        - dict: 包含最短路径的距离和节点顺序。
                如果 source 和 target 顺序反过来，则调整顺序。
                如果无路径，返回 None。
        """
        if not hasattr(self, 'dijkstra_results'):
            self.calculate_shortest_paths()
        
        for entry in self.dijkstra_results:
            if entry['source'] == source and entry['target'] == target:
                return entry['distance'], entry['path']
            elif entry['source'] == target and entry['target'] == source:
                return entry['distance'], list(reversed(entry['path']))
        return None
    
    def get_neighbors(self, city_id):
        """
        获取指定城市的邻接城市。
        
        参数:
            city_id: 城市编号（节点编号）
        
        返回:
            neighbors: 邻接城市列表
        """
        if city_id not in self.G:
            raise ValueError(f"城市 {city_id} 不在图中")
        return list(self.G.neighbors(city_id))
    
    def plot_graph(self):
        """
        可视化图。
        """
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.G)  # 使用 spring 布局
        nx.draw(self.G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold')
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
        plt.show()

    def passby_most(self, orders:list[Order]):
        """输出一组订单中经过城市数目最多的"""
        i = 0
        
        for order in orders:
            _,path_node = self.get_intercity_path(*order.virtual_route())
            if i == 0:
                max_node = path_node
                max_order = order
            if i> 0 and len(path_node) > len(max_node):
                max_node = path_node
                max_order = order
            i+=1
        return path_node, max_order