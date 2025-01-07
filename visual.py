# 安装必要的库
#!pip install matplotlib networkx ffmpeg-python

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# 上传文件
from google.colab import files
uploaded = files.upload()

# 假设文件名为 vehicle_records.csv 和 graph_adjacency.csv
file_names = list(uploaded.keys())
vehicle_file = [f for f in file_names if 'vehicle' in f][0]
graph_file = [f for f in file_names if 'adjacency' in f][0]

# 读取数据
vehicle_data = pd.read_csv(vehicle_file)
graph_data = pd.read_csv(graph_file)

# 构建城市网络图
G = nx.Graph()
for _, row in graph_data.iterrows():
    node = row['Node']
    adjacent_nodes = eval(row['Adjacent Nodes'])  # 将字符串解析为列表
    for adj in adjacent_nodes:
        G.add_edge(node, adj)

# 使用 `spring_layout` 布局减少边交叉
positions = nx.spring_layout(G, seed=42)  # 固定随机种子以保证一致性

# 准备绘图
fig, ax = plt.subplots(figsize=(12, 8))
nx.draw(G, pos=positions, with_labels=True, node_size=700, node_color="lightblue", font_size=10, ax=ax)

# 用于绘制汽车位置的散点图
vehicle_scatter = ax.scatter([], [], s=50)
vehicle_texts = []
global_text = None

# 定义颜色映射
colors = plt.cm.tab20(np.linspace(0, 1, len(vehicle_data['id'].unique())))

# 更新每一帧
def update(frame):
    global vehicle_texts, global_text
    # 清除之前的标注
    for txt in vehicle_texts:
        txt.remove()
    vehicle_texts = []
    if global_text:
        global_text.remove()

    # 筛选当前时间的数据
    current_data = vehicle_data[vehicle_data['time'] == frame]

    # 更新汽车位置
    vehicle_positions = []
    vehicle_colors = []
    for idx, row in current_data.iterrows():
        if row['whether_city']:
            vehicle_pos = positions[row['intercity']]
        else:
            # 在边上的点增加随机扰动以避免重合
            edge_vector = positions[row['intercity']] - positions[row['into_city']]
            offset = (idx % 5) * 0.05  # 通过 idx 控制偏移
            vehicle_pos = positions[row['into_city']] + edge_vector * (0.5 + offset)

        vehicle_positions.append(vehicle_pos)
        vehicle_colors.append(colors[row['id'] % len(colors)])  # 根据车辆 ID 分配颜色

        txt = ax.text(vehicle_pos[0], vehicle_pos[1] + 0.05,
                      f"ID: {row['id']}\nMatched: {row['num_matched']}",
                      fontsize=8, ha='center', color='black')
        vehicle_texts.append(txt)

    # 更新散点图
    vehicle_scatter.set_offsets(vehicle_positions)
    vehicle_scatter.set_color(vehicle_colors)

    # 动态显示全局信息
    orders_unmatched = current_data['orders_unmatched'].iloc[0]
    object_value = current_data['object_value'].iloc[0]
    global_text = ax.text(0.95, 0.95, f"Unmatched Orders: {orders_unmatched}\nObjective Value: {object_value}",
                          fontsize=10, ha='right', va='top', transform=ax.transAxes, color='darkred')

# 获取时间步
time_steps = sorted(vehicle_data['time'].unique())

# 创建动画
ani = FuncAnimation(fig, update, frames=time_steps, repeat=False)

# 保存动画为 MP4，降低帧率使视频播放更慢
ani.save('vehicle_positions.mp4', writer='ffmpeg', fps=2)

# 下载文件到本地
from google.colab import files
files.download('vehicle_positions.mp4')