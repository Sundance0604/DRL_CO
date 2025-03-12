import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
from torch import optim
from itertools import islice


import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    # 请注意改成更多层的了
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.input_dim = state_dim  # 记录超参数
        self.hidden_dim = hidden_dim  # 记录超参数
        self.action_dim = action_dim  # 记录超参数
        self.init_params = {'state_dim':state_dim, 'hidden_dim': hidden_dim,'action_dim': action_dim} 
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)
    

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.input_dim = state_dim  # 记录超参数
        self.hidden_dim = hidden_dim  # 记录超参数
        self.init_params = {'state_dim': state_dim, 'hidden_dim': hidden_dim} 
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
# 定义经验元组
Experience = namedtuple('Experience', [
    'vehicle_states', 'order_states', 'actions', 'selected_log_probs',
    'log_probs', 'probs', 'rewards', 'next_vehicle_states', 'next_order_states','dones'
])   
class ReplayBuffer:
    """存储过去的经验"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self,time, saq_len = 8):
        time = time - 16
        """按时间顺序采样从 time 开始的 saq_len 个连续数据（适用于 deque）"""
        if time + saq_len > len(self.buffer):
            print(time, len(self.buffer))
            raise ValueError("采样范围超出 buffer 大小")

        return list(islice(self.buffer, time, time + saq_len))

    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """清除所有存储的经验"""
        self.buffer.clear()
class MultiAgentAC(torch.nn.Module):
    def __init__(self, device, VEHICLE_STATE_DIM, 
                 ORDER_STATE_DIM, NUM_CITIES, 
                 HIDDEN_DIM, STATE_DIM):
        super(MultiAgentAC, self).__init__()
        self.device = device
        self.NUM_CITIES = NUM_CITIES
        
        # 共享网络
        self.actor = PolicyNet(STATE_DIM, HIDDEN_DIM, NUM_CITIES).to(device)
        self.critic = ValueNet(STATE_DIM, HIDDEN_DIM).to(device)
        
        # 经验回放缓冲区
        self.buffer = ReplayBuffer(capacity=145)  # 增加缓冲区大小，增强训练稳定性

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-3)

        self.batch_size = 16  # 采样批量
        self.gamma = 0.99  # 折扣因子

    def take_action_vehicle(self, vehicle_states, order_states, mask,explore=True, greedy=False):
        """为当前活跃订单生成动作 ⭐"""
        eplison = 0.00001
        mask = torch.from_numpy(mask).to(self.device)
        # 将状态转换为 
       
        v_tensor = torch.FloatTensor(vehicle_states).to(self.device)
        o_tensor = torch.FloatTensor(order_states).to(self.device)
        
        # 分别编码车辆和订单的状态
        v_encoded = v_tensor
        o_encoded = o_tensor
        repeated_global = v_encoded.unsqueeze(0).expand(o_encoded.size(0), -1)
        actor_input = torch.cat([repeated_global, o_encoded], dim=1)
    
        # 计算原始 logits，其形状应为 [num_order, num_city]
        logits = self.actor(actor_input)

        # 利用 mask 屏蔽不允许的动作，将 mask 为 0 的位置设为负无穷
        if mask is not None:
            # mask 为 [num_order, num_city]，1 表示允许，0 表示不允许
            logits = logits.masked_fill(mask == 0, float('-inf'))
        
        # 根据是否探索选择温度参数,这里也改一下
        temperature = 1 if explore else 0.5
        # 计算 softmax 概率，注意温度参数的使用
        probs = F.softmax(logits / temperature, dim=-1)

        # 根据是否使用贪婪策略选择动作
        if greedy:
            # 选择概率最大的动作
            actions = torch.argmax(probs, dim=-1).tolist()
        else:
            # 按照概率采样动作
            torch.manual_seed(114514)
            actions = [torch.multinomial(p, 1).item() for p in probs]

        log_probs = F.log_softmax(logits / temperature, dim=-1)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)       
        selected_log_probs = log_probs.gather(1, actions_tensor.view(-1, 1)).squeeze()
        
        # 防止inf 和 0导致的异常
        probs = torch.nan_to_num(probs, nan= eplison, posinf=0.0, neginf=0.0)
        selected_log_probs = torch.nan_to_num(selected_log_probs, nan= eplison, posinf=0.0, neginf=0.0)
        log_probs = torch.nan_to_num(log_probs, nan= eplison, posinf=0.0, neginf=0.0)
        # 返回动作以及对应的 log 概率
        return actions, selected_log_probs ,log_probs, probs

    def store_experience(self, vehicle_states, order_states, actions, selected_log_probs, log_probs, probs, rewards, next_vehicle_states, next_order_states, dones):
        """存储经验到 buffer"""
        self.buffer.push(vehicle_states, order_states, actions, selected_log_probs, log_probs, probs, rewards, next_vehicle_states, next_order_states, dones)

    def update(self,time):
        """从 buffer 取样进行训练"""
        if len(self.buffer) < self.batch_size:
            return  # 缓冲区数据不足，不更新

        batch = self.buffer.sample(time)

        # 转换成张量
        v_states = torch.tensor([exp.vehicle_states for exp in batch], dtype=torch.float).to(self.device)
        o_states = torch.tensor([exp.order_states for exp in batch], dtype=torch.float).to(self.device)
        # 从 batch 中提取数据并转换为 PyTorch tensor
        rewards = torch.tensor([exp.rewards for exp in batch], dtype=torch.float).to(self.device)
        # 遍历 batch，确保所有状态数据都是 Tensor
        next_v_states = torch.stack([
            torch.tensor(exp.next_vehicle_states, dtype=torch.float) if isinstance(exp.next_vehicle_states, list) else exp.next_vehicle_states
            for exp in batch
        ]).to(self.device)

        next_o_states = torch.stack([
            torch.tensor(exp.next_order_states, dtype=torch.float) if isinstance(exp.next_order_states, (list, np.ndarray)) else exp.next_order_states
            for exp in batch
        ]).to(self.device)

        current_global = self._get_global_state(v_states[0], o_states[0])
        current_v = self.critic(current_global)
        # 计算 Critic 预测值
        next_vs = torch.stack([self.critic(self._get_global_state(s, o)) for s, o in zip(next_v_states, next_o_states)])
        # 计算 n-step TD 目标
        td_target = torch.tensor(0.0, device=rewards[0].device)
        for i in range(len(rewards)):
            td_target += (self.gamma ** i) * rewards[i]

        # 加上 n 步之后的状态值 V(s_{t+n})，考虑 done 终止
        if len(next_vs) > 0:
            td_target += (self.gamma ** len(rewards)) * next_vs[-1].squeeze() 
        # 计算 Critic 损失
        critic_loss = F.mse_loss(current_v, td_target.detach())

        # 计算 Actor 损失
        probs = torch.stack([
            torch.tensor(exp.probs, dtype=torch.float) if isinstance(exp.probs, list) else exp.probs
            for exp in batch
        ]).to(self.device)
        log_probs = torch.stack([
            torch.tensor(exp.log_probs, dtype=torch.float) if isinstance(exp.log_probs, list) else exp.log_probs
            for exp in batch
        ]).to(self.device)
        selected_log_probs = probs = torch.stack([
            torch.tensor(exp.selected_log_probs, dtype=torch.float) if isinstance(exp.selected_log_probs, list) else exp.selected_log_probs
            for exp in batch
        ]).to(self.device)

        # entropy = -torch.sum(probs * log_probs[0], dim=-1).mean()
        advantage = (td_target - current_v).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # 标准化

        actor_loss = -(selected_log_probs[0] * advantage).mean() # - 0.05 * entropy

        # 合并损失
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数

    def _get_global_state(self, v_states, o_states):
        """获取Critic的全局状态表征（无掩码）"""
        
        v_tensor = torch.FloatTensor(v_states).to(self.device)
        v_encoded = v_tensor
        
        # 订单全局特征
        o_tensor = torch.FloatTensor(o_states).to(self.device)
        o_encoded = o_tensor
        global_order = torch.mean(o_encoded, dim=0)
        
        return torch.cat([v_encoded, global_order])