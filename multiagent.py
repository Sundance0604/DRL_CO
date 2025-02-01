import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from torch import optim
import torch.nn.utils.rnn as rnn_utils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class OrderEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(OrderEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, order_states):
        x = F.relu(self.fc1(order_states))
        return F.relu(self.fc2(x))

class VehicleEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VehicleEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vehicle_states):
        x = F.relu(self.fc1(vehicle_states))
        return F.relu(self.fc2(x))
    
class MultiAgentAC:
    def __init__(self, device, VEHICLE_STATE_DIM, 
                 ORDER_STATE_DIM, NUM_CITIES, 
                 HIDDEN_DIM, STATE_DIM):
        self.device = device
        self.NUM_CITIES = NUM_CITIES
        
        # 编码器
        self.vehicle_encoder = VehicleEncoder(VEHICLE_STATE_DIM, HIDDEN_DIM).to(device)
        self.order_encoder = OrderEncoder(ORDER_STATE_DIM, HIDDEN_DIM).to(device)
        
        # 共享网络
        self.actor = PolicyNet(STATE_DIM, HIDDEN_DIM, NUM_CITIES).to(device)
        self.critic = ValueNet(STATE_DIM, HIDDEN_DIM).to(device)
        
        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.vehicle_encoder.parameters()},
            {'params': self.order_encoder.parameters()},
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=3e-4)
        
        # 动态智能体管理 ⭐
        self.active_orders = {}       # 当前活跃订单 {order_id: order_state}
        self.next_order_id = 0        # 订单ID生成器
        self.buffer = deque(maxlen=10000)
        self.batch_size = 64
   
    def get_new_orders(self, order_states):
        """获取新订单⭐"""
        new_orders = {}
        for state in order_states:
            new_orders[self.next_order_id] = state
            self.next_order_id += 1
        return new_orders
        
    def get_active_order_states(self):
        """获取当前活跃订单状态 ⭐"""
        return np.array([state for state in self.active_orders.values()])
    
    def take_action(self, vehicle_states, order_states, explore=True):
        """为当前活跃订单生成动作 ⭐"""
        # 现在先不弄动态的，简单一些
        # if not self.active_orders:
        #    return []
        
        # 获取活跃订单状态
        # order_states = self.get_active_order_states()
        
        # 编码
        v_tensor = torch.FloatTensor(vehicle_states).to(self.device)
        o_tensor = torch.FloatTensor(order_states).to(self.device)
        v_encoded = self.vehicle_encoder(v_tensor)
        o_encoded = self.order_encoder(o_tensor)
        
        # 全局车辆特征
        global_vehicle = torch.mean(v_encoded, dim=0)
        repeated_global = global_vehicle.repeat(o_encoded.size(0), 1)
        
        # 拼接每个订单的输入
        actor_input = torch.cat([repeated_global, o_encoded], dim=1)
        logits = self.actor(actor_input)
        # print(logits)
        # 探索策略
        
        probs = F.softmax(logits / (0.5 if explore else 1.0), dim=-1)
        # print(probs)
        # 采样动作
        
        actions = [torch.multinomial(p, 1).item() for p in probs]
        # print(actions)
        return actions
    
    def update(self, vehicle_states, order_states, actions, rewards, 
           next_vehicle_states, next_order_states, dones=True):

        if len(self.buffer) < self.batch_size:
            return
        """
        batch = random.sample(self.buffer, self.batch_size)
        v_states, o_states, actions, rewards, next_v_states, next_o_states, dones = zip(*batch)
        """
        
        v_states = torch.tensor(vehicle_states, dtype=torch.float).to(self.device)
        o_states = torch.tensor(order_states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_v_states = torch.tensor(next_vehicle_states, dtype=torch.float).to(self.device)
        next_o_states = torch.tensor(next_order_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        # 计算 Critic 损失
        """
        current_global = self.get_global_state(v_states, o_states)
        next_global = self.get_global_state(next_v_states, next_o_states)
        current_v = self.critic(current_global)
        next_v = self.critic(next_global)
        td_target = rewards + 0.95 * next_v * (1 - dones)
        critic_loss = F.mse_loss(current_v, td_target.detach())

        # 计算 Actor 损失
        v_encoded = self.vehicle_encoder(v_states)
        o_encoded = self.order_encoder(o_states)
        global_vehicle = torch.mean(v_encoded, dim=1, keepdim=True).repeat(1, o_encoded.size(1), 1)
        actor_input = torch.cat([global_vehicle, o_encoded], dim=-1)

        logits = self.actor(actor_input.view(-1, 128))
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.view(-1, 1)).squeeze()

        # 熵正则化
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        advantage = (td_target - current_v).detach().repeat_interleave(self.NUM_ORDERS)
        actor_loss = -(selected_log_probs * advantage).mean() - 0.01 * entropy

        # 合并损失
        total_loss = actor_loss + critic_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizer.step()
        """

        # 编码状态
        v_encoded = self.vehicle_encoder(v_states)
        o_encoded = self.order_encoder(o_states)

        # 获取全局状态
        global_vehicle = torch.mean(v_encoded, dim=0, keepdim=True).repeat(o_encoded.shape[0], 1)
        current_global = torch.cat([global_vehicle, o_encoded], dim=1)

        next_v_encoded = self.vehicle_encoder(next_v_states)
        next_o_encoded = self.order_encoder(next_o_states)
        next_global_vehicle = torch.mean(next_v_encoded, dim=0, keepdim=True).repeat(next_o_encoded.shape[0], 1)
        next_global = torch.cat([next_global_vehicle, next_o_encoded], dim=1)

        # 计算 Critic 损失
        current_v = self.critic(current_global)
        next_v = self.critic(next_global)
        td_target = rewards + 0.95 * next_v * (1 - dones)
        critic_loss = F.mse_loss(current_v, td_target.detach())

        # 计算 Actor 损失
        logits = self.actor(current_global)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.view(-1, 1)).squeeze()

        # 熵正则化
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        advantage = (td_target - current_v).detach()
        actor_loss = -(selected_log_probs * advantage).mean() - 0.01 * entropy

        # 反向传播
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizer.step()
        
"""

class DynamicOrderEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DynamicOrderEncoder, self).__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)
        
    def forward(self, x, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return h_n.squeeze(0)  # 取最后一个时间步的隐藏状态
    
class DynamicMultiAgentAC:
    def __init__(self, device, VEHICLE_STATE_DIM, 
                 ORDER_STATE_DIM, NUM_CITIES, 
                 HIDDEN_DIM, STATE_DIM):
        self.device = device
        self.NUM_CITIES = NUM_CITIES

        # 编码器
        self.vehicle_encoder = VehicleEncoder(VEHICLE_STATE_DIM, HIDDEN_DIM).to(device)
        self.order_encoder = DynamicOrderEncoder(ORDER_STATE_DIM, HIDDEN_DIM).to(device)

        # 共享网络
        self.actor = PolicyNet(HIDDEN_DIM * 2, HIDDEN_DIM, NUM_CITIES).to(device)
        self.critic = ValueNet(HIDDEN_DIM * 2, HIDDEN_DIM).to(device)

        # 优化器
        # self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.optimizer = optim.Adam([
            {'params': self.vehicle_encoder.parameters()},
            {'params': self.order_encoder.parameters()},
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=3e-4)

        # 经验回放
        self.buffer = deque(maxlen=10000)
        self.batch_size = 64

    def take_action(self, vehicle_states, order_states, order_lengths, explore=True):
        # 处理动态订单数目，生成动作 
        v_tensor = torch.FloatTensor(vehicle_states).to(self.device)
        o_tensor = torch.FloatTensor(order_states).to(self.device)

        # 编码
        v_encoded = self.vehicle_encoder(v_tensor)
        o_encoded = self.order_encoder(o_tensor, order_lengths)

        # 计算全局车辆特征
        global_vehicle = torch.mean(v_encoded, dim=0, keepdim=True).repeat(o_encoded.shape[0], 1)

        # 拼接全局特征与订单特征
        actor_input = torch.cat([global_vehicle, o_encoded], dim=1)
        logits = self.actor(actor_input)

        # 计算动作概率
        probs = F.softmax(logits / (0.5 if explore else 1.0), dim=-1)
        actions = [torch.multinomial(p, 1).item() for p in probs]

        return actions

    def update(self, vehicle_states, order_states, order_lengths, actions, rewards, 
               next_vehicle_states, next_order_states, next_order_lengths, dones=True):
        # A2C 更新 
        if len(self.buffer) < self.batch_size:
            return
        
        # 转换数据
        v_states = torch.tensor(vehicle_states, dtype=torch.float).to(self.device)
        o_states = torch.tensor(order_states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_v_states = torch.tensor(next_vehicle_states, dtype=torch.float).to(self.device)
        next_o_states = torch.tensor(next_order_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        # 编码状态
        v_encoded = self.vehicle_encoder(v_states)
        o_encoded = self.order_encoder(o_states, order_lengths)

        # 获取全局状态
        global_vehicle = torch.mean(v_encoded, dim=0, keepdim=True).repeat(o_encoded.shape[0], 1)
        current_global = torch.cat([global_vehicle, o_encoded], dim=1)

        next_v_encoded = self.vehicle_encoder(next_v_states)
        next_o_encoded = self.order_encoder(next_o_states, next_order_lengths)
        next_global_vehicle = torch.mean(next_v_encoded, dim=0, keepdim=True).repeat(next_o_encoded.shape[0], 1)
        next_global = torch.cat([next_global_vehicle, next_o_encoded], dim=1)

        # 计算 Critic 损失
        current_v = self.critic(current_global)
        next_v = self.critic(next_global)
        td_target = rewards + 0.95 * next_v * (1 - dones)
        critic_loss = F.mse_loss(current_v, td_target.detach())

        # 计算 Actor 损失
        logits = self.actor(current_global)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.view(-1, 1)).squeeze()

        # 熵正则化
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        advantage = (td_target - current_v).detach()
        actor_loss = -(selected_log_probs * advantage).mean() - 0.01 * entropy

        # 反向传播
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizer.step()
"""