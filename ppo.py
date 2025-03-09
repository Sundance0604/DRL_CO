import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple,deque
from torch import optim
import torch.nn.utils.rnn as rnn_utils
import os
from torch.nn.utils.rnn import pad_sequence

# 定义经验元组
Experience = namedtuple('Experience', [
    'vehicle_states', 'order_states', 'selected_log_probs', 
    'log_probs', 'probs', 'rewards', 'next_vehicle_states', 
    'next_order_states'
])
class OrderEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(OrderEncoder, self).__init__()
        self.input_dim = input_dim  # 记录超参数
        self.hidden_dim = hidden_dim  # 记录超参数
        self.init_params = {'input_dim': input_dim, 'hidden_dim': hidden_dim} 
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, order_states):
        x = F.relu(self.fc1(order_states))
        return F.relu(self.fc2(x))
#   请注意改成GRU而非Linear
class VehicleEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VehicleEncoder, self).__init__()
        self.input_dim = input_dim  # 记录超参数
        self.hidden_dim = hidden_dim  # 记录超参数
        self.init_params = {'input_dim': input_dim, 'hidden_dim': hidden_dim} 
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vehicle_states):
        x = F.relu(self.fc1(vehicle_states))
        return F.relu(self.fc2(x))
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

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
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
#   请注意改成GRU而非Linear

class MultiAgentAC(torch.nn.Module):
    def __init__(self, device, VEHICLE_STATE_DIM, 
                 ORDER_STATE_DIM, NUM_CITIES, 
                 HIDDEN_DIM, STATE_DIM):
        super(MultiAgentAC, self).__init__()
        self.buffer = ReplayBuffer(capacity=64)
       
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
        self.batch_size = 16
        self.active = False
        self.current_order = []
        self.last_order = []
        self.reward = 0
        self.action_key = ''
        self.action = []
        self.v_states = np.array([])
    
    # 改变vehicle_states,不再是平均值，而是其他办法
    def take_action_vehicle(self, vehicle_states, order_states, mask, explore=True, greedy=False):
        """为当前活跃订单生成动作"""
        epsilon = 0.00001
        mask = torch.from_numpy(mask).to(self.device)
        # 将状态转换为 tensor，并放到相应设备上
        v_tensor = torch.FloatTensor(vehicle_states).to(self.device)
        o_tensor = torch.FloatTensor(order_states).to(self.device)
        
        # 分别编码车辆和订单的状态
        v_encoded = v_tensor
        o_encoded = o_tensor

        # 定义一个简单的注意力层
        attention = torch.nn.Linear(v_encoded.size(1), 1).to(self.device)
        attn_weights = torch.softmax(attention(v_encoded), dim=0)  # 计算注意力权重
        global_vehicle = torch.sum(attn_weights * v_encoded, dim=0)  # 加权和
        repeated_global = global_vehicle.repeat(o_encoded.size(0), 1)
        actor_input = torch.cat([repeated_global, o_encoded], dim=1)
        
        # 计算原始 logits
        logits = self.actor(actor_input)

        # 利用 mask 屏蔽不允许的动作
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        
        # 根据是否探索选择温度参数
        temperature = 1 if explore else 0.5
        probs = F.softmax(logits / temperature, dim=-1)

        # 根据是否使用贪婪策略选择动作
        if greedy:
            actions = torch.argmax(probs, dim=-1).tolist()
        else:
            torch.manual_seed(114514)
            actions = [torch.multinomial(p, 1).item() for p in probs]

        log_probs = F.log_softmax(logits / temperature, dim=-1)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)       
        selected_log_probs = log_probs.gather(1, actions_tensor.view(-1, 1)).squeeze()
        
        # 防止 inf 和 0 导致的异常
        probs = torch.nan_to_num(probs, nan=epsilon, posinf=0.0, neginf=0.0)
        selected_log_probs = torch.nan_to_num(selected_log_probs, nan=epsilon, posinf=0.0, neginf=0.0)
        log_probs = torch.nan_to_num(log_probs, nan=epsilon, posinf=0.0, neginf=0.0)
        
        # 返回动作及相关概率信息（这些将成为“旧”策略数据）
        return actions, selected_log_probs, log_probs, probs
   
    def update_third(self, vehicle_states, order_states, actions, selected_log_probs_old, log_probs_old, probs_old, rewards, 
                 next_vehicle_states, next_order_states, dones):
        # 将输入数据转换为 tensor
        v_states = torch.tensor(vehicle_states, dtype=torch.float).to(self.device)
        o_states = torch.tensor(order_states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_v_states = torch.tensor(next_vehicle_states, dtype=torch.float).to(self.device)
        next_o_states = torch.tensor(next_order_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        selected_log_probs_old = selected_log_probs_old.to(self.device)

        # 计算 Critic 损失
        current_global = self._get_global_state(v_states, o_states)
        next_global = self._get_global_state(next_v_states, next_o_states)
        current_v = self.critic(current_global)
        next_v = self.critic(next_global)
        td_target = rewards + 0.95 * next_v * (1 - dones)  # 折扣因子 0.95
        critic_loss = F.mse_loss(current_v, td_target.detach())

        # 计算优势函数
        advantage = (td_target - current_v).detach()

        # 计算新策略的 log_probs
        v_encoded = v_states
        o_encoded = o_states
        attention = torch.nn.Linear(v_encoded.size(1), 1).to(self.device)
        attn_weights = torch.softmax(attention(v_encoded), dim=0)
        global_vehicle = torch.sum(attn_weights * v_encoded, dim=0)
        repeated_global = global_vehicle.repeat(o_encoded.size(0), 1)
        actor_input = torch.cat([repeated_global, o_encoded], dim=1)
        
        logits_new = self.actor(actor_input)
        log_probs_new = F.log_softmax(logits_new, dim=-1)
        selected_log_probs_new = log_probs_new.gather(1, actions.view(-1, 1)).squeeze()

        # 计算概率比 r_t(θ)
        ratio = torch.exp(selected_log_probs_new - selected_log_probs_old)

        # PPO-Clip 参数
        epsilon = 0.2  # 裁剪范围，通常为 0.1 或 0.2

        # 计算裁剪后的 Actor 损失
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        # 计算熵正则化项
        probs_new = F.softmax(logits_new, dim=-1)
        entropy = -torch.sum(probs_new * log_probs_new, dim=-1).mean()
        actor_loss -= 0.01 * entropy  # 熵系数 0.01，与原代码保持一致

        # 合并损失
        total_loss = actor_loss + critic_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()
        
    
    def _get_global_state(self, v_states, o_states):
        """获取Critic的全局状态表征（无掩码）"""
        
        v_tensor = torch.FloatTensor(v_states).to(self.device)
        v_encoded = v_tensor
        global_vehicle = torch.mean(v_encoded, dim=0)
        
        # 订单全局特征
        o_tensor = torch.FloatTensor(o_states).to(self.device)
        o_encoded = o_tensor
        global_order = torch.mean(o_encoded, dim=0)
        
        return torch.cat([global_vehicle, global_order])
    
    
