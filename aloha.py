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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class MultiAgentAC(torch.nn.Module):
    def __init__(self, device, VEHICLE_STATE_DIM, 
                 ORDER_STATE_DIM, NUM_CITIES, 
                 HIDDEN_DIM, STATE_DIM):
        super(MultiAgentAC, self).__init__()
        self.buffer = ReplayBuffer(capacity=64)
       
        self.device = device
        self.NUM_CITIES = NUM_CITIES
        
        # 共享网络
        self.actor = PolicyNet(STATE_DIM, HIDDEN_DIM, NUM_CITIES).to(device)
        self.critic = ValueNet(STATE_DIM, HIDDEN_DIM).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-3)
        
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
   
    def update_third(self, vehicle_states, order_states, actions, selected_log_probs,log_probs, probs,rewards, 
           next_vehicle_states, next_order_states, dones):

        v_states = torch.tensor(vehicle_states, dtype=torch.float).to(self.device)
        o_states = torch.tensor(order_states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_v_states = torch.tensor(next_vehicle_states, dtype=torch.float).to(self.device)
        next_o_states = torch.tensor(next_order_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        # 计算 Critic 损失
        
        current_global = self._get_global_state(v_states, o_states)
        next_global = self._get_global_state(next_v_states, next_o_states)
        current_v = self.critic(current_global)
        next_v = self.critic(next_global)
        td_target = rewards + 0.9 * next_v * (1 - dones)
        critic_loss = F.mse_loss(current_v, td_target.detach())
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        # 不再是num_orders这一固定的
        advantage = (td_target - current_v).detach().repeat_interleave(len(order_states))
        actor_loss = -(selected_log_probs * advantage).mean() - 0.05 * entropy
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
    
    