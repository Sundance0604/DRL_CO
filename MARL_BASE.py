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
#   请注意改成GRU而非Linear
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
    
    def take_action_mask(self, vehicle_states, order_states, mask,explore=True, greedy=False):
        """为当前活跃订单生成动作 ⭐"""
        mask = torch.from_numpy(mask).to(self.device)
        # 将状态转换为 tensor，并放到相应设备上
        v_tensor = torch.FloatTensor(vehicle_states).to(self.device)
        o_tensor = torch.FloatTensor(order_states).to(self.device)
        
        # 分别编码车辆和订单的状态
        v_encoded = self.vehicle_encoder(v_tensor)
        o_encoded = self.order_encoder(o_tensor)

        # 计算全局车辆特征
        global_vehicle = torch.mean(v_encoded, dim=0)
        repeated_global = global_vehicle.repeat(o_encoded.size(0), 1)

        # 拼接每个订单的输入
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

        return actions
    
    
    
    def take_action_third(self, vehicle_states, order_states, mask,explore=True, greedy=False):
        """为当前活跃订单生成动作 ⭐"""
        eplison = 0.00001
        mask = torch.from_numpy(mask).to(self.device)
        # 将状态转换为 tensor，并放到相应设备上
        v_tensor = torch.FloatTensor(vehicle_states).to(self.device)
        o_tensor = torch.FloatTensor(order_states).to(self.device)
        
        # 分别编码车辆和订单的状态
        v_encoded = self.vehicle_encoder(v_tensor)
        o_encoded = self.order_encoder(o_tensor)

        # 计算全局车辆特征
        global_vehicle = torch.mean(v_encoded, dim=0)
        repeated_global = global_vehicle.repeat(o_encoded.size(0), 1)

        # 拼接每个订单的输入
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
    
    def take_action_skyrim(self, vehicle_states, order_states,explore=True, greedy=False):
        """为当前活跃订单生成动作 ⭐"""
     
        
        # 将状态转换为 tensor，并放到相应设备上
        v_tensor = torch.FloatTensor(vehicle_states).to(self.device)
        o_tensor = torch.FloatTensor(order_states).to(self.device)
        
        # 分别编码车辆和订单的状态
        v_encoded = self.vehicle_encoder(v_tensor)
        o_encoded = self.order_encoder(o_tensor)

        # 计算全局车辆特征
        global_vehicle = torch.mean(v_encoded, dim=0)
        repeated_global = global_vehicle.repeat(o_encoded.size(0), 1)

        # 拼接每个订单的输入
        actor_input = torch.cat([repeated_global, o_encoded], dim=1)
        
        # 计算原始 logits，其形状应为 [num_order, num_city]
        logits = self.actor(actor_input)

        
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
            # 这里改一下，改成完全随机的
            # actions = [torch.multinomial(p, 1).item() for p in probs]
            actions = [torch.randint(0, self.NUM_CITIES, (1,)).item() for _ in range(len(order_states))]  

        
        # 返回动作
        return actions 
    
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
        td_target = rewards + 0.95 * next_v * (1 - dones)
        critic_loss = F.mse_loss(current_v, td_target.detach())

        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        # 不再是num_orders这一固定的
        advantage = (td_target - current_v).detach().repeat_interleave(len(order_states))
        actor_loss = -(selected_log_probs * advantage).mean() - 0.01 * entropy

        # 合并损失
        total_loss = actor_loss + critic_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        # 也对critic进行梯度裁剪,这是修改处
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()
        
    def store_experience(self, vehicle_states, order_states, selected_log_probs, 
                         log_probs, probs, rewards, next_vehicle_states, next_order_states):
        if torch.is_tensor(selected_log_probs):
            selected_log_probs = selected_log_probs.cpu().tolist()  # 转为普通列表
        if torch.is_tensor(log_probs):
            log_probs = log_probs.cpu().tolist()  # 同样处理 log_probs
        if torch.is_tensor(probs):
            probs = probs.cpu().tolist()
        # 将经验存入缓冲区
        self.buffer.push(vehicle_states, order_states, selected_log_probs, 
                         log_probs, probs, rewards, next_vehicle_states, next_order_states)

    def update_third_buffer(self):
        # 如果经验不足一个批次，跳过更新
        if len(self.buffer) < self.batch_size:
            return

        # 从缓冲区采样一个批次
        experiences = self.buffer.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # 将批次数据转为张量
        v_states = torch.tensor(batch.vehicle_states, dtype=torch.float).to(self.device)
        o_states = torch.tensor(batch.order_states, dtype=torch.float).to(self.device)
       
        rewards = torch.tensor(batch.rewards, dtype=torch.float).to(self.device)
        next_v_states = torch.tensor(batch.next_vehicle_states, dtype=torch.float).to(self.device)
        next_o_states = torch.tensor(batch.next_order_states, dtype=torch.float).to(self.device)
    
        selected_log_probs = torch.tensor(batch.selected_log_probs, dtype=torch.float).to(self.device)
        log_probs = torch.tensor(batch.log_probs, dtype=torch.float).to(self.device)
        probs = torch.tensor(batch.probs, dtype=torch.float).to(self.device)

        # 计算 Critic 损失
        current_global = self._get_global_state(v_states, o_states)  # [batch_size, global_dim]
        next_global = self._get_global_state(next_v_states, next_o_states)
        current_v = self.critic(current_global)  # [batch_size, 1]
        next_v = self.critic(next_global)
        td_target = rewards + 0.95 * next_v   # [batch_size, 1]
        critic_loss = F.mse_loss(current_v, td_target.detach())

        # 计算 Actor 损失
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()  # 在批次上平均
        advantage = (td_target - current_v).detach()  # [batch_size, 1]
        # 假设 selected_log_probs 是 [batch_size]，advantage 需要展平或调整维度
        actor_loss = -(selected_log_probs * advantage.squeeze()).mean() - 0.01 * entropy

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
        v_encoded = self.vehicle_encoder(v_tensor)
        global_vehicle = torch.mean(v_encoded, dim=0)
        
        # 订单全局特征
        o_tensor = torch.FloatTensor(o_states).to(self.device)
        o_encoded = self.order_encoder(o_tensor)
        global_order = torch.mean(o_encoded, dim=0)
        
        return torch.cat([global_vehicle, global_order])
    
    def update(self, vehicle_states, order_states, actions, rewards, 
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
        td_target = rewards + 0.95 * next_v * (1 - dones)
        critic_loss = F.mse_loss(current_v, td_target.detach())

        # 计算 Actor 损失
        v_encoded = self.vehicle_encoder(v_states)
        o_encoded = self.order_encoder(o_states)
        # global_vehicle = torch.mean(v_encoded, dim=1, keepdim=True).repeat(1, o_encoded.size(1), 1)
        global_vehicle = torch.mean(v_encoded, dim=0)
        repeated_global = global_vehicle.repeat(o_encoded.size(0), 1)
        actor_input = torch.cat([repeated_global, o_encoded], dim=-1)

        logits = self.actor(actor_input.view(-1, 256))
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.view(-1, 1)).squeeze()
        
        # 熵正则化
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        # 不再是num_orders这一固定的
        advantage = (td_target - current_v).detach().repeat_interleave(len(order_states))
        actor_loss = -(selected_log_probs * advantage).mean() - 0.01 * entropy

        # 合并损失
        total_loss = actor_loss + critic_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        # 也对critic进行梯度裁剪,这是修改处
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()

    def update_third_buffer_rnn(self):
        if len(self.buffer) < self.batch_size:
            return
        experiences = self.buffer.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # 处理变长序列
        def pad_states(states_list):
            # states_list 是 [batch_size] 的列表，每个元素是 [num_items, state_dim]
            tensors = [torch.tensor(state, dtype=torch.float) for state in states_list]
            padded = pad_sequence(tensors, batch_first=True)  # [batch_size, max_len, state_dim]
            lengths = torch.tensor([len(state) for state in states_list], dtype=torch.long)
            return padded, lengths

        v_states, v_lengths = pad_states(batch.vehicle_states)
        o_states, o_lengths = pad_states(batch.order_states)
        next_v_states, next_v_lengths = pad_states(batch.next_vehicle_states)
        next_o_states, next_o_lengths = pad_states(batch.next_order_states)
        
        # 转为设备
        v_states = v_states.to(self.device)
        o_states = o_states.to(self.device)
        next_v_states = next_v_states.to(self.device)
        next_o_states = next_o_states.to(self.device)
        v_lengths = v_lengths.to(self.device)
        o_lengths = o_lengths.to(self.device)
        
        # 其他张量
        rewards = torch.tensor(batch.rewards, dtype=torch.float).to(self.device)
        
        max_orders = max(len(probs) for probs in batch.selected_log_probs)
        selected_log_probs_padded = [list(probs) + [0]*(max_orders - len(probs)) 
                                    for probs in batch.selected_log_probs]
        selected_log_probs = torch.tensor(selected_log_probs_padded, dtype=torch.float).to(self.device)
        # GRU 编码
        def encode_states(states, lengths, encoder):
            # states: [batch_size, max_len, state_dim]
            # lengths: [batch_size]
            packed = torch.nn.utils.rnn.pack_padded_sequence(states, lengths, batch_first=True, enforce_sorted=False)
            output, hidden = encoder(packed)  # hidden: [1, batch_size, hidden_dim]
            return hidden.squeeze(0)  # [batch_size, hidden_dim]

        v_encoded = encode_states(v_states, v_lengths, self.vehicle_encoder)
        o_encoded = encode_states(o_states, o_lengths, self.order_encoder)
        next_v_encoded = encode_states(next_v_states, next_v_lengths, self.vehicle_encoder)
        next_o_encoded = encode_states(next_o_states, next_o_lengths, self.order_encoder)

        # 计算全局状态
        global_current = torch.cat([torch.mean(v_encoded, dim=0, keepdim=True).expand_as(o_encoded), o_encoded], dim=-1)
        global_next = torch.cat([torch.mean(next_v_encoded, dim=0, keepdim=True).expand_as(next_o_encoded), next_o_encoded], dim=-1)

        # Critic 和 Actor 计算
        current_v = self.critic(torch.mean(v_encoded, dim=0))  # 全局车辆状态
        next_v = self.critic(torch.mean(next_v_encoded, dim=0))
        td_target = rewards + 0.95 * next_v 
        critic_loss = F.mse_loss(current_v, td_target.detach())

        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        advantage = (td_target - current_v).detach()
        actor_loss = -(selected_log_probs * advantage).mean() - 0.01 * entropy

        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vehicle_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.order_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()

 
