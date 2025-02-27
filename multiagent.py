import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
from torch import optim
from torch.nn.utils.rnn import pad_sequence

# 定义经验元组
Experience = namedtuple('Experience', [
    'vehicle_states', 'order_states', 'selected_log_probs', 
    'log_probs', 'probs', 'rewards', 'next_vehicle_states', 
    'next_order_states'
])

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class OrderEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(OrderEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, order_states):
        # order_states: [batch_size, seq_len, input_dim] 或 PackedSequence
        output, hidden = self.gru(order_states)
        # output: [batch_size, seq_len, hidden_dim] 或 PackedSequence
        # hidden: [1, batch_size, hidden_dim]
        return hidden.squeeze(0)  # [batch_size, hidden_dim]

class VehicleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VehicleEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, vehicle_states):
        output, hidden = self.gru(vehicle_states)
        return hidden.squeeze(0)  # [batch_size, hidden_dim]

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class MultiAgentAC(nn.Module):
    def __init__(self, device, VEHICLE_STATE_DIM, ORDER_STATE_DIM, NUM_CITIES, HIDDEN_DIM, STATE_DIM):
        super(MultiAgentAC, self).__init__()
        self.buffer = ReplayBuffer(capacity=64)
        self.device = device
        self.NUM_CITIES = NUM_CITIES
        self.batch_size = 16
        
        # 编码器
        self.vehicle_encoder = VehicleEncoder(VEHICLE_STATE_DIM, HIDDEN_DIM).to(device)
        self.order_encoder = OrderEncoder(ORDER_STATE_DIM, HIDDEN_DIM).to(device)
        
        # 共享网络
        self.actor = PolicyNet(STATE_DIM, HIDDEN_DIM, NUM_CITIES).to(device)
        self.critic = ValueNet(HIDDEN_DIM, HIDDEN_DIM).to(device)  # 输入调整为 HIDDEN_DIM
        
        self.optimizer = optim.Adam([
            {'params': self.vehicle_encoder.parameters()},
            {'params': self.order_encoder.parameters()},
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=1e-3)
        # 增大学习率，加快收敛
    def take_action_third(self, vehicle_states, order_states, mask, explore=True, greedy=False):
        epsilon = 0.00001
        mask = torch.from_numpy(mask).to(self.device)
        v_tensor = torch.FloatTensor(vehicle_states).unsqueeze(0).to(self.device)  # [1, num_vehicles, state_dim]
        o_tensor = torch.FloatTensor(order_states).unsqueeze(0).to(self.device)  # [1, num_orders, state_dim]
        
        v_encoded = self.vehicle_encoder(v_tensor)  # [1, hidden_dim]
        o_encoded = self.order_encoder(o_tensor)    # [1, hidden_dim]
        
        global_vehicle = v_encoded  # 已压缩为全局表示
        repeated_global = global_vehicle.repeat(o_encoded.size(0), 1)
        
        actor_input = torch.cat([repeated_global, o_encoded], dim=-1)
        logits = self.actor(actor_input)
        
        logits = logits.masked_fill(mask == 0, float('-inf'))
        temperature = 1 if explore else 0.5
        probs = F.softmax(logits / temperature, dim=-1)
        
        if greedy:
            actions = torch.argmax(probs, dim=-1).tolist()
        else:
            torch.manual_seed(114514)
            actions = [torch.multinomial(p, 1).item() for p in probs]
        
        log_probs = F.log_softmax(logits / temperature, dim=-1)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        selected_log_probs = log_probs.gather(1, actions_tensor.view(-1, 1)).squeeze()
        
        probs = torch.nan_to_num(probs, nan=epsilon, posinf=0.0, neginf=0.0)
        selected_log_probs = torch.nan_to_num(selected_log_probs, nan=epsilon, posinf=0.0, neginf=0.0)
        log_probs = torch.nan_to_num(log_probs, nan=epsilon, posinf=0.0, neginf=0.0)
        
        return actions, selected_log_probs, log_probs, probs

    def store_experience(self, vehicle_states, order_states, selected_log_probs, 
                         log_probs, probs, rewards, next_vehicle_states, next_order_states):
        if torch.is_tensor(selected_log_probs):
            selected_log_probs = selected_log_probs.cpu().tolist()
        if torch.is_tensor(log_probs):
            log_probs = log_probs.cpu().tolist()
        if torch.is_tensor(probs):
            probs = probs.cpu().tolist()
        self.buffer.push(vehicle_states, order_states, selected_log_probs, 
                         log_probs, probs, rewards, next_vehicle_states, next_order_states)

    def _get_global_state(self, v_states, o_states):
        v_tensor = torch.FloatTensor(v_states).to(self.device)
        o_tensor = torch.FloatTensor(o_states).to(self.device)
        v_encoded = self.vehicle_encoder(v_tensor)
        o_encoded = self.order_encoder(o_tensor)
        return torch.cat([v_encoded, o_encoded])

    def update_third_buffer_rnn(self):
        if len(self.buffer) < self.batch_size:
            return
        experiences = self.buffer.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        
  
        def pad_states(states_list):
            tensors = [torch.tensor(state, dtype=torch.float) for state in states_list]
            padded = pad_sequence(tensors, batch_first=True)
            lengths = torch.tensor([len(state) for state in states_list], dtype=torch.long)
            return padded, lengths

        v_states, v_lengths = pad_states(batch.vehicle_states)
        o_states, o_lengths = pad_states(batch.order_states)
        next_v_states, next_v_lengths = pad_states(batch.next_vehicle_states)
        next_o_states, next_o_lengths = pad_states(batch.next_order_states)
        
        v_states = v_states.to(self.device)
        o_states = o_states.to(self.device)
        next_v_states = next_v_states.to(self.device)
        next_o_states = next_o_states.to(self.device)
        v_lengths = v_lengths.to(self.device)
        o_lengths = o_lengths.to(self.device)
        
        rewards = torch.tensor(batch.rewards, dtype=torch.float).to(self.device)
        
        max_orders = max(len(probs) for probs in batch.selected_log_probs)
        selected_log_probs_padded = [list(probs) + [0]*(max_orders - len(probs)) 
                                    for probs in batch.selected_log_probs]
        selected_log_probs = torch.tensor(selected_log_probs_padded, dtype=torch.float).to(self.device)
        
        max_orders_log = max(len(p) for p in batch.log_probs)
        max_cities = max(len(p[0]) for p in batch.log_probs)
        log_probs_padded = [[lp + [0]*(max_cities - len(lp)) for lp in p] + [[0]*max_cities]*(max_orders_log - len(p)) 
                        for p in batch.log_probs]
        log_probs = torch.tensor(log_probs_padded, dtype=torch.float).to(self.device)
        
        probs_padded = [[pr + [0]*(max_cities - len(pr)) for pr in p] + [[0]*max_cities]*(max_orders_log - len(p)) 
                    for p in batch.probs]
        probs = torch.tensor(probs_padded, dtype=torch.float).to(self.device)

        # GRU 编码
        def encode_states(states, lengths, encoder):
            packed = torch.nn.utils.rnn.pack_padded_sequence(states, lengths, batch_first=True, enforce_sorted=False)
            encoded = encoder(packed)  # 直接接收单个张量
            return encoded

        v_encoded = encode_states(v_states, v_lengths, self.vehicle_encoder)
        o_encoded = encode_states(o_states, o_lengths, self.order_encoder)
        next_v_encoded = encode_states(next_v_states, next_v_lengths, self.vehicle_encoder)
        next_o_encoded = encode_states(next_o_states, next_o_lengths, self.order_encoder)

        global_current = torch.cat([v_encoded, o_encoded], dim=-1)
        global_next = torch.cat([next_v_encoded, next_o_encoded], dim=-1)

        current_v = self.critic(v_encoded.mean(dim=0, keepdim=True))
        next_v = self.critic(next_v_encoded.mean(dim=0, keepdim=True))
        td_target = rewards + 0.95 * next_v
        # critic_loss = F.mse_loss(current_v, td_target.detach())
        # 对大误差更鲁棒
        critic_loss = F.smooth_l1_loss(current_v, td_target.detach())
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        advantage = (td_target - current_v).detach()
        # 标准化advatage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        # 增大熵系数
        actor_loss = -(selected_log_probs.mean(dim=1) * advantage.squeeze()).mean() - 0.05 * entropy
        

        total_loss = actor_loss + critic_loss
        """
        print(f"Rewards: {rewards.mean().item()}, Current_v: {current_v.item()}, Next_v: {next_v.item()}")
        print(f"Advantage mean: {advantage.mean().item()}, std: {advantage.std().item()}")
        print(f"Actor loss: {actor_loss.item()}, Critic loss: {critic_loss.item()}, Total loss: {total_loss.item()}")
        """
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vehicle_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.order_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()