import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from torch import optim
from typing import List, Dict

class TimeAwareOrder:
    def __init__(self, state: np.ndarray, create_step: int, duration: int):
        """
        时间感知订单结构
        :param state: 订单状态向量
        :param create_step: 订单创建时间步
        :param duration: 订单持续时间步数
        """
        self.state = state
        self.create_step = create_step
        self.end_step = create_step + duration

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
    
class DynamicMultiAgentAC:
    def __init__(self, device, vehicle_state_dim, order_state_dim, 
                 num_cities, hidden_dim, max_orders):
        self.device = device
        self.num_cities = num_cities
        self.max_orders = max_orders  # 最大跟踪订单数
        
        # 编码器
        self.vehicle_encoder = VehicleEncoder(vehicle_state_dim, hidden_dim).to(device)
        self.order_encoder = OrderEncoder(order_state_dim, hidden_dim).to(device)
        
        # 共享网络
        state_dim = hidden_dim * 2  # 车辆全局特征 + 订单特征
        self.actor = PolicyNet(state_dim, hidden_dim, num_cities).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        
        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.vehicle_encoder.parameters()},
            {'params': self.order_encoder.parameters()},
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=3e-4)
        
        # 动态订单管理
        self.total_orders = {}  # 跟踪所有订单
        self.time = 0  # 当前时间步
        
        # 经验回放
        self.buffer = deque(maxlen=10000)
        self.batch_size = 64

    def update_time_step(self, time: int):
        """更新时间步并清除过期订单"""
        self.time = time
        # 移除已过期订单（结束时间 <= 当前时间步）
        # self.total_orders = [o for o in self.total_orders if o.end_step > self.time]

    def generate_mask(self) -> np.ndarray:
        """生成当前激活订单掩码"""
        """生成当前激活订单掩码（返回三维形状）"""
        mask = np.zeros((1, self.max_orders), dtype=bool)  # 添加batch维度
        for i, order in enumerate(self.total_orders.values()):
            if i >= self.max_orders:
                break
            if order.start_time <= self.time < order.end_time and order.matched is False:
                mask[0, i] = True  # [1, max_orders]
        return mask

    def _pad_order_states(self) -> np.ndarray:
        """生成填充后的订单状态矩阵"""
        padded = np.zeros((self.max_orders, self.order_encoder.fc1.in_features))
        for i, order in enumerate(self.total_orders[:self.max_orders]):
            padded[i] = order.state
        return padded

    def track_new_orders(self, new_orders: List[TimeAwareOrder]):
        """跟踪新出现的订单"""
        for order in new_orders:
            if len(self.total_orders) < self.max_orders:
                self.total_orders.append(order)

    def take_action(self, vehicle_states: np.ndarray, order_states, explore=True) -> List[int]:
        """生成当前激活订单的动作"""
        # 准备输入数据
        # order_states = self._pad_order_states()
        mask = self.generate_mask()
        active_indices = np.where(mask)[0]
        
        if len(active_indices) == 0:
            print("No active orders!")
            return []
        # 确保订单状态是三维的 [batch=1, max_orders, feat]
        if order_states.ndim == 2:
            order_states = np.expand_dims(order_states, axis=0)
        # 转换为张量
        v_tensor = torch.FloatTensor(vehicle_states).to(self.device)
        o_tensor = torch.FloatTensor(order_states).to(self.device)
        
        # 编码特征
        v_encoded = self.vehicle_encoder(v_tensor)
        o_encoded = self.order_encoder(o_tensor)
        
        # 计算全局车辆特征
        global_vehicle = torch.mean(v_encoded, dim=0)
        
        # 为每个订单生成输入
        actor_inputs = []
        for i in active_indices:
            order_feat = o_encoded[i]
            combined = torch.cat([global_vehicle, order_feat])
            actor_inputs.append(combined)
        
        # 批量处理
        if len(actor_inputs) > 0:
            batch = torch.stack(actor_inputs)
            logits = self.actor(batch)
            probs = F.softmax(logits / (0.5 if explore else 1.0), dim=-1)
            actions = [torch.multinomial(p, 1).item() for p in probs]
            return actions
        else:
            print("No actor inputs!")
            return []

    def update(self, v_states, o_states, next_v, next_o, actions, rewards, masks,dones=True):
        """训练更新（带掩码处理）"""
        # if len(self.buffer) < self.batch_size:
        #    return
        next_masks = self.generate_mask()
        # 从缓冲区采样
        # batch = random.sample(self.buffer, self.batch_size)
        # v_states, o_states, masks, actions, rewards, next_v, next_o, next_masks, dones = zip(*batch)

        # 转换为张量
        v_states = torch.FloatTensor(v_states).to(self.device)
        o_states = torch.FloatTensor(o_states).to(self.device)
        masks = torch.BoolTensor(masks).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor([rewards]).to(self.device)
        next_v = torch.FloatTensor(next_v).to(self.device)
        next_o = torch.FloatTensor(next_o).to(self.device)
        next_masks = torch.BoolTensor(next_masks).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        # 计算Critic损失
        current_global = self._get_global_state(v_states, o_states, masks)
        next_global = self._get_global_state(next_v, next_o, next_masks)
        current_v = self.critic(current_global)
        next_v_pred = self.critic(next_global)
        td_target = rewards + 0.95 * next_v_pred * (1 - dones) # 要调这里调
        critic_loss = F.mse_loss(current_v, td_target.detach())

        # 计算Actor损失（仅考虑有效订单）
        valid_actions = actions[masks]
        v_encoded = self.vehicle_encoder(v_states)
        o_encoded = self.order_encoder(o_states)
        
        # 生成策略输入
        global_vehicle = torch.mean(v_encoded, dim=1, keepdim=True)
        repeated_global = global_vehicle.expand(-1, o_encoded.size(1), -1)
        actor_input = torch.cat([repeated_global, o_encoded], dim=-1)
        
        # 计算有效位置的损失
        logits = self.actor(actor_input.view(-1, actor_input.size(-1)))
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, valid_actions.view(-1, 1)).squeeze()
        
        # 熵正则化
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        
        # 优势计算
        advantage = (td_target - current_v).detach()
        actor_loss = -(selected_log_probs * advantage).mean() - 0.01 * entropy

        # 合并损失
        total_loss = actor_loss + critic_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizer.step()

    def _get_global_state(self, v_states, o_states, masks):
        """获取Critic的全局状态表征"""
        # 维度修正
        if o_states.dim() == 2:
            o_states = o_states.unsqueeze(0)
        if masks.dim() == 1:
            masks = masks.unsqueeze(0)
        
        # 车辆全局特征
        v_encoded = self.vehicle_encoder(v_states)  # [batch, hidden]
        v_global = torch.mean(v_encoded, dim=1)     # [batch, hidden]
        
        # 订单全局特征（带掩码）
        o_encoded = self.order_encoder(o_states)    # [batch, max_orders, hidden]
        masks = masks.unsqueeze(-1).float()         # [batch, max_orders, 1]
        masked_o = o_encoded * masks
        
        # 安全维度操作
        o_sum = torch.sum(masked_o, dim=1)          # [batch, hidden]
        mask_sum = torch.sum(masks, dim=1).clamp(min=1.0)  # [batch, 1]
        o_global = o_sum / mask_sum
        
        return torch.cat([v_global, o_global], dim=1)  # [batch, 2*hidden]