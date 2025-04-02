import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
from collections import namedtuple,deque
from torch import optim
import torch.nn.utils.rnn as rnn_utils
import os
from torch.nn.utils.rnn import pad_sequence
import copy
import rl_utils

class ComplexPolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, dropout_p=0.2):
        super(ComplexPolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        
        self.fc5 = nn.Linear(hidden_dim, action_dim)
        
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        # 保证输入为二维张量： (batch_size, state_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        out1 = F.relu(self.ln1(self.fc1(x)))
        out2 = F.relu(self.ln2(self.fc2(out1)))
        out2 = self.dropout(out2)
        out2 = out2 + out1  # 残差连接
        
        out3 = F.relu(self.ln3(self.fc3(out2)))
        out3 = self.dropout(out3)
        out3 = out3 + out2
        
        out4 = F.relu(self.ln4(self.fc4(out3)))
        out4 = self.dropout(out4)
        out4 = out4 + out3
        
        logits = self.fc5(out4)
        probs = F.softmax(logits, dim=1)
        return probs

class ComplexValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, dropout_p=0.2):
        super(ComplexValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        out1 = F.relu(self.ln1(self.fc1(x)))
        out2 = F.relu(self.ln2(self.fc2(out1)))
        out2 = self.dropout(out2)
        out2 = out2 + out1
        
        out3 = F.relu(self.ln3(self.fc3(out2)))
        out3 = self.dropout(out3)
        out3 = out3 + out2
        
        value = self.fc4(out3)
        return value

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
    def __init__(self):
        self.v_states = []
        self.o_states = []
        self.rewards = []
        self.probs = []
        self.log_probs = []
        self.selected_log_probs = []
        self.best_invalid = 1000
        self.masks =[]
    def push(self, v_states, o_states, rewards, probs, log_probs, selected_log_probs,mask):
        self.v_states.append(v_states)
        self.o_states.append(o_states)
        self.rewards.append(rewards)
        self.probs.append(probs)
        self.log_probs.append(log_probs)
        self.selected_log_probs.append(selected_log_probs)
        self.masks.append(mask)
    def length(self):
        return len(self.rewards)
    def clear(self):
        """清空所有存储的数据"""
        self.v_states = []
        self.o_states = []
        self.rewards = []
        self.probs = []
        self.log_probs = []
        self.selected_log_probs = []
        self.masks = []
    def restore(self, best_invalid):
        self.best_invalid = best_invalid
        self.betst_v_states = copy.deepcopy(self.v_states)
        self.best_o_states = copy.deepcopy(self.o_states)
        self.best_rewards = copy.deepcopy(self.rewards)
        self.best_probs = copy.deepcopy([p.detach() for p in self.probs])
        self.best_log_probs = copy.deepcopy([lp.detach() for lp in self.log_probs])
        self.best_selected_log_probs = copy.deepcopy([slp.detach() for slp in self.selected_log_probs])
        self.best_masks = copy.deepcopy(self.masks)
    def use_best(self):
        self.v_states = copy.deepcopy(self.betst_v_states)
        self.o_states = copy.deepcopy(self.best_o_states)  
        self.rewards = copy.deepcopy(self.best_rewards)
        self.probs = copy.deepcopy([p.detach() for p in self.best_probs])
        self.log_probs = copy.deepcopy([lp.detach() for lp in self.best_log_probs])
        self.selected_log_probs = copy.deepcopy([slp.detach() for slp in self.best_selected_log_probs])
        self.masks = copy.deepcopy(self.best_masks)

class MultiAgentAC(torch.nn.Module):
    def __init__(self, device, VEHICLE_STATE_DIM, 
                 ORDER_STATE_DIM, NUM_CITIES, 
                 HIDDEN_DIM, STATE_DIM, batch_size):
        super(MultiAgentAC, self).__init__()
        self.buffer = ReplayBuffer()
       
        self.device = device
        self.NUM_CITIES = NUM_CITIES
        
        # 共享网络
        self.actor = PolicyNet(STATE_DIM, HIDDEN_DIM, NUM_CITIES).to(device)
        self.critic = ValueNet(STATE_DIM, HIDDEN_DIM).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        # 动态智能体管理 ⭐
        self.active_orders = {}       # 当前活跃订单 {order_id: order_state}
        self.next_order_id = 0        # 订单ID生成器
        self.batch_size = batch_size
        self.active = False
        self.current_order = []
        self.last_order = []
        self.reward = 0
        self.action_key = ''
        self.action = []
        self.v_states = np.array([])
        self.gamma = 0.95

      
    # 改变vehicle_states,不再是平均值，而是其他办法
    
    def update(self, time, saq_len = 2):
        
        if self.buffer.length() < self.batch_size:
            return
        start_postion = time - self.batch_size+1

        v_states = torch.tensor(self.buffer.v_states[start_postion:start_postion+saq_len], dtype=torch.float).to(self.device)
        # 注意到只能分批转化为张量
        rewards = torch.tensor(self.buffer.rewards[start_postion:start_postion+saq_len], dtype=torch.float).to(self.device)
        probs = self.buffer.probs[start_postion].clone().detach()
        selected_log_probs = self.buffer.selected_log_probs[start_postion].clone().detach()
        log_probs = self.buffer.log_probs[start_postion].clone().detach()
        # 计算 Critic 损失
        current_o_states = torch.from_numpy(self.buffer.o_states[start_postion]).float().to(self.device)
        final_o_states = torch.from_numpy(self.buffer.o_states[start_postion+saq_len-1]).float().to(self.device)
        current_global = self._get_global_state(v_states[0], current_o_states)
        
        current_v = self.critic(current_global)
        cumulative_reward = 0
        
        # 归一化
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        normalized_rewards = (rewards - mean_reward) / std_reward

        # 累积计算
        cumulative_reward = 0
        for normalized_reward in normalized_rewards:
            cumulative_reward = normalized_reward + self.gamma * cumulative_reward
        td_target = cumulative_reward + (self.gamma ** saq_len) * self.critic(self._get_global_state(v_states[-1], final_o_states))
        critic_loss = F.mse_loss(current_v, td_target.detach())

        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        # 不再是num_orders这一固定的
        advantage = (td_target - current_v).detach()
        actor_loss = -(selected_log_probs * advantage).mean() - 0.01 * entropy
        # print("actor_loss:", actor_loss.item(), "critic_loss:", critic_loss.item(), "advantage:", advantage.item(), "current_v:", current_v.item(), "td_target:", td_target.item())

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        actor_loss.requires_grad = True
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
     
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
    
    def store_experience(self, v_states, o_states, rewards, probs, log_probs, selected_log_probs,mask):
        self.buffer.push(v_states, o_states, rewards, probs, log_probs, selected_log_probs,mask)
    def _get_global_state(self, v_states, o_states):
        """获取Critic的全局状态表征（无掩码）"""
        
        v_tensor = torch.FloatTensor(v_states).to(self.device)
        v_encoded = v_tensor
        
        # 订单全局特征
        o_tensor = torch.FloatTensor(o_states).to(self.device)
        o_encoded = o_tensor
        global_order = torch.mean(o_encoded, dim=0)
        
        return torch.cat([v_encoded, global_order])
    
    def update_n_sample(self, time, saq_len=4, n_smaple = 4):
        # 保证缓冲区中有足够的数据连续采样
        if self.buffer.length() < self.batch_size:
            return
        actor_losses = []
        critic_losses = []
        for t in range(time, time+n_smaple):
            start_postion = t - self.batch_size+1
            if start_postion+saq_len > len(self.buffer.v_states):
                break
            v_states = torch.tensor(self.buffer.v_states[start_postion:start_postion+saq_len], dtype=torch.float).to(self.device)
            # 注意到只能分批转化为张量
            rewards = torch.tensor(self.buffer.rewards[start_postion:start_postion+saq_len], dtype=torch.float).to(self.device)
            probs = self.buffer.probs[start_postion].clone().detach()
            selected_log_probs = self.buffer.selected_log_probs[start_postion].clone().detach()
            log_probs = self.buffer.log_probs[start_postion].clone().detach()
            # 计算 Critic 损失
            current_o_states = torch.from_numpy(self.buffer.o_states[start_postion]).float().to(self.device)
            final_o_states = torch.from_numpy(self.buffer.o_states[start_postion+saq_len-1]).float().to(self.device)
            current_global = self._get_global_state(v_states[0], current_o_states)
            
            current_v = self.critic(current_global)
            cumulative_reward = 0
            
            # 归一化
            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-8
            normalized_rewards = (rewards - mean_reward) / std_reward

            # 累积计算
            cumulative_reward = 0
            for normalized_reward in normalized_rewards:
                cumulative_reward = normalized_reward + self.gamma * cumulative_reward
            td_target = cumulative_reward + (self.gamma ** saq_len) * self.critic(self._get_global_state(v_states[-1], final_o_states))
            critic_loss = F.mse_loss(current_v, td_target.detach())

            entropy = -torch.sum(probs * log_probs, dim=-1).mean()
            # 不再是num_orders这一固定的
            advantage = (td_target - current_v).detach()
            actor_loss = -(selected_log_probs * advantage).mean() - 0.01 * entropy
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        average_actor_loss = torch.tensor(sum(actor_losses)/len(actor_losses))
        average_critic_loss =torch.tensor(sum(critic_losses)/len(actor_losses))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
     
        average_actor_loss.backward()  # 计算策略网络的梯度
        average_critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
        # print(self.actor.state_dict())
        # print(self.critic.state_dict())
    
    def update_ppo(self, time, ppo_eposide=30):
        v_states = torch.tensor(self.buffer.v_states[time:time+1], dtype=torch.float).to(self.device)
        reward = torch.tensor(self.buffer.rewards[time], dtype=torch.float).to(self.device)
        old_log_probs = self.buffer.log_probs[time].clone().detach()
        current_o_states = torch.from_numpy(self.buffer.o_states[time]).float().to(self.device)
        final_o_states = torch.from_numpy(self.buffer.o_states[time]).float().to(self.device)
        mask = torch.tensor(self.buffer.masks[time] ,dtype=torch.float).to(self.device)

        current_global = self._get_global_state(v_states[0], current_o_states)
        current_v = self.critic(current_global)
        td_target = reward + self.gamma * self.critic(self._get_global_state(v_states[-1], final_o_states))
        td_delta = td_target - current_v
        advantage = rl_utils.compute_advantage(self.gamma, 0.95, td_delta.cpu()).to(self.device)

        for _ in range(ppo_eposide):
            # 每次迭代重新前向计算所有必要变量
            with torch.no_grad():
                # 重新计算旧概率以避免复用旧计算图
                repeated_global = v_states[0].unsqueeze(0).expand(current_o_states.size(0), -1)
                actor_input = torch.cat([repeated_global, current_o_states], dim=1)
                old_log_probs = old_log_probs.detach()
            
            # 重新计算当前策略的log_probs
            current_logits = self.actor(actor_input)  # 使用更新后的参数计算
            if mask is not None:
                current_logits = current_logits.masked_fill(mask == 0, float('-inf'))
            current_log_probs = F.log_softmax(current_logits, dim=-1)
            
            # 计算PPO损失
            ratio = torch.exp(current_log_probs - old_log_probs)
            surr1 = ratio * advantage.detach()  # 分离advantage的梯度
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantage.detach()
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 重新计算Critic的损失
            current_v = self.critic(current_global)
            critic_loss = F.mse_loss(current_v, td_target.detach())  # 分离td_target
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)  # 保留计算图供Critic使用
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            # 更新Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()  # 不再需要retain_graph
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

    
    