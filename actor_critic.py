import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

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


class ActorCritic:
    def __init__(self, vehicle_dim, order_dim, state_dim ,hidden_dim, action_dim, 
                 actor_lr, critic_lr, gamma, device, num_order, num_city):
        # 初始化车辆和订单编码器
        self.vehicle_encoder = VehicleEncoder(vehicle_dim, hidden_dim).to(device)
        self.order_encoder = OrderEncoder(order_dim, hidden_dim).to(device)

        # 策略和价值网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.encoder_optimizer = torch.optim.Adam(
            list(self.vehicle_encoder.parameters()) + list(self.order_encoder.parameters()),
            lr=actor_lr
        )

        self.gamma = gamma
        self.device = device
        self.num_order = num_order
        self.num_city = num_city

    def take_action(self, vehicle_states, order_states):
        # 将输入数据转换为 PyTorch Tensor
        vehicle_states = torch.tensor(vehicle_states, dtype=torch.float32)
        order_states = torch.tensor(order_states, dtype=torch.float32)

        # 编码车辆和订单状态
        vehicle_encoded = self.vehicle_encoder(vehicle_states)  # 编码车辆状态
        order_encoded = self.order_encoder(order_states)  # 编码订单状态

        # 确保订单编码后的形状是 [num_orders, order_state_dim]
        order_encoded = order_encoded.view(self.num_order, -1)

        vehicle_encoded = vehicle_encoded.view(-1)
        order_encoded = order_encoded.view(-1)

        # 合并车辆和订单状态
        # 车辆编码是 [num_vehicles, vehicle_state_dim]
        # 订单编码是 [num_orders, order_state_dim]
        combined_state = torch.cat([vehicle_encoded.flatten(), order_encoded.flatten()], dim=0)
        # 添加 batch 维度
        combined_state = combined_state.unsqueeze(0)  # 形状为 [1, combined_state_length]

        # 使用策略网络计算每个订单的动作概率
        probs = self.actor(combined_state)  # 获取每个订单的城市选择概率

        # 确保 actor 输出的是 [num_orders, num_cities]
        action_probs = probs.view(self.num_order, self.num_city)  # 转换为 [订单数，城市数] 的概率矩阵

        # 使用 softmax 确保每个订单的选择是概率分布
        action_probs = F.softmax(action_probs, dim=-1)  # 对最后一维进行softmax，确保每个订单的概率和为1

        # 对每个订单选择一个城市
        selected_action = torch.multinomial(action_probs, 1)  # 基于概率分布选择城市

        # 创建一个动作矩阵表示选择的城市
        action_matrix = torch.zeros_like(action_probs)  # 创建一个和action_probs同样大小的全零矩阵
        action_matrix.scatter_(1, selected_action, 1)  # 对每个订单选择的城市位置设置为 1

        return action_matrix  # 返回动作矩阵



    # 这是我的
    def my_update(self, vehicle_states, order_states, reward, next_vehicle_states, 
           next_order_states, done, action):
        
    
        vehicle_encoded = self.vehicle_encoder(torch.tensor(vehicle_states, dtype=torch.float).to(self.device))
        order_encoded = self.order_encoder(torch.tensor(order_states, dtype=torch.float).to(self.device))
        # 确保订单编码后的形状是 [num_orders, order_state_dim]
        order_encoded = order_encoded.view(self.num_order, -1)

        vehicle_encoded = vehicle_encoded.view(-1)
        order_encoded = order_encoded.view(-1)

        # 订单编码是 [num_orders, order_state_dim]
        combined_state = torch.cat([vehicle_encoded.flatten(), order_encoded.flatten()], dim=0)
        # 添加 batch 维度
        combined_state = combined_state.unsqueeze(0)  # 形状为 [1, combined_state_length]

        next_vehicle_encoded = self.vehicle_encoder(torch.tensor(next_vehicle_states, dtype=torch.float).to(self.device))
        next_order_encoded = self.order_encoder(torch.tensor(next_order_states, dtype=torch.float).to(self.device))
        next_order_encoded = next_order_encoded.view(self.num_order, -1)

        next_vehicle_encoded = next_vehicle_encoded.view(-1)
        next_order_encoded = next_order_encoded.view(-1)

        next_combined_state = torch.cat([next_vehicle_encoded.flatten(), next_order_encoded.flatten()],dim = 0)
        next_combined_state = next_combined_state.unsqueeze(0)

        # 计算价值
        value = self.critic(combined_state)
        next_value = self.critic(next_combined_state)

        # TD目标和优势
        td_target = reward + self.gamma * next_value * (1 - done)
        advantage = td_target - value

        # 更新策略网络
        action_probs = self.actor(combined_state)
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(torch.tensor(action).to(self.device))
        actor_loss = -log_prob * advantage.detach()

        # 更新价值网络
        critic_loss = F.mse_loss(value, td_target.detach())
        # print(critic_loss)
        # 反向传播
        self.actor_optimizer.zero_grad()
        actor_loss.sum().backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    # 这是含有gpt的调试
    def update(self, vehicle_states, order_states, reward, next_vehicle_states, 
           next_order_states, done, action):
        # 将输入数据转换为浮动类型并移到设备上
        vehicle_tensor = torch.tensor(vehicle_states, dtype=torch.float).to(self.device)
        order_tensor = torch.tensor(order_states, dtype=torch.float).to(self.device)
        
        # 编码车辆和订单
        vehicle_encoded = self.vehicle_encoder(vehicle_tensor)
        order_encoded = self.order_encoder(order_tensor)
        
        # 确保订单编码后的形状是 [num_orders, order_state_dim]
        order_encoded = order_encoded.view(self.num_order, -1)

        # 扁平化编码的车辆状态和订单状态
        vehicle_encoded = vehicle_encoded.view(-1)
        order_encoded = order_encoded.view(-1)

        # 合并车辆和订单编码后的状态
        combined_state = torch.cat([vehicle_encoded.flatten(), order_encoded.flatten()], dim=0)
        combined_state = combined_state.unsqueeze(0)  # 形状为 [1, combined_state_length]

        #print(f"combined_state shape: {combined_state.shape}")  # 检查合并状态的形状

        # 对下一个状态进行编码
        next_vehicle_tensor = torch.tensor(next_vehicle_states, dtype=torch.float).to(self.device)
        next_order_tensor = torch.tensor(next_order_states, dtype=torch.float).to(self.device)

        next_vehicle_encoded = self.vehicle_encoder(next_vehicle_tensor)
        next_order_encoded = self.order_encoder(next_order_tensor)

        next_order_encoded = next_order_encoded.view(self.num_order, -1)
        next_vehicle_encoded = next_vehicle_encoded.view(-1)
        next_order_encoded = next_order_encoded.view(-1)

        # 合并下一个状态
        next_combined_state = torch.cat([next_vehicle_encoded.flatten(), next_order_encoded.flatten()], dim=0)
        next_combined_state = next_combined_state.unsqueeze(0)

        #print(f"next_combined_state shape: {next_combined_state.shape}")  # 检查下一个合并状态的形状

        # 计算价值估计
        value = self.critic(combined_state)
        next_value = self.critic(next_combined_state)

        #print(f"value: {value}")  # 打印当前价值估计
        #print(f"next_value: {next_value}")  # 打印下一个价值估计

        # 计算 TD目标和优势
        td_target = reward + self.gamma * next_value * (1 - done)
        advantage = td_target - value

        #print(f"td_target: {td_target}")  # 打印 TD目标
        #print(f"advantage: {advantage}")  # 打印优势

        # 更新策略网络
        action_probs = self.actor(combined_state)
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(torch.tensor(action).to(self.device))
        actor_loss = -log_prob * advantage.detach()

        #print(f"actor_loss: {actor_loss}")  # 打印策略网络的损失

        # 更新价值网络
        critic_loss = F.mse_loss(value, td_target.detach())

        #print(f"critic_loss: {critic_loss}")  # 打印价值网络的损失

        
        
        # 反向传播
        self.actor_optimizer.zero_grad()
        actor_loss.sum().backward(retain_graph=True)  # 确保 actor_loss 是标量
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
         # 在每次更新后，检查参数的梯度
        """
        for name, param in self.critic.named_parameters():
            if param.grad is not None:
                print(f"{name} grad max: {param.grad.max()}, grad min: {param.grad.min()}")

        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                print(f"{name} grad max: {param.grad.max()}, grad min: {param.grad.min()}")
        """
        
        
