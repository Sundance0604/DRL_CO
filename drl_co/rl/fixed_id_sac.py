# Legacy implementation retained below for checkpoint compatibility.
import copy, random, torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

EPS = 1e-8

def masked_softmax(logits, mask=None, dim=-1, eps=EPS):
    if mask is None:
        return F.softmax(logits, dim=dim)
    # 数值稳定 & 将无效动作概率置零后再归一化
    logits = logits - logits.max(dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits) * mask
    denom = exp_logits.sum(dim=dim, keepdim=True)
    return exp_logits / (denom + eps)

class DiscretePolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, dropout_p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        h = F.relu(self.ln1(self.fc1(x)))
        h2 = F.relu(self.ln2(self.fc2(h))); h2 = self.dropout(h2); h = h + h2
        h3 = F.relu(self.ln3(self.fc3(h))); h3 = self.dropout(h3); h = h + h3
        return self.fc4(h)  # logits

class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, dropout_p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        h = F.relu(self.ln1(self.fc1(x)))
        h2 = F.relu(self.ln2(self.fc2(h))); h2 = self.dropout(h2); h = h + h2
        h3 = F.relu(self.ln3(self.fc3(h))); h3 = self.dropout(h3); h = h + h3
        return self.fc4(h)  # Q(s,·)

class SACReplay:
    def __init__(self, capacity=500000):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s_next, done_boot, mask, mask_next):
        self.buf.append((s, a, r, s_next, done_boot, mask, mask_next))
    def sample(self, batch_size, device):
        import torch
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, done, mask, mask_next = zip(*batch)
        s = torch.stack(s).to(device)
        a = torch.tensor(a, dtype=torch.long, device=device).view(-1,1)
        r = torch.tensor(r, dtype=torch.float, device=device).view(-1,1)
        s_next = torch.stack(s_next).to(device)
        done = torch.tensor(done, dtype=torch.float, device=device).view(-1,1)
        mask = torch.stack(mask).to(device)
        mask_next = torch.stack(mask_next).to(device)
        return s, a, r, s_next, done, mask, mask_next
    def __len__(self): return len(self.buf)

class MultiOrderSAC(nn.Module):
    """
    离散 SAC（支持掩码），按“每个订单一条样本”展平存入回放池。
    状态 = concat(vehicle_state, order_state_i)
    """
    def __init__(self, device,
                 VEHICLE_STATE_DIM, ORDER_STATE_DIM, ACTION_DIM,
                 HIDDEN_DIM=256, gamma=0.95, tau=0.01,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 target_entropy_ratio=0.9, auto_alpha=True,
                 replay_capacity=500000, batch_size=2048,
                 dropout_p=0.2):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.state_dim = VEHICLE_STATE_DIM + ORDER_STATE_DIM
        self.action_dim = ACTION_DIM

        self.actor = DiscretePolicyNet(self.state_dim, HIDDEN_DIM, ACTION_DIM, dropout_p).to(device)
        self.q1 = QNet(self.state_dim, HIDDEN_DIM, ACTION_DIM, dropout_p).to(device)
        self.q2 = QNet(self.state_dim, HIDDEN_DIM, ACTION_DIM, dropout_p).to(device)
        self.q1_tgt = copy.deepcopy(self.q1).to(device);  [p.requires_grad_(False) for p in self.q1_tgt.parameters()]
        self.q2_tgt = copy.deepcopy(self.q2).to(device);  [p.requires_grad_(False) for p in self.q2_tgt.parameters()]

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)

        self.auto_alpha = auto_alpha
        self.log_alpha = torch.tensor(0.0, device=device, requires_grad=True)  # α=1 初始
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy_ratio = float(target_entropy_ratio)

        self.replay = SACReplay(replay_capacity)
        self.last_train_info = {}

    @torch.no_grad()
    def take_action_vehicle(self, vehicle_states, order_states, mask, explore=True, greedy=False):
        v = torch.tensor(vehicle_states, dtype=torch.float, device=self.device)
        o = torch.tensor(order_states,   dtype=torch.float, device=self.device)
        m = torch.tensor(mask,           dtype=torch.float, device=self.device)  # [N,A]
        if o.ndim == 1:  # N=1 时防形状问题
            o = o.unsqueeze(0)
            m = m.unsqueeze(0)
        if o.size(0) == 0:
            return [], None, None, None

        repeated_v = v.unsqueeze(0).expand(o.size(0), -1)     # [N,V]
        s = torch.cat([repeated_v, o], dim=1)                 # [N,V+O]
        logits = self.actor(s)                                # [N,A]
        pi = masked_softmax(logits, m, dim=-1)                # [N,A]

        if greedy or not explore:
            actions = torch.argmax(pi, dim=-1).tolist()
        else:
            actions = []
            for i in range(pi.size(0)):
                if m[i].sum() <= 0:     # 无可行动作：退化为0
                    actions.append(0)
                else:
                    actions.append(torch.multinomial(pi[i], 1).item())
        return actions, torch.log(pi + EPS), pi, logits

    @torch.no_grad()
    def sac_add_step(self,
                     vehicle_states, order_states, mask,
                     actions, rewards,
                     next_vehicle_states, next_order_states, next_mask,
                     done_global=False):
        v  = torch.tensor(vehicle_states, dtype=torch.float, device=self.device)
        o  = torch.tensor(order_states,   dtype=torch.float, device=self.device)
        m  = torch.tensor(mask,           dtype=torch.float, device=self.device)
        if o.ndim == 1:  # N=1
            o = o.unsqueeze(0); m = m.unsqueeze(0)

        nv = torch.tensor(next_vehicle_states, dtype=torch.float, device=self.device)
        no = torch.tensor(next_order_states,   dtype=torch.float, device=self.device)
        nm = torch.tensor(next_mask,           dtype=torch.float, device=self.device)
        if no.ndim == 1:
            no = no.unsqueeze(0); nm = nm.unsqueeze(0)

        if not isinstance(rewards, (list, tuple)):
            rewards = [float(rewards)] * o.size(0)

        for i, a in enumerate(actions):
            s      = torch.cat([v,  o[i]], dim=-1)     # [S]
            s_next = torch.cat([nv, no[i]], dim=-1)    # [S]
            r      = float(rewards[i])
            # “自然终止”：下一时刻该订单无任何可选动作 -> 不引导
            term_i = 1.0 if nm[i].sum().item() <= 0 else 0.0
            done_boot = 1.0 if term_i > 0 else (1.0 if done_global else 0.0)
            self.replay.push(
                s.detach().cpu(), a, r, s_next.detach().cpu(),
                done_boot, m[i].detach().cpu(), nm[i].detach().cpu()
            )

    def update_sac(self, iters=1):
        if len(self.replay) < max(1000, self.batch_size):
            return None
        info = {}
        for _ in range(iters):
            s, a, r, s_next, done, mask, mask_next = self.replay.sample(self.batch_size, self.device)

            with torch.no_grad():
                logits_next = self.actor(s_next)                         # [B,A]
                pi_next = masked_softmax(logits_next, mask_next, dim=-1) # [B,A]
                log_pi_next = torch.log(pi_next + EPS)

                q1_next = self.q1_tgt(s_next)
                q2_next = self.q2_tgt(s_next)
                q_min_next = torch.min(q1_next, q2_next)

                alpha = self.log_alpha.exp()
                v_next = (pi_next * (q_min_next - alpha * log_pi_next)).sum(dim=-1, keepdim=True)  # [B,1]
                y = r + (1.0 - done) * self.gamma * v_next

            q1 = self.q1(s).gather(1, a)
            q2 = self.q2(s).gather(1, a)
            q1_loss = F.mse_loss(q1, y)
            q2_loss = F.mse_loss(q2, y)
            self.q1_opt.zero_grad(); q1_loss.backward(); nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0); self.q1_opt.step()
            self.q2_opt.zero_grad(); q2_loss.backward(); nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0); self.q2_opt.step()

            logits = self.actor(s)
            pi = masked_softmax(logits, mask, dim=-1)
            log_pi = torch.log(pi + EPS)
            q_min = torch.min(self.q1(s), self.q2(s))
            actor_loss = (pi * (self.log_alpha.exp() * log_pi - q_min)).sum(dim=-1).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.actor_opt.step()

            if self.auto_alpha:
                eff_actions = mask.sum(dim=-1).clamp(min=1)  # 每样本有效动作数量
                target_entropy = self.target_entropy_ratio * torch.log(eff_actions).mean().detach()
                entropy = -(pi * log_pi).sum(dim=-1).mean().detach()
                alpha_loss = (self.log_alpha.exp() * (-entropy - target_entropy)).mean()
                self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
                info['alpha'] = float(self.log_alpha.exp().item())
                info['alpha_loss'] = float(alpha_loss.item())

            with torch.no_grad():
                for p, p_tgt in zip(self.q1.parameters(), self.q1_tgt.parameters()):
                    p_tgt.mul_(1 - self.tau).add_(self.tau * p)
                for p, p_tgt in zip(self.q2.parameters(), self.q2_tgt.parameters()):
                    p_tgt.mul_(1 - self.tau).add_(self.tau * p)

            info.update({
                'q1_loss': float(q1_loss.item()),
                'q2_loss': float(q2_loss.item()),
                'actor_loss': float(actor_loss.item()),
                'replay_size': len(self.replay),
            })
        self.last_train_info = info
        return info


# The maintained implementation overrides the historical definitions above.
# Keeping this shim lets old notebooks continue importing ``sac_agent``.
from drl_co.rl.discrete_sac import (  # noqa: E402,F401
    DiscretePolicyNet,
    MultiOrderSAC,
    QNet,
    SACReplay,
    masked_softmax,
)
