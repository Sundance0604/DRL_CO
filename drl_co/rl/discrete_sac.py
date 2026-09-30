"""Maintained masked discrete SAC implementation for the dispatch model."""

from __future__ import annotations

import copy
import random
from collections import deque
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 1e-8


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    """Create a stable distribution and fail fast on an invalid empty mask."""
    if mask is None:
        return F.softmax(logits, dim=dim)
    if logits.shape != mask.shape:
        raise ValueError(f"logits/mask shape mismatch: {tuple(logits.shape)} != {tuple(mask.shape)}")
    valid = mask.to(dtype=torch.bool)
    if (~valid.any(dim=dim)).any():
        raise ValueError("every policy row must contain at least one valid action")
    return F.softmax(logits.masked_fill(~valid, torch.finfo(logits.dtype).min), dim=dim)


class _ResidualMLP(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, output_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        hidden = F.relu(self.ln1(self.fc1(x)))
        hidden = hidden + self.dropout(F.relu(self.ln2(self.fc2(hidden))))
        hidden = hidden + self.dropout(F.relu(self.ln3(self.fc3(hidden))))
        return self.out(hidden)


class DiscretePolicyNet(_ResidualMLP):
    pass


class QNet(_ResidualMLP):
    pass


class SACReplay:
    def __init__(self, capacity: int = 500_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask, next_mask) -> None:
        self.buf.append((state, action, reward, next_state, done, mask, next_mask))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buf, batch_size)
        state, action, reward, next_state, done, mask, next_mask = zip(*batch)
        return (
            torch.stack(state).to(device),
            torch.tensor(action, dtype=torch.long, device=device).view(-1, 1),
            torch.tensor(reward, dtype=torch.float32, device=device).view(-1, 1),
            torch.stack(next_state).to(device),
            torch.tensor(done, dtype=torch.float32, device=device).view(-1, 1),
            torch.stack(mask).to(device),
            torch.stack(next_mask).to(device),
        )

    def __len__(self) -> int:
        return len(self.buf)


class MultiOrderSAC(nn.Module):
    """Discrete SAC with masks and one replay item per active order."""

    def __init__(
        self,
        device,
        VEHICLE_STATE_DIM: int,
        ORDER_STATE_DIM: int,
        ACTION_DIM: int,
        HIDDEN_DIM: int = 256,
        gamma: float = 0.95,
        tau: float = 0.01,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        initial_alpha: float = 0.1,
        target_entropy_ratio: float = 0.9,
        auto_alpha: bool = True,
        replay_capacity: int = 500_000,
        batch_size: int = 256,
        learning_starts: int | None = None,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts or batch_size)
        self.state_dim = int(VEHICLE_STATE_DIM + ORDER_STATE_DIM)
        self.action_dim = int(ACTION_DIM)

        self.actor = DiscretePolicyNet(self.state_dim, HIDDEN_DIM, ACTION_DIM, dropout_p).to(self.device)
        self.q1 = QNet(self.state_dim, HIDDEN_DIM, ACTION_DIM, dropout_p).to(self.device)
        self.q2 = QNet(self.state_dim, HIDDEN_DIM, ACTION_DIM, dropout_p).to(self.device)
        self.q1_tgt = copy.deepcopy(self.q1).to(self.device).eval()
        self.q2_tgt = copy.deepcopy(self.q2).to(self.device).eval()
        for parameter in (*self.q1_tgt.parameters(), *self.q2_tgt.parameters()):
            parameter.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)
        self.auto_alpha = bool(auto_alpha)
        if initial_alpha <= 0:
            raise ValueError("initial_alpha must be positive")
        self.log_alpha = nn.Parameter(
            torch.tensor(float(np.log(initial_alpha)), device=self.device)
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy_ratio = float(target_entropy_ratio)
        self.replay = SACReplay(replay_capacity)
        self.last_train_info: dict[str, float] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _state_batch(self, vehicle_states, order_states) -> torch.Tensor:
        vehicle = torch.as_tensor(vehicle_states, dtype=torch.float32, device=self.device).flatten()
        orders = torch.as_tensor(order_states, dtype=torch.float32, device=self.device)
        if orders.numel() == 0:
            return torch.empty((0, self.state_dim), dtype=torch.float32, device=self.device)
        if orders.ndim == 1:
            orders = orders.unsqueeze(0)
        state = torch.cat([vehicle.unsqueeze(0).expand(orders.size(0), -1), orders], dim=1)
        if state.size(1) != self.state_dim:
            raise ValueError(f"state has {state.size(1)} features; expected {self.state_dim}")
        return state

    @torch.no_grad()
    def take_action_vehicle(self, vehicle_states, order_states, mask, explore: bool = True, greedy: bool = False):
        state = self._state_batch(vehicle_states, order_states)
        if state.size(0) == 0:
            return [], None, None, None
        action_mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        logits = self.actor(state)
        probabilities = masked_softmax(logits, action_mask)
        if greedy or not explore:
            actions = probabilities.argmax(dim=-1).tolist()
        else:
            actions = torch.multinomial(probabilities, 1).squeeze(1).tolist()
        return actions, torch.log(probabilities.clamp_min(EPS)), probabilities, logits

    @torch.no_grad()
    def add_transition(self, state, action: int, reward: float, next_state, done: bool, mask, next_mask) -> None:
        state_t = torch.as_tensor(state, dtype=torch.float32).flatten().cpu()
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32).flatten().cpu()
        mask_t = torch.as_tensor(mask, dtype=torch.bool).flatten().cpu()
        next_mask_t = torch.as_tensor(next_mask, dtype=torch.bool).flatten().cpu()
        if state_t.numel() != self.state_dim or next_state_t.numel() != self.state_dim:
            raise ValueError("transition state dimension does not match the agent")
        if mask_t.numel() != self.action_dim or next_mask_t.numel() != self.action_dim:
            raise ValueError("transition mask dimension does not match the action space")
        if not mask_t.any() or not next_mask_t.any():
            raise ValueError("transition masks must include at least one valid action")
        if not mask_t[int(action)]:
            raise ValueError(f"action {action} is masked out")
        self.replay.push(state_t, int(action), float(reward), next_state_t, float(done), mask_t, next_mask_t)

    @torch.no_grad()
    def sac_add_order_transitions(
        self,
        vehicle_states,
        order_ids: Sequence[int],
        order_states,
        mask,
        actions: Sequence[int],
        rewards: Sequence[float] | Mapping[int, float] | float,
        next_vehicle_states,
        next_order_ids: Sequence[int],
        next_order_states,
        next_mask,
        done_global: bool = False,
    ) -> None:
        """Store dynamic order transitions, pairing current and next rows by id."""
        current = self._state_batch(vehicle_states, order_states).cpu()
        current_mask = torch.as_tensor(mask, dtype=torch.bool).cpu()
        if current_mask.ndim == 1:
            current_mask = current_mask.unsqueeze(0)
        following = self._state_batch(next_vehicle_states, next_order_states).cpu()
        following_mask = torch.as_tensor(next_mask, dtype=torch.bool).cpu()
        if following_mask.ndim == 1:
            following_mask = following_mask.unsqueeze(0)

        if not (len(order_ids) == len(actions) == len(current) == len(current_mask)):
            raise ValueError("order_ids, states, masks and actions must have equal lengths")
        if not (len(next_order_ids) == len(following) == len(following_mask)):
            raise ValueError("next order ids, states and masks must have equal lengths")
        if isinstance(rewards, Mapping):
            reward_values = [float(rewards[order_id]) for order_id in order_ids]
        elif isinstance(rewards, (int, float)):
            reward_values = [float(rewards)] * len(order_ids)
        else:
            reward_values = [float(value) for value in rewards]
        if len(reward_values) != len(order_ids):
            raise ValueError("one reward is required per current order")

        next_by_id = {order_id: index for index, order_id in enumerate(next_order_ids)}
        terminal_state = torch.zeros(self.state_dim)
        terminal_mask = torch.zeros(self.action_dim, dtype=torch.bool)
        terminal_mask[0] = True
        for index, (order_id, action, reward) in enumerate(zip(order_ids, actions, reward_values)):
            next_index = next_by_id.get(order_id)
            terminal = done_global or next_index is None
            state_next = terminal_state if terminal else following[next_index]
            mask_next = terminal_mask if terminal else following_mask[next_index]
            self.add_transition(
                current[index], action, reward, state_next, terminal,
                current_mask[index], mask_next,
            )

    @torch.no_grad()
    def sac_add_step(
        self, vehicle_states, order_states, mask, actions, rewards,
        next_vehicle_states, next_order_states, next_mask, done_global: bool = False,
    ) -> None:
        """Compatibility helper for fixed, position-aligned order sets."""
        self.sac_add_order_transitions(
            vehicle_states, list(range(len(actions))), order_states, mask, actions, rewards,
            next_vehicle_states, list(range(len(next_order_states))), next_order_states,
            next_mask, done_global,
        )

    def update_sac(self, iters: int = 1):
        if len(self.replay) < max(self.learning_starts, self.batch_size):
            return None
        info: dict[str, float] = {}
        for _ in range(iters):
            state, action, reward, next_state, done, mask, next_mask = self.replay.sample(
                self.batch_size, self.device
            )
            with torch.no_grad():
                next_prob = masked_softmax(self.actor(next_state), next_mask)
                next_log_prob = torch.log(next_prob.clamp_min(EPS))
                next_q = torch.minimum(self.q1_tgt(next_state), self.q2_tgt(next_state))
                next_value = (next_prob * (next_q - self.alpha.detach() * next_log_prob)).sum(-1, keepdim=True)
                target = reward + (1.0 - done) * self.gamma * next_value

            q1 = self.q1(state).gather(1, action)
            q2 = self.q2(state).gather(1, action)
            q1_loss = F.mse_loss(q1, target)
            q2_loss = F.mse_loss(q2, target)
            self.q1_opt.zero_grad(set_to_none=True)
            q1_loss.backward()
            q1_grad = nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
            self.q1_opt.step()
            self.q2_opt.zero_grad(set_to_none=True)
            q2_loss.backward()
            q2_grad = nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
            self.q2_opt.step()

            probabilities = masked_softmax(self.actor(state), mask)
            log_probabilities = torch.log(probabilities.clamp_min(EPS))
            with torch.no_grad():
                q_min = torch.minimum(self.q1(state), self.q2(state))
            actor_loss = (probabilities * (self.alpha.detach() * log_probabilities - q_min)).sum(-1).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            entropy = -(probabilities * log_probabilities).sum(-1).mean()
            alpha_loss = torch.zeros((), device=self.device)
            if self.auto_alpha:
                target_entropy = self.target_entropy_ratio * torch.log(mask.sum(-1).clamp(min=1).float()).mean()
                alpha_loss = self.log_alpha * (entropy - target_entropy).detach()
                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()
                with torch.no_grad():
                    self.log_alpha.clamp_(-20.0, 2.0)

            with torch.no_grad():
                for online, target_parameter in zip(self.q1.parameters(), self.q1_tgt.parameters()):
                    target_parameter.lerp_(online, self.tau)
                for online, target_parameter in zip(self.q2.parameters(), self.q2_tgt.parameters()):
                    target_parameter.lerp_(online, self.tau)

            info = {
                "q1_loss": float(q1_loss.item()),
                "q2_loss": float(q2_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "alpha_loss": float(alpha_loss.item()),
                "alpha": float(self.alpha.item()),
                "entropy": float(entropy.item()),
                "q_mean": float(torch.minimum(q1, q2).mean().item()),
                "target_mean": float(target.mean().item()),
                "actor_grad_norm": float(actor_grad),
                "q1_grad_norm": float(q1_grad),
                "q2_grad_norm": float(q2_grad),
                "replay_size": float(len(self.replay)),
            }
        self.last_train_info = info
        return info
