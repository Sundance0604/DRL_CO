"""Permutation-equivariant candidate-scoring discrete SAC.

The legacy model emits one output neuron per city id.  This model instead runs
the same scorer over every candidate city and uses a DeepSets-style pooled
context, so relabelling cities relabels actions without changing their scores.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_families.legacy.engine.rl.discrete_sac import EPS, masked_softmax


class CandidateScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim == 2:
            candidates = candidates.unsqueeze(0)
        if candidates.ndim != 3:
            raise ValueError("candidate state must have shape [batch, cities, features]")
        encoded = self.encoder(candidates)
        mean_context = encoded.mean(dim=1, keepdim=True).expand_as(encoded)
        max_context = encoded.amax(dim=1, keepdim=True).expand_as(encoded)
        return self.head(torch.cat([encoded, mean_context, max_context], dim=-1)).squeeze(-1)


class CandidateReplay:
    def __init__(self, capacity: int = 500_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask, next_mask):
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

    def __len__(self):
        return len(self.buf)


class CandidateSAC(nn.Module):
    """Masked discrete SAC using shared candidate scorers.

    Orders are selected sequentially by urgency.  Each provisional decision
    reserves aggregate seat supply in the state seen by later orders, exposing
    within-step competition that the independent legacy policy could not see.
    """

    def __init__(
        self,
        device,
        feature_dim: int = 20,
        hidden_dim: int = 128,
        gamma: float = 0.95,
        tau: float = 0.01,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        initial_alpha: float = 0.1,
        target_entropy_ratio: float = 0.2,
        auto_alpha: bool = True,
        replay_capacity: int = 500_000,
        batch_size: int = 128,
        learning_starts: int | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.feature_dim = int(feature_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts or batch_size)
        self.actor = CandidateScorer(feature_dim, hidden_dim).to(self.device)
        self.q1 = CandidateScorer(feature_dim, hidden_dim).to(self.device)
        self.q2 = CandidateScorer(feature_dim, hidden_dim).to(self.device)
        self.q1_tgt = copy.deepcopy(self.q1).to(self.device).eval()
        self.q2_tgt = copy.deepcopy(self.q2).to(self.device).eval()
        for parameter in (*self.q1_tgt.parameters(), *self.q2_tgt.parameters()):
            parameter.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)
        self.auto_alpha = bool(auto_alpha)
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(initial_alpha)), device=self.device))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy_ratio = float(target_entropy_ratio)
        self.replay = CandidateReplay(replay_capacity)
        self.last_train_info: dict[str, float] = {}

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _states(self, value) -> torch.Tensor:
        states = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if states.numel() == 0:
            return torch.empty((0, 0, self.feature_dim), device=self.device)
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if states.ndim != 3 or states.shape[-1] != self.feature_dim:
            raise ValueError(f"expected [orders, cities, {self.feature_dim}] candidate state")
        return states

    @torch.no_grad()
    def take_action_candidates(self, candidate_states, mask, explore=True, greedy=False,
                               sequential=True):
        states = self._states(candidate_states)
        if states.shape[0] == 0:
            return [], None, None, None, states
        masks = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if masks.ndim == 1:
            masks = masks.unsqueeze(0)
        if masks.shape != states.shape[:2]:
            raise ValueError("mask must have shape [orders, cities]")

        decision_states = states.clone()
        actions = torch.zeros(states.shape[0], dtype=torch.long, device=self.device)
        if sequential:
            # Small slack first, then high normalized revenue.
            priority = states[:, 0, 1] - 0.1 * states[:, 0, 4]
            order_sequence = torch.argsort(priority)
            mutable = states.clone()
            remaining = set(range(states.shape[0]))
            for order_index_t in order_sequence:
                order_index = int(order_index_t.item())
                remaining.discard(order_index)
                decision_states[order_index] = mutable[order_index]
                probability = masked_softmax(
                    self.actor(mutable[order_index:order_index + 1]),
                    masks[order_index:order_index + 1],
                )[0]
                if greedy or not explore:
                    action = int(probability.argmax().item())
                else:
                    action = int(torch.multinomial(probability, 1).item())
                actions[order_index] = action
                # Column 6 is normalized aggregate available seat supply.
                demand = float(mutable[order_index, 0, 0].item())
                if remaining:
                    remaining_index = torch.tensor(sorted(remaining), device=self.device)
                    mutable[remaining_index, action, 6] = torch.clamp(
                        mutable[remaining_index, action, 6] - demand, min=0.0
                    )
        else:
            probability = masked_softmax(self.actor(states), masks)
            if greedy or not explore:
                actions = probability.argmax(dim=-1)
            else:
                actions = torch.multinomial(probability, 1).squeeze(1)

        logits = self.actor(decision_states)
        probabilities = masked_softmax(logits, masks)
        return (
            actions.tolist(),
            torch.log(probabilities.clamp_min(EPS)),
            probabilities,
            logits,
            decision_states.cpu(),
        )

    @torch.no_grad()
    def candidate_actor_values(self, candidate_states) -> torch.Tensor:
        """Return actor logits for downstream candidate screening.

        Unlike critic values, logits are only used for within-order ranking.  A
        downstream optimizer should not interpret their absolute scale as
        economic value.
        """
        states = self._states(candidate_states)
        if states.shape[0] == 0:
            return torch.empty(states.shape[:2])
        return self.actor(states).cpu()

    @torch.no_grad()
    def candidate_q_values(self, candidate_states) -> torch.Tensor:
        """Return conservative Q estimates for use by a downstream optimizer."""
        states = self._states(candidate_states)
        if states.shape[0] == 0:
            return torch.empty(states.shape[:2])
        return torch.minimum(self.q1(states), self.q2(states)).cpu()

    def behavior_clone(self, states, masks, actions, epochs: int = 10, batch_size: int = 256):
        states = torch.as_tensor(states, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.bool)
        actions = torch.as_tensor(actions, dtype=torch.long)
        if len(states) == 0:
            return {"bc_loss": 0.0, "bc_accuracy": 0.0, "bc_examples": 0}
        losses = []
        for _ in range(epochs):
            for indices in torch.randperm(len(states)).split(batch_size):
                batch_states = states[indices].to(self.device)
                batch_masks = masks[indices].to(self.device)
                batch_actions = actions[indices].to(self.device)
                logits = self.actor(batch_states).masked_fill(
                    ~batch_masks, torch.finfo(torch.float32).min
                )
                loss = F.cross_entropy(logits, batch_actions)
                self.actor_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_opt.step()
                losses.append(float(loss.item()))
        with torch.no_grad():
            logits = self.actor(states.to(self.device)).masked_fill(
                ~masks.to(self.device), torch.finfo(torch.float32).min
            )
            accuracy = (logits.argmax(-1).cpu() == actions).float().mean().item()
        return {
            "bc_loss": float(np.mean(losses)),
            "bc_accuracy": float(accuracy),
            "bc_examples": int(len(states)),
        }

    @torch.no_grad()
    def add_order_transitions(
        self,
        states,
        order_ids: Sequence[int],
        mask,
        actions: Sequence[int],
        rewards: Sequence[float] | Mapping[int, float] | float,
        next_states,
        next_order_ids: Sequence[int],
        next_mask,
        done_global: bool = False,
    ):
        current = self._states(states).cpu()
        following = self._states(next_states).cpu()
        current_mask = torch.as_tensor(mask, dtype=torch.bool).cpu()
        following_mask = torch.as_tensor(next_mask, dtype=torch.bool).cpu()
        if isinstance(rewards, Mapping):
            reward_values = [float(rewards[order_id]) for order_id in order_ids]
        elif isinstance(rewards, (int, float)):
            reward_values = [float(rewards)] * len(order_ids)
        else:
            reward_values = [float(value) for value in rewards]
        next_by_id = {order_id: index for index, order_id in enumerate(next_order_ids)}
        num_cities = current.shape[1]
        terminal_state = torch.zeros((num_cities, self.feature_dim))
        terminal_mask = torch.zeros(num_cities, dtype=torch.bool)
        terminal_mask[0] = True
        for index, (order_id, action, reward) in enumerate(zip(order_ids, actions, reward_values)):
            next_index = next_by_id.get(order_id)
            terminal = done_global or next_index is None
            self.replay.push(
                current[index], int(action), reward,
                terminal_state if terminal else following[next_index],
                float(terminal), current_mask[index],
                terminal_mask if terminal else following_mask[next_index],
            )

    def update_sac(self, iters: int = 1):
        if len(self.replay) < max(self.learning_starts, self.batch_size):
            return None
        info = {}
        for _ in range(iters):
            state, action, reward, next_state, done, mask, next_mask = self.replay.sample(
                self.batch_size, self.device
            )
            with torch.no_grad():
                next_probability = masked_softmax(self.actor(next_state), next_mask)
                next_log_probability = torch.log(next_probability.clamp_min(EPS))
                next_q = torch.minimum(self.q1_tgt(next_state), self.q2_tgt(next_state))
                next_value = (
                    next_probability * (next_q - self.alpha.detach() * next_log_probability)
                ).sum(-1, keepdim=True)
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

            probability = masked_softmax(self.actor(state), mask)
            log_probability = torch.log(probability.clamp_min(EPS))
            with torch.no_grad():
                q_min = torch.minimum(self.q1(state), self.q2(state))
            actor_loss = (
                probability * (self.alpha.detach() * log_probability - q_min)
            ).sum(-1).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            entropy = -(probability * log_probability).sum(-1).mean()
            alpha_loss = torch.zeros((), device=self.device)
            if self.auto_alpha:
                target_entropy = self.target_entropy_ratio * torch.log(
                    mask.sum(-1).clamp(min=1).float()
                ).mean()
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
                "actor_grad_norm": float(actor_grad),
                "q1_grad_norm": float(q1_grad),
                "q2_grad_norm": float(q2_grad),
                "replay_size": float(len(self.replay)),
            }
        self.last_train_info = info
        return info
