"""Hierarchical masked set encoding and tanh Gaussian actor/return critic."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ...strategic.observation import dimensions

NETWORK_VERSION = "single-hierarchical-set-ppo/v1"


class SetEncoder(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.width, self.hidden = width, hidden
        self.layers = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, rows):
        # No padding is needed in the serial prototype: empty sets are masked
        # explicitly and all public rows are retained. Count complements the mean.
        if not rows:
            return torch.zeros(self.hidden+1, device=next(self.parameters()).device)
        x = torch.tensor(rows, dtype=torch.float32, device=next(self.parameters()).device)
        return torch.cat((self.layers(x).mean(0), x.new_tensor([math.log1p(len(rows))])))


class PolicyNetwork(nn.Module):
    def __init__(self, hub_count, hidden_size, with_critic=True):
        super().__init__()
        self.hub_count = hub_count
        d, h = dimensions(hub_count), hidden_size
        self.order_encoder = SetEncoder(d["order_dim"], h)
        self.arc_encoder = SetEncoder(d["arc_dim"], h)
        self.vehicle_encoder = SetEncoder(d["vehicle_dim"]+2*(h+1), h)
        self.body = nn.Sequential(nn.Linear(d["global_dim"]+hub_count*d["hub_dim"]+2*(h+1), h),
                                  nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.actor = nn.Linear(h, 2*hub_count)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        self.log_std = nn.Parameter(torch.full((2*hub_count,), -.5))
        self.critic = nn.Linear(h, 1) if with_critic else None

    def encode(self, observation):
        if len(observation.hubs) != self.hub_count:
            raise ValueError("network hub dimension mismatch")
        device = self.log_std.device
        vehicle_vectors = []
        for v in observation.vehicles:
            row = torch.tensor(v.features, dtype=torch.float32, device=device)
            vehicle_vectors.append(torch.cat((row, self.arc_encoder(v.arcs),
                                               self.order_encoder(v.commitments))))
        if vehicle_vectors:
            x = torch.stack(vehicle_vectors)
            vehicle_set = torch.cat((self.vehicle_encoder.layers(x).mean(0),
                                     x.new_tensor([math.log1p(len(vehicle_vectors))])))
        else:
            vehicle_set = torch.zeros(self.vehicle_encoder.hidden+1, device=device)
        public = torch.tensor((*observation.global_features,
                               *(v for row in observation.hub_features for v in row)),
                              dtype=torch.float32, device=device)
        return self.body(torch.cat((public, self.order_encoder(observation.orders), vehicle_set)))

    def forward(self, observations):
        encoded = torch.stack([self.encode(o) for o in observations])
        mean = self.actor(encoded)
        std = self.log_std.clamp(-5, 2).exp().expand_as(mean)
        value = self.critic(encoded).squeeze(-1) if self.critic is not None else None
        return mean, std, value


def squashed_log_probability(mean, std, latent):
    normal = -.5*((latent-mean)/std).square()-std.log()-.5*math.log(2*math.pi)
    # log(1-tanh(z)^2), stable even when float32 tanh saturates to exactly +/-1.
    jacobian = 2*(math.log(2)-latent-F.softplus(-2*latent))
    return (normal-jacobian).sum(-1)


def sample_action(network, observation, generator, deterministic=False):
    with torch.no_grad():
        mean, std, value = network([observation])
        latent = mean if deterministic else mean+std*torch.randn(mean.shape, generator=generator)
        action = torch.tanh(latent)
        logp = squashed_log_probability(mean, std, latent)
        tensors = [mean, std, latent, action, logp] + ([value] if value is not None else [])
        if not all(torch.isfinite(x).all() for x in tensors):
            raise FloatingPointError("nonfinite policy output/log probability/value")
    return action[0], latent[0], logp[0], value[0] if value is not None else None
