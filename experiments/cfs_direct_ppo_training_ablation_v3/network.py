"""Actor initialization exactly matches A; independent D value encoder."""
import torch
from torch import nn
from experiments.cfs_direct_ppo_v3.network import DirectNetwork, DirectPolicy


class ValueNetwork(DirectNetwork):
    def __init__(self, hubs, hidden):
        super().__init__(hubs, hidden)
        del self.actor
        del self.log_std
        # Original encoding uses log_std only to find its device.
        self.register_buffer('log_std', torch.zeros(1))

    def forward(self, observations):
        return self.critic(torch.stack([self.encode(o) for o in observations])).squeeze(-1)


class DecoupledNetwork(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor_model, self.value_model = actor, critic

    def forward(self, observations):
        mean, std, _ = self.actor_model(observations)
        return mean, std, self.value_model(observations)


def build(variant, seed, hubs=5, hidden=64):
    if variant not in ('A', 'B', 'C', 'D'):
        raise ValueError('unknown ablation')
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        actor = DirectNetwork(hubs, hidden, True)
    if variant in ('A', 'B'):
        return actor
    actor.critic = None
    if variant == 'C':
        return actor
    with torch.random.fork_rng():
        torch.manual_seed(seed + 2001)
        critic = ValueNetwork(hubs, hidden)
    return DecoupledNetwork(actor, critic)


def actor_network(net):
    return net.actor_model if isinstance(net, DecoupledNetwork) else net


def actor_weights(net):
    return {k: v.detach().clone() for k, v in actor_network(net).state_dict().items()
            if not k.startswith('critic.')}


def optimizers(net, learning_rate):
    if isinstance(net, DecoupledNetwork):
        return (torch.optim.Adam(net.actor_model.parameters(), lr=learning_rate),
                torch.optim.Adam(net.value_model.parameters(), lr=learning_rate))
    return (torch.optim.Adam(net.parameters(), lr=learning_rate),)
