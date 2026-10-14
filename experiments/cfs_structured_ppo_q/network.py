"""Same public encoders/critic; only strategic action head is enlarged."""
import torch
from torch import nn
from experiments.cfs_direct_ppo_v3.network import DirectNetwork
from .control import VERSION


class QNetwork(DirectNetwork):
    def __init__(self,hubs=5,hidden_size=64,with_critic=True):
        super().__init__(hubs,hidden_size,with_critic)
        old=self.actor;n=2*hubs
        self.actor=nn.Linear(hidden_size,2*n)
        with torch.no_grad():
            self.actor.weight.zero_();self.actor.bias.zero_()
            self.actor.weight[:n].copy_(old.weight);self.actor.bias[:n].copy_(old.bias)
        self.log_std=nn.Parameter(torch.cat((torch.full((n,),-.5),torch.full((n,),-1.5))))
        self.action_version=VERSION


def build(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed);return QNetwork()
