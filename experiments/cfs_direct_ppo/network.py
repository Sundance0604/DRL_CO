"""Same set encoders and PPO math, twenty fewer shared-trunk input features."""
import torch
from torch import nn
from model_families.single.learning.ppo.networks import PolicyNetwork, SetEncoder, sample_action
from .observation import dimensions
from .contracts import OBSERVATION_VERSION


class DirectNetwork(PolicyNetwork):
    def __init__(self,hub_count,hidden_size=64,with_critic=True):
        nn.Module.__init__(self)
        self.hub_count=hub_count
        d,h=dimensions(hub_count),hidden_size
        self.order_encoder=SetEncoder(d['order_dim'],h);self.arc_encoder=SetEncoder(d['arc_dim'],h)
        self.vehicle_encoder=SetEncoder(d['vehicle_dim']+2*(h+1),h)
        self.body=nn.Sequential(nn.Linear(d['global_dim']+hub_count*d['hub_dim']+2*(h+1),h),
            nn.Tanh(),nn.Linear(h,h),nn.Tanh())
        self.actor=nn.Linear(h,2*hub_count);nn.init.zeros_(self.actor.weight);nn.init.zeros_(self.actor.bias)
        self.log_std=nn.Parameter(torch.full((2*hub_count,),-.5))
        self.critic=nn.Linear(h,1) if with_critic else None

    def encode(self,obs):
        if obs.version!=OBSERVATION_VERSION:raise ValueError('Direct network rejects residual observations')
        return super().encode(obs)


class DirectPolicy:
    def __init__(self,checkpoint,expected_source=None):
        from pathlib import Path
        from model_families.single.strategic.artifacts import read_json
        from experiments.cfs_ppo_strategic_control.study_common import sha
        cp=Path(checkpoint);m=read_json(cp/'manifest.json')
        if m['kind']!='direct-no-mc-inference/v1' or (expected_source is not None and m['source']!=expected_source):
            raise ValueError('incompatible Direct checkpoint')
        if sha(cp/'actor.pt')!=m['actor_sha256']:raise ValueError('Direct actor hash mismatch')
        self.manifest=m
        with torch.random.fork_rng():self.network=DirectNetwork(len(m['action']['hubs']),m['hidden_size'],False)
        self.network.load_state_dict(torch.load(cp/'actor.pt',map_location='cpu',weights_only=True),strict=True)
        self.network.eval();self.network.requires_grad_(False)
        self.generator=torch.Generator().manual_seed(0)

    def act(self,obs):
        if tuple(self.manifest['action']['hubs'])!=obs.hubs:raise ValueError('Direct policy hub mismatch')
        return tuple(sample_action(self.network,obs,self.generator,True)[0].tolist())
