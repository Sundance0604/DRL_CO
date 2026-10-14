import copy
import sys
import torch
import pytest
from torch import nn
from model_families.single.learning.ppo.config import PPOConfig
from model_families.single.learning.ppo.networks import squashed_log_probability
from experiments.cfs_direct_ppo_training_ablation_v6.backend import update
from experiments.cfs_structured_ppo_q_mechanism.kl import gaussian_parts,Capture,fingerprint
from experiments.cfs_structured_ppo_q_mechanism.states import FeasibilityPlanner
from experiments.cfs_structured_ppo_q.control import episode,QProvider,QSignal
from model_families.single.rolling.contracts import WindowRequest
from model_families.single.rolling.controller import observed_state
from model_families.single.runner import state_snapshot
from test_direct_ppo_v3 import direct_cfg
from test_single_strategic_ppo import scenario


def test_exact_KL_identity_sum_and_mean_std_split():
    a=torch.zeros((2,20));s=torch.ones_like(a);b=a.clone();b[:,10:]=.1;t=s.clone();t[:,:10]=1.1
    mean,scale,total=gaussian_parts(a,s,b,t)
    assert torch.allclose(total,mean+scale)
    assert mean[:,:10].sum()==0 and scale[:,10:].sum()==0
    assert total.sum(1).tolist()==pytest.approx((total[:,:10].sum(1)+total[:,10:].sum(1)).tolist())


def test_smaller_sigma_increases_mean_sensitivity_without_changing_policy():
    a=torch.zeros((1,10));delta=torch.ones_like(a)*.1
    big=torch.ones_like(a)*torch.exp(torch.tensor(-.5));small=torch.ones_like(a)*torch.exp(torch.tensor(-1.5))
    m1,_,_=gaussian_parts(a,big,delta,big);m2,_,_=gaussian_parts(a,small,delta,small)
    assert (m2.sum()/m1.sum()).item()==pytest.approx(float(torch.exp(torch.tensor(2.))),rel=1e-6)


class ToyNetwork(nn.Module):
    def __init__(self):
        super().__init__();self.body=nn.Linear(1,4);self.actor=nn.Linear(4,20);self.critic=nn.Linear(4,1)
        self.log_std=nn.Parameter(torch.tensor([-.5]*10+[-1.5]*10))
    def forward(self,obs):
        x=self.body(torch.tensor(obs).reshape(-1,1));mean=self.actor(x);std=self.log_std.exp().expand_as(mean)
        return mean,std,self.critic(x).squeeze(-1)


def test_actual_check_capture_does_not_change_optimizer_rng_or_results(tmp_path):
    torch.manual_seed(100);net=ToyNetwork();plain=copy.deepcopy(net);rows=[]
    gen=torch.Generator().manual_seed(201)
    for i in range(8):
        o=float(i)/8
        with torch.no_grad():mu,std,_=net([o]);z=(mu+std*torch.randn(mu.shape,generator=gen))[0]
        rows.append(dict(observation=o,latent=z,logp=squashed_log_probability(mu,std,z[None]).item(),advantage=(-1.)**i,**{'return':10.}))
    cfg=PPOConfig(epochs=4,minibatch_size=4,target_kl=.0001,learning_rate=.02)
    a=(torch.optim.Adam(net.parameters(),lr=.02),);b=(torch.optim.Adam(plain.parameters(),lr=.02),)
    g1=torch.Generator().manual_seed(300);g2=torch.Generator().manual_seed(300);global_rng=torch.get_rng_state().clone()
    with Capture(net,rows,tmp_path,1) as capture:r1=update(net,a,rows,cfg,g1,'B')
    r2=update(plain,b,rows,cfg,g2,'B')
    assert r1==r2 and fingerprint(net)==fingerprint(plain)
    assert torch.equal(g1.get_state(),g2.get_state()) and torch.equal(global_rng,torch.get_rng_state())
    assert sys.gettrace() is None
    assert len(capture.checks)==r1['minibatches']+int(r1['kl_early_stop'])
    assert r1['kl_early_stop'] and capture.checks[-1]['stopping_trigger']
    assert capture.checks[-1]['approximate_KL']==r1['trigger']['KL']
    snap=torch.load(tmp_path/'trigger-u01.pt',weights_only=False)
    assert snap['parameter_sha256']==fingerprint(net)


class Count:
    def __init__(self):self.calls=0
    def __call__(self):return False
    def claim(self):self.calls+=1


@pytest.mark.parametrize('fleet,active',[(1,False),(3,True)])
def test_qactive_certificate_uses_identical_integer_constraints(fleet,active):
    s=scenario();s['hubs']=['A']*fleet;c=direct_cfg();ep=episode(s,c);o=ep.reset();before=state_snapshot(ep._sim)
    p=QProvider(ep.schema,QSignal((0.,)*12,0,o.revision,o.hash),3,fleet);budget=Count()
    planner=FeasibilityPlanner(p,dict(period=0,T=3),budget)
    planner.plan(WindowRequest(observed_state(ep._sim),0,1,1,0,ep._check))
    cert=planner.certificate
    assert cert['q_active']==active
    assert cert['constraint_sha256']==cert['copy_constraint_sha256']
    assert cert['status']=='OPTIMAL' and budget.calls==1 and state_snapshot(ep._sim)==before
    assert any(r['q']==2 and r['occupied'] for r in cert['slot_occupancy'])==active
