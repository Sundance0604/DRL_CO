import copy
import math
import pytest
import torch
import numpy as np
from test_direct_ppo_v3 import direct_cfg
from test_single_strategic_ppo import scenario
from experiments.cfs_structured_ppo_q.control import QProvider,QSignal,episode,replay,VERSION,basis
from experiments.cfs_direct_ppo_v3.control import DirectProvider,episode as direct_episode
from experiments.cfs_direct_ppo_v3.contracts import DirectSchema,DirectSignal
from experiments.cfs_direct_ppo_v3.network import DirectNetwork
from experiments.cfs_structured_ppo_q.network import QNetwork
from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
from model_families.single.learning.ppo.networks import squashed_log_probability,sample_action
from experiments.cfs_structured_ppo_q.analysis import crossed


@pytest.mark.parametrize('fleet',[1,3,8])
def test_exact_nested_query_and_terminal(fleet):
    schema=DirectSchema(('A','B'),(10.,20.));a=(.2,-.3,.4,-.5)
    old=DirectProvider(schema,DirectSignal(a,2,0,'hash'),24)
    new=QProvider(schema,QSignal(a+(0.,)*4,2,0,'hash'),24,fleet)
    for t in (2,6,23,24,25):
        for h in schema.hubs:
            for q in range(1,fleet+1):assert new.marginal_release_value(t,h,q,{})==old.marginal_release_value(t,h,q,{})


def test_q_slope_boundaries_and_no_forced_monotonicity():
    p=QProvider(DirectSchema(('A',),(10.,)),QSignal((.2,.2,.6,.6),0,0,'x'),24,3)
    assert [p.marginal_release_value(3,'A',q,{}) for q in (1,2,3)]==pytest.approx([2,5,8])
    for q in (0,4,1.2):
        with pytest.raises(ValueError):p.marginal_release_value(3,'A',q,{})
    with pytest.raises(ValueError):p.marginal_release_value(-1,'A',1,{})
    with pytest.raises(ValueError):QSignal((float('nan'),)*4,0,0,'x')
    with pytest.raises(ValueError):QProvider(p.schema,QSignal((0.,)*2,0,0,'x'),24,3)


def test_day_nesting_cash_reward_and_independent_tamper_check():
    cfg=direct_cfg();s=scenario();a=direct_episode(s,cfg);b=episode(s,cfg)
    a.reset();b.reset()
    with no_mc_guard():
        while True:
            ta=a.step((.2,)*6);tb=b.step((.2,)*6+(0.,)*6)
            assert ta.terminated==tb.terminated
            assert a.records[-1]['action_hash']==b.records[-1]['action_hash']
            assert a.records[-1]['plan']['solver']['window_objective']==b.records[-1]['plan']['solver']['window_objective']
            if ta.terminated:break
        r=dict(transitions=b.records,ledger=b.final_ledger)
        assert replay(s,cfg,r)['status']=='PASSED'
    assert a.final_ledger==b.final_ledger
    assert sum(t['reward_USD'] for t in b.records)==pytest.approx(b.final_ledger['realized_profit'])
    tampered=copy.deepcopy(r);tampered['transitions'][0]['coefficients'][0]['value']+=1
    with pytest.raises(RuntimeError,match='q price'):replay(s,cfg,tampered)


def test_public_hidden_future_invariance_and_stale_control():
    cfg=direct_cfg();a=scenario();b=copy.deepcopy(a);b['orders'][0]['revenue']+=1e6
    x,y=episode(a,cfg),episode(b,cfg);o=x.reset();p=y.reset()
    assert o==p
    with no_mc_guard():assert x.probe((.2,)*12)['action_hash']==y.probe((.2,)*12)['action_hash']
    with pytest.raises(RuntimeError,match='stale'):x.step(QSignal((0.,)*12,1,o.revision,o.hash))


def test_same_shared_initialization_new_head_and_logstd():
    with torch.random.fork_rng():torch.manual_seed(33);old=DirectNetwork(3,8)
    with torch.random.fork_rng():torch.manual_seed(33);new=QNetwork(3,8)
    for k,v in old.state_dict().items():
        if k not in ('actor.weight','actor.bias','log_std'):assert torch.equal(v,new.state_dict()[k])
    assert torch.equal(new.actor.weight[:6],old.actor.weight)
    assert torch.all(new.actor.weight[6:]==0)
    assert new.log_std.tolist()==[-.5]*6+[-1.5]*6


def test_joint_log_probability_sum_and_sample_once():
    z=torch.linspace(-1,1,20)[None];mu=torch.zeros_like(z);std=torch.ones_like(z)
    p=squashed_log_probability(mu,std,z)
    independent=torch.distributions.Normal(mu,std).log_prob(z)-torch.log(1-torch.tanh(z)**2)
    assert p.item()==pytest.approx(independent.sum().item(),abs=1e-5)
    assert p.item()!=pytest.approx(independent.mean().item())
    obs=episode(scenario(),direct_cfg()).reset();net=QNetwork(3,8)
    a,z,logp,_=sample_action(net,obs,torch.Generator().manual_seed(9))
    assert len(a)==12 and logp.isfinite()
    schema=DirectSchema(**direct_cfg()['action']);p=QProvider(schema,QSignal(tuple(a.tolist()),0,0,obs.hash),3,1)
    v=p.marginal_release_value(1,'A',1,{})
    assert v==p.marginal_release_value(1,'A',1,{}) and len(p.queries)==1


def test_recovery_restores_adam_rng_and_exact_prediction(tmp_path):
    net=QNetwork(3,8);op=torch.optim.Adam(net.parameters(),lr=.0003)
    obs=episode(scenario(),direct_cfg()).reset();gen=torch.Generator().manual_seed(99)
    mean,std,value=net([obs]);(mean.square().sum()+value.square().sum()).backward();op.step()
    path=tmp_path/'q.pt';torch.save(dict(network=net.state_dict(),optimizer=op.state_dict(),rng=gen.get_state()),path)
    next1=sample_action(net,obs,gen)
    other=QNetwork(3,8);newop=torch.optim.Adam(other.parameters(),lr=.0003)
    saved=torch.load(path,weights_only=False);other.load_state_dict(saved['network']);newop.load_state_dict(saved['optimizer'])
    g=torch.Generator();g.set_state(saved['rng']);next2=sample_action(other,obs,g)
    assert all(torch.equal(a,b) for a,b in zip(next1,next2))
    for key,state in op.state_dict()['state'].items():
        for k,v in state.items():assert torch.equal(v,newop.state_dict()['state'][key][k])


def test_structural_q_basis_is_not_constant_copy():
    schema=DirectSchema(('A',),(1.,));cs=[dict(hub='A',time=4,supply_slot=q) for q in (1,2,3)]
    b=basis(schema,0,24,3,cs)
    assert np.linalg.matrix_rank(b[:,:2])==1
    assert np.linalg.matrix_rank(b)==2
    assert np.linalg.matrix_rank(basis(schema,0,24,1,cs)[:,2:])==0


def test_crossed_small_sample_not24independent_runs():
    r=crossed(np.full((3,8),7.))
    assert r['mean_USD']==7 and r['CI95']==[7,7] and r['Bonferroni3']==[7,7]
    with pytest.raises(ValueError):crossed(np.ones((24,1)))
