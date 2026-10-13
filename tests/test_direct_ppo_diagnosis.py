import copy
import pytest
from experiments.cfs_direct_ppo_diagnosis.common import canonical,observation,restore,configuration,scenes,PARENT,read_json
from experiments.cfs_direct_ppo_diagnosis.numerics import critic_statistics

def state():return dict(vehicles=[dict(id='0',hub='A',phase='idle',assigned_orders=[],route=[]),dict(id='1',hub='A',phase='idle',assigned_orders=[],route=[])])
def plan():return dict(period=0,assignments=[dict(vehicle_id='0',order_id='x')],routes=[dict(vehicle_id='0',B=['A','B'])],repositions=[],expired=[])
def test_homogeneous_permutation():
    p=plan();q=copy.deepcopy(p)
    for row in q['assignments']+q['routes']:row['vehicle_id']='1'
    assert canonical(p,state())==canonical(q,state())
def test_nonhomogeneous_not_permutable():
    s=state();s['vehicles'][1]['hub']='C';p=plan();q=copy.deepcopy(p)
    for row in q['assignments']+q['routes']:row['vehicle_id']='1'
    assert canonical(p,s)!=canonical(q,s)
def test_route_internal_order():
    q=plan();q['routes'][0]['B']=['B','A'];assert canonical(plan(),state())!=canonical(q,state())
def test_order_identity():
    q=plan();q['assignments'][0]['order_id']='y';assert canonical(plan(),state())!=canonical(q,state())
def test_commitment_not_permutable():
    s=state();s['vehicles'][1]['assigned_orders']=['z'];q=plan()
    for row in q['assignments']+q['routes']:row['vehicle_id']='1'
    assert canonical(plan(),s)!=canonical(q,s)
def test_top_level_order_irrelevant():
    p=plan();p['assignments'].append(dict(vehicle_id='0',order_id='y'));q=copy.deepcopy(p);q['assignments'].reverse()
    assert canonical(p,state())==canonical(q,state())
def test_observation_roundtrip():
    name='western_multicluster-n80-k3';r=read_json(PARENT/'training'/name/'30001/validation-0008-s21008.json')
    assert observation(r['transitions'][0]['observation']).hash==r['transitions'][0]['observation_hash']
@pytest.mark.parametrize('period',[0,12,23])
def test_exact_frozen_state_restore(period):
    name='western_multicluster-n80-k3';s=scenes(name,'validation')[0]
    r=read_json(PARENT/'training'/name/'30001/validation-0008-s21008.json')
    ep,obs=restore(s,configuration(name),r,period)
    assert obs.period==period and obs.hash==r['transitions'][period]['observation_hash']
    assert len(ep._rows)==period
def test_critic_mc_not_lambda_return():
    rewards=[1.]*24;values=[float(24-i) for i in range(24)]
    r=dict(transitions=[dict(reward_USD=1000.) for _ in range(24)],learning=[dict(value=v,advantage=0.,**{'return':v}) for v in values])
    stats=critic_statistics([r]);assert stats['critic_MC_EV']==1 and stats['critic_MC_RMSE_reward_units']==0
def test_corrupt_advantage_rejected():
    r=dict(transitions=[dict(reward_USD=1000.) for _ in range(24)],learning=[dict(value=24-i,advantage=1.,**{'return':24-i}) for i in range(24)])
    with pytest.raises(RuntimeError,match='GAE'):critic_statistics([r])
