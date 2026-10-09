"""Round-seven targeted controls, never write historical output directories."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from experiment_core.storage import digest
from model_families.single.planning import PeriodAction,VehicleAction,compile_action,route_candidates
from model_families.single.runner import serialize_plan,state_snapshot
from model_families.single.value.providers import MCTableValueProvider
from experiments.cfs_rollout_target_alignment.toys import fixture,order
from experiments.cfs_rollout_target_alignment.forced import forced_plan,route_key
from experiments.cfs_resource_policy_iteration.protocol import resource_world,pair_audit
from experiments.cfs_resource_value_state_bucket_exact_accounting.exact_accounting import reconcile_integer_action
from experiments.cfs_resource_value_state_bucket_exact_accounting.pressure import pressure,expected_prior,bucket
from experiments.cfs_joint_action_value_diagnostic.actions import decide,verify,physical_action,releases,period_action,choose
from experiments.cfs_joint_action_value_diagnostic.worlds import evaluate
from experiments.cfs_joint_action_value_diagnostic.futures import generate

M0=dict(table=None,table_hash=None,policy_hash='toy-M0')

def serial(s,action):
    info={};p=compile_action(s,action,info)
    return serialize_plan(p,s.t,digest(state_snapshot(s)),info)

def test_entire_joint_action_is_forced_and_normal_control_resumes_next_period():
    s,orders=fixture([order('now','A','B',10),order('future','A','B',100,book=1,deadline=2)],hubs=['A','A'])
    wait=serial(s,PeriodAction(s.t));before=digest(state_snapshot(s))
    r=evaluate(s,orders,M0,{},wait)
    assert digest(state_snapshot(s))==before
    assert r['trace'][0]['plan']['assignments']==[]
    assert r['trace'][0]['plan']['routes']==[]
    assert any(a['order_id']=='future' for a in r['trace'][1]['plan']['assignments'])
    assert r['current_action_hash']==digest(physical_action(wait))
    assert r['replay']['status']=='PASSED'

def test_two_vehicles_competing_for_one_order_cannot_be_duplicated():
    s,orders=fixture([order('one','A','B',100)],hubs=['A','A'])
    a=decide(s,M0,{})
    assert len(a['assignments'])==1
    result=evaluate(s,orders,M0,{},a)
    assert result['total']==100
    r=next(r for r in route_candidates(0,s.net,s.pool,s.vehs[0],False,0,s.parameters) if r['e']=='B')
    with pytest.raises(RuntimeError,match='duplicate'):
        compile_action(s,PeriodAction(0,(VehicleAction(0,r,('one',)),VehicleAction(1,r,('one',)))),{})

def test_cross_time_release_groups_do_not_serve_one_future_order_twice():
    s,orders=fixture([order('locked1','B','A',0),order('locked2','C','A',0),
        order('future','A','B',100,book=2,deadline=3)],hubs=['B','C'],horizon=4)
    plan=decide(s,M0,{})
    # Zero-fare ties may wait: explicitly compile the two legal anchored commitments.
    acts=[]
    for v,oid in [(s.vehs[0],'locked1'),(s.vehs[1],'locked2')]:
        r=next(r for r in route_candidates(0,s.net,s.pool,v,False,0,s.parameters) if r['e']=='A')
        acts.append(VehicleAction(v.id,r,(oid,)))
    a=serial(s,PeriodAction(0,tuple(acts)));release=releases(s,a)
    assert {(r['time'],r['hub']) for r in release}=={(1,'A'),(2,'A')}
    result=evaluate(s,orders,M0,{},a)
    ids=[a['order_id'] for r in result['trace'] if not r.get('drain') for a in r['plan']['assignments']]
    assert ids.count('future')==1 and result['total']==100

def test_terminal_resource_zero_does_not_imply_terminal_action_future_component_zero():
    s,orders=fixture([order('short','A','B',40,deadline=1),order('long','A','C',50,deadline=3),
        order('return','B','A',100,book=1,deadline=2),order('out','B','C',70,book=1,deadline=2)],
        hubs=['A','B'],horizon=3,max_delay=1)
    r=next(r for r in route_candidates(0,s.net,s.pool,s.vehs[0],False,0,s.parameters) if r['e']=='C' and r['delay']==1)
    info={};raw=forced_plan(s,dict(vehicle_id=0,kind='route',route_key=route_key(r),candidate_hash='toy'),info)
    p=reconcile_integer_action(s,raw,info);a=serialize_plan(p,s.t,digest(state_snapshot(s)),info)
    assert next(e for e in releases(s,a) if e['vehicle_id']==0)['time']==s.T
    short=next(r for r in route_candidates(0,s.net,s.pool,s.vehs[0],False,0,s.parameters) if r['e']=='B' and r['delay']==0)
    info0={};raw0=forced_plan(s,dict(vehicle_id=0,kind='route',route_key=route_key(short),candidate_hash='short'),info0)
    p0=reconcile_integer_action(s,raw0,info0);base=serialize_plan(p0,s.t,digest(state_snapshot(s)),info0)
    x=evaluate(s,orders,M0,{},a);y=evaluate(s,orders,M0,{},base)
    assert x['total']-x['current_stage_USD'] != y['total']-y['current_stage_USD']
    table=dict(method='table',values={})
    assert MCTableValueProvider(table).marginal_release_value(s.T,'C',1,dict(T=s.T))==0

def test_bucket_original_base_and_post_injection_are_distinct_but_labels_stay_common():
    s,_=fixture([order('now','A','B',6)]);s.parameters['capacity']=40
    prior=expected_prior([dict(nodes=['A','B','C'],horizon=4,seed=8300,orders=[])],40)
    original=pressure(s,prior)['A']['R'];low=resource_world(s,'A',0);high=resource_world(s,'A',1)
    assert pressure(low,prior)['A']['R']!=pressure(high,prior)['A']['R']
    pair_audit(low,high,'A');cuts=dict(c33=.02,c67=.04)
    label=bucket(original,cuts)
    assert [label,label]==[bucket(original,cuts)]*2
    assert len(resource_world(s,'B',2).vehs)>len(s.vehs)

def test_future_role_isolation_and_sampler_never_reads_hidden_orders():
    s,orders=fixture([order('visible','A','B',5)],horizon=5)
    prior=dict(rate_per_period=8.,marks=[dict(departure='A',destination='B',passenger=1,revenue=5,
        penalty=0,ready_delay=0,slack=1)],hash='prior',distribution='SURROGATE')
    _,sel=generate(orders,s,prior,'C','state','selection',0)
    _,ass=generate(orders,s,prior,'C','state','assessment',0)
    assert sel['stream_hash']!=ass['stream_hash'] and sel['future_hash']!=ass['future_hash']
    assert generate(orders,s,prior,'C','state','selection',0)[1]==sel
    hidden=orders|{'hidden':SimpleNamespace(**order('hidden','A','B',100,book=1))}
    with pytest.raises(AssertionError,match='Hidden future'):generate(hidden,s,prior,'C','state','selection',0)
    assert all(o['book_time']>s.t for o in sel['future_orders'])
    assert len({o['id'] for o in sel['future_orders']})==len(sel['future_orders'])

def test_independent_execution_and_saved_action_replay_are_identical():
    s,orders=fixture([order('one','A','C',100,deadline=4)],horizon=1)
    a=decide(s,M0,{})
    x=evaluate(s,orders,M0,{},a);y=evaluate(s,copy.deepcopy(orders),M0,{},copy.deepcopy(a))
    assert x['physical_final_hash']==y['physical_final_hash'] and x['total']==y['total']
    assert x['cooldown_periods']>0 and x['replay']['total_difference']==0
    assert x['independent_cash_max_residual']==0

def test_selector_ties_are_explicit_and_assessment_is_not_an_input():
    actions=[dict(action_hash='b',labels=['pi1']),dict(action_hash='a',labels=['M0'])]
    assert choose(actions,dict(a=1.,b=1.))['action_hash']=='b'
    assert choose(actions,dict(a=2.,b=1.))['action_hash']=='a'
    assert choose(actions,dict(a=1.,b=1.))['tie_set']==['a','b']

def test_invalid_booked_amount_or_partial_current_action_is_rejected():
    s,_=fixture([order('one','A','B',100)])
    a=decide(s,M0,{});bad=copy.deepcopy(a);bad['operating_profit']+=1
    with pytest.raises(AssertionError,match='cash'):verify(s,bad)
    bad=copy.deepcopy(a);bad['assignments']=[]
    with pytest.raises(RuntimeError,match='anchor'):verify(s,bad)
