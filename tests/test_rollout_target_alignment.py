"""Fifth targets: one-period forcing, no future leak, J-g and replay."""
import copy
import pytest
from experiments.cfs_rollout_target_alignment.toys import fixture,order,find,audit
from experiments.cfs_rollout_target_alignment.queries import choose_candidates,pack_state,restore_state
from experiments.cfs_rollout_target_alignment.forced import forced_plan,_build_forced,route_key
from experiments.cfs_rollout_target_alignment.protocol import world,paired_target,state_identity,replay_world

def test_exact_toy_accounting_and_control():assert audit()['status']=='PASSED'

def test_force_keeps_joint_assignment_and_other_vehicle_free():
    s,o=fixture([order('low','A','B',10),order('high','A','B',20),order('other','B','C',30)],hubs=['A','B'])
    cs,_,_=choose_candidates(s);c=find(cs,'B');before=state_identity(s)
    result=forced_plan(s,c,{})
    assert ('high',0) in result[2]['x'] and ('other',1) in result[2]['x']
    assert ('low',0) not in result[2]['x'] and state_identity(s)==before

def test_wait_force_does_not_force_other_vehicles_to_wait():
    s,o=fixture([order('a','A','B',20),order('b','B','C',30)],hubs=['A','B'])
    cs,_,_=choose_candidates(s);wait=next(c for c in cs if c['vehicle_id']==0 and c['kind']=='wait')
    r=forced_plan(s,wait,{})
    assert not any(k==0 for _,k in r[2]['x']) and ('b',1) in r[2]['x']

def test_next_period_reoptimizes_and_intervention_not_persistent():
    s,o=fixture([order('a','A','B',20),order('future','A','C',30,book=1)],horizon=3)
    cs,_,_=choose_candidates(s);wait=next(c for c in cs if c['kind']=='wait')
    r=world(s,o,wait)
    assert r['solver_periods']==[0,1,2]
    assert r['trace'][1]['plan']['assignments']
    assert 'one_period_force' not in r['trace'][1]['plan']['solver']
    assert replay_world(s,o,r)['status']=='PASSED'

def test_future_price_change_cannot_affect_frozen_query_or_current_solution():
    s,o=fixture([order('a','A','B',20),order('future','B','C',30,book=1)])
    cs,missing,plan=choose_candidates(s)
    o['future'].revenue=99999
    ds,missing2,plan2=choose_candidates(s)
    assert cs==ds and missing==missing2 and plan['assignments']==plan2['assignments']
    assert all('future' not in c['compatible_order_ids'] for c in cs)

def test_unforced_copy_exact_and_business_profit_excludes_tie():
    s,o=fixture([order('a','A','B',20)],max_delay=1)
    p=s.parameters
    r=_build_forced(s.t,s.net,s.pool,s.vehs,p['cross_hub'],p['empty_cost'],False,parameters=p,solver_parameters=s.solver_parameters,solve_info={})
    assert r[:3]==forced_plan(s,None,{})[:3]
    cs,_,_=choose_candidates(s);c=find(cs,'B',1);r=forced_plan(s,c,{})
    assert r[0]==20 and r[1]==19

def test_baseline_self_target_exact_zero():
    s,o=fixture([order('a','A','B',20)])
    cs,_,_=choose_candidates(s);c=next(c for c in cs if c['baseline'])
    a=world(s,o,c);b=world(s,o)
    assert paired_target(a,b)==0 and a['current_action_hash']==b['current_action_hash']

def test_full_cooldown_delivers_late_commitments():
    s,o=fixture([order('late','A','C',20)],horizon=1)
    r=world(s,o)
    assert r['cooldown_periods']>=2 and r['ledger']['completed_orders']==1
    assert r['ledger']['credited_unrealized_fare']==0 and replay_world(s,o,r)['status']=='PASSED'

def test_current_metadata_selection_limit_and_missing_reason():
    s,o=fixture([order('a','A','B',20),order('b','A','C',30)],max_delay=2)
    cs,missing,_=choose_candidates(s,3)
    assert len(cs)<=3 and sum(c['baseline'] for c in cs)==1
    s,o=fixture([]);cs,missing,_=choose_candidates(s)
    assert not cs and missing[0]['missing_reason']=='no_feasible_alternative'

def test_replay_tolerates_only_submicrodollar_cash_not_physics():
    from experiments.cfs_rollout_target_alignment.protocol import assert_replay_snapshot
    a=dict(profit=37378.93371399542,period=8,vehicles=['same'])
    b=dict(profit=37378.933713995415,period=8,vehicles=['same'])
    assert_replay_snapshot(a,b)
    with pytest.raises(AssertionError):assert_replay_snapshot(a,b|{'vehicles':['changed']})
    with pytest.raises(AssertionError):assert_replay_snapshot(a,b|{'profit':b['profit']+.001})
