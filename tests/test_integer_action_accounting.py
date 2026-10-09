"""Accounting amendment tests: same actions, strict replay, real stopped state."""
import copy
import pytest
from experiment_core.storage import read_json, digest
from model_families.single.engine.mt_prototype import DecisionPlan
from model_families.single.runner import state_snapshot, serialize_plan
from experiments.cfs_rollout_target_alignment.toys import fixture, order
from experiments.cfs_resource_policy_iteration.action_analysis import canonical
from experiments.cfs_resource_policy_iteration.protocol import restore, resource_world, read_compressed, state_identity
from experiments.cfs_resource_value_state_bucket_exact_accounting.common import FAILED_B
from experiments.cfs_resource_value_state_bucket_exact_accounting.exact_accounting import reconcile_integer_action
from experiments.cfs_resource_value_state_bucket_exact_accounting.target_protocol import continuation


def toy_plan():
    s,_=fixture([order('now','A','B',100,deadline=2)],horizon=2)
    s.parameters.update(cross_hub=False,empty_cost=0,reposition=False)
    info={};p=s.solve_plan(False,0,False,None,info)
    return s,p,info


def test_near_integer_solver_cash_is_not_executed_and_raw_values_are_retained():
    s,p,info=toy_plan();before=state_identity(s)
    perturbed=DecisionPlan((p[0]+.0002744,p[1]+.0002744,*p[2:]),p.period,p.revision)
    original=canonical(serialize_plan(perturbed,s.t,digest(state_snapshot(s)),info))
    fixed=reconcile_integer_action(s,perturbed,info)
    assert state_identity(s)==before
    assert canonical(serialize_plan(fixed,s.t,digest(state_snapshot(s)),info))==original
    assert fixed[2] is perturbed[2] and fixed[3] is perturbed[3] and fixed[4] is perturbed[4]
    assert fixed[0]==pytest.approx(p[0],abs=1e-9)
    diagnostic=info['integer_action_accounting']
    assert diagnostic['raw_solver_profit_USD']==perturbed[0]
    assert diagnostic['integer_minus_raw_profit_USD']==pytest.approx(-.0002744)
    assert info['accounting']['profit_identity_residual']==0
    s.commit_plan(fixed);assert s.J==fixed[0]


def test_nonoptimal_stale_and_invalid_original_actions_are_not_accepted():
    s,p,info=toy_plan()
    with pytest.raises(RuntimeError,match='OPTIMAL'):
        reconcile_integer_action(s,p,info|dict(status='TIME_LIMIT'))
    with pytest.raises(RuntimeError,match='stale'):
        reconcile_integer_action(s,DecisionPlan(tuple(p),p.period,p.revision+1),info)
    badsol=copy.deepcopy(p[2]);badsol['x']*=2
    bad=DecisionPlan((p[0],p[1],badsol,*p[3:]),p.period,p.revision)
    with pytest.raises(RuntimeError,match='duplicate'):
        reconcile_integer_action(s,bad,copy.deepcopy(info))


def test_real_stopped_seed8405_q3_passes_strict_full_cooldown_replay():
    queries=read_json(FAILED_B/'manifests/diagnostic_query_freeze.json')['queries']
    query=next(q for q in queries if q['id']=='california_corridor-n80-k3-s8405')
    previous=read_compressed(FAILED_B/'raw/diagnostic/raw/traces/construction-iter0-california_corridor-n80-k3-s8405.json.gz')
    scenario=read_json(query['scenario_ref'])['scenarios'][0]
    state,orders=restore(scenario,query['parameters'],query['solver'],previous['state_packets']['7'])
    state=resource_world(state,'06-348',3);before=state_identity(state);info={}
    original=state.solve_plan(state.parameters['cross_hub'],state.parameters['empty_cost'],state.parameters['reposition'],None,info)
    fixed=reconcile_integer_action(state,original,info)
    assert state_identity(state)==before
    assert abs(info['integer_action_accounting']['integer_minus_raw_profit_USD'])>1e-5
    assert info['accounting']['profit_identity_residual']==0
    assert canonical(serialize_plan(original,state.t,'',{}))==canonical(serialize_plan(fixed,state.t,'',{}))
    result=continuation(state,orders,dict(table=None,policy_hash='exact-accounting-regression',table_hash=None),{})
    assert result['replay']['periods']>0 and result['metrics']['unfinished_committed']==0
    for row in result['trace']:
        if not row.get('drain'):
            assert row['plan']['solver']['integer_action_accounting']['integer_action_preserved']
            assert row['invariants']['profit_reconciles']
