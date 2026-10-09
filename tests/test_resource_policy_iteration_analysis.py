"""Analysis checks do not select, fit or alter any experimental policy."""
import numpy as np
from experiments.cfs_resource_policy_iteration.analysis import describe,holm

def test_oscillation_counts_are_strict_json_serializable_without_changing_comparisons():
    from experiments.cfs_resource_policy_iteration.analysis import oscillation_metrics
    from experiment_core.storage import canonical
    result=oscillation_metrics([-2.,3.,-1.,4.])
    assert result==dict(sign_flip_count=3,range_USD=6.,update_direction_reversal_count=2)
    assert type(result['sign_flip_count']) is type(result['update_direction_reversal_count']) is int
    assert canonical(result)

def test_paired_sample_is_a_scenario_not_a_trace_row():
    cfg=dict(bootstrap_seed=10,bootstrap_replications=1000)
    result=describe([1,2,3,4],cfg)
    assert result['n']==4 and result['mean']==2.5
    assert np.isclose(result['SD'],np.std([1,2,3,4],ddof=1))
    assert result['CI95_lower']<result['mean']<result['CI95_upper']

def test_holm_stepdown_and_capped_probability():
    rows=[dict(p_two_sided=p) for p in [.01,.04,.50]];holm(rows)
    assert np.allclose([r['Holm_p'] for r in rows],[.03,.08,.5])
    rows=[dict(p_two_sided=p) for p in [.01,.01,.9]];holm(rows)
    assert np.allclose([r['Holm_p'] for r in rows],[.03,.03,.9])

def test_zero_paired_effect_keeps_zero_without_nan():
    result=describe([0]*40,dict(bootstrap_seed=0,bootstrap_replications=100))
    assert result['n']==40 and result['SD']==0 and result['p_two_sided']==1
    assert result['bootstrap95_lower']==result['bootstrap95_upper']==0

def test_canonical_action_does_not_count_array_permutation():
    from copy import deepcopy
    from experiments.cfs_resource_policy_iteration.action_analysis import canonical
    a=dict(assignments=[dict(vehicle_id='0',order_id='a'),dict(vehicle_id='1',order_id='b')],routes=[],repositions=[],expired=['c','d'])
    b=deepcopy(dict(a,assignments=list(reversed(a['assignments'])),expired=list(reversed(a['expired']))))
    assert canonical(a)==canonical(b)
    b['assignments'][0]['vehicle_id']='9'
    assert canonical(a)!=canonical(b)

def test_wait_component_distinguishes_delayed_route_from_immediate_route():
    from experiments.cfs_resource_policy_iteration.action_analysis import control_modes
    a=dict(routes=[dict(vehicle_id='0',delay=0)],repositions=[])
    b=dict(routes=[dict(vehicle_id='0',delay=1)],repositions=[])
    assert control_modes(a)!=control_modes(b)
    assert control_modes(dict(routes=[],repositions=[]))!=control_modes(a)

def test_action_hash_matches_equality_for_equivalent_json_representations():
    from experiments.cfs_resource_policy_iteration.action_analysis import canonical
    from experiment_core.storage import digest
    a=dict(assignments=[],routes=[dict(vehicle_id='0',B=('A','B'),delay=1,new=True,cost=-0.0)],repositions=[],expired=[])
    b=dict(assignments=[],routes=[dict(vehicle_id='0',B=['A','B'],delay=1.0,new=1,cost=0)],repositions=[],expired=[])
    assert canonical(a)==canonical(b) and digest(canonical(a))==digest(canonical(b))
    b['routes'][0]['cost']=1e-10
    assert canonical(a)!=canonical(b) and digest(canonical(a))!=digest(canonical(b))

def test_terminal_candidate_advantage_is_not_terminal_free_resource_price():
    from experiments.cfs_resource_policy_iteration.analysis import terminal_estimand_distinction
    evidence=terminal_estimand_distinction()
    assert evidence['candidate_release_time']>=evidence['horizon']
    assert evidence['terminal_candidate_Yroll']==-100
    assert evidence['free_resource_at_T_marginal']==0
    assert not evidence['used_in_policy_evaluation'] and not evidence['used_in_policy_improvement']

def test_lineage_audit_rejects_zero_rollout_disguised_as_value_policy():
    import pytest
    from experiments.cfs_rollout_target_alignment.toys import fixture,order
    from experiments.cfs_resource_policy_iteration.phase0 import toy_policy,toy_table
    from experiments.cfs_resource_policy_iteration.protocol import continuation
    from experiments.cfs_resource_policy_iteration.delivery import value_audit
    s,orders=fixture([order('now','A','B',6,deadline=1),order('later','A','B',10,book=1,deadline=2)],horizon=2)
    p1=toy_policy(toy_table(6,10),1)
    correct=continuation(s,orders,p1,dict(rate={}))
    assert value_audit(correct['trace'],p1)[0]==correct['value_calls']>0
    wrong=continuation(s,orders,toy_policy(None,0),dict(rate={}))
    with pytest.raises(AssertionError):value_audit(wrong['trace'],p1)

def test_reverse_policy_table_hash_rejects_changed_source(tmp_path,monkeypatch):
    import pytest
    from experiments.cfs_resource_policy_iteration import policies
    from experiment_core.storage import atomic_json,digest
    from experiments.cfs_resource_policy_iteration.common import file_hash
    monkeypatch.setattr(policies,'OUTPUT',tmp_path)
    table=tmp_path/'tables/V0/test.json';atomic_json(table,dict(method='table',values={}))
    record=dict(policy_id='pi1',table_ref='tables/V0/test.json',table_hash=file_hash(table))
    record['policy_hash']=digest(record);atomic_json(tmp_path/'policies/pi1.json',dict(test=record))
    assert policies.load_policies(1)['test']['table_hash']==record['table_hash']
    atomic_json(table,dict(method='table',values={},changed=True))
    with pytest.raises(AssertionError):policies.load_policies(1)

def test_query_provenance_does_not_call_interpolation_an_observation():
    from experiments.cfs_resource_policy_iteration.analysis import query_support
    from experiments.cfs_resource_policy_iteration.phase0 import toy_table
    model=toy_table(6,10)
    assert query_support(model,0,'A',1,2,[1,2],2)['support_status']=='observed'
    result=query_support(model,.5,'A',1,2,[1,2],2)
    assert result['support_status']=='interpolated_time' and result['time_weight_right']==.5
    result=query_support(model,0,'A',2,3,[1,3],2)
    assert result['support_status']=='interpolated_q' and result['q_weight_right']==.5
    assert query_support(model,0,'A',3,2,[1,2],2)['support_status']=='unsupported_q_endpoint'
    assert query_support(model,2,'missing',99,2,[1,2],2)['support_status']=='terminal_exact_zero'

def test_negative_prefix_price_cannot_be_omitted_or_cherry_picked():
    import gurobipy as gp
    from model_families.single.value.contracts import add_release_slots
    from experiments.cfs_resource_policy_iteration.protocol import TrackedProvider
    from experiments.cfs_resource_policy_iteration.phase0 import toy_table
    for count,expected in [(1,-6),(2,4)]:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            with gp.Model(env=env) as model:
                model.setParam('Threads',1);model.setParam('Seed',0)
                indicators=[model.addVar(lb=1,ub=1,vtype=gp.GRB.BINARY) for _ in range(count)]
                terms,coefficients=add_release_slots(model,{(0,'A'):indicators},
                    TrackedProvider(toy_table(-6,10),'table','policy'),dict(T=2),2,2)
                model.setObjective(gp.quicksum(terms),gp.GRB.MAXIMIZE);model.optimize()
                assert model.Status==gp.GRB.OPTIMAL and model.ObjVal==expected
                assert [r['value'] for r in coefficients]==[-6,10]
