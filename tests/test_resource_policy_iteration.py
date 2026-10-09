"""Independent sixth-study boundary/lineage tests, not a profit expectation test."""
import pytest
from experiment_core.storage import digest
from experiments.cfs_rollout_target_alignment.toys import fixture,order
from experiments.cfs_resource_policy_iteration.protocol import resource_world,pair_audit,TrackedProvider,restore,pack_state,continuation
from experiments.cfs_resource_policy_iteration.phase0 import toy_policy,toy_table
from experiments.cfs_resource_policy_iteration.common import defaults,INTERPOLATION,TERMINAL

def test_pair_preserves_every_commitment_and_other_hub():
    s,_=fixture([order('current','A','B',10)],hubs=['A','B'])
    info={};s.commit_plan(s.solve_plan(False,0,False,None,info))
    pair_audit(resource_world(s,'A',2),resource_world(s,'A',3),'A')
    assert resource_world(s,'A',2).vehs[0].segments==s.vehs[0].segments

def test_pair_rejects_nonresource_change():
    s,_=fixture([]);a,b=resource_world(s,'A',0),resource_world(s,'A',1);b.J+=1
    with pytest.raises(AssertionError):pair_audit(a,b,'A')

def test_negative_and_nonmonotone_are_not_clipped():
    p=TrackedProvider(toy_table(-6,10),'table','policy');s=dict(T=2)
    assert p.marginal_release_value(0,'A',1,s)==-6
    assert p.marginal_release_value(0,'A',2,s)==10
    assert p.marginal_release_value(2,'A',1,s)==0
    assert p.marginal_release_value(20,'not-a-hub',999,s)==0

def test_no_unrecorded_zero_fallback_for_missing_slot():
    p=TrackedProvider(toy_table(6,10),'table','policy')
    with pytest.raises(KeyError):p.marginal_release_value(0,'A',99,dict(T=2))

def test_parent_provider_changes_both_future_decision_and_marginal():
    s,o=fixture([order('now','A','B',6,deadline=1),order('later','A','B',10,book=1,deadline=2)],horizon=2)
    a=continuation(s,o,toy_policy(None,0),dict(rate={}))
    b=continuation(s,o,toy_policy(toy_table(6,10),1),dict(rate={}))
    c=continuation(s,o,toy_policy(toy_table(10,6),2),dict(rate={}))
    assert [r['metrics']['realized_profit'] for r in [a,b,c]]==[6,10,10]
    assert b['value_calls']>0 and c['table_hash']!=b['table_hash']
    assert b['policy_hash']!=a['policy_hash']

def test_split_fixed_four_no_formal_tuning():
    c=defaults();assert c['iterations']==4 and c['paths_per_source']==1
    splits=[set(c[k]) for k in ['construction_seeds','diagnostic_seeds','formal_seeds','pilot_seeds']]
    assert not any(a&b for i,a in enumerate(splits) for b in splits[:i])
    assert len(c['construction_seeds'])==60 and len(c['formal_seeds'])==40
    assert c['interpolation']==INTERPOLATION and c['terminal']==TERMINAL

def test_restored_counterfactual_can_have_more_than_original_fleet():
    s,o=fixture([order('now','A','B',6,deadline=1)],horizon=2)
    w=resource_world(s,'A',3)
    scenario=dict(id='restore-test',seed=0,horizon=2,nodes=['A','B','C'],edges=[['A','B',1],['B','C',1]],hubs=['A'],
        orders=[order('now','A','B',6,deadline=1)])
    restored,_=restore(scenario,s.parameters,s.solver_parameters,pack_state(w))
    assert len(restored.vehs)==3 and digest(pack_state(restored))==digest(pack_state(w))

def test_profiled_table_is_exact_frozen_table_constructor():
    from experiments.cfs_classical_value.fit_models import table_model
    from experiments.cfs_resource_policy_iteration.targets import profiled_table_model
    rows=[dict(time=t,hub=h,q=q,target=float(t+q*seed)) for t in [0,2] for h in ['A','B'] for q in [1,3] for seed in [1,2]]
    model,aggregate,interpolate=profiled_table_model(rows,6)
    assert model==table_model(rows,6) and aggregate>=0 and interpolate>=0

def test_profiled_simulation_preserves_frozen_decisions():
    from experiments.cfs_classical_value.simulation import simulate as frozen_simulate
    from experiments.cfs_resource_policy_iteration.simulation import simulate
    from experiments.cfs_local_post_bayesian_value.baseline_regression import physical
    a,o=fixture([order('now','A','B',6,deadline=1),order('later','A','B',10,book=1,deadline=2)],horizon=2)
    b=a.clone();ma,ta,_=frozen_simulate(a,o,trace=True,first_begun=True);mb,tb,_=simulate(b,o,trace=True,first_begun=True)
    assert ma['realized_profit']==mb['realized_profit'] and physical(ta)==physical(tb)
    for field in ['context_preparation_seconds','state_transition_seconds','realized_accounting_seconds','provider_lookup_seconds']:
        assert mb[field]>=0
