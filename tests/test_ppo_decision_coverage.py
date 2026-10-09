"""All-period study integrity; physical solver tests remain independent synthetic cases."""
import copy
import pytest
from experiments.cfs_ppo_decision_coverage.run import axis_controls, decision_identity, ensure_disjoint
from experiments.cfs_ppo_decision_coverage.analyse import canonical_hash, summarize
from experiments.cfs_ppo_signal_action_diagnostic.run import Budget, probe_at
from test_single_strategic_ppo import setup, scenario


def test_all_axes_are_unchanged_and_not_joint_search():
    rows = axis_controls(10)
    assert len(rows)==20 and len({tuple(r['signal']) for r in rows})==20
    assert all(sum(abs(v) for v in r['signal'])==1 for r in rows)
    with pytest.raises(ValueError):
        axis_controls(6)


def test_primary_action_ignores_only_top_level_enumeration_order():
    p = dict(period=0, assignments=[dict(order_id='x',vehicle_id='1'),dict(order_id='y',vehicle_id='2')],
        routes=[dict(vehicle_id='1', B=['A','B','C'])], repositions=[], expired=['b','a'])
    q = copy.deepcopy(p)
    q['assignments'].reverse(); q['expired'].reverse()
    assert decision_identity(p)==decision_identity(q)==canonical_hash(p)==canonical_hash(q)
    q['routes'][0]['B'].reverse()
    assert decision_identity(p)!=decision_identity(q)


def test_period_subsets_keep_zero_response_and_nested_denominators():
    rows = [dict(scenario_seed=s,period=t,responsive=t==3,coefficient_responsive=t<23,
        resource_active=t in (3,4)) for s in (1,2) for t in range(24)]
    results = {r['subset']:r for r in summarize(rows, (4,8,12,16,20,23)) if r['scenario_seed'] is None}
    assert results['all']['states']==48 and results['all']['responsive']==2
    assert results['fixed_six']['states']==12 and results['fixed_six']['responsive']==0
    assert results['omitted_eighteen']['states']==36 and results['omitted_eighteen']['responsive']==2
    with pytest.raises(RuntimeError,match='duplicate'):
        summarize(rows+[rows[0]], (4,8,12,16,20,23))


def test_no_old_scene_reuse_by_seed_or_content():
    old = scenario(seed=1)
    with pytest.raises(RuntimeError,match='overlap'):
        ensure_disjoint([old],[old])
    new = copy.deepcopy(old); new['seed']=9; new['id']='new-id'
    with pytest.raises(RuntimeError,match='overlap'):
        ensure_disjoint([new],[old])


def test_actual_all_period_probes_never_change_mc_state_and_full_terminal():
    ep=setup(); obs=ep.reset(); periods=[]
    budget=Budget(seconds=60,solves=30)
    while True:
        cache={}; zero=(0.,)*6
        _, ref, _=probe_at(ep,obs,zero,cache,budget)
        for sign in (-1.,1.):
            probe_at(ep,obs,(sign,0.,0.,0.,0.,0.),cache,budget)
        periods.append(obs.period)
        tr=ep.step(zero)
        assert ep.records[-1]['action_hash']==ref['action_hash']
        if tr.terminated:
            break
        obs=tr.observation
    assert periods==[0,1,2]
    assert ep.final_ledger['unfinished_committed']==0
    assert ep.records[-1]['coefficients']==[]
