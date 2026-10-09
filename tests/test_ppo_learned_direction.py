"""New replay helper and prespecified nested descriptive denominators."""
import copy
from dataclasses import asdict
import pytest
from test_single_strategic_ppo import setup
from experiments.cfs_ppo_learned_direction.run import advance_frozen_mc, Budget
from experiments.cfs_ppo_learned_direction.analyse import summarize
from experiments.cfs_ppo_signal_action_diagnostic.run import BudgetLimited, scaled_signal
from model_families.single.runner import state_snapshot


def reference():
    ep=setup();obs=ep.reset();states=[]
    while True:
        states.append((asdict(obs),state_snapshot(ep._sim)))
        tr=ep.step((0.,)*6)
        if tr.terminated:break
        obs=tr.observation
    return states,ep.records


def test_reconstruction_uses_no_decoder_and_matches_every_boundary():
    states,records=reference();ep=setup();obs=ep.reset()
    ep.step=lambda *_:pytest.fail('replay must not solve')
    ep.probe=lambda *_:pytest.fail('replay must not probe')
    for expected,tr in zip(states,records,strict=True):
        assert (asdict(obs),state_snapshot(ep._sim))==expected
        obs=advance_frozen_mc(ep,tr)
    assert obs is None and all(v.phase=='idle' for v in ep._sim.vehs)


@pytest.mark.parametrize('field',['before_state_hash','after_state_hash'])
def test_reconstruction_rejects_changed_boundary(field):
    _,records=reference();tr=copy.deepcopy(records[0]);tr[field]='tampered'
    ep=setup();ep.reset()
    with pytest.raises(RuntimeError,match='mismatch'):
        advance_frozen_mc(ep,tr)


def test_reconstruction_rejects_changed_cash():
    _,records=reference();tr=copy.deepcopy(records[0]);tr['plan']['operating_profit']+=1
    ep=setup();ep.reset()
    with pytest.raises(RuntimeError,match='cash mismatch'):
        advance_frozen_mc(ep,tr)


def test_scaling_is_fixed_and_does_not_change_raw_direction():
    raw=(.4,-.8,.01)
    assert scaled_signal(raw,4)==((1.,-1.,.04),2)
    assert raw==(.4,-.8,.01)


def test_budget_counts_only_new_solves_and_enforces_limit():
    b=Budget(seconds=60,solves=1);b.claim()
    with pytest.raises(BudgetLimited):b.claim()
    b.deadline=0
    with pytest.raises(BudgetLimited,match='ten-minute'):b.check()


def test_nested_denominators_preserve_nonresponsive_zero_states():
    rows=[dict(scenario_seed=s,period=t,training_seed=seed,multiplier=scale,
        prior_axis_responsive=t==0,changed=t==0 and scale==4,coefficient_changed=t<2,
        clipped_components=0,max_abs_residual_USD=1.)
        for s in (1,2) for t in range(3) for seed in (30001,30002,30003) for scale in (1.,4.)]
    summary=[r for r in summarize(rows) if r['scenario_seed'] is None and r['training_seed']==30001 and r['multiplier']==4]
    counts={r['subset']:(r['states'],r['changed']) for r in summary}
    assert counts=={'all':(6,2),'axis_responsive':(2,2),'axis_nonresponsive':(4,0)}
    with pytest.raises(RuntimeError,match='duplicate'):summarize(rows+[rows[0]])
