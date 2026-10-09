"""Mechanism probe integrity and budget boundaries, including actual MILP."""
import copy
from dataclasses import asdict
import time
import pytest
from experiments.cfs_ppo_signal_action_diagnostic.run import (
    Budget, BudgetLimited, scaled_signal, probe_at, audit_probe)
from experiments.cfs_ppo_signal_action_diagnostic.analyse import interpolate
from test_single_strategic_ppo import setup


def test_multiplier_clipping_is_explicit_and_does_not_mutate_actor():
    signal = [.4, -.8, 0.]
    actual, clipped = scaled_signal(signal, 2)
    assert actual == (.8, -1., 0.) and clipped == 1
    assert signal == [.4, -.8, 0.]
    assert scaled_signal(signal, 0) == ((0., 0., 0.), 0)
    with pytest.raises(ValueError):
        scaled_signal([float("nan")], 1)
    with pytest.raises(ValueError):
        scaled_signal([.1], -1)


def test_expired_wall_and_solver_count_cannot_create_fake_completed_samples():
    budget = Budget(seconds=60, solves=1)
    budget.claim()
    with pytest.raises(BudgetLimited):
        budget.claim()
    budget.deadline = time.monotonic()-1
    with pytest.raises(BudgetLimited):
        budget.check()


def test_independent_interpolation_endpoints_and_negative_values():
    assert interpolate([0,2,4], [10,-10,20], 1) == 0
    assert interpolate([0,2,4], [10,-10,20], 3) == 5
    assert interpolate([0,2,4], [10,-10,20], -1) == 10
    assert interpolate([0,2,4], [10,-10,20], 9) == 20


def test_actual_probe_cache_readonly_and_zero_action_execution():
    ep = setup()
    obs = ep.reset()
    cache, budget = {}, Budget(seconds=60, solves=5)
    key, raw, hit = probe_at(ep, obs, (0.,)*6, cache, budget)
    assert not hit and budget.solves == 1
    again, same, hit = probe_at(ep, obs, (0.,)*6, cache, budget)
    assert hit and again == key and same == raw and budget.solves == 1
    ep.step((0.,)*6)
    assert ep.records[-1]["action_hash"] == raw["action_hash"]


def test_actual_probe_cash_tampering_is_detected():
    ep = setup()
    obs = ep.reset()
    record = ep.probe((0.,)*6)
    bad = copy.deepcopy(record)
    bad["plan"]["operating_profit"] += 1
    with pytest.raises(RuntimeError, match="cash/action"):
        audit_probe(ep, obs, (0.,)*6, bad)


def test_probe_cache_does_not_merge_different_signals():
    ep = setup()
    obs = ep.reset()
    cache, budget = {}, Budget(seconds=60, solves=5)
    zero, _, _ = probe_at(ep, obs, (0.,)*6, cache, budget)
    other, _, _ = probe_at(ep, obs, (.5,0.,0.,0.,0.,0.), cache, budget)
    assert zero != other and budget.solves == 2
