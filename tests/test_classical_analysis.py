"""Postprocessing checks remain separate from frozen numerical source identity."""
import numpy as np
import pytest
from experiments.cfs_classical_value.analysis import policy_statistics
from experiments.cfs_classical_value.run_value_policy import LADDER


def fixture_rows():
    labels=[p[0] for p in LADDER]
    rows=[]
    for j,p in enumerate(labels):
        for seed in [1,2,3,4]:
            rows.append(dict(condition="toy-n4-k2",policy=p,status="COMPLETED",seed=seed,
                scenario_hash=str(seed),realized_profit=100*seed+j*10,completed_fare=500,
                realized_cost=100,service_rate=.8,actual_loaded_distance=10,actual_empty_distance=2,
                actual_order_waiting=1,pending_at_T=1,daily_wall_seconds=.1,daily_solver_seconds=.01,
                online_value_eval_seconds=.001,build_seconds=.01))
    cfg=dict(conditions=[dict(region="toy",orders=4,fleet=2)],formal_seeds=[1,2,3,4],
        bootstrap_seed=1,bootstrap_replications=10000)
    return cfg,rows


def test_paired_profit_not_difference_of_unpaired_cis(tmp_path,monkeypatch):
    import experiments.cfs_classical_value.analysis as analysis
    monkeypatch.setattr(analysis,"OUTPUT",tmp_path)
    cfg,rows=fixture_rows()
    summaries,pairs,ratios=policy_statistics(cfg,rows)
    p=next(r for r in pairs if r["policy_a"]=="P1-Myopic-Fluid" and r["metric"]=="realized_profit")
    assert p["mean"]==p["CI_low"]==p["CI_high"]==10
    basis=next(r for r in pairs if r["policy_a"]=="P8-RH4-Chebyshev" and
        r["policy_b"]=="P9-RH4-Spline" and r["metric"]=="realized_profit")
    assert basis["mean"]==basis["CI_low"]==basis["CI_high"]==-10
    assert all(r["eligible"] for r in ratios)


def test_duplicate_seed_rejected(tmp_path,monkeypatch):
    import experiments.cfs_classical_value.analysis as analysis
    monkeypatch.setattr(analysis,"OUTPUT",tmp_path)
    cfg,rows=fixture_rows()
    with pytest.raises(AssertionError,match="duplicate"):
        policy_statistics(cfg,rows+[rows[0]])


def test_different_data_same_seed_rejected(tmp_path,monkeypatch):
    import experiments.cfs_classical_value.analysis as analysis
    monkeypatch.setattr(analysis,"OUTPUT",tmp_path)
    cfg,rows=fixture_rows();rows[4]["scenario_hash"]="different"
    with pytest.raises(AssertionError):policy_statistics(cfg,rows)


def test_failed_sample_is_not_dropped(tmp_path,monkeypatch):
    import experiments.cfs_classical_value.analysis as analysis
    monkeypatch.setattr(analysis,"OUTPUT",tmp_path)
    cfg,rows=fixture_rows();rows[0]["status"]="FAILED"
    with pytest.raises(RuntimeError,match="failed sample"):policy_statistics(cfg,rows)


def test_uncertain_denominator_no_ratio(tmp_path,monkeypatch):
    import experiments.cfs_classical_value.analysis as analysis
    monkeypatch.setattr(analysis,"OUTPUT",tmp_path)
    cfg,rows=fixture_rows()
    rh4={r["seed"]:r["realized_profit"] for r in rows if r["policy"]=="P6-RH4"}
    for r in rows:
        if r["policy"]=="P10-RH8":r["realized_profit"]=rh4[r["seed"]]+(1000 if r["seed"]%2 else -900)
    _,_,ratios=policy_statistics(cfg,rows)
    ratios=[r for r in ratios if r["kind"]=="horizon_compression"]
    assert all(not r["eligible"] and r["ratio"] is None for r in ratios)
