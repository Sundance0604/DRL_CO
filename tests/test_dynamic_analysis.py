"""Independent statistical checks; postprocessing is not simulation runtime."""
import numpy as np
import pytest
from scipy.stats import t
from experiments.cfs_dynamic_lookahead.analysis import statistics,paired,summarize


CONFIG={'bootstrap_seed':20261005,'bootstrap_replications':10000}


def row(seed,profit,scenario='frozen',**extra):
    return dict(region='western_multicluster',orders=40,fleet=8,rho_label='rho_max',
                policy='myopic',H=0,seed=seed,status='COMPLETED',scenario_hash=scenario,
                realized_profit=profit,service_rate=.5,actual_empty_distance=100.,
                empty_pickup_distance=30.,deliberate_deferral_rate=.1,actual_order_waiting=5.,**extra)


def test_statistics_sample_SD_and_student_t_not_population_normal_CI():
    values=np.array([1.,2.,4.,7.]);s=statistics(values)
    half=t.ppf(.975,3)*values.std(ddof=1)/2
    assert s['n']==4 and s['SD']==pytest.approx(values.std(ddof=1))
    assert s['CI_low']==pytest.approx(values.mean()-half)
    assert s['CI_high']==pytest.approx(values.mean()+half)
    assert s['median']==3.


def test_skew_bootstrap_is_deterministic_and_retains_extreme_sample():
    a=[0.]*29+[100.]
    first=statistics(a);second=statistics(a)
    assert first==second and first['n']==30 and first['skew']>1
    assert first['bootstrap_low'] is not None and first['bootstrap_high'] is not None


@pytest.mark.parametrize('a',[[],[1.,float('nan')],[float('inf')]])
def test_statistics_rejects_missing_nonfinite_samples(a):
    with pytest.raises(ValueError):statistics(a)


def test_pairing_is_by_seed_and_checks_frozen_scenario():
    reg=('western_multicluster',40,8)
    groups={(*reg,'rho_max','known_supply',2):[row(2,104.),row(1,12.)],
            (*reg,'rho_max','myopic',0):[row(1,10.),row(2,100.)]}
    p=paired(groups,CONFIG,'gain',reg,'rho_max','known_supply',2,'myopic',0)
    profit=next(r for r in p if r['metric']=='realized_profit')
    assert profit['mean']==3 and profit['SD']==pytest.approx(np.sqrt(2))
    groups[(*reg,'rho_max','myopic',0)][0]['scenario_hash']='other'
    with pytest.raises(RuntimeError,match='frozen scenario'):
        paired(groups,CONFIG,'gain',reg,'rho_max','known_supply',2,'myopic',0)


def test_null_deferral_success_is_not_imputed_as_failure():
    rows=[row(1,1.,deferral_success_rate=None),row(2,2.,deferral_success_rate=1.)]
    summaries,_=summarize(rows,CONFIG)
    s=next(r for r in summaries if r['metric']=='deferral_success_rate')
    assert s['n']==1 and s['missing_sample_count']==1 and s['mean']==1.
    assert s['CI_low'] is None


@pytest.mark.parametrize('delta',[1e-12,1.])
def test_case_replay_only_preserves_recorded_rounding_not_real_cost_changes(monkeypatch,delta):
    from types import SimpleNamespace
    from experiments.cfs_dynamic_lookahead import case_study
    sim=SimpleNamespace(t=0,revision=7)
    compiled=(10.,9.,{'x':[],'y':[],'z':[]},{},set(),0)
    monkeypatch.setattr(case_study,'compile_action',lambda *args:compiled)
    row={'plan':{'period':0,'routes':[],'assignments':[],'repositions':[],
                 'operating_profit':10.+delta,'augmented_solver_objective':9.+delta}}
    if delta>1e-6:
        with pytest.raises(RuntimeError,match='floating tolerance'):case_study.replay_plan(sim,row)
    else:
        plan=case_study.replay_plan(sim,row)
        assert plan[0]==row['plan']['operating_profit'] and plan[2:]==compiled[2:]
        assert plan.period==0 and plan.revision==7
