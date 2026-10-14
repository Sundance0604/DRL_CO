import numpy as np
import pytest
from experiments.cfs_direct_ppo_training_ablation_v4.analysis import crossed,error


def test_bootstrap_constant_and_family_correction():
    r=crossed(np.full((3,24),7.),9,replications=2000)
    assert r['mean_USD']==7 and r['low95_USD']==7 and r['high_adjusted_USD']==7
    assert r['count_weighted_recalculation']


def test_bootstrap_does_not_flatten_model_scene_crossing():
    x=np.repeat(np.array([[-2.],[0.],[5.]]),24,axis=1)
    r=crossed(x,9,replications=2000)
    assert r['low_adjusted_USD']<=r['low95_USD'] and r['high_adjusted_USD']>=r['high95_USD']
    with pytest.raises(ValueError):crossed(x.reshape(72,1),9)


def test_USD_critic_errors_distinguish_bias_and_dispersion():
    r=error([1000,2000],[2000,3000])
    assert r['RMSE_USD']==1000 and r['MAE_USD']==1000 and r['EV']==1
    assert error([1,1],[2,2])['EV'] is None


def test_analysis_C_no_critic_is_missing_not_zero():
    from experiments.cfs_direct_ppo_training_ablation_v4.network import build
    assert build('C',1).critic is None
