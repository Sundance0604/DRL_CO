import copy
import pytest
from experiments.cfs_resource_policy_iteration_damping.evaluation import blend
from experiments.cfs_resource_policy_iteration_damping.common import configuration,ALPHAS

def surface(x):return dict(method='table',values={'A':{'1':dict(times=[0,24],values=[x,0.])}})

@pytest.mark.parametrize('alpha',[.25,.5,.75,1.])
def test_exact_blend_negative_terminal_and_no_mutation(alpha):
    a,b=surface(10),surface(-30);original=copy.deepcopy([a,b])
    result=blend(a,b,alpha)
    assert result['values']['A']['1']['values']==[(1-alpha)*10+alpha*(-30),0.]
    assert [a,b]==original

def test_grid_mismatch_fails():
    a,b=surface(1),surface(2);b['values']['A']['1']['times']=[1,24]
    with pytest.raises(AssertionError):blend(a,b,.5)

def test_true_self_consistency_and_split_boundaries():
    c=configuration();assert c['alphas']==ALPHAS
    sets=[set(c[k]) for k in ['construction_seeds','diagnostic_seeds','pilot_seeds','formal_seeds']]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[:i])
    assert c['diagnostic_seeds']==list(range(8200,8220))
    assert c['formal_seeds']==list(range(9100,9140))
    assert 'ACTUAL' in c['clarification']

def test_fixed_frozen_smoothing_is_not_self_consistent():
    # E changes with the deployed policy: later raw evaluations cannot be frozen.
    E=lambda x:surface(-20 if x['values']['A']['1']['values'][0]>0 else 30)
    v0=surface(10);v1=blend(v0,E(v0),.25)
    raw2=E(v1)
    assert raw2!=E(E(v0))
