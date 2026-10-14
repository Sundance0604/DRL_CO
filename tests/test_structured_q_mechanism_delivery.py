"""Non-numerical delivery regression; not part of the frozen optimization source."""
import copy
import pytest
import numpy as np
from experiments.cfs_structured_ppo_q_mechanism.delivery import compute_kl
from experiments.cfs_structured_ppo_q_mechanism.physical_qa import public_episode,validate_plan
from experiments.cfs_structured_ppo_q_mechanism.common import TARGET,read_json


@pytest.mark.parametrize('dimension',[10,20])
def test_scalar_per_dimension_sigma_broadcasts_over_minibatch(dimension):
    old=np.zeros((3,dimension));new=old+.1
    c=dict(old_mu=old.tolist(),new_mu=new.tolist(),old_sigma=[.5]*dimension,new_sigma=[.5]*dimension,latent=old.tolist())
    r=compute_kl(c)
    assert r['exact_KL']==pytest.approx(dimension*.02)
    assert r['alpha_exact_KL']==pytest.approx(.2)
    assert r['beta_exact_KL']==(pytest.approx(.2) if dimension==20 else None)
    assert r['alpha_std_KL']==0


@pytest.mark.parametrize('tamper',[False,True])
def test_saved_witness_projection_checks_actual_prefix_without_solving(tamper):
    path=TARGET/'raw/selection/s21009-t4.json'
    assert path.exists(), 'run numerical stage before final delivery tests'
    d=read_json(path);ep,*_=public_episode(d['spec']);c=d['certificate'];slots=copy.deepcopy(c['slot_occupancy'])
    if tamper:
        slot=next(s for s in slots if s['time']==5 and s['hub']=='06-41740' and s['q']==1)
        slot['occupied']=1
        with pytest.raises(RuntimeError,match='physical release prefix differs'):
            validate_plan(ep._sim,c['witness_plan'],slots,c['original_coefficients'],objective=False)
    else:
        validate_plan(ep._sim,c['witness_plan'],slots,c['original_coefficients'],objective=False)
