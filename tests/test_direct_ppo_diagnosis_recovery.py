import copy
import math
import numpy as np
import pytest
from experiments.cfs_direct_ppo_diagnosis_recovery import run as recovery
from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
from experiments.cfs_direct_ppo_diagnosis.analysis import bootstrap

def test_parent_validation_outside_no_mc_guard(monkeypatch):
    from model_families.single.strategic.baseline import FrozenBaseline
    calls=[]
    def verify(full):
        assert full and FrozenBaseline.__init__.__name__!='forbidden'
        calls.append(True);return {'source':{'verified':True}}
    monkeypatch.setattr(recovery.c,'verify_parent',verify)
    # setup rebinds imported targets/functions; restore all monkeypatched bindings.
    monkeypatch.setattr(recovery.c,'TARGET',recovery.c.TARGET)
    monkeypatch.setattr(recovery.n,'TARGET',recovery.n.TARGET)
    monkeypatch.setattr(recovery.n,'source_files',recovery.n.source_files)
    monkeypatch.setattr(recovery.n,'verify_parent',recovery.n.verify_parent)
    recovery.setup()
    with no_mc_guard():assert recovery.n.verify_parent()=={'source':{'verified':True}}
    assert calls==[True]

def test_bootstrap_constant_crossed():
    r=bootstrap(np.full((3,24),7.));assert r['mean_USD']==7 and r['CI95']==[7,7] and r['CI_adjusted']==[7,7]
def test_mc_does_not_invent_training_seeds():
    r=bootstrap(np.arange(24)[None,:]);assert r['crossed_models']==1 and r['scenarios']==24
def test_joint_uncertainty_contains_seed_variation():
    x=np.vstack([np.zeros(24),np.ones(24)*10,np.ones(24)*20]);r=bootstrap(x)
    assert r['CI95'][0]<10<r['CI95'][1] and r['mean_USD']==10
def test_recovery_reuses_original_numeric_source():
    assert 'experiments/cfs_direct_ppo_diagnosis/numerics.py' in recovery.sources()
    assert recovery.TARGET!=recovery.OLD
