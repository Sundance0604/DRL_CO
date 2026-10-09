"""Outcome-separation checks use disposable fixtures, never research outputs."""
import pytest
from experiment_core.storage import atomic_json,digest
from experiments.cfs_resource_policy_iteration_damping import analysis
from experiments.cfs_resource_policy_iteration_damping.common import configuration,file_hash

def test_formal_before_alpha_selection_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'OUTPUT',tmp_path)
    atomic_json(tmp_path/'processed/formal_runs.json',[])
    with pytest.raises(AssertionError):analysis.select_alpha(configuration())

def frozen(tmp_path,cfg):
    diagnostic=tmp_path/'processed/diagnostic_runs.json';atomic_json(diagnostic,[])
    atomic_json(tmp_path/'manifests/alpha_freeze.json',dict(selected_alpha='050',
        selection_config_hash=digest(cfg['selection']),diagnostic_results_hash=file_hash(diagnostic)))
    return diagnostic

def test_frozen_alpha_is_never_reselected_after_formal(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'OUTPUT',tmp_path);cfg=configuration();frozen(tmp_path,cfg)
    atomic_json(tmp_path/'processed/formal_runs.json',[dict(profit=999999999.)])
    def forbidden(*args):raise RuntimeError('must not revisit diagnostic or formal ranking')
    monkeypatch.setattr(analysis,'table_metrics',forbidden)
    assert analysis.select_alpha(cfg)=='050'

def test_altered_diagnostic_cache_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'OUTPUT',tmp_path);cfg=configuration();path=frozen(tmp_path,cfg)
    atomic_json(path,[dict(changed=True)])
    with pytest.raises(AssertionError):analysis.select_alpha(cfg)

def test_shared_U_policy_and_D_alias_are_not_new_scenarios():
    lookup={('c',9100,'U1',None):dict(scenario_seed=9100,profit=10),
        ('c',9100,'D1','050'):dict(scenario_seed=9100,profit=10,alias_of='U1')}
    assert analysis.run_for(lookup,'c',9100,'U1','075')==lookup['c',9100,'U1',None]
    assert analysis.run_for(lookup,'c',9100,'D1','050')['scenario_seed']==9100
