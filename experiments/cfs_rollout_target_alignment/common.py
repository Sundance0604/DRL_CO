"""Read-only historical provenance and isolated fifth-study outputs."""
import os, platform, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
from experiment_core.storage import atomic_json, read_json, digest
from experiments.cfs_classical_value.common import ROOT, key, parameters, map_jobs
from experiments.cfs_dynamic_lookahead.common import freeze_data, rho_levels
from experiments.cfs_local_post_bayesian_value.common import write_csv
from experiments.cfs_rho_endofday.run_experiment import file_hash

OUTPUT = ROOT / 'experiments/cfs_rollout_target_alignment'
CONFIG = OUTPUT / 'config/main.json'
FOURTH = ROOT / 'experiments/cfs_local_post_bayesian_value'

def now(): return datetime.now(timezone.utc).isoformat()

def config_defaults():
    old = read_json(FOURTH/'config/main.json')
    return {k:old[k] for k in ['baseline_config','conditions','placement_rule',
        'target_supply_slots','mc_paths_per_state','mc_donor_offset','degrees',
        'ridge_penalties','spline_internal_knots','spline_penalties','max_parallel',
        'run_timeout_seconds','chebyshev_node_count']} | dict(
        schema_version='cfs-target-alignment/v1',phase1_seeds=list(range(6000,6020)),
        construction_seeds=list(range(6200,6240)),validation_seeds=list(range(6300,6315)),
        formal_seeds=list(range(7000,7040)),heldout_value_seeds=list(range(7000,7012)),
        query_times=[3,8,12,16,20],max_candidates_per_vehicle=8,
        phase1_condition=dict(region='western_multicluster',orders=40,fleet=8),
        continuation_policy='myopic-zero',policy_seed=0,training_seed=None,
        bootstrap_seed=20261007,bootstrap_replications=10000,tie_tolerance_usd=1e-6,
        control_contrast='same forced current joint action; freeze t+1 until candidate release versus rolling t+1 onwards',
        estimand='(J_candidate-g_candidate)-(J_baseline-g_baseline)',
        historical_label='Separate fourth-style release resource marginal J(q)-J(q-1); never used for primary gate',
        user_approved_estimand_alignment=True,
        gating=dict(T0_spearman_min=.95,T0_pair_flip_max=.05,T0_top1_max=.05,
                    T0_sign_flip_max=.05,T1_spearman_below=.90,T1_pair_flip_min=.10,
                    T1_top1_min=.10,T1_sign_flip_min=.10,
                    subgroup_min_scenarios=10,subgroup_strict_inversion_min=.10,
                    duration_association_min=.30,duration_positive_cluster_ci=True,
                    gray_zone='exploratory Phase 2 if not T0 or T1; never claim confirmatory significance'),
        scope='No DRL, no neural, no new V families, no route-slot coupling changes; no RH4+V',
        phase2_residual='Y_roll minus original absolute slot base; local additive/q-average approximation retained, not exact joint action value')

def source_hashes():
    old = read_json(FOURTH/'manifests/source_manifest.json')
    h={name:file_hash(ROOT/name) for name in old['source_code_sha256']}
    for p in OUTPUT.rglob('*.py'):
        h[p.relative_to(ROOT).as_posix()]=file_hash(p)
    for p in (ROOT/'tests').glob('test_rollout_target_alignment*.py'):
        h[p.relative_to(ROOT).as_posix()]=file_hash(p)
    h['doc/main_v2.tex']=file_hash(ROOT/'doc/main_v2.tex')
    return h

def historical_index():
    """Hash every old study artifact, including raw observations, without writing it."""
    folders=['experiments/cfs_rho_endofday','experiments/cfs_dynamic_lookahead',
        'experiments/cfs_classical_value','experiments/cfs_local_post_bayesian_value',
        'results/cfs_rho_endofday','results/cfs_dynamic_lookahead','results/cfs_classical_value',
        'reports/cfs_rho_endofday','reports/cfs_dynamic_lookahead','reports/cfs_classical_value']
    h={}
    for folder in folders:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                h[p.relative_to(ROOT).as_posix()]=file_hash(p)
    fourth=read_json(FOURTH/'manifests/source_manifest.json')
    for name,sha in fourth['source_code_sha256'].items():
        assert file_hash(ROOT/name)==sha, f'frozen numerical source changed: {name}'
        h[name]=sha
    final=read_json(FOURTH/'manifests/final_manifest.json')
    assert final['status']=='COMPLETED'
    for name,sha in final.get('deliverables',{}).items():
        assert file_hash(FOURTH/name)==sha, f'fourth deliverable changed: {name}'
    return h

def initialize():
    assert os.environ.get('PYTHONHASHSEED')=='0','Set PYTHONHASHSEED=0 before launch'
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for d in ['config','manifests','raw','qa','processed','targets','models','figures']:
        (OUTPUT/d).mkdir(exist_ok=True)
    if not CONFIG.exists(): atomic_json(CONFIG,config_defaults())
    cfg=read_json(CONFIG);base=read_json(ROOT/cfg['baseline_config'])
    if (OUTPUT/'manifest.json').exists():
        m=read_json(OUTPUT/'manifest.json')
        assert m['config_hash']==digest(cfg),'Frozen fifth config changed'
        # Numerical identity is frozen after tests/toys, before any query target.
        return cfg,base,m
    print('Freezing read-only historical files and CFS identity...',flush=True)
    historical=historical_index()
    source=read_json(FOURTH/'manifest.json')['source']
    assert file_hash(Path(base['data_source']))==source['sha256']
    atomic_json(OUTPUT/'manifests/historical_readonly.json',dict(files=historical,created_at=now()))
    m=dict(schema_version=cfg['schema_version'],config_hash=digest(cfg),source=source,
        config=cfg,baseline_config=base,status='PHASE0',created_at=now(),python=platform.python_version(),
        policy_seed=0,training_seed=None,hash_seed=0,historical_index_hash=file_hash(OUTPUT/'manifests/historical_readonly.json'),
        primary_definition_approved_by_user=True,main_model_document='doc/main_v2.tex',
        terminology_note='No file literally named main_v2_DRL exists in this checkout; frozen single main_v2.tex and engine were inspected.')
    atomic_json(OUTPUT/'manifest.json',m)
    return cfg,base,m

def freeze_numerical_source(m):
    if 'code_hash' in m:
        h={n:file_hash(ROOT/n) for n in m['source_code_sha256']}
        assert m['code_hash']==digest(h),'Numerical code changed after target freeze; do not reuse outcomes'
        return
    h=source_hashes();sha=digest(h)
    p=OUTPUT/'raw/source.snapshot.zip'
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for name in h:z.write(ROOT/name,name)
        z.write(CONFIG,'fifth-config.json')
    m.update(code_hash=sha,source_code_sha256=h,source_snapshot_sha256=file_hash(p),source_frozen_at=now())
    atomic_json(OUTPUT/'manifests/source_manifest.json',m)
    atomic_json(OUTPUT/'manifest.json',m)

def datasets(cfg,base,seeds,conditions):
    out={}
    for c in conditions:
        for seed in seeds:
            p=OUTPUT/'raw/datasets'/f'{key(c)}-s{seed}.dataset.json'
            if p.exists():out[c['region'],c['orders'],c['fleet'],seed]=read_json(p)
    if len(out)==len(conditions)*len(seeds):return out
    return freeze_data(cfg,base,OUTPUT,conditions,seeds)

def baseline_regression(cfg):
    p=OUTPUT/'qa/baseline_regression.json'
    if p.exists():assert read_json(p)['status']=='PASSED';return read_json(p)
    from experiments.cfs_local_post_bayesian_value.baseline_regression import job
    tasks=[(key(c),label,policy) for c in cfg['conditions'] for label,policy in
        [('P0-Myopic','myopic'),('P6-RH4','RH4'),('P10-RH8','RH8')]]
    rows,wall=map_jobs(job,tasks,'Fifth baseline reproduction',cfg['max_parallel'])
    result=dict(status='PASSED',runs=len(rows),rows=sorted(rows,key=lambda r:(r['condition'],r['policy'])),wall_seconds=wall)
    atomic_json(p,result);return result

def verify_historical():
    old=read_json(OUTPUT/'manifests/historical_readonly.json')['files']
    bad=[n for n,sha in old.items() if not (ROOT/n).exists() or file_hash(ROOT/n)!=sha]
    assert not bad,f'Historical files modified: {bad[:5]}'
    result=dict(status='PASSED',files=len(old),checked_at=now())
    atomic_json(OUTPUT/'qa/historical_integrity.json',result);return result
