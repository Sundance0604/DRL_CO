"""Frozen protocol, independent split identity, immutable historical inputs."""
import os, time, zipfile, platform
from datetime import datetime, timezone
from pathlib import Path
from experiment_core.storage import atomic_json, read_json, digest
from experiments.cfs_classical_value.common import ROOT, key, parameters, nodes, map_jobs
from experiments.cfs_dynamic_lookahead.common import freeze_data, rho_levels
from experiments.cfs_local_post_bayesian_value.common import write_csv
from experiments.cfs_rho_endofday.run_experiment import file_hash

OUTPUT=ROOT/'experiments/cfs_resource_policy_iteration'
FIFTH=ROOT/'experiments/cfs_rollout_target_alignment'
INTERPOLATION=dict(time='piecewise linear; endpoint constant below first node',
    q='piecewise linear at integer q; endpoint constant beyond measured K',
    counterfactual_max_q='2K: retained nonqueried resources plus inserted queried supply',
    negative='retain',monotonic='no enforcement',missing_hub_time='fail')
TERMINAL=dict(free_resource='exact zero at time>=T',commitments='full physical cooldown; no new controls')
COMPARISONS=[('P1','P0'),('P2','P1'),('P3','P2'),('P4','P3'),
    ('P2','P0'),('P3','P0'),('P4','P0'),('P4','R0'),('P4','R1'),('D','P4')]

def now():return datetime.now(timezone.utc).isoformat()

def defaults():
    old=read_json(FIFTH/'config/main.json')
    return {k:old[k] for k in ['baseline_config','conditions','placement_rule','target_supply_slots',
        'chebyshev_node_count','max_parallel','run_timeout_seconds']}|dict(
        schema_version='cfs-resource-policy-iteration/v1',construction_seeds=list(range(8000,8060)),
        diagnostic_seeds=list(range(8100,8120)),pilot_seeds=[8900,8901],formal_seeds=list(range(9000,9040)),
        iterations=4,policy_seed=0,training_seed=None,bootstrap_seed=20261007,bootstrap_replications=10000,
        paths_per_source=1,future_sampling='own scenario only; no shared donor; same CRN scenario across iterations',
        state_sampling='each iteration recollect current parent-policy on-policy states; fixed seed/time/hub grid',
        action_comparison='pi0 diagnostic state bank at ALL t; adjacent policies solved on identical begun state',
        damping='optional diagnostic not run in this fixed undamped protocol; no alpha selection',
        convergence=dict(relative_table_l2=.05,action_disagreement=.05,mean_abs_SMD=.1,
            paired_gain_fraction=.01,steps=2,note='descriptive joint criteria, not a convergence theorem'),
        interpolation=INTERPOLATION,terminal=TERMINAL,comparisons=COMPARISONS,
        scientific_scope='approximate separable resource-value policy iteration; no action correction, RL or oracle evaluation')

def numerical_hashes():
    old=read_json(FIFTH/'manifests/source_manifest.json')
    hashes={n:file_hash(ROOT/n) for n in old['source_code_sha256']}
    phase2=read_json(FIFTH/'manifests/phase2_source_manifest.json')
    for name,sha in phase2['files'].items():
        assert file_hash(ROOT/name)==sha
        hashes[name]=sha
    for name in ['common.py','protocol.py','simulation.py','targets.py','policies.py','phase0.py','run_experiment.py']:
        p=OUTPUT/name
        if p.exists():hashes[p.relative_to(ROOT).as_posix()]=file_hash(p)
    p=ROOT/'tests/test_resource_policy_iteration.py'
    if p.exists():hashes[p.relative_to(ROOT).as_posix()]=file_hash(p)
    return hashes

def initialize():
    assert os.environ.get('PYTHONHASHSEED')=='0'
    for d in ['config','manifests','targets','tables','policies','raw','processed','qa','figures']:
        (OUTPUT/d).mkdir(parents=True,exist_ok=True)
    path=OUTPUT/'config/main.json'
    if not path.exists():atomic_json(path,defaults())
    cfg=read_json(path);base=read_json(ROOT/cfg['baseline_config'])
    splits=[set(cfg[k]) for k in ['construction_seeds','diagnostic_seeds','pilot_seeds','formal_seeds']]
    assert not any(a&b for i,a in enumerate(splits) for b in splits[:i])
    if (OUTPUT/'manifest.json').exists():
        m=read_json(OUTPUT/'manifest.json');assert m['config_hash']==digest(cfg)
        return cfg,base,m
    archive=OUTPUT/'raw/failed_attempts/timing-v1/manifest.json'
    if archive.exists() and (OUTPUT/'manifests/historical_readonly.json').exists():
        old=read_json(archive);assert old['config_hash']==digest(cfg)
        m={k:old[k] for k in ['config','config_hash','source','baseline_config','python','historical_files','policy_seed','training_seed']}
        m.update(status='PHASE0',created_at=now(),prior_attempt='raw/failed_attempts/timing-v1',
            repair='observational disjoint timers only; frozen physics and decision statements; no formal outcomes seen',
            reused_history_index_sha256=file_hash(OUTPUT/'manifests/historical_readonly.json'))
        atomic_json(OUTPUT/'manifest.json',m)
        return cfg,base,m
    fifth=read_json(FIFTH/'manifests/final_manifest.json');assert fifth['status']=='COMPLETED'
    print('Freeze previous five studies and read-only CFS source...',flush=True)
    from experiments.cfs_rollout_target_alignment.common import historical_index
    historical=historical_index()
    for p in sorted(FIFTH.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:historical[p.relative_to(ROOT).as_posix()]=file_hash(p)
    atomic_json(OUTPUT/'manifests/historical_readonly.json',dict(files=historical,created_at=now()))
    source=read_json(FIFTH/'manifest.json')['source'];assert file_hash(Path(base['data_source']))==source['sha256']
    m=dict(status='PHASE0',config=cfg,config_hash=digest(cfg),source=source,baseline_config=base,
        created_at=now(),python=platform.python_version(),historical_files=len(historical),policy_seed=0,training_seed=None)
    atomic_json(OUTPUT/'manifest.json',m)
    for filename,payload in [('base.yaml',base),('construction.yaml',cfg),('formal.yaml',dict(
        seeds=cfg['formal_seeds'],conditions=cfg['conditions'],strategies=['P0','P1','P2','P3','P4','R0','R1','D'],
        comparisons=COMPARISONS,sample_unit='paired independent scenario',Holm_family='36 main comparisons across four conditions; Oracle gap separate descriptive',
        created_at=now(),formal_seen=False))]:
        # JSON is a valid YAML 1.2 subset; no additional dependency is required.
        atomic_json(OUTPUT/'config'/filename,payload)
    atomic_json(OUTPUT/'manifests/formal_manifest.json',dict(status='PREREGISTERED',created_at=now(),formal_seen=False,
        protocol=cfg,formal_config_hash=file_hash(OUTPUT/'config/formal.yaml')))
    return cfg,base,m

def freeze(m):
    for olddir in [ROOT/'experiments/cfs_local_post_bayesian_value',FIFTH]:
        final=read_json(olddir/'manifests/final_manifest.json')
        for name,sha in final['deliverable_sha256'].items():
            assert file_hash(olddir/name)==sha,f'Historical final deliverable changed: {name}'
    hashes=numerical_hashes()
    if 'code_hash' in m:
        assert hashes==m['source_code_sha256'],'Numerical code changed: archive attempt before rerun'
        return
    snapshot=OUTPUT/'raw/source.snapshot.zip'
    task=Path('D:/download/Codex_第六次数值实验A_SelfConsistent_MC_Policy_Iteration.md')
    with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:z.write(ROOT/name,name)
        z.write(OUTPUT/'config/main.json','sixth-config.json')
        z.write(task,'sixth-task-document.md')
    m.update(code_hash=digest(hashes),source_code_sha256=hashes,source_frozen_at=now(),snapshot_sha256=file_hash(snapshot),
        task_document_sha256=file_hash(task))
    atomic_json(OUTPUT/'manifests/source_manifest.json',m);atomic_json(OUTPUT/'manifest.json',m)

def datasets(cfg,base,seeds):
    data={}
    for c in cfg['conditions']:
        for seed in seeds:
            p=OUTPUT/'raw/datasets'/f'{key(c)}-s{seed}.dataset.json'
            if p.exists():data[c['region'],c['orders'],c['fleet'],seed]=read_json(p)
    if len(data)==len(cfg['conditions'])*len(seeds):return data
    return freeze_data(cfg,base,OUTPUT,cfg['conditions'],seeds)

def verify_history():
    h=read_json(OUTPUT/'manifests/historical_readonly.json')['files']
    bad=[n for n,v in h.items() if not (ROOT/n).exists() or file_hash(ROOT/n)!=v]
    assert not bad,bad[:5]
    base=read_json(OUTPUT/'manifest.json');assert file_hash(Path(base['baseline_config']['data_source']))==base['source']['sha256']
    result=dict(status='PASSED',files=len(h),CFS_sha256_verified=True,checked_at=now())
    atomic_json(OUTPUT/'qa/historical_integrity.json',result);return result
