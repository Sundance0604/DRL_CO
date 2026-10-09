"""Configuration, immutable provenance and content-addressed experiment storage."""
import csv
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import time
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from experiment_core.storage import atomic_json, read_json, digest
from experiments.cfs_resource_policy_iteration.common import parameters, rho_levels, nodes, key
from experiments.cfs_resource_value_state_bucket_exact_accounting.common import OUTPUT as B
from experiments.cfs_resource_policy_iteration_damping.common import OUTPUT as A2, HISTORY as A
from experiments.cfs_dynamic_lookahead.common import freeze_data

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=Path(__file__).resolve().parent
TASK=Path('D:/download/Codex_第七轮_联合动作价值接入诊断与独立MC复核.md')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def config():
    old=read_json(B/'config/main.json')
    return dict(schema_version='joint-action-independent-mc/v1',baseline_config=old['baseline_config'],
        conditions=old['conditions'][:3],placement_rule=old['placement_rule'],
        pilot_seeds=list(range(10000,10004)),diagnostic_seeds=list(range(10100,10120)),
        formal_seeds=list(range(10200,10240)),bank_seeds=[list(range(a,a+60)) for a in (10400,10500,10600,10700)],
        deployment_seeds=list(range(10800,10840)),query_times=[3,8,12,16,20],sources=['M0','pi1'],
        max_candidates=6,selection_paths=32,nested_selection_paths=8,assessment_paths=64,
        interaction_paths=64,maximum_interaction_pairs=2,sensitivity_seeds=list(range(10100,10104)),
        fixed_evaluation_seeds=list(range(10100,10104)),fixed_evaluation_paths=64,
        bank_nodes=7,bank_supply_slots=old['target_supply_slots'],bank_hub_sampling='A rotating hub rule, unchanged for all four banks',
        max_parallel=3,policy_seed=0,training_seed=None,bootstrap_replications=10000,bootstrap_seed=20261008,
        future_distribution='surrogate-distribution rollout: empirical construction marked Poisson arrivals',
        future_rule='rate estimated ONLY from B construction 8300:8359; future count Poisson(rate*(T-t-1)); '
            'iid construction OD/load/fare marks; booking uniform t+1:T-1; ready/slack inherited distributions; '
            'does NOT condition on hidden remaining shipments, actual OD quotas or remaining count',
        future_namespace='sha256(condition,state_id,role,replicate); candidate-independent CRN',
        price_reference='pi1=A2 U1=G(A V0); NOT B1 whose construction hub sampling differs',
        candidates=['M0','pi1','D4_alpha025','B2_exact','up to two deterministic original H1 MILP constrained-route alternatives'],
        primary='six two-sided Holm tests: three conditions x two state sources, select32 minus pi1-price selector',
        secondary=['select32-M0','select32-select8','select32-B2-price','pi1 continuation sensitivity',
            'current-free-resource interactions','four independent banks','fixed-state pi0/pi1/pi3 marginals'],
        intervals='scenario paired t95 and 10000 bootstrap; unadjusted intervals, Holm p separate',
        tie_rule='scores within 1e-8 USD; prefer base action then physical action hash, all tie sets retained',
        semantic_correction='NONE preregistered: historical relative buckets and q proxy are preserved controls',
        cross_time_injection='NOT_IMPLEMENTED: only current FREE resources; no conversion of committed arrival into free supply',
        wall_budget_hours=72,storage_budget_GiB=200,staged_priority=['CA/W40 joint','W80 joint','four banks'],
        pilot_decision='fixed 64 assessment retained unless a uniform pre-formal amendment is separately frozen; no formal extension')

def numerical_hashes():
    inherited=read_json(B/'manifests/source_manifest.json')['source_code_sha256']
    out=dict(inherited)
    for name,h in inherited.items():assert sha(ROOT/name)==h,name
    for p in (ROOT/'model_families/single').rglob('*.py'):out[p.relative_to(ROOT).as_posix()]=sha(p)
    for family in ['cfs_rollout_target_alignment','cfs_resource_policy_iteration','cfs_resource_policy_iteration_damping',
                   'cfs_resource_value_state_bucket_exact_accounting','cfs_classical_value','cfs_dynamic_lookahead']:
        for p in (ROOT/'experiments'/family).glob('*.py'):
            out[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in OUTPUT.glob('*.py'):
        if p.name not in ['analysis.py','figures.py','delivery.py']:out[p.relative_to(ROOT).as_posix()]=sha(p)
    p=ROOT/'tests/test_joint_action_value_diagnostic.py'
    out[p.relative_to(ROOT).as_posix()]=sha(p)
    return out

def gate():
    proofs={}
    for folder,expected in [(A2,'1e35175f351179cd8c459fee7950be618045e1c5b14d6147ef06c3876255c803'),
                            (B,'e6ceef1754ace83ba350e1fa3735293efd96bac28ed0b6cf5f1218065b17a9cf')]:
        p=folder/'manifests/final_manifest.json';assert sha(p)==expected
        m=read_json(p);assert m['status']=='COMPLETED'
        for name,h in m['deliverable_sha256'].items():assert sha(folder/name)==h,str(folder/name)
        proofs[p.relative_to(ROOT).as_posix()]=expected
    failed=ROOT/'experiments/cfs_resource_value_state_bucket/qa/stop_condition_failure.json'
    assert read_json(failed)['status']=='STOP_CONDITION_TRIGGERED'
    proofs[failed.relative_to(ROOT).as_posix()]=sha(failed)
    return proofs

def initialize():
    for name,val in dict(PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1').items():
        assert os.environ.get(name)==val,(name,os.environ.get(name))
    cfg=config();base=read_json(ROOT/cfg['baseline_config']);p=OUTPUT/'manifest.json'
    if p.exists():
        m=read_json(p);assert m['config_hash']==digest(cfg)
        assert m['code_hash']==digest(numerical_hashes()),'Frozen numerical source changed'
        assert m['status']!='COMPLETED','COMPLETED is immutable; no rerun'
        gate();return cfg,base,m
    for d in ['config','manifests','raw/states','raw/actions','raw/futures','raw/worlds','raw/banks','processed','qa','figures']:
        (OUTPUT/d).mkdir(parents=True,exist_ok=True)
    proposed=set(cfg['pilot_seeds']+cfg['diagnostic_seeds']+cfg['formal_seeds']+cfg['deployment_seeds']+
                 [s for bank in cfg['bank_seeds'] for s in bank])
    historical_seeds=set()
    for family in (ROOT/'experiments').iterdir():
        if family==OUTPUT or not family.is_dir():continue
        for p0 in [family/'config/main.json',family/'configs/main.json']:
            if p0.exists():
                obj=read_json(p0)
                for k,v in obj.items():
                    if 'seed' in k and isinstance(v,list):historical_seeds.update(x for x in v if isinstance(x,int))
    assert not proposed&historical_seeds,sorted(proposed&historical_seeds)
    source=read_json(B/'manifest.json')['source']
    print('Checking original CFS SHA256 (read only)...',flush=True)
    assert sha(base['data_source'])==source['sha256'],'CFS source changed'
    m=dict(status='INITIALIZED',created_at=now(),config_hash=digest(cfg),source=source,
        historical_final_hashes=gate(),source_code_sha256=numerical_hashes(),git_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_status=subprocess.check_output(
        ['git','status','--porcelain','--untracked-files=normal'],cwd=ROOT,text=True),dirty_worktree=True,
        environment=dict(python=platform.python_version(),executable=os.sys.executable,platform=platform.platform(),
            packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','gurobipy','matplotlib']}))
    m['code_hash']=digest(m['source_code_sha256'])
    atomic_json(OUTPUT/'config/main.json',cfg)
    atomic_json(OUTPUT/'manifests/seed_collision_audit.json',dict(status='PASSED',proposed=sorted(proposed),
        historical_config_seeds=sorted(historical_seeds),note='No reuse of historical numerical scenarios'))
    with zipfile.ZipFile(OUTPUT/'raw/source.snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in m['source_code_sha256']:z.write(ROOT/name,name)
        z.write(TASK,'task.md');z.write(OUTPUT/'config/main.json','config.json')
    m.update(task_sha256=sha(TASK),snapshot_sha256=sha(OUTPUT/'raw/source.snapshot.zip'))
    atomic_json(OUTPUT/'manifest.json',m);atomic_json(OUTPUT/'manifests/source_manifest.json',m)
    atomic_json(OUTPUT/'manifests/analysis_preregistration.json',dict(created_at=now(),formal_seen=False,
        diagnostic_seen=False,config=cfg,analysis_sha256=sha(OUTPUT/'analysis.py')))
    return cfg,base,m

def datasets(cfg,base,seeds):
    paths=[OUTPUT/'raw/datasets'/f'{key(c)}-s{s}.dataset.json' for c in cfg['conditions'] for s in seeds]
    if all(p.exists() for p in paths):
        return {(c['region'],c['orders'],c['fleet'],s):read_json(OUTPUT/'raw/datasets'/f'{key(c)}-s{s}.dataset.json')
                for c in cfg['conditions'] for s in seeds}
    return freeze_data(cfg,base,OUTPUT,cfg['conditions'],seeds)

def csv_output(name,rows):
    assert rows,'Do not create empty placeholder CSVs'
    p=OUTPUT/'processed'/f'{name}.csv';columns=list(dict.fromkeys(k for row in rows for k in row))
    import json
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    atomic_json(p.with_suffix('.json'),rows)

def parallel(fn,jobs,label,workers=3):
    started=time.perf_counter();items=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        from concurrent.futures import as_completed
        fs={pool.submit(fn,j):j for j in jobs}
        for i,f in enumerate(as_completed(fs),1):
            try:items.append(f.result())
            except Exception as e:
                atomic_json(OUTPUT/'qa/stop_condition_failure.json',dict(status='STOP_CONDITION_TRIGGERED',
                    label=label,job=fs[f],error=repr(e),at=now()));raise
            if i%5==0 or i==len(jobs):print(f'{label}: {i}/{len(jobs)}',flush=True)
    return items,time.perf_counter()-started
