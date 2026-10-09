"""Fourth-study identity and isolated output paths."""
import os, platform, zipfile, json
from datetime import datetime, timezone
from pathlib import Path
from experiment_core.storage import atomic_json, read_json, digest
from experiments.cfs_classical_value.common import ROOT, key, parameters, nodes, map_jobs, PROTECTED
from experiments.cfs_dynamic_lookahead.common import freeze_data, rho_levels
from experiments.cfs_rho_endofday.run_experiment import file_hash, write_csv as _write_csv

OUTPUT=ROOT/'experiments/cfs_local_post_bayesian_value'
CONFIG=OUTPUT/'config/main.json'
OLD=ROOT/'results/cfs_classical_value'

def write_csv(path,rows):
    """Typed scalar columns; structured provenance is valid JSON, never Python repr."""
    _write_csv(path,[{k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v
                      for k,v in r.items()} for r in rows])

def now(): return datetime.now(timezone.utc).isoformat()

def hashes():
    old=read_json(OLD/'manifest.json')['source_code_sha256']
    result={name:file_hash(ROOT/name) for name in old}
    for directory in [OUTPUT,ROOT/'model_families/single/local_value']:
        for p in directory.rglob('*.py'):
            if p.name=='analysis.py': continue
            result[p.relative_to(ROOT).as_posix()]=file_hash(p)
    result['tests/test_local_post_bayesian.py']=file_hash(ROOT/'tests/test_local_post_bayesian.py')
    return result

def initialize():
    assert os.environ.get('PYTHONHASHSEED')=='0'
    cfg=read_json(CONFIG); base=read_json(ROOT/cfg['baseline_config'])
    assert read_json(OUTPUT/'manifests/baseline_regression.json')['status']=='PASSED'
    old=read_json(OLD/'manifest.json')
    for name,sha in old['source_code_sha256'].items():
        assert file_hash(ROOT/name)==sha, f'historical numerical source changed: {name}'
    for name,sha in read_json(OLD/'manifests/model_freeze.json')['models'].items():
        assert file_hash(OLD/'models'/name)==sha
    for d in ['config','manifests','targets','models','raw','processed','figures','qa']:
        (OUTPUT/d).mkdir(exist_ok=True)
    h=hashes(); identity=dict(config_hash=digest(cfg),code_hash=digest(h))
    if (OUTPUT/'manifest.json').exists():
        m=read_json(OUTPUT/'manifest.json'); assert all(m[k]==v for k,v in identity.items())
        return cfg,base,m
    assert file_hash(Path(base['data_source']))==old['source']['sha256']
    snapshot=OUTPUT/'raw/source.snapshot.zip'
    with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as z:
        for name in h: z.write(ROOT/name,name)
        z.write(CONFIG,'fourth-config.json')
    m=dict(**identity,source_code_sha256=h,source=old['source'],config=cfg,baseline_config=base,
           created_at=now(),python=platform.python_version(),status='INITIALIZED',policy_seed=0,
           training_seed=None,hash_seed=0,source_snapshot_sha256=file_hash(snapshot),
           historical_manifest_sha256=file_hash(OLD/'manifest.json'),
           historical_model_freeze_sha256=file_hash(OLD/'manifests/model_freeze.json'))
    atomic_json(OUTPUT/'manifest.json',m); atomic_json(OUTPUT/'manifests/source_manifest.json',m)
    return cfg,base,m

def datasets(cfg,base,seeds):
    out={}
    for c in cfg['conditions']:
        for seed in seeds:
            p=OUTPUT/'raw/datasets'/f'{key(c)}-s{seed}.dataset.json'
            if p.exists(): out[c['region'],c['orders'],c['fleet'],seed]=read_json(p)
    if len(out)==len(cfg['conditions'])*len(seeds): return out
    return freeze_data(cfg,base,OUTPUT,cfg['conditions'],seeds)
