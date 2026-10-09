"""B has separate splits, models, lineage and an explicit A2-completion gate."""
import os
import zipfile
from pathlib import Path
from experiment_core.storage import atomic_json,read_json,digest
from experiments.cfs_resource_policy_iteration_damping.common import (
    ROOT,HISTORY,OUTPUT as A2,parameters,rho_levels,nodes,map_jobs,key,write_csv,now,file_hash,INTERPOLATION,TERMINAL)
from experiments.cfs_dynamic_lookahead.common import freeze_data

OUTPUT=ROOT/'experiments/cfs_resource_value_state_bucket_exact_accounting'
FAILED_B=ROOT/'experiments/cfs_resource_value_state_bucket'

def failed_attempt_gate():
    """Verify the stopped attempt without restarting or modifying its files."""
    stop=read_json(FAILED_B/'qa/stop_condition_failure.json')
    assert stop['status']=='STOP_CONDITION_TRIGGERED'
    assert file_hash(FAILED_B/'manifest.json')==stop['manifest_sha256_at_stop']
    source=read_json(FAILED_B/'manifests/source_manifest.json')
    assert source['code_hash']==stop['numerical_code_hash']
    for name,sha in source['source_code_sha256'].items():assert file_hash(ROOT/name)==sha,name
    assert file_hash(FAILED_B/'analysis.py')==stop['analysis_source_sha256']
    assert file_hash(A2/'manifests/final_manifest.json')==stop['A2_final_manifest_sha256']
    return dict(original_directory=FAILED_B.relative_to(ROOT).as_posix(),
        original_stop_sha256=file_hash(FAILED_B/'qa/stop_condition_failure.json'),
        original_manifest_sha256=stop['manifest_sha256_at_stop'],
        original_code_hash=stop['numerical_code_hash'],original_analysis_sha256=stop['analysis_source_sha256'],
        original_attempt_readonly=True,approved_change='Independent integer-action cash compilation; original actions and solver diagnostics preserved',
        authorization='User reply: 好; approved separate attempt, not rewriting frozen B',
        full_B_rerun_no_old_numerical_cache=True)

def configuration():
    cfg=read_json(HISTORY/'config/main.json')
    cfg.update(schema_version='cfs-minimal-state-bucket-exact-accounting/v1',
        construction_seeds=list(range(8300,8360)),diagnostic_seeds=list(range(8400,8420)),
        pilot_seeds=[8960,8961],formal_seeds=list(range(9200,9240)),iterations=1,
        damping=None,parent='M0 for BOTH base and bucket targets',Hb=4,eta=1.,
        buckets=3,minimum_support=10,threshold_quantiles=[1/3,2/3],
        target_hub_sampling='ALL five hubs at the SAME seven A time nodes, BOTH base and bucket; '
            '60 independent construction scenarios per raw tuq before splitting into b',
        unit='Bload and Dnear in passenger/capacity vehicle equivalents; supply in vehicles',
        pressure='Bready + (known booked near-ready + construction expected unbooked near-ready) over idle+known near incoming+1',
        bucket_boundary='R<=c33 surplus; c33<R<=c67 balanced; R>c67 scarce; ties retained and reported',
        online_bucket='current observation hub bucket prices future release slot; no candidate-dependent post-state',
        fallback='ANY active nonterminal interpolation source corner count<10 OR q>K: use SAME base table at query; record all calls/reasons',
        effectiveness=dict(key_conditions=['CA80/3','W40/8','W80/3'],majority_positive=2,
            problematic_gain_fraction=.01,W80_noninferiority_margin_fraction=.01,
            maximum_formal_fallback_fraction=.5,
            within_variance='construction decomposition descriptive; independent diagnostic target '
                'variance corroboration required, not automatic train-group variance decrease',
            state_aliasing_evidence='positive formal/diagnostic paired gains, correct Holm support in CA or W40, '
                'W80 noninferiority, variance corroboration and adequate support; unmet threshold is inconclusive'),
        comparisons=[['B2','B1'],['B1','B0'],['B2','B0'],['B2','R0'],['B2','R1'],['D','B2']],
        scientific_scope='one M0 parent policy improvement, no bucket PI, no damping, no ACTION correction, no RL')
    cfg['accounting_amendment']=dict(attempt='independent exact-accounting B',
        selected_actions='unchanged original solver extraction; independent compilation verifies equality',
        cash='recomputed from actual integer actions, no relaxed guards or solver tolerances',
        diagnostics='raw floating solver profit/objective/accounting retained',
        previous_attempt_readonly=FAILED_B.relative_to(ROOT).as_posix())
    return cfg

def code_hashes():
    old=read_json(HISTORY/'manifests/source_manifest.json')['source_code_sha256'];out=dict(old)
    for name,sha in old.items():assert file_hash(ROOT/name)==sha,name
    # B directly imports readonly path/utility adapters from A2 common.py.
    # Include that actual dependency in B's reproducible numerical snapshot.
    a2common=A2/'common.py';out[a2common.relative_to(ROOT).as_posix()]=file_hash(a2common)
    for name in ['analysis.py','action_analysis.py']:
        helper=HISTORY/name;out[helper.relative_to(ROOT).as_posix()]=file_hash(helper)
    for p in OUTPUT.glob('*.py'):
        if p.name not in ['analysis.py','figures.py','delivery.py']:
            out[p.relative_to(ROOT).as_posix()]=file_hash(p)
    p=ROOT/'tests/test_resource_value_state_bucket_exact_accounting.py';out[p.relative_to(ROOT).as_posix()]=file_hash(p)
    p=ROOT/'tests/test_integer_action_accounting.py';out[p.relative_to(ROOT).as_posix()]=file_hash(p)
    return out

def initialize():
    a2final=read_json(A2/'manifests/final_manifest.json')
    assert a2final['status']=='COMPLETED','B numerical execution requires final frozen A2'
    amendment=failed_attempt_gate()
    for name,sha in a2final['source_code_sha256'].items():assert file_hash(ROOT/name)==sha,name
    assert os.environ.get('PYTHONHASHSEED')=='0' and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
    cfg=configuration();base=read_json(ROOT/cfg['baseline_config']);path=OUTPUT/'manifest.json'
    if path.exists():
        m=read_json(path);assert m['config_hash']==digest(cfg)
        if m.get('code_hash'):assert m['source_code_sha256']==code_hashes()
        return cfg,base,m
    for d in ['config','manifests','raw','targets','tables','policies','processed','qa','figures']:
        (OUTPUT/d).mkdir(parents=True,exist_ok=True)
    atomic_json(OUTPUT/'config/main.json',cfg)
    atomic_json(OUTPUT/'manifests/analysis_preregistration.json',dict(created_at=now(),formal_seen=False,
        diagnostic_seen=False,protocol=cfg,formal_unit='40 independent scenarios per condition',
        primary_Holm='four B2-B1 comparisons across conditions',secondary_Holm='16 B1-B0/B2-B0/B2-R0/B2-R1 comparisons',
        oracle='D-B2 descriptive; PreviewH8 not certified full-day bound',
        interval='paired t95 and 10000 bootstrap unadjusted; Holm p separate',
        support='all interpolation source corners audited; missing b never silent',
        diagnostic_targets='independent M0 target bank8400–8419 for variance corroboration only; not table fitting or tuning',
        variance_corroboration='predict diagnostic targets with frozen construction base/bucket tables; '
            'mean squared-error differences first averaged within scenario, then paired t95/bootstrap/Holm across four conditions; '
            'requires positive base-minus-bucket mean, lower t95>0 and Holm<=.05; '
            'within-diagnostic group centering alone is NOT confirmatory',
        fallback_denominator='all provider coefficient queries; also report nonterminal denominator; '
            'not selected/occupied slots only',
        bucket_contrast='same tuq, three raw conditional means and actual supported/fallback lookup prices; '
            'unsupported raw means not evidence of deployed conditional pricing'))
    m=dict(status='INITIALIZED',created_at=now(),config_hash=digest(cfg),baseline_config=base,
        source=read_json(HISTORY/'manifest.json')['source'],A2_final_sha256=file_hash(A2/'manifests/final_manifest.json'),
        A2_completed_at=read_json(A2/'manifests/final_manifest.json')['completed_at'],
        training_seed=None,policy_seed=0)
    m['approved_accounting_amendment']=amendment
    atomic_json(path,m);return cfg,base,m

def datasets(cfg,base,seeds):
    found={}
    for c in cfg['conditions']:
        for seed in seeds:
            p=OUTPUT/'raw/datasets'/f'{key(c)}-s{seed}.dataset.json'
            if p.exists():found[c['region'],c['orders'],c['fleet'],seed]=read_json(p)
    if len(found)==len(cfg['conditions'])*len(seeds):return found
    result=freeze_data(cfg,base,OUTPUT,cfg['conditions'],seeds)
    for c in cfg['conditions']:
        for seed in seeds:
            previous=FAILED_B/'raw/datasets'/f'{key(c)}-s{seed}.dataset.json'
            if previous.exists():
                assert result[c['region'],c['orders'],c['fleet'],seed]['scenarios']==read_json(previous)['scenarios'],previous
    return result

def freeze(m):
    hashes=code_hashes()
    if m.get('code_hash'):assert m['source_code_sha256']==hashes;return
    with zipfile.ZipFile(OUTPUT/'raw/source.snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:z.write(ROOT/name,name)
        z.write(OUTPUT/'config/main.json','B-config.json')
        task=Path('D:/download/Codex_第六轮后续_A2_Damping与B_MinimalStateExpansion.md')
        z.write(task,'task.md')
    m.update(code_hash=digest(hashes),source_code_sha256=hashes,source_frozen_at=now(),
        snapshot_sha256=file_hash(OUTPUT/'raw/source.snapshot.zip'),task_document_sha256=file_hash(task))
    atomic_json(OUTPUT/'manifest.json',m);atomic_json(OUTPUT/'manifests/source_manifest.json',m)

def history_index():
    """B never calls an A/A2 writer, and freezes all their existing bytes."""
    path=OUTPUT/'manifests/historical_readonly.json'
    if path.exists():return read_json(path)
    final=read_json(A2/'manifests/final_manifest.json');assert final['status']=='COMPLETED'
    for name,sha in final['source_code_sha256'].items():assert file_hash(ROOT/name)==sha,name
    earlier=read_json(A2/'manifests/historical_readonly.json');index={}
    for i,(name,sha) in enumerate(earlier['files'].items()):
        assert file_hash(ROOT/name)==sha,name;index[name]=sha
        if (i+1)%20000==0:print(f'B historical freeze {i+1}/{len(earlier["files"])} prior files',flush=True)
    for p in sorted(A2.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:index[p.relative_to(ROOT).as_posix()]=file_hash(p)
    failed_attempt_gate()
    for p in sorted(FAILED_B.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:index[p.relative_to(ROOT).as_posix()]=file_hash(p)
    source=read_json(A2/'manifest.json')['baseline_config']['data_source']
    assert file_hash(Path(source))==earlier['CFS_sha256']
    result=dict(created_at=now(),files=index,CFS_sha256=earlier['CFS_sha256'],
        A2_final_sha256=file_hash(A2/'manifests/final_manifest.json'),A2_readonly=True,
        original_B_readonly=True,approved_accounting_amendment=failed_attempt_gate())
    atomic_json(path,result);return result
