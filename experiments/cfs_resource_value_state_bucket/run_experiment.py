"""B starts only after immutable A2; one parent/one improvement, never bucket PI."""
import argparse
from experiment_core.storage import atomic_json,read_json
from .common import OUTPUT,A2,initialize,freeze,datasets,now,history_index
from .phase0 import run as phase0
from .targets import prepare,evaluate,build_tables
from .policies import create,run_matrix,action_matrix

def main(stage):
    cfg,base,m=initialize()
    # A2 final deliverables remain read-only. Full prior/raw integrity is audited
    # separately; this immediate check catches modified tables/report/QA at entry.
    from .common import file_hash
    a2=read_json(A2/'manifests/final_manifest.json')
    for name,sha in a2['deliverable_sha256'].items():assert file_hash(A2/name)==sha,name
    history_index()
    phase0(cfg)
    if stage=='phase0':return
    freeze(m)
    construction=datasets(cfg,base,cfg['construction_seeds'])
    priors=prepare(cfg,base,construction,m)
    evaluate(cfg,base,construction,priors,m,'construction');build_tables(cfg,m);create(cfg,m)
    if stage=='construction':return
    analysis_freeze=OUTPUT/'manifests/analysis_source_freeze.json'
    analysis_sha=file_hash(OUTPUT/'analysis.py')
    if analysis_freeze.exists():assert read_json(analysis_freeze)['sha256']==analysis_sha
    else:atomic_json(analysis_freeze,dict(created_at=now(),formal_seen=False,diagnostic_seen=False,sha256=analysis_sha))
    diagnostic=datasets(cfg,base,cfg['diagnostic_seeds'])
    evaluate(cfg,base,diagnostic,priors,m,'diagnostic')
    run_matrix(cfg,base,diagnostic,priors,m,'diagnostic');action_matrix(cfg,'diagnostic')
    frozen=OUTPUT/'manifests/policy_freeze.json'
    if not frozen.exists():atomic_json(frozen,dict(status='FROZEN',created_at=now(),formal_seen=False,
        diagnostic_seen=True,no_hyperparameter_selection=True,one_step_only=True,
        tables_sha256=file_hash(OUTPUT/'manifests/table_freeze.json'),threshold_sha256=file_hash(OUTPUT/'manifests/threshold_freeze.json')))
    if stage=='diagnostic':return
    for split in ['pilot','formal']:
        data=datasets(cfg,base,cfg[f'{split}_seeds']);run_matrix(cfg,base,data,priors,m,split)
    action_matrix(cfg,'formal')
    m['status']='NUMERICAL_COMPLETE_ANALYSIS_QA_PENDING';atomic_json(OUTPUT/'manifest.json',m)
    print('B numerical matrix complete; analysis, QA and final freeze remain.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['phase0','construction','diagnostic','all'],default='all')
    main(p.parse_args().stage)
