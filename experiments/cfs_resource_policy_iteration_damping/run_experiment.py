"""Strict A2 stage order; B is never dispatched before frozen A2 completion."""
import argparse
from experiment_core.storage import atomic_json,read_json
from .common import OUTPUT,ALPHAS,initialize,freeze,datasets,history_index,key,now,file_hash
from .phase0 import run as phase0
from .evaluation import evaluate,deploy
from .policies import create,create_damped,load,run_matrix,action_matrix

def main(stage):
    cfg,base,m=initialize()
    if stage=='history':
        history_index();phase0(cfg);return
    history_index();phase0(cfg);freeze(m)
    data=datasets(cfg,base,cfg['construction_seeds'])
    from .common import HISTORY
    priors=read_json(HISTORY/'tables/priors.json')
    create(cfg,m)
    for alpha in ALPHAS:
        for n in range(4):
            if n>=2:
                policies={key(c):load(f'D{n}',key(c),alpha) for c in cfg['conditions']}
                evaluate(cfg,base,data,priors,m,n,alpha,policies)
            deploy(cfg,m,n,alpha);create_damped(cfg,m,n+1,alpha)
            m['status']=f'ALPHA_{alpha}_D{n+1}_FROZEN';atomic_json(OUTPUT/'manifest.json',m)
            print(f'alpha {alpha}, raw/deployed V{n} and D{n+1} frozen.',flush=True)
    atomic_json(OUTPUT/'manifests/policy_freeze.json',dict(status='FROZEN',created_at=now(),formal_seen=False,
        code_hash=m['code_hash'],alphas=ALPHAS,all_paths_complete=True))
    if stage=='construction':return
    selection_source=OUTPUT/'manifests/selection_source_freeze.json'
    if selection_source.exists():
        assert read_json(selection_source)['sha256']==file_hash(OUTPUT/'analysis.py')
    else:
        assert not (OUTPUT/'processed/diagnostic_runs.json').exists()
        atomic_json(selection_source,dict(created_at=now(),diagnostic_seen=False,formal_seen=False,
            sha256=file_hash(OUTPUT/'analysis.py'),rule=cfg['selection']))
    diagnostic=datasets(cfg,base,cfg['diagnostic_seeds'])
    run_matrix(cfg,base,diagnostic,priors,m,cfg['diagnostic_seeds'],'diagnostic',list(ALPHAS))
    action_matrix(cfg,'diagnostic',list(ALPHAS))
    from .analysis import select_alpha
    selected=select_alpha(cfg)
    if stage=='diagnostic':return
    for label in ['pilot','formal']:
        stage_data=datasets(cfg,base,cfg[f'{label}_seeds'])
        run_matrix(cfg,base,stage_data,priors,m,cfg[f'{label}_seeds'],label,[selected])
        m['status']=label.upper()+'_COMPLETED';atomic_json(OUTPUT/'manifest.json',m)
    action_matrix(cfg,'formal',[selected])
    m['status']='NUMERICAL_COMPLETE_ANALYSIS_QA_PENDING';atomic_json(OUTPUT/'manifest.json',m)
    print('A2 numerical matrix complete; final analysis/QA/freeze still required.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['history','construction','diagnostic','all'],default='all')
    main(parser.parse_args().stage)
