"""Stage-gated, resumable numerical entry. COMPLETED and failed attempts cannot restart."""
import argparse
import time
import shutil
from experiment_core.storage import atomic_json,read_json,digest
from .common import OUTPUT,initialize,config,datasets,sha,now

def run(stage):
    if (OUTPUT/'qa/stop_condition_failure.json').exists():raise RuntimeError('STOP_CONDITION_TRIGGERED: preserve failed attempt, do not resume it')
    if stage=='regression':
        from .phase0 import run as regression
        regression(config());return
    cfg,base,m=initialize()
    if stage=='prepare':
        from .phase0 import run as regression
        regression(cfg)
        return
    assert read_json(OUTPUT/'qa/historical_regression.json')['status']=='PASSED'
    from .states import freeze_bank
    from .experiment import run_joint
    if stage in ['pilot','all']:
        data=datasets(cfg,base,cfg['pilot_seeds']);manifest=freeze_bank(cfg,base,data,cfg['pilot_seeds'],'pilot')
        pilot=run_joint(cfg,manifest,'pilot')
        estimate=pilot['worker_seconds']/pilot['worlds']
        # Conservative fixed upper count, not conditional on pilot gains or favorable states.
        future_worlds=(20+40)*3*2*(6*96+2*4*64)+50000
        projected=estimate*future_worlds/cfg['max_parallel']/3600
        budget=dict(status='FROZEN_BEFORE_DIAGNOSTIC_FORMAL',created_at=now(),pilot_sha256=sha(OUTPUT/'manifests/pilot_joint.json'),
            mean_world_pipeline_seconds=estimate,conservative_world_upper_bound=future_worlds,
            projected_wall_hours=projected,allowed_wall_hours=cfg['wall_budget_hours'],
            free_storage_GiB=shutil.disk_usage(OUTPUT).free/2**30,assessment_paths=64,
            rule='all planned stages if conservative runtime<=72h and free storage>=200GiB; otherwise BUDGET_STOP before formal',
            all_conditions_all_methods_same_paths=True,pilot_profit_not_used_for_allocation=True)
        p=OUTPUT/'manifests/budget_freeze.json'
        if p.exists():
            prior=read_json(p);assert prior['pilot_sha256']==budget['pilot_sha256']
        else:atomic_json(p,budget)
        if projected>cfg['wall_budget_hours'] or budget['free_storage_GiB']<cfg['storage_budget_GiB']:
            m.update(status='BUDGET_STOP',budget=budget);atomic_json(OUTPUT/'manifest.json',m)
            raise RuntimeError('Prespecified uniform budget gate stopped before diagnostic/formal; user decision required')
        if stage=='pilot':return
    assert read_json(OUTPUT/'manifests/budget_freeze.json')['status']=='FROZEN_BEFORE_DIAGNOSTIC_FORMAL'
    assert sha(OUTPUT/'analysis.py')==read_json(OUTPUT/'manifests/analysis_preregistration.json')['analysis_sha256']
    for name in ['diagnostic','formal']:
        if stage not in [name,'all']:continue
        data=datasets(cfg,base,cfg[name+'_seeds']);manifest=freeze_bank(cfg,base,data,cfg[name+'_seeds'],name)
        run_joint(cfg,manifest,name)
        m.update(status=name.upper()+'_NUMERICAL_COMPLETE',last_updated_at=now());atomic_json(OUTPUT/'manifest.json',m)
    if stage in ['banks','all']:
        assert (OUTPUT/'manifests/formal_joint.json').exists()
        from .banks import construct,deploy
        seeds=[s for bank in cfg['bank_seeds'] for s in bank]
        data=datasets(cfg,base,seeds);tables=construct(cfg,base,data)
        data=datasets(cfg,base,cfg['deployment_seeds']);deploy(cfg,base,data,tables)
        m.update(status='NUMERICAL_COMPLETE_ANALYSIS_QA_PENDING',numerical_completed_at=now())
        atomic_json(OUTPUT/'manifest.json',m)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['regression','prepare','pilot','diagnostic','formal','banks','all'],default='all')
    args=parser.parse_args()
    try:run(args.stage)
    except Exception as e:
        if not (OUTPUT/'qa/stop_condition_failure.json').exists() and 'budget gate' not in str(e) and 'COMPLETED' not in str(e):
            atomic_json(OUTPUT/'qa/entry_failure.json',dict(status='ENTRY_FAILED',stage=args.stage,error=repr(e),at=now()))
        raise
