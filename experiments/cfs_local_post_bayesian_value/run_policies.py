"""Frozen formal matrix, all failures retained; no policy outcome selection."""
import time
from pathlib import Path
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from model_families.single.local_value.providers import load_local
from model_families.single.value.providers import load_provider
from experiments.cfs_classical_value.simulation import simulate as old_simulate
from .simulation import simulate
from .common import OUTPUT,OLD,key,parameters,rho_levels,map_jobs,write_csv,file_hash

LADDER=[('P0','myopic',None),('P1','myopic+V','legacy_fluid_reference'),('P2','myopic+V','table'),
    ('P3','myopic+V','linear'),('P4','myopic+V','chebyshev'),('P5','myopic+V','spline'),('P6','myopic+V','bayesian'),
    ('P7','RH4',None),('P8','RH4+V','legacy_fluid_reference'),('P9','RH4+V','table'),
    ('P10','RH4+V','linear'),('P11','RH4+V','chebyshev'),('P12','RH4+V','spline'),('P13','RH4+V','bayesian'),
    ('P14','RH8',None),('D','OracleH8',None)]

def policy_job(job):
    target=OUTPUT/'raw/runs'/job['id'];identity=digest(job)
    if (target/'summary.json').exists():
        old=read_json(target/'summary.json');assert old['identity']==identity
        if old['status']=='COMPLETED':return old
    row=dict(id=job['id'],condition=job['condition'],seed=job['scenario']['seed'],policy=job['label'],
        stage=job['stage'],scenario_hash=digest(job['scenario']),identity=identity,code_hash=job['code_hash'],
        model_hash=job['model_hash'],policy_seed=0,training_seed=None,information='oracle' if job['policy']=='OracleH8' else 'online')
    try:
        sim,orders=matching_state(job['scenario'],job['parameters'],job['solver'])
        provider=(load_provider(Path(job['model'])) if job.get('old_summary') else load_local(Path(job['model']))) if job['model'] else None
        fn=old_simulate if job.get('old_summary') else simulate
        metrics,trace,_=fn(sim,orders,job['policy'],provider,job['prior'],trace=True,timeout=job['timeout'])
        assert metrics['unfinished_committed']==0 and metrics['assigned']==metrics['delivered']
        row.update({k:v for k,v in metrics.items() if isinstance(v,(float,int))},status='COMPLETED')
        atomic_json(target/'metrics.json',metrics);atomic_json(target/'trace.json',trace)
        atomic_json(target/'decision-ledger.json',[dict(period=r['period'],accounting=r['plan']['accounting']) for r in trace if not r.get('drain')])
        atomic_json(target/'realized-ledger.json',[dict(period=r['period'],ledger=r['realized_ledger']) for r in trace])
        row['trace_hash']=file_hash(target/'trace.json')
    except Exception as exc:
        row.update(status='FAILED',error=str(exc))
    atomic_json(target/'spec.json',job);atomic_json(target/'summary.json',row)
    return row

def run_ladder(cfg,base,data,priors,manifest,seeds,stage,ablation=False):
    frozen=read_json(OUTPUT/'manifests/model_freeze.json')
    assert frozen['code_hash']==manifest['code_hash'] and not frozen['formal_seed_profit_seen']
    for name,sha in frozen['models'].items():assert file_hash(OUTPUT/name)==sha
    ladder=[('OldM-Cheb','myopic+V','chebyshev'),('OldM-Spline','myopic+V','spline'),
            ('OldRH4-Cheb','RH4+V','chebyshev'),('OldRH4-Spline','RH4+V','spline')] if ablation else LADDER
    jobs=[]
    oldpriors=read_json(OLD/'models/priors.json') if ablation else None
    for c in cfg['conditions']:
        cid=key(c)
        for seed in seeds:
            s=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
            for label,policy,method in ladder:
                model=(OLD/'models'/f'{cid}-{method}.json' if ablation else OUTPUT/'models'/method/f'{cid}.json') if method else None
                jobs.append(dict(id=f'{stage}-{cid}-{seed}-{label}',stage=stage,condition=cid,scenario=s,label=label,
                    policy=policy,model=str(model) if model else None,model_hash=file_hash(model) if model else None,
                    parameters=parameters(base,rho_levels(s,base)['rho_max']),solver=base['solver'],
                    prior=oldpriors[cid] if ablation else priors[cid],code_hash=manifest['code_hash'],
                    timeout=cfg['run_timeout_seconds'],old_summary=ablation))
    rows,wall=map_jobs(policy_job,jobs,stage,cfg['max_parallel']); rows.sort(key=lambda r:r['id'])
    write_csv(OUTPUT/'processed'/f'{stage}_policies.csv',rows);atomic_json(OUTPUT/'processed'/f'{stage}_policies.json',rows)
    assert all(r['status']=='COMPLETED' for r in rows),[r for r in rows if r['status']!='COMPLETED'][:5]
    return dict(runs=len(rows),failures=0,wall_seconds=wall,seeds=seeds)
