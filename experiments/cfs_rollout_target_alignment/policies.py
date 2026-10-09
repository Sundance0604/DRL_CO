"""Frozen 14-strategy matrix; same physical model and identical paired scenarios."""
import time
import numpy as np
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from model_families.single.value.providers import load_provider
from model_families.single.local_value.providers import LegacyAdapter
from experiments.cfs_local_post_bayesian_value.simulation import simulate
from .common import OUTPUT,key,parameters,rho_levels,map_jobs,file_hash,write_csv
from .features import AlignedValueProvider
from .protocol import write_compressed,replay_world,physical_identity

LADDER=[('P0','myopic',None),('P1','myopic+V','legacy_fluid_reference'),('P2','myopic+V','table'),
    ('P3','myopic+V','linear_OLD'),('P4','myopic+V','linear_ACTION'),
    ('P5','myopic+V','chebyshev_OLD'),('P6','myopic+V','chebyshev_ACTION'),
    ('P7','myopic+V','spline_OLD'),('P8','myopic+V','spline_ACTION'),
    ('P9','myopic+V','bayesian_OLD'),('P10','myopic+V','bayesian_ACTION'),
    ('R0','RH4',None),('R1','RH8',None),('D','OracleH8',None)]

def job_run(job):
    target=OUTPUT/'raw/runs'/job['id'];identity=digest(job)
    if (target/'summary.json').exists():
        r=read_json(target/'summary.json');assert r['identity']==identity
        if r['status']=='COMPLETED':return r
    r=dict(id=job['id'],condition=job['condition'],seed=job['scenario']['seed'],policy=job['label'],
        identity=identity,scenario_hash=digest(job['scenario']),model_hash=job['model_hash'],code_hash=job['code_hash'],
        stage=job['stage'],policy_seed=0,training_seed=None,information='oracle-diagnostic' if job['policy']=='OracleH8' else 'online')
    started=time.perf_counter()
    try:
        sim,orders=matching_state(job['scenario'],job['parameters'],job['solver'])
        provider=None
        if job['model']:
            modelpath=OUTPUT/job['model']
            provider=LegacyAdapter(load_provider(modelpath)) if job['method'] in ['table','legacy_fluid_reference'] else AlignedValueProvider(read_json(modelpath))
        metrics,trace,_=simulate(sim,orders,job['policy'],provider,job['prior'],trace=True,timeout=job['timeout'])
        assert metrics['unfinished_committed']==0 and metrics['assigned']==metrics['delivered']
        initial,_=matching_state(job['scenario'],job['parameters'],job['solver'])
        initial.begin_period([o for o in orders.values() if o.book_time==0])
        replay=replay_world(initial,orders,dict(trace=trace,total=metrics['realized_profit'],physical_final_hash=physical_identity(sim)))
        assert all(all(row['invariants'].values()) for row in trace)
        r.update({k:v for k,v in metrics.items() if isinstance(v,(int,float))},status='COMPLETED',
            replay_steps=replay['periods'],replay_profit_difference=replay['total_difference'],wall_seconds=time.perf_counter()-started)
        write_compressed(target/'observations.json.gz',dict(metrics=metrics,trace=trace,
            decision_ledger=[dict(period=x['period'],accounting=x['plan']['accounting']) for x in trace if not x.get('drain')],replay=replay))
        r['observations_sha256']=file_hash(target/'observations.json.gz')
    except Exception as exc:r.update(status='FAILED',error=repr(exc),wall_seconds=time.perf_counter()-started)
    atomic_json(target/'spec.json',job);atomic_json(target/'summary.json',r)
    return r

def run_ladder(cfg,base,data,priors,m,seeds,stage):
    frozen=read_json(OUTPUT/'manifests/model_freeze.json')
    assert not frozen['formal_seen'] and frozen['code_hash']==m['phase2_code_hash']
    for path,sha in frozen['models'].items():assert file_hash(OUTPUT/path)==sha
    jobs=[]
    for c in cfg['conditions']:
        cid=key(c)
        for seed in seeds:
            s=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
            for label,policy,method in LADDER:
                model=f'models/{method}/{cid}.json' if method else None
                jobs.append(dict(id=f'{stage}-{cid}-{seed}-{label}',stage=stage,condition=cid,scenario=s,label=label,method=method,
                    policy=policy,model=model,model_hash=file_hash(OUTPUT/model) if model else None,
                    parameters=parameters(base,rho_levels(s,base)['rho_max']),solver=base['solver'],prior=priors[cid],
                    code_hash=m['phase2_code_hash'],timeout=cfg['run_timeout_seconds']))
    rows,wall=map_jobs(job_run,jobs,stage,cfg['max_parallel']);rows.sort(key=lambda r:r['id'])
    atomic_json(OUTPUT/'processed'/f'{stage}_policies.json',rows);write_csv(OUTPUT/'processed'/f'{stage}_policies.csv',rows)
    assert all(r['status']=='COMPLETED' for r in rows),[r for r in rows if r['status']!='COMPLETED'][:5]
    return dict(runs=len(rows),failures=0,wall_seconds=wall,seeds=seeds)
