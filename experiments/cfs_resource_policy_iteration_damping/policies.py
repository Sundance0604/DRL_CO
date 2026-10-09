"""Owned policy lineage and scenario-paired simulations with explicit aliases."""
import time
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from experiments.cfs_rollout_target_alignment.queries import pack_state
from experiments.cfs_resource_policy_iteration.protocol import (
    continuation,write_compressed,read_compressed,restore,decide_same_state)
from experiments.cfs_resource_policy_iteration.action_analysis import canonical,control_modes
from .common import OUTPUT,HISTORY,key,parameters,rho_levels,map_jobs,write_csv,now,file_hash

def make_policy(m,cid,label,table_path=None,alpha=None):
    row=dict(policy_id=label,condition=cid,alpha=alpha,model_code_hash=m['code_hash'],
        table_ref=None if table_path is None else str(table_path.resolve()),
        table_hash=None if table_path is None else file_hash(table_path),
        policy_seed=0,training_seed=None,action_correction=False,framework='pure H1 slot-prefix')
    row['policy_hash']=digest(row)
    return row

def create(cfg,m):
    for c in cfg['conditions']:
        cid=key(c)
        for n in range(5):
            p=OUTPUT/'policies'/f'U{n}'/(cid+'.json')
            row=make_policy(m,cid,f'U{n}',None if n==0 else HISTORY/'tables'/f'V{n-1}'/(cid+'.json'))
            if p.exists(): assert read_json(p)==row
            else: atomic_json(p,row)

def create_damped(cfg,m,n,alpha):
    for c in cfg['conditions']:
        cid=key(c);path=OUTPUT/'policies'/f'alpha_{alpha}'/f'D{n}'/(cid+'.json')
        row=make_policy(m,cid,f'D{n}',OUTPUT/'tables'/f'alpha_{alpha}'/f'deployed_V{n-1}'/(cid+'.json'),cfg['alphas'][alpha])
        if path.exists(): assert read_json(path)==row
        else: atomic_json(path,row)

def load(label,cid,alpha=None):
    path=OUTPUT/'policies'
    if label.startswith('D'): path=path/f'alpha_{alpha}'
    row=read_json(path/label/(cid+'.json'))
    assert digest({k:v for k,v in row.items() if k!='policy_hash'})==row['policy_hash']
    if row['table_ref']: assert file_hash(row['table_ref'])==row['table_hash']
    return row|dict(table=None if row['table_ref'] is None else read_json(row['table_ref']))

def run_job(job):
    ident=digest(job);path=OUTPUT/'raw/runs'/job['id'];sp=path/'summary.json'
    if sp.exists():
        r=read_json(sp);assert r['identity']==ident and file_hash(OUTPUT/r['trace_ref'])==r['trace_sha256'];return r
    started=time.perf_counter()
    s,orders=matching_state(job['scenario'],job['parameters'],job['solver'])
    s.begin_period([o for o in orders.values() if o.book_time==0])
    result=continuation(s,orders,job['policy'],job['prior'],collect_times=range(s.T))
    metrics=result['metrics']
    assert metrics['limited_window_count']==0 and metrics['unfinished_committed']==0
    assert metrics['assigned']==metrics['delivered']
    trace=OUTPUT/'raw/traces'/(job['id']+'.json.gz')
    write_compressed(trace,{k:v for k,v in result.items() if k!='states'}|
        dict(state_packets={str(t):pack_state(v) for t,v in result['states'].items()}))
    row=dict(id=job['id'],stage=job['stage'],condition=job['condition'],alpha=job.get('alpha'),
        scenario_seed=job['scenario']['seed'],scenario_hash=digest(job['scenario']),policy=job['label'],
        identity=ident,code_hash=job['code_hash'],policy_hash=job['policy']['policy_hash'],
        table_hash=job['policy'].get('table_hash'),status='COMPLETED',policy_seed=0,training_seed=None,
        **{k:v for k,v in metrics.items() if isinstance(v,(float,int))},
        replay_steps=result['replay']['periods'],replay_profit_difference=result['replay']['total_difference'],
        value_calls=result['value_calls'],trace_ref=trace.relative_to(OUTPUT).as_posix(),
        trace_sha256=file_hash(trace),pipeline_seconds=time.perf_counter()-started,alias_of=None)
    atomic_json(path/'spec.json',job);atomic_json(sp,row);return row

def run_matrix(cfg,base,data,priors,m,seeds,stage,alphas):
    manifest=OUTPUT/'manifests'/f'{stage}_matrix.json'
    if manifest.exists():
        result=read_json(manifest);assert result['alphas']==alphas
        assert file_hash(OUTPUT/result['results_ref'])==result['results_hash'];return result
    jobs=[]
    for c in cfg['conditions']:
        cid=key(c)
        for seed in seeds:
            scenario=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
            specs=[(f'U{n}',None) for n in range(5)]+[(f'D{n}',a) for a in alphas for n in range(2,5)]
            for label,alpha in specs:
                policy=load(label,cid,alpha)
                jobs.append(dict(id=f'{stage}-{cid}-s{seed}-{label}'+('' if alpha is None else f'-a{alpha}'),
                    label=label,alpha=alpha,stage=stage,condition=cid,scenario=scenario,policy=policy,
                    prior=priors[cid],parameters=parameters(base,rho_levels(scenario,base)['rho_max']),
                    solver=base['solver'],code_hash=m['code_hash']))
    rows,wall=map_jobs(run_job,jobs,f'{stage} closed-loop',cfg['max_parallel'])
    aliases=[]
    for r in rows:
        if r['policy']!='U1':continue
        for alpha in alphas:
            j=read_json(OUTPUT/'raw/runs'/r['id']/'spec.json')
            dp=load('D1',r['condition'],alpha)
            assert dp['table']['values']==j['policy']['table']['values']
            assert dp['table_hash']==j['policy']['table_hash']
            label=r|dict(id=r['id'].replace('-U1',f'-D1-a{alpha}'),alpha=alpha,policy='D1',
                policy_hash=dp['policy_hash'],alias_of=r['id'],
                alias_proof='identical table bytes, same scenario, planner, parameters, seeds; no extra replication')
            aliases.append(label)
            atomic_json(OUTPUT/'raw/runs'/label['id']/'spec.json',j|dict(label='D1',alpha=alpha,policy=dp,alias_of=r['id']))
            atomic_json(OUTPUT/'raw/runs'/label['id']/'summary.json',label)
    rows+=aliases;rows.sort(key=lambda r:r['id'])
    ref=f'processed/{stage}_runs.json';atomic_json(OUTPUT/ref,rows);write_csv(OUTPUT/'processed'/f'{stage}_runs.csv',rows)
    result=dict(status='COMPLETED',alphas=alphas,runs=len(rows),unique_simulations=len(jobs),aliases=len(aliases),
        wall_seconds=wall,results_ref=ref,results_hash=file_hash(OUTPUT/ref),completed_at=now())
    atomic_json(manifest,result);return result

def action_job(request):
    path=OUTPUT/'raw/action_comparisons'/(request['id']+'.json.gz');ident=digest(request)
    if path.exists():
        cache=read_compressed(path);assert cache['identity']==ident;return cache['rows']
    bank=read_compressed(OUTPUT/request['trace_ref'])['state_packets'];rows=[]
    policies={f'U{n}':load(f'U{n}',request['condition']) for n in range(5)}
    for alpha in request['alphas']:
        for n in range(1,5):policies[f'a{alpha}D{n}']=load(f'D{n}',request['condition'],alpha)
    for t in range(request['scenario']['horizon']):
        state,_=restore(request['scenario'],request['parameters'],request['solver'],bank[str(t)])
        plans={label:canonical(decide_same_state(state,p,request['prior'])) for label,p in policies.items() if not label.endswith('D1')}
        for alpha in request['alphas']:plans[f'a{alpha}D1']=plans['U1']
        pairs=[('undamped',n,f'U{n-1}',f'U{n}') for n in range(1,5)]
        for alpha in request['alphas']:
            pairs.extend((alpha,n,'U0' if n==1 else f'a{alpha}D{n-1}',f'a{alpha}D{n}') for n in range(1,5))
            pairs.extend((alpha,n,f'U{n}',f'a{alpha}D{n}') for n in range(2,5))
        for alpha,n,a,b in pairs:
            x,y=plans[a],plans[b]
            rows.append(dict(stage=request['stage'],condition=request['condition'],scenario_seed=request['scenario']['seed'],
                time=t,alpha=alpha,iteration=n,kind='between_paths' if a==f'U{n}' else 'adjacent',
                policy_a=a,policy_b=b,state_hash=digest(bank[str(t)]),reference='U0',
                action_hash_a=digest(x),action_hash_b=digest(y),full_action_disagree=int(x!=y),
                routes_disagree=int(x['routes']!=y['routes']),assignments_disagree=int(x['assignments']!=y['assignments']),
                wait_reposition_disagree=int(control_modes(x)!=control_modes(y)),
                release_hub_disagree=int({r['vehicle_id']:r['e'] for r in x['routes']}!={r['vehicle_id']:r['e'] for r in y['routes']})))
    write_compressed(path,dict(identity=ident,rows=rows));return rows

def action_matrix(cfg,stage,alphas):
    resultpath=OUTPUT/'processed'/f'{stage}_action_disagreement.json'
    if resultpath.exists():return read_json(resultpath)
    rows=read_json(OUTPUT/'processed'/f'{stage}_runs.json');jobs=[]
    for r in rows:
        if r['policy']!='U0':continue
        spec=read_json(OUTPUT/'raw/runs'/r['id']/'spec.json')
        jobs.append(spec|dict(alphas=alphas,trace_ref=r['trace_ref']))
    items,wall=map_jobs(action_job,jobs,f'{stage} same-state actions',cfg['max_parallel'])
    flat=[r for item in items for r in item]
    atomic_json(resultpath,flat);write_csv(resultpath.with_suffix('.csv'),flat)
    atomic_json(OUTPUT/'qa'/f'{stage}_actions.json',dict(status='PASSED',states=len(jobs)*24,
        comparisons=len(flat),wall_seconds=wall,reference='U0 ALL periods',no_future_input=True))
    return flat
