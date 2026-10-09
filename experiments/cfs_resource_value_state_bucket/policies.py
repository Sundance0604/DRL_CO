"""Independent B policy registry and complete paired scenario simulations."""
import time
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from experiments.cfs_rollout_target_alignment.queries import pack_state
from experiments.cfs_resource_policy_iteration.action_analysis import canonical,control_modes
from .common import OUTPUT,key,parameters,rho_levels,map_jobs,write_csv,now,file_hash
from .protocol import continuation,read_compressed,write_compressed,physical_identity,replay_world,restore,decide_same_state
from .simulation import simulate

LABELS=['B0','B1','B2','R0','R1','D']

def create(cfg,m):
    priors_sha=file_hash(OUTPUT/'tables/priors.json')
    for c in cfg['conditions']:
        cid=key(c)
        for label in LABELS:
            ref=f'tables/{"base_tuq" if label=="B1" else "bucket_tuqb"}/{cid}.json' if label in ['B1','B2'] else None
            row=dict(label=label,policy_id='M0' if label=='B0' else label,condition=cid,code_hash=m['code_hash'],
                table_ref=ref,table_hash=None if ref is None else file_hash(OUTPUT/ref),context_prior_sha256=priors_sha,
                framework='original M0' if label=='B0' else 'pure H1 slot-prefix' if label.startswith('B') else label,
                damping=False,action_correction=False,training_seed=None,policy_seed=0)
            row['policy_hash']=digest(dict(condition=cid,parent='M0',code=m['code_hash'])) if label=='B0' else digest(row)
            row['record_hash']=digest(row)
            path=OUTPUT/'policies'/label/(cid+'.json')
            if path.exists():assert read_json(path)==row
            else:atomic_json(path,row)

def load(label,cid):
    r=read_json(OUTPUT/'policies'/label/(cid+'.json'))
    assert digest({k:v for k,v in r.items() if k!='record_hash'})==r['record_hash']
    assert file_hash(OUTPUT/'tables/priors.json')==r['context_prior_sha256']
    if r['table_ref']:assert file_hash(OUTPUT/r['table_ref'])==r['table_hash']
    return r|dict(table=None if r['table_ref'] is None else read_json(OUTPUT/r['table_ref']))

def run_job(j):
    ident=digest(j);path=OUTPUT/'raw/runs'/j['id']
    if (path/'summary.json').exists():
        r=read_json(path/'summary.json');assert r['identity']==ident and file_hash(OUTPUT/r['trace_ref'])==r['trace_sha256'];return r
    start=time.perf_counter();s,orders=matching_state(j['scenario'],j['parameters'],j['solver'])
    s.begin_period([o for o in orders.values() if o.book_time==0]);initial=s.clone()
    if j['label'].startswith('B'):
        result=continuation(s,orders,j['policy'],j['prior'],collect_times=range(s.T))
    else:
        metrics,trace,states=simulate(s,orders,{'R0':'RH4','R1':'RH8','D':'OracleH8'}[j['label']],
            prior=j['prior'],collect_times=range(s.T),trace=True,first_begun=True)
        replay=replay_world(initial,orders,dict(trace=trace,total=metrics['realized_profit'],physical_final_hash=physical_identity(s)))
        result=dict(metrics=metrics,trace=trace,states=states,replay=replay,final_hash=physical_identity(s),
            lookup_queries=[],value_calls=0,fallback_calls=0)
    metrics=result['metrics'];assert metrics['limited_window_count']==0 and metrics['unfinished_committed']==0
    assert metrics['assigned']==metrics['delivered']
    tracepath=OUTPUT/'raw/traces'/(j['id']+'.json.gz')
    write_compressed(tracepath,{k:v for k,v in result.items() if k!='states'}|
        dict(state_packets={str(t):pack_state(v) for t,v in result['states'].items()}))
    r=dict(id=j['id'],stage=j['stage'],condition=j['condition'],scenario_seed=j['scenario']['seed'],
        scenario_hash=digest(j['scenario']),policy=j['label'],policy_hash=j['policy']['policy_hash'],
        table_hash=j['policy']['table_hash'],code_hash=j['code_hash'],identity=ident,status='COMPLETED',
        **{k:v for k,v in metrics.items() if isinstance(v,(int,float))},
        value_calls=result['value_calls'],fallback_calls=result['fallback_calls'],
        replay_steps=result['replay']['periods'],replay_profit_difference=result['replay']['total_difference'],
        trace_ref=tracepath.relative_to(OUTPUT).as_posix(),trace_sha256=file_hash(tracepath),pipeline_seconds=time.perf_counter()-start,
        policy_seed=0,training_seed=None)
    atomic_json(path/'spec.json',j);atomic_json(path/'summary.json',r);return r

def run_matrix(cfg,base,data,priors,m,stage):
    mp=OUTPUT/'manifests'/f'{stage}_run_manifest.json'
    if mp.exists():
        r=read_json(mp);assert file_hash(OUTPUT/r['results_ref'])==r['results_sha256'];return r
    assert read_json(OUTPUT/'manifests/table_freeze.json')['status']=='FROZEN'
    jobs=[]
    for c in cfg['conditions']:
        cid=key(c)
        for seed in cfg[f'{stage}_seeds']:
            scenario=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
            for label in LABELS:
                jobs.append(dict(id=f'{stage}-{cid}-s{seed}-{label}',stage=stage,condition=cid,scenario=scenario,
                    label=label,policy=load(label,cid),prior=priors[cid],
                    parameters=parameters(base,rho_levels(scenario,base)['rho_max']),solver=base['solver'],code_hash=m['code_hash']))
    rows,wall=map_jobs(run_job,jobs,f'B {stage} closed loop',cfg['max_parallel']);rows.sort(key=lambda r:r['id'])
    ref=f'processed/{stage}_runs.json';atomic_json(OUTPUT/ref,rows);write_csv(OUTPUT/'processed'/f'{stage}_runs.csv',rows)
    r=dict(status='COMPLETED',created_at=now(),results_ref=ref,results_sha256=file_hash(OUTPUT/ref),
        runs=len(rows),wall_seconds=wall,replay_steps=sum(r['replay_steps'] for r in rows))
    atomic_json(mp,r);return r

def action_job(j):
    path=OUTPUT/'raw/action_comparisons'/(j['id']+'.json.gz');ident=digest(j)
    if path.exists():
        r=read_compressed(path);assert r['identity']==ident;return r['rows']
    states=read_compressed(OUTPUT/j['trace_ref'])['state_packets'];policies=[load(p,j['condition']) for p in ['B0','B1','B2']];rows=[]
    for t in range(j['scenario']['horizon']):
        state,_=restore(j['scenario'],j['parameters'],j['solver'],states[str(t)])
        plans=[canonical(decide_same_state(state,p,j['prior'])) for p in policies]
        for aa,bb in [(0,1),(1,2)]:
            a,b=plans[aa],plans[bb]
            rows.append(dict(stage=j['stage'],condition=j['condition'],scenario_seed=j['scenario']['seed'],time=t,
                policy_a=f'B{aa}',policy_b=f'B{bb}',reference='B0 SAME complete state',state_hash=digest(states[str(t)]),
                action_hash_a=digest(a),action_hash_b=digest(b),full_action_disagree=int(a!=b),
                routes_disagree=int(a['routes']!=b['routes']),assignments_disagree=int(a['assignments']!=b['assignments']),
                wait_reposition_disagree=int(control_modes(a)!=control_modes(b)),
                release_hub_disagree=int({r['vehicle_id']:r['e'] for r in a['routes']}!={r['vehicle_id']:r['e'] for r in b['routes']})))
    write_compressed(path,dict(identity=ident,rows=rows));return rows

def action_matrix(cfg,stage):
    path=OUTPUT/'processed'/f'{stage}_action_disagreement.json'
    if path.exists():return read_json(path)
    jobs=[]
    for r in read_json(OUTPUT/'processed'/f'{stage}_runs.json'):
        if r['policy']=='B0':jobs.append(read_json(OUTPUT/'raw/runs'/r['id']/'spec.json')|dict(trace_ref=r['trace_ref']))
    items,wall=map_jobs(action_job,jobs,f'B {stage} same-state comparison',cfg['max_parallel'])
    rows=[r for item in items for r in item];atomic_json(path,rows);write_csv(path.with_suffix('.csv'),rows)
    atomic_json(OUTPUT/'qa'/f'{stage}_same_state_actions.json',dict(status='PASSED',states=len(jobs)*24,
        comparisons=len(rows),wall_seconds=wall,no_future_input=True))
    return rows
