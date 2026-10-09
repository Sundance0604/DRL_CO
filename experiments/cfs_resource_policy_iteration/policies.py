"""G(V), immutable reverse lineage, common-state action and run protocols."""
import time
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from .simulation import simulate,summary
from experiments.cfs_rollout_target_alignment.queries import pack_state
from .common import OUTPUT,key,parameters,rho_levels,map_jobs,write_csv,file_hash,now
from .protocol import continuation,write_compressed,read_compressed,restore,decide_same_state,action_hash,physical_identity,replay_world

LADDER=['P0','P1','P2','P3','P4','R0','R1','D']

def create_policy(cfg,m,n):
    path=OUTPUT/'policies'/f'pi{n}.json'
    if path.exists():return load_policies(n)
    records={}
    for c in cfg['conditions']:
        cid=key(c);model=None if n==0 else f'tables/V{n-1}/{cid}.json'
        row=dict(policy_id=f'pi{n}',condition=cid,model_code_hash=m['code_hash'],parent_table_id=None if n==0 else f'V{n-1}',
            table_ref=model,table_hash=None if model is None else file_hash(OUTPUT/model),
            planner='frozen original M0' if n==0 else 'single-route-macro-action-slot-value/v1; H=1; pure slots only',
            created_at=now(),action_correction=False,policy_seed=0,training_seed=None)
        row['policy_hash']=digest(row);records[cid]=row
    atomic_json(path,records);return load_policies(n)

def load_policies(n):
    records=read_json(OUTPUT/'policies'/f'pi{n}.json');out={}
    for cid,r in records.items():
        assert digest({k:v for k,v in r.items() if k!='policy_hash'})==r['policy_hash']
        if r['table_ref']:
            assert file_hash(OUTPUT/r['table_ref'])==r['table_hash']
        out[cid]=r|dict(table=read_json(OUTPUT/r['table_ref']) if r['table_ref'] else None)
    return out

def run_job(job):
    identity=digest(job);p=OUTPUT/'raw'/('formal_runs' if job['stage']=='formal' else 'runs')/job['id']
    if (p/'summary.json').exists():
        row=read_json(p/'summary.json');assert row['identity']==identity and row['status']=='COMPLETED'
        assert file_hash(OUTPUT/row['trace_ref'])==row['trace_sha256'];return row
    started=time.perf_counter();s,orders=matching_state(job['scenario'],job['parameters'],job['solver'])
    s.begin_period([o for o in orders.values() if o.book_time==0]);initial=s.clone()
    if job['label'].startswith('P'):
        result=continuation(s,orders,job['policy'],job['prior'],collect_times=range(s.T) if job['stage']=='diagnostic' else ())
        metrics=result['metrics'];trace=result['trace'];replay=result['replay']
    else:
        policy={'R0':'RH4','R1':'RH8','D':'OracleH8'}[job['label']]
        metrics,trace,_=simulate(s,orders,policy=policy,trace=True,first_begun=True)
        replay=replay_world(initial,orders,dict(trace=trace,total=metrics['realized_profit'],physical_final_hash=physical_identity(s)))
        result=dict(metrics=metrics,trace=trace,replay=replay,states={},value_calls=0)
    assert metrics['limited_window_count']==0 and metrics['unfinished_committed']==0 and metrics['assigned']==metrics['delivered']
    data={k:v for k,v in result.items() if k!='states'}
    if result['states']:data['state_packets']={str(t):pack_state(v) for t,v in result['states'].items()}
    tracepath=OUTPUT/'raw/traces'/(job['id']+'.json.gz');write_compressed(tracepath,data)
    row=dict(id=job['id'],stage=job['stage'],condition=job['condition'],scenario_seed=job['scenario']['seed'],policy=job['label'],
        scenario_hash=digest(job['scenario']),identity=identity,code_hash=job['code_hash'],policy_hash=job.get('policy',{}).get('policy_hash'),
        model_hash=job.get('policy',{}).get('table_hash'),status='COMPLETED',policy_seed=0,training_seed=None,
        **{k:v for k,v in metrics.items() if isinstance(v,(float,int))},replay_steps=replay['periods'],
        replay_profit_difference=replay['total_difference'],value_calls=result['value_calls'],
        trace_ref=tracepath.relative_to(OUTPUT).as_posix(),trace_sha256=file_hash(tracepath),pipeline_seconds=time.perf_counter()-started)
    atomic_json(p/'spec.json',job);atomic_json(p/'summary.json',row);return row

def run_matrix(cfg,base,data,priors,m,seeds,stage):
    policies={f'P{i}':load_policies(i) for i in range(5)};jobs=[]
    for c in cfg['conditions']:
        cid=key(c)
        for seed in seeds:
            scenario=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
            for label in LADDER if stage!='diagnostic' else LADDER[:5]:
                jobs.append(dict(id=f'{stage}-{cid}-s{seed}-{label}',stage=stage,condition=cid,scenario=scenario,label=label,
                    policy=policies[label][cid] if label in policies else {},prior=priors[cid],
                    parameters=parameters(base,rho_levels(scenario,base)['rho_max']),solver=base['solver'],code_hash=m['code_hash']))
    rows,wall=map_jobs(run_job,jobs,stage,cfg['max_parallel']);rows.sort(key=lambda r:r['id'])
    atomic_json(OUTPUT/'processed'/f'{stage}_runs.json',rows);write_csv(OUTPUT/'processed'/f'{stage}_runs.csv',rows)
    return dict(runs=len(rows),wall_seconds=wall,completed_at=now(),replay_steps=sum(r['replay_steps'] for r in rows))

def action_job(job):
    packet=read_compressed(OUTPUT/job['trace_ref'])['state_packets'];rows=[]
    policy=[load_policies(i)[job['condition']] for i in range(5)]
    for t in range(job['scenario']['horizon']):
        s,_=restore(job['scenario'],job['parameters'],job['solver'],packet[str(t)])
        plans=[decide_same_state(s,p,job['prior']) for p in policy]
        for n in range(4):
            a,b=plans[n],plans[n+1]
            rows.append(dict(condition=job['condition'],scenario_seed=job['scenario']['seed'],time=t,iteration=n+1,
                reference_state_policy='pi0',state_hash=digest(packet[str(t)]),policy_a=f'pi{n}',policy_b=f'pi{n+1}',
                full_action_disagree=int(action_hash(a)!=action_hash(b)),
                routes_disagree=int(a['routes']!=b['routes']),assignments_disagree=int(a['assignments']!=b['assignments']),
                wait_reposition_disagree=int(a['repositions']!=b['repositions'] or {r['vehicle_id'] for r in a['routes']}!={r['vehicle_id'] for r in b['routes']}),
                release_hub_disagree=int({r['vehicle_id']:r['e'] for r in a['routes']}!={r['vehicle_id']:r['e'] for r in b['routes']})))
    return rows

def action_diagnostics(cfg,base,priors):
    rows=read_json(OUTPUT/'processed/diagnostic_runs.json');jobs=[]
    for r in rows:
        if r['policy']!='P0':continue
        path=OUTPUT/'raw/runs'/r['id']/'spec.json';j=read_json(path);jobs.append(j|dict(trace_ref=r['trace_ref']))
    results,wall=map_jobs(action_job,jobs,'same-state action comparison',cfg['max_parallel'])
    out=[r for item in results for r in item]
    write_csv(OUTPUT/'processed/policy_action_disagreement.csv',out)
    atomic_json(OUTPUT/'processed/policy_action_disagreement.json',out)
    atomic_json(OUTPUT/'qa/same_state_action_audit.json',dict(status='PASSED',states=len(jobs)*base['horizon'],
        pair_comparisons=len(out),reference='pi0 diagnostic states, all periods; fixed before outcomes',wall_seconds=wall))
