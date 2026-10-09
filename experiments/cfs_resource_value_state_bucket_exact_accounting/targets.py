"""Shared M0 target worlds for base/bucket; bucket fixed BEFORE injection."""
import time
from pathlib import Path
import numpy as np
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from experiments.cfs_classical_value.simulation import feature_row,summary
from experiments.cfs_rollout_target_alignment.queries import pack_state
from experiments.cfs_resource_policy_iteration.targets import profiled_table_model
from experiments.cfs_resource_policy_iteration.protocol import resource_world,pair_audit,write_compressed,read_compressed,state_identity
from .target_protocol import continuation
from .common import OUTPUT,key,parameters,rho_levels,nodes,map_jobs,write_csv,now,file_hash,INTERPOLATION,TERMINAL
from .pressure import pressure,bucket

def target_job(job):
    out=Path(job['output_root']);start=time.perf_counter();identity=digest(job)
    root=out/'targets'/f'iter{job["iteration"]}'/'raw'
    path=root/(job['id']+'.json')
    if path.exists():
        saved=read_json(path);assert saved['identity']==identity
        return saved
    scenario=job['scenario'];s,orders=matching_state(scenario,job['parameters'],job['solver'])
    s.begin_period([o for o in orders.values() if o.book_time==0])
    parent=continuation(s,orders,job['policy'],job['prior'],collect_times=job['times'])
    parentpath=out/'raw/traces'/f'construction-iter{job["iteration"]}-{job["id"]}.json.gz'
    write_compressed(parentpath,{k:v for k,v in parent.items() if k!='states'}|
        dict(state_packets={str(t):pack_state(v) for t,v in parent['states'].items()}))
    future_hash=digest([vars(orders[k]) for k in sorted(orders)])
    rows=[];world_index=[];solver_count=0;calls=0;replay_steps=0;simulation_seconds=0.;serialization_seconds=0.;qa_seconds=0.
    for index,t in enumerate(job['times']):
        state=parent['states'][t];hubs=sorted(state.net.H)
        query=hubs  # identical all-hub density for BOTH base and bucket estimators
        online=summary(state,job['prior'])
        for hub in query:
            pr=pressure(state,job['pressure_prior'],online)[hub]
            pre_bucket=bucket(pr['R'],job['thresholds'])
            slots=sorted({q for q in job['slots'] if q<=job['fleet']}|{job['fleet']})
            allq=sorted(set(slots)|{q-1 for q in slots});profits={};worlds={}
            for q in allq:
                w=resource_world(state,hub,q);identifier=f'{job["id"]}-t{t}-h{hub}-q{q}'
                wp=out/'raw/worlds'/f'iter{job["iteration"]}'/(identifier+'.json.gz')
                wi=digest(dict(job=identity,time=t,hub=hub,q=q,state=state_identity(w),future=future_hash))
                if wp.exists():
                    result=read_compressed(wp);assert result['identity']==wi
                else:
                    evaluated=continuation(w,orders,job['policy'],job['prior'])
                    assert evaluated['policy_hash']==job['policy']['policy_hash']
                    if job['iteration']>0 and t<state.T:
                        assert evaluated['table_hash']==job['policy']['table_hash']
                        # A call may be absent only with zero vehicles/no pre-T release slots.
                    result={k:v for k,v in evaluated.items() if k!='states'}|dict(identity=wi,
                        initial=pack_state(w),scenario_ref=job['scenario_ref'],
                        condition=job['condition'],iteration=job['iteration'],time=t,hub=hub,q=q,
                        future_hash=future_hash,policy_seed=0,training_seed=None,
                        pre_injection_bucket=pre_bucket,pre_injection_pressure=pr,
                        pre_injection_state_hash=state_identity(state))
                    ts=time.perf_counter();write_compressed(wp,result);serialization_seconds+=time.perf_counter()-ts
                profits[q]=result['metrics']['realized_profit'];worlds[q]=w
                calls+=result['value_calls'];simulation_seconds+=result['metrics']['daily_wall_seconds'];qa_seconds+=result['replay_seconds']
                replay_steps+=result['replay']['periods'];solver_count+=state.T-t
                world_index.append(dict(path=wp.relative_to(out).as_posix(),sha256=file_hash(wp),identity=wi,policy_hash=result['policy_hash'],
                    table_hash=result['table_hash'],value_calls=result['value_calls'],replay_steps=result['replay']['periods']))
            for q in slots:
                qa_started=time.perf_counter();audit=pair_audit(worlds[q-1],worlds[q],hub);qa_seconds+=time.perf_counter()-qa_started
                rows.append(dict(condition=job['condition'],iteration=job['iteration'],scenario_seed=scenario['seed'],
                    policy_seed=0,training_seed=None,time=t,hub=hub,q=q,supply_slot=q,
                    profit_q=profits[q],profit_previous=profits[q-1],target=profits[q]-profits[q-1],
                    continuation_policy_id=job['policy']['policy_id'],continuation_policy_hash=job['policy']['policy_hash'],
                    continuation_table_hash=job['policy'].get('table_hash'),state_hash=state_identity(state),
                    future_hash=future_hash,paired_common_hash=audit['common_hash'],paired_state_check=True,
                    terminal_anchor=False,bucket=pre_bucket,pre_injection_pressure=pr,
                    bucket_computed_before_resource_replacement=True,**{k:v for k,v in feature_row(t,hub,q,online).items() if k not in ['hub','q']}))
    for hub in sorted(s.net.H):
        for q in sorted({q for q in job['slots'] if q<=job['fleet']}|{job['fleet']}):
            rows.append(dict(condition=job['condition'],iteration=job['iteration'],scenario_seed=scenario['seed'],
                policy_seed=0,training_seed=None,time=s.T,hub=hub,q=q,supply_slot=q,target=0.,
                continuation_policy_id=job['policy']['policy_id'],continuation_policy_hash=job['policy']['policy_hash'],
                continuation_table_hash=job['policy'].get('table_hash'),terminal_anchor=True,
                anchor_proof='paired terminal resource toy, no dispatch at time>=T; waiting_cost=0'))
    saved=dict(identity=identity,rows=rows,worlds=world_index,parent_trace=parentpath.relative_to(out).as_posix(),
        parent_value_calls=parent['value_calls'],parent_table_hash=parent['table_hash'],
        parent_seconds=parent['seconds'],simulation_seconds=simulation_seconds,serialization_seconds=serialization_seconds,
        replay_steps=replay_steps,solver_periods=solver_count,qa_seconds=qa_seconds,seconds=time.perf_counter()-start)
    atomic_json(path,saved);return saved

def survey_job(job):
    ident=digest(job);path=OUTPUT/'raw/survey'/(job['id']+'.json.gz')
    if path.exists():
        r=read_compressed(path);assert r['identity']==ident;return r['pressure_rows']
    s,orders=matching_state(job['scenario'],job['parameters'],job['solver'])
    s.begin_period([o for o in orders.values() if o.book_time==0])
    result=continuation(s,orders,job['policy'],job['prior'],collect_times=job['times'])
    rows=[]
    for t,state in result['states'].items():
        pp=pressure(state,job['pressure_prior'])
        for h,r in pp.items():
            rows.append(dict(condition=job['condition'],scenario_seed=job['scenario']['seed'],
                time=t,hub=h,state_hash=state_identity(state),**r))
    write_compressed(path,{k:v for k,v in result.items() if k!='states'}|dict(identity=ident,
        state_packets={str(t):pack_state(v) for t,v in result['states'].items()},pressure_rows=rows))
    return rows

def prepare(cfg,base,data,m):
    """Freeze expected prior and construction quantile thresholds BEFORE labels."""
    path=OUTPUT/'manifests/threshold_freeze.json'
    if path.exists():
        r=read_json(path)
        for name,sha in r['files'].items():assert file_hash(OUTPUT/name)==sha
        return read_json(OUTPUT/'tables/priors.json')
    assert not (OUTPUT/'targets/construction/targets.json').exists()
    from .pressure import expected_prior,thresholds
    from experiments.cfs_classical_value.build_fluid_value import demand_prior
    priors={};jobs=[]
    for c in cfg['conditions']:
        cid=key(c);scenarios=[data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0] for seed in cfg['construction_seeds']]
        pp=expected_prior(scenarios,base['capacity']);bp=demand_prior(scenarios)
        priors[cid]=dict(base_prior=bp,pressure_prior=pp)
        for scenario in scenarios:
            jobs.append(dict(id=f'{cid}-s{scenario["seed"]}',condition=cid,scenario=scenario,
                parameters=parameters(base,rho_levels(scenario,base)['rho_max']),solver=base['solver'],
                policy=dict(policy_id='M0',policy_hash=digest(dict(condition=cid,parent='M0',code=m['code_hash'])),table=None,table_hash=None),
                prior=bp,pressure_prior=pp,times=nodes(scenario['horizon'],cfg['chebyshev_node_count']),code_hash=m['code_hash']))
    items,wall=map_jobs(survey_job,jobs,'B construction pressure survey',cfg['max_parallel'])
    rows=[r for item in items for r in item];cuts=[]
    for c in cfg['conditions']:
        cid=key(c);record=thresholds([r for r in rows if r['condition']==cid])
        priors[cid]['thresholds']=record;cuts.append(dict(condition=cid,**record))
    write_csv(OUTPUT/'processed/pressure_construction.csv',rows)
    write_csv(OUTPUT/'processed/bucket_thresholds.csv',cuts)
    atomic_json(OUTPUT/'tables/priors.json',priors)
    atomic_json(path,dict(status='FROZEN',created_at=now(),formal_seen=False,targets_seen=False,
        survey_wall_seconds=wall,construction_samples=60,time_hub_rows=len(rows),
        files={n:file_hash(OUTPUT/n) for n in ['tables/priors.json','processed/pressure_construction.csv','processed/bucket_thresholds.csv']},
        pressure_unit=cfg['unit'],source='construction only; M0 before injection; all five hubs and seven time nodes'))
    return priors

def evaluate(cfg,base,data,priors,m,stage):
    mp=OUTPUT/'manifests'/f'{stage}_target_manifest.json'
    if mp.exists():
        r=read_json(mp);assert file_hash(OUTPUT/r['targets_ref'])==r['targets_sha256'];return r
    assert read_json(OUTPUT/'manifests/threshold_freeze.json')['status']=='FROZEN'
    seeds=cfg[f'{stage}_seeds'];jobs=[];out=OUTPUT/'raw'/stage
    for c in cfg['conditions']:
        cid=key(c)
        for seed in seeds:
            scenario=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
            jobs.append(dict(id=f'{cid}-s{seed}',iteration=0,output_root=str(out),condition=cid,
                scenario=scenario,scenario_ref=str(OUTPUT/'raw/datasets'/f'{cid}-s{seed}.dataset.json'),fleet=c['fleet'],
                policy=dict(policy_id='M0',policy_hash=digest(dict(condition=cid,parent='M0',code=m['code_hash'])),table=None,table_hash=None),
                prior=priors[cid]['base_prior'],pressure_prior=priors[cid]['pressure_prior'],thresholds=priors[cid]['thresholds'],
                parameters=parameters(base,rho_levels(scenario,base)['rho_max']),solver=base['solver'],
                slots=cfg['target_supply_slots'],times=nodes(scenario['horizon'],cfg['chebyshev_node_count']),model_code_hash=m['code_hash']))
    queries=[{k:v for k,v in j.items() if k not in ['scenario','prior','pressure_prior','policy']}|dict(
        scenario_hash=digest(j['scenario']),parent_policy_hash=j['policy']['policy_hash'],pressure_prior_hash=digest(j['pressure_prior'])) for j in jobs]
    qp=OUTPUT/'manifests'/f'{stage}_query_freeze.json'
    if qp.exists():assert read_json(qp)['queries']==queries
    else:atomic_json(qp,dict(created_at=now(),queries=queries,formal_seen=False))
    items,wall=map_jobs(target_job,jobs,f'B {stage} shared M0 paired targets',cfg['max_parallel'])
    rows=sorted([r for item in items for r in item['rows']],key=lambda r:(r['condition'],r['scenario_seed'],r['time'],r['hub'],r['q']))
    ref=f'targets/{stage}/targets.json';atomic_json(OUTPUT/ref,rows);write_csv(OUTPUT/'targets'/stage/'targets.csv',rows)
    result=dict(status='FROZEN',created_at=now(),code_hash=m['code_hash'],targets_ref=ref,
        targets_sha256=file_hash(OUTPUT/ref),worlds=sum(len(i['worlds']) for i in items),
        replay_steps=sum(i['replay_steps'] for i in items),wall_seconds=wall,worker_seconds=sum(i['seconds'] for i in items),
        shared_parent_M0=True,shared_future_base_bucket=True,all_hubs=True,formal_seen=False)
    atomic_json(mp,result);return result

def build_tables(cfg,m):
    path=OUTPUT/'manifests/table_freeze.json'
    if path.exists():
        r=read_json(path)
        for name,sha in r['tables'].items():assert file_hash(OUTPUT/name)==sha
        return r
    rows=read_json(OUTPUT/'targets/construction/targets.json');tables={};audit=[]
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:
        if not r['terminal_anchor']:groups[r['condition'],r['hub'],r['time'],r['q'],r['bucket']].append(r['target'])
    for c in cfg['conditions']:
        cid=key(c);rr=[r for r in rows if r['condition']==cid]
        base,_,_=profiled_table_model(rr,2*c['fleet'])
        base.update(interpolation=INTERPOLATION,terminal=TERMINAL,observed_max_q=c['fleet'],source_condition=cid)
        measured=sorted({q for q in cfg['target_supply_slots'] if q<=c['fleet']}|{c['fleet']})
        support={};buckets={}
        for b in range(3):
            # Unsupported corners carry base placeholders; they are NEVER used
            # as bucket prices without the explicit support gate in provider.py.
            import copy
            bm=copy.deepcopy(base)
            for h,qs in base['values'].items():
                support.setdefault(h,{})[str(b)]={}
                for q in measured:
                    support[h][str(b)][str(q)]={}
                    for t in qs['1']['times']:
                        yy=groups[cid,h,t,q,b]
                        support[h][str(b)][str(q)][str(t)]=len(yy)
                        audit.append(dict(condition=cid,time=t,hub=h,q=q,bucket=b,sample_count=len(yy),
                            MC_SE=float(np.std(yy,ddof=1)/np.sqrt(len(yy))) if len(yy)>1 else None,
                            raw_value=float(np.mean(yy)) if yy else None,cell_count=1,
                            interpolated_flag=False,unsupported_flag=len(yy)<10 and t<24,
                            terminal_anchor=t==24,fallback='base_tuq' if len(yy)<10 and t<24 else None))
                for q,rec in bm['values'][h].items():
                    vals=[]
                    for t in rec['times']:
                        qmeans=[]
                        for qq in measured:
                            yy=groups[cid,h,t,qq,b]
                            qmeans.append(float(np.mean(yy)) if len(yy)>=10 else base['values'][h][str(qq)]['values'][rec['times'].index(t)])
                        vals.append(float(np.interp(int(q),measured,qmeans)))
                    rec['values']=vals
            buckets[str(b)]=bm
        model=dict(method='bucket_table',base=base,buckets=buckets,support=support,measured_q=measured,
            observed_max_q=c['fleet'],minimum_support=10,conditional_on='ORIGINAL pre-injection bucket',
            interpolation=INTERPOLATION,terminal=TERMINAL,code_hash=m['code_hash'],
            base_and_bucket_same_targets_sha256=file_hash(OUTPUT/'targets/construction/targets.json'))
        for ref,payload in [(f'tables/base_tuq/{cid}.json',base),(f'tables/bucket_tuqb/{cid}.json',model)]:
            atomic_json(OUTPUT/ref,payload);tables[ref]=file_hash(OUTPUT/ref)
    write_csv(OUTPUT/'processed/support_audit.csv',audit)
    result=dict(status='FROZEN',created_at=now(),formal_seen=False,diagnostic_seen=False,
        tables=tables,target_sha256=file_hash(OUTPUT/'targets/construction/targets.json'),
        threshold_freeze_sha256=file_hash(OUTPUT/'manifests/threshold_freeze.json'),shared_M0_parent=True)
    atomic_json(path,result);return result
