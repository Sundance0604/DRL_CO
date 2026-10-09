"""E(pi): fresh on-policy states and full-cooldown paired q marginal worlds."""
import time
import numpy as np
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from experiments.cfs_classical_value.fit_models import table_model
from experiments.cfs_classical_value.simulation import feature_row,summary
from experiments.cfs_rollout_target_alignment.queries import pack_state
from .common import OUTPUT,key,parameters,rho_levels,nodes,map_jobs,write_csv,now,file_hash,INTERPOLATION,TERMINAL
from .protocol import resource_world,pair_audit,continuation,write_compressed,read_compressed,state_identity

def profiled_table_model(rows,fleet):
    """Same ordered means/np.interp as frozen table_model, separate timings."""
    started=time.perf_counter();groups={}
    for r in rows:groups.setdefault((r['hub'],r['time'],r['q']),[]).append(r['target'])
    means={cell:float(np.mean(values)) for cell,values in groups.items()}
    aggregation=time.perf_counter()-started;started=time.perf_counter()
    times=sorted({r['time'] for r in rows});hubs=sorted({r['hub'] for r in rows});values={}
    for h in hubs:
        values[h]={}
        for q in range(1,fleet+1):
            vals=[]
            for t in times:
                qs=sorted(qq for hh,tt,qq in means if hh==h and tt==t)
                if not qs:raise RuntimeError('MC table missing time/hub cell')
                vals.append(float(np.interp(q,qs,[means[h,t,qq] for qq in qs])))
            values[h][str(q)]=dict(times=times,values=vals)
    model=dict(method='table',values=values,target='mean paired marginal by time/hub/q; time and unobserved integer q interpolated',
        state_dependence='current backlog/other idle/incoming ignored; construction-only surface')
    return model,aggregation,time.perf_counter()-started

def target_job(job):
    start=time.perf_counter();identity=digest(job)
    root=OUTPUT/'targets'/f'iter{job["iteration"]}'/'raw'
    path=root/(job['id']+'.json')
    if path.exists():
        saved=read_json(path);assert saved['identity']==identity
        return saved
    scenario=job['scenario'];s,orders=matching_state(scenario,job['parameters'],job['solver'])
    s.begin_period([o for o in orders.values() if o.book_time==0])
    parent=continuation(s,orders,job['policy'],job['prior'],collect_times=job['times'])
    parentpath=OUTPUT/'raw/traces'/f'construction-iter{job["iteration"]}-{job["id"]}.json.gz'
    write_compressed(parentpath,{k:v for k,v in parent.items() if k!='states'}|
        dict(state_packets={str(t):pack_state(v) for t,v in parent['states'].items()}))
    future_hash=digest([vars(orders[k]) for k in sorted(orders)])
    rows=[];world_index=[];solver_count=0;calls=0;replay_steps=0;simulation_seconds=0.;serialization_seconds=0.;qa_seconds=0.
    for index,t in enumerate(job['times']):
        state=parent['states'][t];hubs=sorted(state.net.H)
        query=[hubs[(scenario['seed']+index)%len(hubs)]]
        if index%3==0:query.append(hubs[(scenario['seed']+index+1)%len(hubs)])
        online=summary(state,job['prior'])
        for hub in query:
            slots=sorted({q for q in job['slots'] if q<=job['fleet']}|{job['fleet']})
            allq=sorted(set(slots)|{q-1 for q in slots});profits={};worlds={}
            for q in allq:
                w=resource_world(state,hub,q);identifier=f'{job["id"]}-t{t}-h{hub}-q{q}'
                wp=OUTPUT/'raw/worlds'/f'iter{job["iteration"]}'/(identifier+'.json.gz')
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
                        initial=pack_state(w),scenario_ref=f'raw/datasets/{scenario["id"]}.dataset.json',
                        condition=job['condition'],iteration=job['iteration'],time=t,hub=hub,q=q,
                        future_hash=future_hash,policy_seed=0,training_seed=None)
                    ts=time.perf_counter();write_compressed(wp,result);serialization_seconds+=time.perf_counter()-ts
                profits[q]=result['metrics']['realized_profit'];worlds[q]=w
                calls+=result['value_calls'];simulation_seconds+=result['metrics']['daily_wall_seconds'];qa_seconds+=result['replay_seconds']
                replay_steps+=result['replay']['periods'];solver_count+=state.T-t
                world_index.append(dict(path=wp.relative_to(OUTPUT).as_posix(),identity=wi,policy_hash=result['policy_hash'],
                    table_hash=result['table_hash'],value_calls=result['value_calls'],replay_steps=result['replay']['periods']))
            for q in slots:
                qa_started=time.perf_counter();audit=pair_audit(worlds[q-1],worlds[q],hub);qa_seconds+=time.perf_counter()-qa_started
                rows.append(dict(condition=job['condition'],iteration=job['iteration'],scenario_seed=scenario['seed'],
                    policy_seed=0,training_seed=None,time=t,hub=hub,q=q,supply_slot=q,
                    profit_q=profits[q],profit_previous=profits[q-1],target=profits[q]-profits[q-1],
                    continuation_policy_id=job['policy']['policy_id'],continuation_policy_hash=job['policy']['policy_hash'],
                    continuation_table_hash=job['policy'].get('table_hash'),state_hash=state_identity(state),
                    future_hash=future_hash,paired_common_hash=audit['common_hash'],paired_state_check=True,
                    terminal_anchor=False,**{k:v for k,v in feature_row(t,hub,q,online).items() if k not in ['hub','q']}))
    for hub in sorted(s.net.H):
        for q in sorted({q for q in job['slots'] if q<=job['fleet']}|{job['fleet']}):
            rows.append(dict(condition=job['condition'],iteration=job['iteration'],scenario_seed=scenario['seed'],
                policy_seed=0,training_seed=None,time=s.T,hub=hub,q=q,supply_slot=q,target=0.,
                continuation_policy_id=job['policy']['policy_id'],continuation_policy_hash=job['policy']['policy_hash'],
                continuation_table_hash=job['policy'].get('table_hash'),terminal_anchor=True,
                anchor_proof='paired terminal resource toy, no dispatch at time>=T; waiting_cost=0'))
    saved=dict(identity=identity,rows=rows,worlds=world_index,parent_trace=parentpath.relative_to(OUTPUT).as_posix(),
        parent_value_calls=parent['value_calls'],parent_table_hash=parent['table_hash'],
        parent_seconds=parent['seconds'],simulation_seconds=simulation_seconds,serialization_seconds=serialization_seconds,
        replay_steps=replay_steps,solver_periods=solver_count,qa_seconds=qa_seconds,seconds=time.perf_counter()-start)
    atomic_json(path,saved);return saved

def evaluate(cfg,base,data,priors,m,iteration,policies):
    manifestpath=OUTPUT/'manifests'/f'iteration_{iteration}_manifest.json'
    if manifestpath.exists():
        frozen=read_json(manifestpath)
        for path,sha in frozen['tables'].items():assert file_hash(OUTPUT/path)==sha
        return frozen
    jobs=[]
    for c in cfg['conditions']:
        cid=key(c)
        for seed in cfg['construction_seeds']:
            scenario=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
            jobs.append(dict(id=f'{cid}-s{seed}',iteration=iteration,condition=cid,scenario=scenario,fleet=c['fleet'],
                policy=policies[cid],prior=priors[cid],parameters=parameters(base,rho_levels(scenario,base)['rho_max']),
                solver=base['solver'],slots=cfg['target_supply_slots'],times=nodes(scenario['horizon'],cfg['chebyshev_node_count']),
                model_code_hash=m['code_hash']))
    atomic_json(OUTPUT/'manifests'/f'iteration_{iteration}_query_manifest.json',dict(created_at=now(),
        status='FROZEN_BEFORE_CURRENT_TARGETS',queries=[{k:v for k,v in j.items() if k not in ['scenario','policy','prior']}|
        dict(scenario_hash=digest(j['scenario']),parent_policy_hash=j['policy']['policy_hash']) for j in jobs],
        future_sampling=cfg['future_sampling'],state_sampling=cfg['state_sampling']))
    items,wall=map_jobs(target_job,jobs,f'E(pi{iteration})',cfg['max_parallel'])
    ts=time.perf_counter();rows=sorted([r for item in items for r in item['rows']],key=lambda r:(r['condition'],r['scenario_seed'],r['time'],r['hub'],r['q']))
    collection_seconds=time.perf_counter()-ts;ts=time.perf_counter()
    atomic_json(OUTPUT/'targets'/f'iter{iteration}'/'targets.json',rows)
    write_csv(OUTPUT/'targets'/f'iter{iteration}'/'targets.csv',rows)
    target_serialization_seconds=time.perf_counter()-ts;ts=time.perf_counter();tables={};lineage={};cells=[]
    table_aggregation_seconds=table_interpolation_seconds=table_serialization_seconds=0.
    for c in cfg['conditions']:
        cid=key(c);rr=[r for r in rows if r['condition']==cid]
        model,ta,ti=profiled_table_model(rr,2*c['fleet']);table_aggregation_seconds+=ta;table_interpolation_seconds+=ti
        model.update(interpolation=INTERPOLATION,terminal=TERMINAL,
            iteration=iteration,source_condition=cid,observed_max_q=c['fleet'])
        tablepath=f'tables/V{iteration}/{cid}.json';serialization_started=time.perf_counter()
        atomic_json(OUTPUT/tablepath,model);tables[tablepath]=file_hash(OUTPUT/tablepath);table_serialization_seconds+=time.perf_counter()-serialization_started
        for h,qs in model['values'].items():
            for q,record in qs.items():
                for t,value in zip(record['times'],record['values']):
                    support=[r for r in rr if r['hub']==h and r['time']==t and r['q']==int(q)]
                    observed=int(q)<=c['fleet'] and bool(support)
                    yy=[r['target'] for r in support]
                    cells.append(dict(condition=cid,iteration=iteration,time=t,hub=h,q=int(q),value=value,
                        support_status='terminal' if t==base['horizon'] else 'observed' if observed else
                            'unsupported_endpoint' if int(q)>c['fleet'] else 'interpolated_q',
                        independent_scenarios=len(support),MC_SD=float(np.std(yy,ddof=1)) if len(yy)>1 else None,
                        MC_SE=float(np.std(yy,ddof=1)/np.sqrt(len(yy))) if len(yy)>1 else None))
        lineage[cid]=dict(iteration_id=iteration,parent_policy_id=policies[cid]['policy_id'],
            continuation_policy_hash=policies[cid]['policy_hash'],continuation_table_hash=policies[cid].get('table_hash'),
            construction_seed_set=cfg['construction_seeds'],network_hash=digest(jobs[[j['condition'] for j in jobs].index(cid)]['scenario']['edges']),
            model_code_hash=m['code_hash'],value_provider_hash=m['source_code_sha256']['model_families/single/value/providers.py'],
            target_generator_hash=file_hash(OUTPUT/'targets.py'),table_hash=tables[tablepath],
            interpolation_rule_hash=digest(INTERPOLATION),terminal_rule_hash=digest(TERMINAL),created_at=now())
    write_csv(OUTPUT/'tables'/f'V{iteration}'/'cells.csv',cells)
    result=dict(status='FROZEN',iteration_id=iteration,created_at=now(),tables=tables,lineage=lineage,
        target_hash=file_hash(OUTPUT/'targets'/f'iter{iteration}'/'targets.json'),jobs=len(jobs),paired_targets=sum(not r['terminal_anchor'] for r in rows),
        terminal_anchors=sum(r['terminal_anchor'] for r in rows),worlds=sum(len(i['worlds']) for i in items),
        wall_seconds=wall,worker_seconds=sum(i['seconds'] for i in items),simulation_worker_seconds=sum(i['simulation_seconds'] for i in items),
        parent_policy_worker_seconds=sum(i['parent_seconds'] for i in items),serialization_worker_seconds=sum(i['serialization_seconds'] for i in items),
        aggregate_interpolate_serialize_seconds=time.perf_counter()-ts,replay_steps=sum(i['replay_steps'] for i in items),
        table_aggregation_seconds=table_aggregation_seconds,table_interpolation_seconds=table_interpolation_seconds,
        table_serialization_seconds=table_serialization_seconds,target_collection_seconds=collection_seconds,
        target_serialization_seconds=target_serialization_seconds,paired_world_QA_worker_seconds=sum(i['qa_seconds'] for i in items),
        value_calls=sum(sum(w['value_calls'] for w in i['worlds']) for i in items),formal_seen=False)
    atomic_json(manifestpath,result);return result
