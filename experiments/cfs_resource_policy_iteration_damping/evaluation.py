"""E(actual deployed parent), identical A sampling and full-cooldown physics.

The worker below is mechanically copied from frozen A; only explicit owned output
paths differ. No module globals in A are patched and no historical writer is called.
"""
import time
from pathlib import Path
import numpy as np
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from experiments.cfs_classical_value.simulation import feature_row,summary
from experiments.cfs_rollout_target_alignment.queries import pack_state
from experiments.cfs_resource_policy_iteration.targets import profiled_table_model
from experiments.cfs_resource_policy_iteration.protocol import resource_world,pair_audit,continuation,write_compressed,read_compressed,state_identity
from .common import OUTPUT,key,parameters,rho_levels,nodes,map_jobs,write_csv,now,file_hash,INTERPOLATION,TERMINAL

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
        query=[hubs[(scenario['seed']+index)%len(hubs)]]
        if index%3==0:query.append(hubs[(scenario['seed']+index+1)%len(hubs)])
        online=summary(state,job['prior'])
        for hub in query:
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
                        future_hash=future_hash,policy_seed=0,training_seed=None)
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
                    terminal_anchor=False,**{k:v for k,v in feature_row(t,hub,q,online).items() if k not in ['hub','q']}))
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

def evaluate(cfg,base,data,priors,m,n,alpha,policies):
    """Fresh on-policy E(Dn); only V0/V1 have an undamped reuse proof."""
    out=OUTPUT/'raw'/f'alpha_{alpha}'
    manifest=OUTPUT/'manifests'/f'alpha_{alpha}_raw_V{n}.json'
    if manifest.exists():
        result=read_json(manifest)
        assert result['code_hash']==m['code_hash']
        for ref,sha in result['tables'].items(): assert file_hash(OUTPUT/ref)==sha
        return result
    jobs=[]
    for c in cfg['conditions']:
        cid=key(c)
        for seed in cfg['construction_seeds']:
            scenario=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
            from .common import HISTORY
            jobs.append(dict(id=f'{cid}-s{seed}',iteration=n,alpha=alpha,
                output_root=str(out),scenario_ref=str(HISTORY/'raw/datasets'/f'{cid}-s{seed}.dataset.json'),
                condition=cid,scenario=scenario,fleet=c['fleet'],policy=policies[cid],prior=priors[cid],
                parameters=parameters(base,rho_levels(scenario,base)['rho_max']),solver=base['solver'],
                slots=cfg['target_supply_slots'],times=nodes(scenario['horizon'],cfg['chebyshev_node_count']),
                model_code_hash=m['code_hash']))
    qp=OUTPUT/'manifests'/f'alpha_{alpha}_V{n}_queries.json'
    queries=[{k:v for k,v in j.items() if k not in ['scenario','policy','prior']} |
        dict(scenario_hash=digest(j['scenario']),parent_policy_hash=j['policy']['policy_hash'],
            parent_table_hash=j['policy'].get('table_hash')) for j in jobs]
    if qp.exists(): assert read_json(qp)['queries']==queries
    else: atomic_json(qp,dict(created_at=now(),queries=queries,formal_seen=False))
    items,wall=map_jobs(target_job,jobs,f'alpha={cfg["alphas"][alpha]} E(D{n})',cfg['max_parallel'])
    rows=sorted([r for item in items for r in item['rows']],key=lambda r:(r['condition'],r['scenario_seed'],r['time'],r['hub'],r['q']))
    targetpath=OUTPUT/'targets'/f'alpha_{alpha}'/f'raw_V{n}'
    atomic_json(targetpath/'targets.json',rows);write_csv(targetpath/'targets.csv',rows)
    tables={};cells=[];lineage={};groups={}
    for r in rows: groups.setdefault((r['condition'],r['hub'],r['time'],r['q']),[]).append(r['target'])
    for c in cfg['conditions']:
        cid=key(c);rr=[r for r in rows if r['condition']==cid]
        model,ta,ti=profiled_table_model(rr,2*c['fleet'])
        model.update(interpolation=INTERPOLATION,terminal=TERMINAL,iteration=n,
            source_condition=cid,observed_max_q=c['fleet'])
        ref=f'tables/alpha_{alpha}/raw_V{n}/{cid}.json'
        atomic_json(OUTPUT/ref,model);tables[ref]=file_hash(OUTPUT/ref)
        lineage[cid]=dict(parent_policy_hash=policies[cid]['policy_hash'],
            parent_table_hash=policies[cid].get('table_hash'),table_hash=tables[ref],
            alpha=cfg['alphas'][alpha],iteration=n,code_hash=m['code_hash'],
            interpolation_hash=digest(INTERPOLATION),terminal_hash=digest(TERMINAL),
            construction_seeds=cfg['construction_seeds'],aggregation_seconds=ta,interpolation_seconds=ti)
        for h,qs in model['values'].items():
            for q,rec in qs.items():
                for t,v in zip(rec['times'],rec['values']):
                    yy=groups.get((cid,h,t,int(q)),[])
                    cells.append(dict(condition=cid,alpha=alpha,iteration=n,time=t,hub=h,q=int(q),value=v,
                        sample_count=len(yy),MC_SE=float(np.std(yy,ddof=1)/np.sqrt(len(yy))) if len(yy)>1 else None,
                        support_status='terminal' if t==base['horizon'] else 'observed' if yy else
                            'unsupported_endpoint' if int(q)>c['fleet'] else 'interpolated_q'))
    write_csv(targetpath/'cells.csv',cells)
    result=dict(status='FROZEN',code_hash=m['code_hash'],tables=tables,lineage=lineage,
        targets_ref=(targetpath/'targets.json').relative_to(OUTPUT).as_posix(),
        target_sha256=file_hash(targetpath/'targets.json'),world_root=out.relative_to(OUTPUT).as_posix(),
        worlds=sum(len(i['worlds']) for i in items),replay_steps=sum(i['replay_steps'] for i in items),
        value_calls=sum(sum(w['value_calls'] for w in i['worlds']) for i in items),
        jobs=len(items),wall_seconds=wall,worker_seconds=sum(i['seconds'] for i in items),
        parent_worker_seconds=sum(i['parent_seconds'] for i in items),
        simulation_worker_seconds=sum(i['simulation_seconds'] for i in items),
        serialization_worker_seconds=sum(i['serialization_seconds'] for i in items),
        QA_worker_seconds=sum(i['qa_seconds'] for i in items),created_at=now(),formal_seen=False)
    atomic_json(manifest,result);return result

def blend(previous,raw,alpha):
    """Exact scalar under-relaxation, no clipping/monotonicity intervention."""
    assert previous['values'].keys()==raw['values'].keys()
    import copy
    model=copy.deepcopy(raw)
    for h,qs in raw['values'].items():
        assert previous['values'][h].keys()==qs.keys()
        for q,rec in qs.items():
            p=previous['values'][h][q]
            assert p['times']==rec['times'] and len(p['values'])==len(rec['values'])
            model['values'][h][q]['values']=[(1-alpha)*a+alpha*b for a,b in zip(p['values'],rec['values'])]
    return model

def deploy(cfg,m,n,alpha):
    from .common import HISTORY
    tables={};proof=[]
    path=OUTPUT/'manifests'/f'alpha_{alpha}_deployed_V{n}.json'
    if path.exists():
        result=read_json(path)
        for ref,sha in result['tables'].items(): assert file_hash(OUTPUT/ref)==sha
        return result
    for c in cfg['conditions']:
        cid=key(c)
        source=HISTORY/'tables'/f'V{n}'/(cid+'.json') if n<2 else OUTPUT/'tables'/f'alpha_{alpha}'/f'raw_V{n}'/(cid+'.json')
        raw=read_json(source)
        if n<2:
            rawref=OUTPUT/'tables'/f'alpha_{alpha}'/f'raw_V{n}'/(cid+'.json')
            atomic_json(rawref,raw)
            assert file_hash(rawref)==file_hash(source), 'Shared V0/V1 must be byte-identical to A'
        previous=None if n==0 else OUTPUT/'tables'/f'alpha_{alpha}'/f'deployed_V{n-1}'/(cid+'.json')
        model=raw if previous is None else blend(read_json(previous),raw,cfg['alphas'][alpha])
        ref=f'tables/alpha_{alpha}/deployed_V{n}/{cid}.json';atomic_json(OUTPUT/ref,model)
        tables[ref]=file_hash(OUTPUT/ref)
        proof.append(dict(condition=cid,iteration=n,alpha=cfg['alphas'][alpha],
            raw_ref=str(source),raw_sha256=file_hash(source),
            previous_ref=None if previous is None else previous.relative_to(OUTPUT).as_posix(),
            previous_sha256=None if previous is None else file_hash(previous),
            deployed_ref=ref,deployed_sha256=tables[ref],
            reused_raw_reason='E(M0) identical' if n==0 else 'D1=U1, same construction paths and numerical E' if n==1 else None,
            code_hash=m['code_hash']))
    result=dict(status='FROZEN',tables=tables,proof=proof,created_at=now(),formal_seen=False)
    atomic_json(path,result);return result
