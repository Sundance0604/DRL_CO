"""Four independent one-step MC improvements; no selected winning bank."""
import time
from experiment_core.storage import read_json,atomic_json,digest
from model_families.single.data import matching_state
from model_families.single.runner import state_snapshot
from experiments.cfs_rollout_target_alignment.queries import pack_state
from experiments.cfs_resource_policy_iteration.protocol import resource_world,pair_audit
from experiments.cfs_resource_policy_iteration.targets import profiled_table_model
from experiments.cfs_rollout_target_alignment.protocol import write_compressed,read_compressed
from .common import OUTPUT,key,parameters,rho_levels,nodes,parallel,sha,now,A
from .actions import historical,decide,verify,physical_action
from .worlds import evaluate
from .states import restore_row

def target_job(job):
    p=OUTPUT/'raw/banks'/f'bank{job["bank"]}'/(job['id']+'.json')
    if p.exists():
        r=read_json(p);assert r['identity']==digest(job);return r
    started=time.perf_counter();s,orders=matching_state(job['scenario'],job['parameters'],job['solver'])
    policies,prior=historical(job['condition']);targets=[];worlds=[];states={};source_trace=[]
    for t in range(s.T):
        s.begin_period([o for o in orders.values() if o.book_time==t])
        if t in job['times']:states[t]=s.clone()
        action=decide(s,policies['M0'],prior);plan,serial=verify(s,action);s.commit_plan(plan)
        source_trace.append(dict(period=t,plan=serial))
    for i,t in enumerate(job['times']):
        state=states[t];hubs=sorted(state.net.H);hs=[hubs[(job['seed']+i)%len(hubs)]]
        if i%3==0:hs.append(hubs[(job['seed']+i+1)%len(hubs)])
        qs=sorted({q for q in job['slots'] if q<=job['fleet']}|{job['fleet']});allq=sorted(set(qs)|{q-1 for q in qs})
        for h in hs:
            amounts={};physical={}
            for q in allq:
                state_q=resource_world(state,h,q);physical[q]=state_q
                wp=OUTPUT/'raw/banks'/f'bank{job["bank"]}'/'worlds'/f'{job["id"]}-t{t}-h{h}-q{q}.json.gz'
                wi=digest(dict(job=digest(job),time=t,hub=h,q=q))
                if wp.exists():result=read_compressed(wp);assert result['identity']==wi
                else:
                    result=evaluate(state_q,orders,policies['M0'],prior)|dict(initial=pack_state(state_q),
                        scenario_ref=job['scenario_ref'],identity=wi,time=t,hub=h,q=q,bank=job['bank'])
                    write_compressed(wp,result)
                amounts[q]=result['total'];worlds.append(dict(path=str(wp),sha256=sha(wp),identity=wi,
                    replay_steps=result['replay']['periods']))
            for q in qs:
                pair_audit(physical[q-1],physical[q],h)
                targets.append(dict(condition=job['condition'],scenario_seed=job['seed'],time=t,hub=h,q=q,
                    target=amounts[q]-amounts[q-1],terminal_anchor=False,bank=job['bank'],parent='M0'))
    for h in sorted(s.net.H):
        for q in sorted({q for q in job['slots'] if q<=job['fleet']}|{job['fleet']}):
            targets.append(dict(condition=job['condition'],scenario_seed=job['seed'],time=s.T,hub=h,q=q,
                target=0.,terminal_anchor=True,bank=job['bank'],parent='M0'))
    row=dict(identity=digest(job),spec=job,targets=targets,worlds=worlds,source_trace=source_trace,
        seconds=time.perf_counter()-started);atomic_json(p,row);return row

def construct(cfg,base,data):
    manifest=OUTPUT/'manifests/banks_tables.json'
    if manifest.exists():return read_json(manifest)
    tables=[];runtime=[]
    for bank,seeds in enumerate(cfg['bank_seeds'],1):
        jobs=[]
        for c in cfg['conditions']:
            cid=key(c)
            for seed in seeds:
                scene=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
                jobs.append(dict(id=f'{cid}-s{seed}',bank=bank,condition=cid,seed=seed,scenario=scene,
                    scenario_ref=str(OUTPUT/'raw/datasets'/f'{cid}-s{seed}.dataset.json'),fleet=c['fleet'],
                    times=nodes(scene['horizon'],cfg['bank_nodes']),slots=cfg['bank_supply_slots'],
                    parameters=parameters(base,rho_levels(scene,base)['rho_max']),solver=base['solver'],
                    numeric_code_hash=read_json(OUTPUT/'manifests/source_manifest.json')['code_hash']))
        items,wall=parallel(target_job,jobs,f'Independent bank {bank} E(M0)',cfg['max_parallel'])
        for c in cfg['conditions']:
            cid=key(c);rr=[r for x in items for r in x['targets'] if r['condition']==cid]
            model,ta,ti=profiled_table_model(rr,2*c['fleet'])
            model.update(parent='M0',bank=bank,seeds=seeds,observed_max_q=c['fleet'])
            path=OUTPUT/'tables'/f'bank{bank}'/f'{cid}.json';atomic_json(path,model)
            tables.append(dict(bank=bank,condition=cid,path=str(path),sha256=sha(path),
                target_hash=digest(rr),policy_hash=digest(dict(table_sha256=sha(path),framework='pure H1 slot-prefix'))))
        runtime.append(dict(bank=bank,wall_seconds=wall,worker_seconds=sum(r['seconds'] for r in items),
            worlds=sum(len(r['worlds']) for r in items)))
    m=dict(status='FROZEN_BEFORE_DEPLOYMENT',tables=tables,runtime=runtime,created_at=now(),
        no_bank_selection=True,one_evaluation_one_improvement=True)
    atomic_json(manifest,m);return m

def deploy_job(j):
    path=OUTPUT/'raw/deployment'/f'{j["condition"]}-s{j["seed"]}-{j["label"]}.json.gz'
    ident=digest(j)
    if path.exists():r=read_compressed(path);assert r['identity']==ident;return r['summary']
    s,orders=matching_state(j['scenario'],j['parameters'],j['solver']);s.begin_period([o for o in orders.values() if o.book_time==0])
    policies,prior=historical(j['condition'])
    policy=policies['M0'] if j['label']=='M0' else dict(table=read_json(j['table']['path']),
        table_hash=j['table']['sha256'],policy_hash=j['table']['policy_hash'])
    result=evaluate(s,orders,policy,prior)
    summary=dict(condition=j['condition'],scenario_seed=j['seed'],policy=j['label'],profit_USD=result['total'],
        table_sha256=policy.get('table_hash'),world_ref=str(path),policy_hash=policy['policy_hash'],
        replay_steps=result['replay']['periods'],replay_difference_USD=result['replay']['total_difference'],
        seconds=result['wall_seconds'])
    write_compressed(path,result|dict(identity=ident,initial=pack_state(s),scenario=j['scenario'],
        parameters=j['parameters'],solver=j['solver'],summary=summary));return summary

def deploy(cfg,base,data,tables):
    path=OUTPUT/'manifests/banks_deployment.json'
    if path.exists():return read_json(path)
    jobs=[]
    for c in cfg['conditions']:
        cid=key(c)
        for seed in cfg['deployment_seeds']:
            scene=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
            for label in ['M0','bank1','bank2','bank3','bank4']:
                tab=None if label=='M0' else next(x for x in tables['tables'] if x['condition']==cid and x['bank']==int(label[-1]))
                jobs.append(dict(condition=cid,seed=seed,label=label,scenario=scene,table=tab,
                    parameters=parameters(base,rho_levels(scene,base)['rho_max']),solver=base['solver']))
    rows,wall=parallel(deploy_job,jobs,'Four frozen banks paired deployment',cfg['max_parallel'])
    # Compare all bank policies on the already frozen complete diagnostic states.
    fixed=[]
    for ref in read_json(OUTPUT/'manifests/diagnostic_actions.json')['actions']:
        ar=read_json(ref['path']);row=read_json(ar['state_ref']['path']);s,_=restore_row(row)
        _,prior=historical(row['condition'])
        for tab in tables['tables']:
            if tab['condition']!=row['condition']:continue
            policy=dict(table=read_json(tab['path']),table_hash=tab['sha256'],policy_hash=tab['policy_hash'])
            action=decide(s,policy,prior)
            fixed.append(dict(state_id=row['state_id'],source=row['source'],condition=row['condition'],bank=tab['bank'],
                action_hash=digest(physical_action(action)),state_hash=row['state_hash'],table_sha256=tab['sha256']))
    atomic_json(OUTPUT/'processed/bank_fixed_state_actions.json',fixed)
    atomic_json(path,dict(status='NUMERICAL_COMPLETE',runs=rows,wall_seconds=wall,created_at=now(),
        conditional_CI_unit='40 shared scenarios per frozen model; not 160 independent samples',
        four_training_banks_descriptive_only=True));return read_json(path)
