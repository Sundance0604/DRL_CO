"""Independent streams, whole-action CRN and pre-outcome resource-pair selection."""
import copy
import time
from pathlib import Path
from experiment_core.storage import atomic_json,read_json,digest
from experiments.cfs_rollout_target_alignment.protocol import write_compressed,read_compressed
from experiments.cfs_rollout_target_alignment.queries import pack_state
from model_families.single.engine.mt_prototype import Veh
from .common import OUTPUT,sha,parallel,now
from .states import restore_row
from .actions import historical,choose
from .futures import generate,frozen_prior
from .worlds import evaluate
from .semantics import audit

def resource_pairs(actions):
    """Choose before futures; hubs implicated by price versus baseline release changes."""
    from collections import Counter
    price=next(a for a in actions if 'pi1' in a['labels']);base=next(a for a in actions if 'M0' in a['labels'])
    ea=Counter((r['time'],r['hub']) for r in price['scores']['pi1']['release_events'])
    eb=Counter((r['time'],r['hub']) for r in base['scores']['pi1']['release_events'])
    hubs=sorted({h for t,h in set(ea)|set(eb) if ea[t,h]!=eb[t,h]})
    if not hubs:return []
    # If only one hub differs, examine two independent one-unit additions there.
    import itertools
    pairs=list(itertools.combinations_with_replacement(hubs,2))[:2]
    return [dict(hub_i=a,hub_j=b,quantity_i=1,quantity_j=1,time='current',
        selection='lexicographic hubs from release differences, before interaction outcomes') for a,b in pairs]

def world_cached(row,state,orders,spec,policy,prior,action,tag):
    wi=digest(dict(state=pack_state(state),future=spec['stream_hash'],action=None if action is None else action['action_hash'],
        continuation=policy['policy_hash'],numeric_code_hash=row['numeric_code_hash']))
    path=OUTPUT/'raw/worlds'/wi[:2]/(wi+'.json.gz')
    if path.exists():
        r=read_compressed(path);assert r['identity']==wi
    else:
        result=evaluate(state,orders,policy,prior,None if action is None else action['serial'])
        r=result|dict(identity=wi,initial=pack_state(state),state_ref=row['state_ref'],future_ref=spec['ref'],
            tag=tag,action_hash=None if action is None else action['action_hash'],numeric_code_hash=row['numeric_code_hash'])
        write_compressed(path,r)
    return dict(path=str(path),sha256=sha(path),identity=wi,total_USD=r['total'],
        replay_steps=r['replay']['periods'],replay_difference_USD=r['replay']['total_difference'],
        simulation_seconds=r['wall_seconds'],replay_seconds=r['replay_seconds'],solver_seconds=r['solver_seconds'])

def future_for(row,state,known,prior,role,k):
    orders,spec=generate(known,state,prior,row['condition'],row['state_id'],role,k)
    path=OUTPUT/'raw/futures'/f'{row["stage"]}-{row["state_id"]}-{role}-{k}.json'
    if path.exists():assert read_json(path)==spec
    else:atomic_json(path,spec)
    return orders,spec|dict(ref=str(path))

def joint_job(job):
    ref=job['ref'];packet=read_json(ref['path']);assert sha(ref['path'])==ref['sha256']
    state_ref=packet['state_ref'];row=read_json(state_ref['path']);assert sha(state_ref['path'])==state_ref['sha256']
    row['state_ref']=state_ref;state,known=restore_row(row);actions=packet['actions'];cfg=job['config']
    policies,prior=historical(row['condition']);future_prior=frozen_prior(row['condition'])
    out=OUTPUT/'raw/joint'/f'{row["state_id"]}.json';selection_file=OUTPUT/'raw/selections'/f'{row["state_id"]}.json'
    if out.exists():return read_json(out)
    started=time.perf_counter();select={a['action_hash']:[] for a in actions};worlds=[]
    pairs=resource_pairs(actions)
    pairfile=OUTPUT/'raw/interactions'/f'{row["state_id"]}-pairs.json'
    pairspec=dict(state_id=row['state_id'],pairs=pairs,future_seen=False)
    if pairfile.exists():assert read_json(pairfile)==pairspec
    else:atomic_json(pairfile,pairspec)
    for k in range(cfg['selection_paths']):
        orders,spec=future_for(row,state,known,future_prior,'selection',k)
        for a in actions:
            w=world_cached(row,state,orders,spec,policies['M0'],prior,a,'joint')
            worlds.append(w);select[a['action_hash']].append(w['total_USD'])
    import numpy as np
    selections=dict(price=choose(actions,{a['action_hash']:a['scores']['pi1']['exact_optimizer_score_USD'] for a in actions}),
        bucket_price=choose(actions,{a['action_hash']:a['scores']['B2']['exact_optimizer_score_USD'] for a in actions}),
        select8=choose(actions,{h:float(np.mean(v[:cfg['nested_selection_paths']])) for h,v in select.items()}),
        select32=choose(actions,{h:float(np.mean(v)) for h,v in select.items()}),
        M0=dict(action_hash=next(a['action_hash'] for a in actions if 'M0' in a['labels'])))
    selection=dict(state_id=row['state_id'],choices=selections,selection_returns=select,
        selection_role='selection',assessment_seen=False,action_manifest_sha256=ref['sha256'])
    if selection_file.exists():assert read_json(selection_file)==selection
    else:atomic_json(selection_file,selection)
    # Assessment values do not flow into any selector, threshold or candidate generation.
    assess={a['action_hash']:[] for a in actions};interaction=[];fixed=[];sensitivity=[]
    for k in range(cfg['assessment_paths']):
        orders,spec=future_for(row,state,known,future_prior,'assessment',k)
        for a in actions:
            w=world_cached(row,state,orders,spec,policies['M0'],prior,a,'joint');worlds.append(w)
            assess[a['action_hash']].append(w['total_USD'])
    if row['scenario_seed'] in cfg['sensitivity_seeds']:
        for k in range(cfg['assessment_paths']):
            orders,spec=future_for(row,state,known,future_prior,'assessment',k)
            chosen={selections[s]['action_hash'] for s in ['price','select8','select32','M0']}
            vals={}
            for a in actions:
                if a['action_hash'] in chosen:
                    w=world_cached(row,state,orders,spec,policies['pi1'],prior,a,'continuation_sensitivity');worlds.append(w)
                    vals[a['action_hash']]=w['total_USD']
            sensitivity.append(dict(replicate=k,continuation='pi1',returns=vals))
    for pindex,pair in enumerate(pairs):
        values=[]
        for k in range(cfg['interaction_paths']):
            orders,spec=future_for(row,state,known,future_prior,'interaction',k);v=[]
            for mask in range(4):
                s=state.clone()
                if mask&1:s.vehs.append(Veh(len(s.vehs),pair['hub_i']))
                if mask&2:s.vehs.append(Veh(len(s.vehs),pair['hub_j']))
                w=world_cached(row,s,orders,spec,policies['M0'],prior,None,f'interaction-{pindex}-{mask}')
                worlds.append(w);v.append(w['total_USD'])
            values.append(dict(replicate=k,J00=v[0],J10=v[1],J01=v[2],J11=v[3],interaction_USD=v[3]-v[1]-v[2]+v[0]))
        interaction.append(dict(pair=pair,values=values,mean_USD=float(np.mean([v['interaction_USD'] for v in values])),
            MC_SE_USD=float(np.std([v['interaction_USD'] for v in values],ddof=1)/np.sqrt(len(values))),
            counterfactual_fleet_exceeds_K=True,cross_time_injection=False))
    if row['scenario_seed'] in cfg['fixed_evaluation_seeds']:
        from experiments.cfs_resource_policy_iteration.protocol import resource_world,pair_audit
        h=sorted({r['hub'] for a in actions for r in a['scores']['pi1']['release_events']})[0]
        # Deterministic first action-relevant hub/q=1, identical physical base for all parent policies.
        low=resource_world(state,h,0);high=resource_world(state,h,1);pair_audit(low,high,h)
        for label in ['M0','pi1']+(['pi3'] if row['condition'].startswith('california_') else []):
            vals=[]
            for k in range(cfg['fixed_evaluation_paths']):
                orders,spec=future_for(row,state,known,future_prior,'fixed_evaluation',k)
                lo=world_cached(row,low,orders,spec,policies[label],prior,None,'fixed-resource-0')
                hi=world_cached(row,high,orders,spec,policies[label],prior,None,'fixed-resource-1')
                worlds.extend([lo,hi]);vals.append(hi['total_USD']-lo['total_USD'])
            fixed.append(dict(continuation=label,hub=h,q=1,values_USD=vals,mean_USD=float(np.mean(vals)),
                physical_base_hash=digest(pack_state(state)),policy_hash=policies[label]['policy_hash']))
    qa,alignment,fallback=audit(row,actions)
    result=dict(state_id=row['state_id'],condition=row['condition'],source=row['source'],stage=row['stage'],
        scenario_seed=row['scenario_seed'],time=row['time'],state_ref=state_ref,action_ref=ref,
        selection_sha256=sha(selection_file),selections=selections,selection_returns=select,assessment_returns=assess,
        interactions=interaction,fixed=fixed,sensitivity=sensitivity,worlds=worlds,
        query_semantics=qa,bucket_base_alignment=alignment,fallback_audit=fallback,missing=packet['missing'],
        seconds=time.perf_counter()-started,distribution=future_prior['distribution'])
    atomic_json(out,result);return result

def run_joint(cfg,manifest,stage):
    path=OUTPUT/'manifests'/f'{stage}_joint.json'
    if path.exists():return read_json(path)
    jobs=[dict(ref=r,config=cfg) for r in manifest['actions']]
    rows,wall=parallel(joint_job,jobs,stage+' independent full-action MC',cfg['max_parallel'])
    refs=[]
    for r in sorted(rows,key=lambda r:r['state_id']):
        p=OUTPUT/'raw/joint'/f'{r["state_id"]}.json';refs.append(dict(path=str(p),sha256=sha(p),state_id=r['state_id']))
    m=dict(status='NUMERICAL_COMPLETE',stage=stage,results=refs,worlds=sum(len(r['worlds']) for r in rows),
        wall_seconds=wall,worker_seconds=sum(r['seconds'] for r in rows),created_at=now())
    atomic_json(path,m);return m
