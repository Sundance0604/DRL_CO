"""Fresh paired targets and explicit, reproducible candidate interventions.

Action labels are NOT old labels with contexts pasted onto them. We commit a
fixed feasible observed coalition, hold other controls to the candidate release,
then perturb only FREE queried resources. The continuation is frozen Myopic V0.
This identifies an intervention-conditional marginal, not a joint optimal V.
"""
import itertools, time
import numpy as np
from experiment_core.storage import atomic_json, read_json, digest
from model_families.single.data import matching_state
from model_families.single.runner import state_snapshot
from model_families.single.planning import (route_candidates,decision_vehicles,committed_arc_load,
    compile_action,VehicleAction,PeriodAction)
from model_families.single.local_value.providers import context,features
from experiments.cfs_classical_value.build_mc_targets import resource_world,future_orders
from experiments.cfs_classical_value.simulation import simulate,summary,feature_row
from .common import OUTPUT,key,parameters,nodes,map_jobs,rho_levels,write_csv

def coalition(state,vehicle,route):
    orders=sorted(route['serve'],key=lambda o:(o.end_time,-o.revenue,o.id))
    anchors=[]
    if route['new']:
        if vehicle.phase=='idle': anchors.append([o for o in orders if o.departure==route['p']])
        anchors.append([o for o in orders if o.destination==route['e']])
    for combination in itertools.product(*anchors):
        chosen={o.id:o for o in combination}; loads=committed_arc_load(state.net,vehicle)
        for o in chosen.values():
            for arc in state.net.arcs(o.departure,o.destination): loads[arc]=loads.get(arc,0)+o.passenger
        if max(loads.values(),default=0)>state.parameters['capacity']: continue
        for o in orders:
            if o.id in chosen: continue
            arcs=state.net.arcs(o.departure,o.destination)
            if all(loads.get(a,0)+o.passenger<=state.parameters['capacity'] for a in arcs):
                chosen[o.id]=o
                for a in arcs: loads[a]=loads.get(a,0)+o.passenger
        return tuple(chosen)
    return None

def interventions(state,seed,index):
    out=[]; idle=[v for v in state.vehs if v.phase=='idle']
    if idle:
        v=idle[(seed+index)%len(idle)]
        ctx=context(state,v,None,state.t,state.t+1,v.loc,f'wait-{v.id}-{state.t}')
        out.append((ctx,PeriodAction(state.t)))
    candidates=[]
    for v in decision_vehicles(state.t,state.vehs):
        for r in route_candidates(state.t,state.net,state.pool,v,True,state.parameters['empty_cost'],state.parameters):
            ids=coalition(state,v,r)
            if ids is None: continue
            finish=r['t0']+state.net.tau[r['p'],r['e']]
            ctx=context(state,v,r,state.t,finish,r['e'],digest((seed,index,v.id,r['p'],r['e'],r['delay']))[:20])
            payload={k:value for k,value in r.items() if k!='serve'}
            action=PeriodAction(state.t,(VehicleAction(v.id,payload,ids),))
            candidates.append((digest(ctx),ctx,action))
    candidates.sort(key=lambda a:a[0])
    for _,ctx,action in candidates[:max(0,2-len(out))]: out.append((ctx,action))
    return out

def project_intervention(state,action,future,release):
    """Observed assignment fixed first; no reassignment while projecting to release."""
    s=state.clone(); t0=s.t; audit=None
    while s.t<min(release,s.T):
        if s.t!=t0: s.begin_period([o for o in future.values() if o.book_time==s.t])
        info={}; plan=compile_action(s,action if s.t==t0 else PeriodAction(s.t),info)
        s.commit_plan(plan)
        if audit is None:
            post=summary(s,{'rate':{}})
            ids=[oid for oid,_ in plan[2]['x']]
            audit=dict(backlog=post['backlog'],idle=post['idle'],incoming=post['incoming'],
                       served_count=len(ids),served_load=sum(future[o].passenger for o in ids))
    if s.t<s.T: s.begin_period([o for o in future.values() if o.book_time==s.t])
    return s,audit

def target_job(job):
    path=OUTPUT/'targets/raw'/f'{job["id"]}.json'; identity=digest(job)
    if path.exists():
        prior=read_json(path); assert prior['identity']==identity; return prior
    start=time.perf_counter();s=job['scenario'];T=s['horizon']
    sim,orders=matching_state(s,job['parameters'],job['solver'])
    _,_,states=simulate(sim,orders,collect_times=job['times'])
    donors=[matching_state(d,job['parameters'],job['solver'])[1] for d in job['donors']]
    hubs=sorted(sim.net.H);base=[];action_rows=[]
    slots=sorted(set(q for q in job['slots'] if q<=job['fleet'])|{job['fleet']})
    allq=sorted(set(slots+[q-1 for q in slots]))
    def paired(state,future,hub):
        profits={q:simulate(resource_world(state,hub,q),future,policy=job['continuation'],first_begun=True)[0]['realized_profit'] for q in allq}
        return profits
    for index,t in enumerate(job['times']):
        state=states[t]; online=summary(state,job['prior'])
        query_hubs=[hubs[(s['seed']+index)%len(hubs)],hubs[(s['seed']+index+1)%len(hubs)]]
        choices=interventions(state,s['seed'],index)
        for rep,donor in enumerate(donors):
            future=future_orders(orders,donor,t,rep)
            common=dict(condition=job['condition'],split=job['split'],scenario_seed=s['seed'],
                donor_seed=job['donors'][rep]['seed'],mc_replication=rep,decision_time=t,
                future_hash=digest([vars(o) for o in future.values()]),continuation_policy=job['continuation'],
                parent_state_hash=digest(state_snapshot(state)))
            for hub in query_hubs:
                profits=paired(state,future,hub)
                for q in slots:
                    base.append(common|dict(state_hash=common['parent_state_hash'],time=t,supply_slot=q,
                        profit_q=profits[q],profit_previous=profits[q-1],target=profits[q]-profits[q-1],**feature_row(t,hub,q,online)))
            for ctx,act in choices:
                release=ctx['release_time']; projected,audit=project_intervention(state,act,future,release)
                hub=ctx['release_hub']; f=features(ctx,online); post=summary(projected,job['prior'])
                profits=paired(projected,future,hub) if release<T else {q:0. for q in allq}
                for q in slots:
                    action_rows.append(common|f|dict(q=q,supply_slot=q,time=release,
                        candidate_id=ctx['candidate_id'],action_context=ctx,context_hash=digest(ctx),
                        intervention_assignment=list(act.vehicles[0].order_ids) if act.vehicles else [],
                        state_hash=digest(state_snapshot(projected)),profit_q=profits[q],profit_previous=profits[q-1],
                        target=profits[q]-profits[q-1],terminal_anchor=release>=T,
                        actual_post_backlog=audit['backlog'][hub],actual_post_idle=audit['idle'][hub],
                        actual_served_count=audit['served_count'],actual_served_load=audit['served_load'],
                        actual_release_backlog=post['backlog'][hub],actual_release_supply=post['idle'][hub],
                        actual_release_imbalance=post['backlog'][hub]-post['idle'][hub],
                        local_origin_idle=online['idle'][ctx['vehicle_origin_hub']]+ctx['origin_idle_delta'],
                        actual_post_origin_idle=audit['idle'][ctx['vehicle_origin_hub']]))
    for hub in hubs:
        for q in slots:
            base.append(dict(condition=job['condition'],split=job['split'],scenario_seed=s['seed'],
                time=T,hub=hub,q=q,supply_slot=q,mc_replication=-1,target=0.,terminal_anchor=True))
    saved=dict(identity=identity,base=base,actions=action_rows,seconds=time.perf_counter()-start)
    atomic_json(path,saved); return saved

def build_targets(cfg,baseline,data,priors,split,seeds,continuation='myopic'):
    jobs=[]; conditions=cfg['conditions'] if continuation=='myopic' else [cfg['secondary_condition']]
    for c in conditions:
        cid=key(c)
        for i,seed in enumerate(seeds):
            s=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
            jobs.append(dict(id=f'{split}-{cid}-{seed}-{continuation}',condition=cid,scenario=s,split=split,
                donors=[data[c['region'],c['orders'],c['fleet'],d]['scenarios'][0] for d in [seed,seeds[(i+7)%len(seeds)]]],
                parameters=parameters(baseline,rho_levels(s,baseline)['rho_max']),solver=baseline['solver'],
                times=nodes(s['horizon']) if continuation=='myopic' else [3,12,20],
                fleet=c['fleet'],slots=cfg['target_supply_slots'],prior=priors[cid],continuation=continuation))
    saved,wall=map_jobs(target_job,jobs,'Targets '+split, cfg['max_parallel'])
    base=[r for a in saved for r in a['base']]; actions=[r for a in saved for r in a['actions']]
    base.sort(key=lambda r:(r['condition'],r['scenario_seed'],r['time'],r['hub'],r['q'],r['mc_replication']))
    actions.sort(key=lambda r:(r['condition'],r['scenario_seed'],r['decision_time'],r['candidate_id'],r['q'],r['mc_replication']))
    atomic_json(OUTPUT/'targets'/f'{split}.json',dict(base=base,actions=actions))
    write_csv(OUTPUT/'targets'/f'{split}_slot.csv',base);write_csv(OUTPUT/'targets'/f'{split}_action.csv',actions)
    return dict(jobs=len(jobs),base_rows=len(base),action_rows=len(actions),wall_seconds=wall,
                worker_seconds=sum(a['seconds'] for a in saved))

def averaged_actions(rows):
    groups={}
    for r in rows: groups.setdefault((r['condition'],r['scenario_seed'],r['candidate_id'],r['q']),[]).append(r)
    return [a[0]|{'target':float(np.mean([r['target'] for r in a])),
                   'actual_release_backlog':float(np.mean([r['actual_release_backlog'] for r in a])),
                   'actual_release_supply':float(np.mean([r['actual_release_supply'] for r in a])),
                   'actual_release_imbalance':float(np.mean([r['actual_release_imbalance'] for r in a]))} for a in groups.values()]
