"""Freeze candidate selection before evaluating any continuation outcome."""
import time
from types import SimpleNamespace
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from model_families.single.runner import state_snapshot,serialize_plan
from model_families.single.planning import decision_vehicles,route_candidates
from model_families.single.local_value.providers import context,features
from experiments.cfs_classical_value.simulation import simulate,summary
from experiments.cfs_local_post_bayesian_value.targets import coalition
from .common import OUTPUT,key,parameters,rho_levels,map_jobs,write_csv,now
from .forced import route_key,route_payload,forced_plan
from .protocol import state_identity,action_hash,write_compressed,read_compressed

def pack_state(s):
    vehicles=[]
    for v in s.vehs:
        vehicles.append(vars(v)|dict(orders=list(v.orders),onboard=sorted(v.onboard)))
    return dict(t=s.t,T=s.T,J=s.J,S=s.S,revision=s.revision,pool=list(s.pool),vehicles=vehicles)

def restore_state(scenario,p,solver,packet):
    s,orders=matching_state(scenario,p,solver)
    s.t=packet['t'];s.J=packet['J'];s.S=packet['S'].copy();s.revision=packet['revision']
    s.pool={oid:orders[oid] for oid in packet['pool']}
    for v,data in zip(s.vehs,packet['vehicles']):
        for name,val in data.items():setattr(v,name,val)
        v.orders={oid:orders[oid] for oid in data['orders']};v.onboard=set(data['onboard'])
    return s,orders

def choose_candidates(state,limit=8):
    """Only current-state feasibility/metadata. No orders schedule argument."""
    info={};baseline=forced_plan(state,None,info)
    serial=serialize_plan(baseline,state.t,digest(state_snapshot(state)),info)
    selected_routes={kid:baseline[3][kid][ri] for kid,ri in baseline[2]['y']}
    selected=[];missing=[]
    online=summary(state,{'rate':{}})
    deciding=decision_vehicles(state.t,state.vehs)
    if not deciding:missing.append(dict(vehicle_id=None,missing_reason='no_deciding_vehicle'))
    for v in deciding:
        options=[]
        def add(route):
            kind='wait' if route is None else 'route'
            release=state.t+1 if route is None else route['t0']+state.net.tau[route['p'],route['e']]
            hub=v.loc if route is None else route['e']
            payload=dict(vehicle_id=v.id,kind=kind,route_key=None if route is None else route_key(route),
                route=None if route is None else route_payload(route),release_time=release,release_hub=hub)
            ch=digest(payload);cid=ch[:24]
            ctx=context(state,v,route,state.t,release,hub,cid)
            baseline_choice=(route is None and v.id not in selected_routes) or (
                route is not None and v.id in selected_routes and route_key(route)==route_key(selected_routes[v.id]))
            options.append(payload|dict(candidate_hash=ch,candidate_id=cid,baseline=baseline_choice,context=ctx,
                route_type='wait' if route is None else 'same_hub' if hub==v.loc else 'cross_hub',
                metadata_features=features(ctx,online),compatible_order_ids=[] if route is None else sorted(o.id for o in route['serve'])))
        if v.phase=='idle':add(None)
        for route in route_candidates(state.t,state.net,state.pool,v,state.parameters['cross_hub'],state.parameters['empty_cost'],state.parameters):
            # The old coalition routine is ONLY an existence certificate, not a fixed assignment.
            if coalition(state,v,route) is not None:add(route)
        options.sort(key=lambda a:a['candidate_hash'])
        base=[a for a in options if a['baseline']];assert len(base)==1
        if len(options)<2:
            missing.append(dict(vehicle_id=v.id,missing_reason='no_feasible_alternative',available_candidates=len(options)))
            continue
        chosen=base.copy()
        def take(items):
            if len(chosen)>=limit:return
            new=[a for a in items if a['candidate_id'] not in {b['candidate_id'] for b in chosen}]
            if new:chosen.append(new[0])
        take([a for a in options if a['kind']=='wait'])
        take(sorted(options,key=lambda a:(a['context']['route_duration'],a['candidate_hash'])))
        take(sorted(options,key=lambda a:(-a['context']['route_duration'],a['candidate_hash'])))
        for hub in sorted({a['release_hub'] for a in options}):
            if not any(a['release_hub']==hub for a in chosen):take([a for a in options if a['release_hub']==hub])
        take([a for a in options if a['route_type']=='same_hub'])
        while len(chosen)<min(limit,len(options)):take(options)
        selected.extend(chosen)
    return selected,missing,serial

def freeze_job(job):
    start=time.perf_counter();scenario=job['scenario'];p=job['parameters'];solver=job['solver']
    s,orders=matching_state(scenario,p,solver)
    _,_,states=simulate(s,orders,collect_times=job['times'])
    records=[];missing=[];packets={}
    futurehash=digest([vars(orders[k]) for k in sorted(orders)])
    for t in job['times']:
        state=states[t];before=state_identity(state);candidates,absent,baseline=choose_candidates(state,job['limit'])
        assert state_identity(state)==before
        sh=before;packets[str(t)]=pack_state(state)
        common=dict(condition=job['condition'],scenario_seed=scenario['seed'],time=t,state_hash=sh,
            future_hash=futurehash,network_hash=digest(scenario['edges']),cost_hash=digest(p),
            commitments_hash=digest([dict(id=v.id,orders=sorted(v.orders),route=v.route,hold=v.hold,dead=v.dead,arrive=v.arrive) for v in state.vehs]),
            baseline_action_hash=action_hash(baseline),remaining_horizon=state.T-t,
            current_backlog_load=sum(o.passenger for o in state.pool.values())/p['capacity'],
            fleet_tightness=sum(v.phase!='idle' for v in state.vehs)/len(state.vehs))
        for c in candidates:
            hub=c['release_hub'];release_idle=sum(v.phase=='idle' and v.loc==hub for v in state.vehs)
            records.append(common|c|dict(query_id=digest((job['condition'],scenario['seed'],t,c['candidate_hash']))[:28],
                release_hub_current_supply=release_idle,policy_seed=0,training_seed=None))
        missing.extend(common|a for a in absent)
    file=OUTPUT/'raw/query_states'/f'{job["condition"]}-{scenario["seed"]}.json.gz'
    write_compressed(file,dict(scenario=scenario,parameters=p,solver=solver,states=packets))
    return dict(rows=records,missing=missing,seconds=time.perf_counter()-start,
        state_file=file.relative_to(OUTPUT).as_posix())

def freeze_queries(cfg,base,data,seeds,conditions,label='phase1'):
    path=OUTPUT/'manifests'/f'{label}_query_manifest.json'
    if path.exists():return read_json(path)
    assert not any((OUTPUT/'raw/targets').glob(f'{label}-*')),'Queries must precede target outcomes'
    jobs=[]
    for c in conditions:
        for seed in seeds:
            s=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
            jobs.append(dict(scenario=s,condition=key(c),parameters=parameters(base,rho_levels(s,base)['rho_max']),
                solver=base['solver'],times=cfg['query_times'],limit=cfg['max_candidates_per_vehicle']))
    items,wall=map_jobs(freeze_job,jobs,'Freeze '+label+' queries',cfg['max_parallel'])
    rows=sorted([r for item in items for r in item['rows']],key=lambda r:r['query_id'])
    missing=[r for item in items for r in item['missing']]
    manifest=dict(status='FROZEN_BEFORE_TARGETS',label=label,created_at=now(),queries=rows,missing=missing,
        query_digest=digest(rows),missing_digest=digest(missing),state_files=[i['state_file'] for i in items],
        selection='all deciding vehicles; canonical baseline; wait/hub/duration coverage; current metadata and deterministic hash only',
        future_used_for_selection=False,selection_wall_seconds=wall,selection_worker_seconds=sum(i['seconds'] for i in items),
        scenario_seeds=seeds,conditions=conditions)
    atomic_json(path,manifest)
    if label=='phase1':atomic_json(OUTPUT/'query_manifest.json',manifest)
    write_csv(OUTPUT/'processed'/f'{label}_missing_queries.csv',missing)
    return manifest
