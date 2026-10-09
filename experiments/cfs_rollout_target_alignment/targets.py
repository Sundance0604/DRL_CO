"""Same frozen query, joint current action, paired freeze/rolling worlds."""
import time
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.planning import route_candidates,PeriodAction,VehicleAction
from experiments.cfs_local_post_bayesian_value.targets import coalition,project_intervention
from experiments.cfs_classical_value.build_mc_targets import resource_world
from experiments.cfs_classical_value.simulation import simulate
from .common import OUTPUT,map_jobs,write_csv
from .queries import restore_state
from .forced import route_key,route_payload
from .protocol import world,paired_target,replay_world,state_identity,write_compressed,read_compressed

def old_resource_diagnostic(state,orders,candidate):
    """Fourth procedure, not the primary estimand: fixed coalition, release q=1-0."""
    if candidate['kind']=='wait':action=PeriodAction(state.t);ids=[]
    else:
        v=state.vehs[candidate['vehicle_id']]
        route=next(r for r in route_candidates(state.t,state.net,state.pool,v,state.parameters['cross_hub'],state.parameters['empty_cost'],state.parameters)
            if route_key(r)==candidate['route_key'])
        ids=coalition(state,v,route);assert ids is not None
        action=PeriodAction(state.t,(VehicleAction(v.id,route_payload(route),ids),))
    release=candidate['release_time'];hub=candidate['release_hub']
    projected,_=project_intervention(state,action,orders,release)
    if release>=state.T:profits={0:0.,1:0.}
    else:profits={q:simulate(resource_world(projected,hub,q),orders,first_begun=True)[0]['realized_profit'] for q in [0,1]}
    return dict(fourth_q=1,fourth_fixed_order_subset=list(ids),fourth_projected_state_hash=state_identity(projected),
        fourth_profit_q=profits[1],fourth_profit_previous=profits[0],fourth_resource_marginal=profits[1]-profits[0],
        fourth_diagnostic_only=True)

def target_job(job):
    start=time.perf_counter();packet=read_compressed(OUTPUT/job['state_file'])
    outfile=OUTPUT/'raw/targets'/f'{job["label"]}-{job["condition"]}-{job["seed"]}.json'
    identity=digest(job)
    if outfile.exists():
        saved=read_json(outfile);assert saved['identity']==identity;return saved
    rows=[];audits=[];runtime={'roll':0.,'freeze':0.,'historical':0.};replay_periods=0;world_count=0
    for t in sorted({q['time'] for q in job['queries']}):
        state,orders=restore_state(packet['scenario'],packet['parameters'],packet['solver'],packet['states'][str(t)])
        assert state_identity(state)==next(q['state_hash'] for q in job['queries'] if q['time']==t)
        cache={}
        def get(candidate,mode,release=None):
            nonlocal replay_periods,world_count
            cid='baseline' if candidate is None or candidate['baseline'] else candidate['query_id']
            cachekey=(cid,mode,release)
            if cachekey in cache:return cache[cachekey]
            name=f't{t}-{cid}-{mode}'+('' if release is None else f'-h{release}')+'.json.gz'
            path=OUTPUT/'raw/worlds'/job['label']/f'{job["condition"]}-{job["seed"]}'/name
            expected=digest(dict(state=state_identity(state),candidate=cid,mode=mode,release=release,
                future=digest([vars(orders[k]) for k in sorted(orders)]),code=job['code_hash']))
            if path.exists():
                saved=read_compressed(path);assert saved['identity']==expected;w=saved['world']
            else:
                w=world(state,orders,candidate,release,timeout=job['timeout'])
                ra=replay_world(state,orders,w)
                write_compressed(path,dict(identity=expected,world=w,replay=ra))
            assert w['source_state_hash']==state_identity(state)
            ra=replay_world(state,orders,w);replay_periods+=ra['periods'];world_count+=1
            runtime[mode]+=w['wall_seconds'];cache[cachekey]=(w,path.relative_to(OUTPUT).as_posix())
            return cache[cachekey]
        baseline_roll,brref=get(None,'roll')
        for q in [q for q in job['queries'] if q['time']==t]:
            assert q['future_hash']==baseline_roll['future_hash']
            assert q['baseline_action_hash']==baseline_roll['current_action_hash']
            cr,crref=get(q,'roll')
            if job['paired']:
                bf,bfref=get(None,'freeze',q['release_time']);cf,cfref=get(q,'freeze',q['release_time'])
                assert cr['current_action_hash']==cf['current_action_hash'] and baseline_roll['current_action_hash']==bf['current_action_hash']
                assert cr['current_stage']==cf['current_stage'] and baseline_roll['current_stage']==bf['current_stage']
                yfreeze=paired_target(cf,bf)
                oldstart=time.perf_counter();legacy=old_resource_diagnostic(state,orders,q);runtime['historical']+=time.perf_counter()-oldstart
            else:bf=cf={};bfref=cfref=None;yfreeze=None;legacy={}
            yroll=paired_target(cr,baseline_roll)
            assert cr['solver_periods']==list(range(t,state.T))
            if q['baseline']:assert yroll==0 and (yfreeze is None or yfreeze==0)
            row=q|dict(candidate_release_time=q['release_time'],candidate_release_hub=q['release_hub'],
                candidate_duration=q['context']['route_duration'],candidate_deadhead=q['context']['deadhead_distance'],
                candidate_current_stage=cr['current_stage'],baseline_current_stage=baseline_roll['current_stage'],
                candidate_total_roll=cr['total'],baseline_total_roll=baseline_roll['total'],
                candidate_total_freeze=cf.get('total'),baseline_total_freeze=bf.get('total'),
                Y_roll=yroll,Y_freeze=yfreeze,target=yroll,
                solve_count_after_intervention=len(cr['solver_periods'])-1,
                candidate_action_hash=cr['current_action_hash'],candidate_roll_ref=crref,baseline_roll_ref=brref,
                candidate_freeze_ref=cfref,baseline_freeze_ref=bfref,
                candidate_realized_minus_booked=cr['realized_minus_booked'],baseline_realized_minus_booked=baseline_roll['realized_minus_booked'],
                candidate_roll_seconds=cr['wall_seconds'],candidate_freeze_seconds=cf.get('wall_seconds'),
                candidate_stage_identity_residual=cr['current_plan']['accounting']['profit_identity_residual'],**legacy)
            rows.append(row)
            audits.append(dict(query_id=q['query_id'],state_hash=q['state_hash'],future_hash=q['future_hash'],
                next_period_reoptimized=True,current_reward_count_once=True,pair_hash_identical=True,
                candidate_current_action_identical_between_modes=not job['paired'] or cr['current_action_hash']==cf['current_action_hash']))
    saved=dict(identity=identity,rows=rows,audits=audits,runtime=runtime,wall_seconds=time.perf_counter()-start,
        unique_worlds=world_count,replay_periods=replay_periods)
    atomic_json(outfile,saved);return saved

def build_targets(cfg,manifest,query_manifest,label='phase1',paired=True):
    grouped={}
    for q in query_manifest['queries']:grouped.setdefault((q['condition'],q['scenario_seed']),[]).append(q)
    jobs=[]
    for (cid,seed),queries in grouped.items():
        jobs.append(dict(label=label,condition=cid,seed=seed,queries=queries,paired=paired,
            state_file=f'raw/query_states/{cid}-{seed}.json.gz',code_hash=manifest['code_hash'],timeout=cfg['run_timeout_seconds']))
    results,wall=map_jobs(target_job,jobs,'Targets '+label,cfg['max_parallel'])
    rows=sorted([r for a in results for r in a['rows']],key=lambda r:r['query_id'])
    audits=[r for a in results for r in a['audits']]
    atomic_json(OUTPUT/'targets'/f'{label}.json',rows)
    write_csv(OUTPUT/'targets'/f'{label}_paired_targets.csv',rows)
    write_csv(OUTPUT/'targets'/f'{label}_roll_targets.csv',rows)
    if paired:
        write_csv(OUTPUT/'targets/freeze_targets.csv',rows)
        write_csv(OUTPUT/'targets/rolling_targets.csv',rows)
        write_csv(OUTPUT/'processed/paired_target_differences.csv',[r|dict(difference=r['Y_roll']-r['Y_freeze'],absolute_difference=abs(r['Y_roll']-r['Y_freeze'])) for r in rows])
    atomic_json(OUTPUT/'qa'/f'{label}_target_audit.json',dict(status='PASSED',rows=len(audits),queries=audits,
        unique_worlds=sum(a['unique_worlds'] for a in results),replay_periods=sum(a['replay_periods'] for a in results)))
    runtime={mode:sum(a['runtime'][mode] for a in results) for mode in ['roll','freeze','historical']}
    return dict(rows=len(rows),jobs=len(jobs),wall_seconds=wall,worker_seconds=sum(a['wall_seconds'] for a in results),
        runtime=runtime,rolling_to_freeze_ratio=runtime['roll']/runtime['freeze'] if runtime['freeze'] else None)
