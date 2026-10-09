"""Scheduler-only future, original physical execution, auditable J minus g."""
import gzip, json, os, time
from experiment_core.storage import digest
from model_families.single.runner import state_snapshot,serialize_plan
from model_families.single.accounting import realized_ledger,completion_bound,cooldown_tick
from model_families.single.planning import PeriodAction,VehicleAction,compile_action
from .forced import forced_plan

def state_identity(state):
    return digest(dict(snapshot=state_snapshot(state),revision=state.revision,
        vehicles=[dict(id=v.id,orders={oid:vars(o) for oid,o in v.orders.items()},
            segments=v.segments,pickups=v.pickups,deliveries=v.deliveries) for v in state.vehs]))

def action_hash(plan):
    return digest({k:plan[k] for k in ['assignments','routes','repositions','expired']})

def physical_identity(state):
    snapshot=state_snapshot(state);snapshot.pop('profit')
    return digest(dict(snapshot=snapshot,revision=state.revision,
        vehicles=[dict(id=v.id,orders={oid:vars(o) for oid,o in v.orders.items()},
            segments=v.segments,pickups=v.pickups,deliveries=v.deliveries) for v in state.vehs]))

def assert_replay_snapshot(actual,expected):
    assert abs(actual['profit']-expected['profit'])<=1e-6,'replay cash discrepancy'
    assert {k:v for k,v in actual.items() if k!='profit'}=={k:v for k,v in expected.items() if k!='profit'},'physical replay discrepancy'

def write_compressed(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:
            f.write(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode('utf-8'))
    os.replace(temp,path)

def read_compressed(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)

def world(state,orders,candidate=None,freeze_release=None,timeout=600,keep_trace=True):
    """Current action is jointly optimized; only subsequent control may be frozen.

    Full ledger includes common pre-t committed cash/history. Subtract that same
    pre-t realized ledger to expose J_t; it cancels again in the paired contrast.
    g is business profit, NOT the solver delay tie-break or a value coefficient.
    J-g includes reconciliation of booked route cost with actual occupancy cost.
    """
    started=time.perf_counter();s=state.clone();initial=s.t
    origin=realized_ledger(s,orders,[],cutoff=initial)['realized_profit']
    rows=[];trace=[];solves=[]
    def check():
        if time.perf_counter()-started>timeout:raise TimeoutError('rollout wall budget exceeded')
    for t in range(initial,s.T):
        check()
        if t!=initial:s.begin_period([o for o in orders.values() if o.book_time==t])
        before=state_snapshot(s);identity=state_identity(s);info={}
        if t==initial:plan=forced_plan(s,candidate,info)
        elif freeze_release is not None and t<freeze_release:
            plan=compile_action(s,PeriodAction(t),info)
            info['status']='FROZEN_CONTINUE_WAIT'
        else:
            plan=forced_plan(s,None,info)
        if info.get('status')=='FEASIBLE_LIMIT':raise RuntimeError('Target uses a nonoptimal M0 incumbent')
        assert abs(info['accounting']['profit_identity_residual'])<=1e-6
        serialized=serialize_plan(plan,t,digest(before),info)
        assert all(orders[a['order_id']].book_time<=t for a in serialized['assignments'])
        assert state_identity(s)==identity,'solver mutated simulator state'
        if t==initial:
            g=float(plan[0]);current_hash=action_hash(serialized);current_plan=serialized
        if t==initial or freeze_release is None or t>=freeze_release:solves.append(t)
        s.commit_plan(plan);after=state_snapshot(s)
        rows.append(dict(period=t,plan=serialized))
        if keep_trace:trace.append(dict(period=t,before=before,after=after,plan=serialized,
            before_identity=identity,after_identity=state_identity(s),visible_order_ids=sorted(o['id'] for o in before['pool'])))
    pending=len(s.pool);cap=completion_bound(s);s.pool={}
    while any(v.phase!='idle' for v in s.vehs):
        check();assert s.t<cap,'unfinished commitment'
        before=state_snapshot(s);period=s.t;cooldown_tick(s)
        if keep_trace:trace.append(dict(period=period,drain=True,before=before,after=state_snapshot(s)))
    assert all(not v.orders and not v.onboard for v in s.vehs)
    ledger=realized_ledger(s,orders,rows)
    prior_ids={oid for v in state.vehs for oid in v.orders}
    new_ids=[a['order_id'] for r in rows for a in r['plan']['assignments']]
    assert len(new_ids)==len(set(new_ids)) and not set(new_ids)&prior_ids
    delivered={oid for v in s.vehs for oid in v.deliveries}
    assert set(new_ids)|prior_ids <= delivered,'assigned commitments not completed'
    total=ledger['realized_profit']-origin
    stage_sum=sum(r['plan']['operating_profit'] for r in rows)
    # Reconcile actual minus booked cash, including common old commitments.
    booked_fare=sum(r['plan']['accounting']['assignment_revenue'] for r in rows)
    assert abs(booked_fare-sum(orders[oid].revenue for oid in new_ids))<1e-6
    return dict(total=total,current_stage=g,future_component=total-g,current_action_hash=current_hash,
        current_plan=current_plan,solver_periods=solves,decisions=rows,trace=trace,
        ledger=ledger,origin_realized_common=origin,stage_profit_sum=stage_sum,
        realized_minus_booked=total-stage_sum,pending_at_T=pending,
        source_state_hash=state_identity(state),future_hash=digest([vars(orders[k]) for k in sorted(orders)]),
        execution_events=[dict(vehicle_id=str(v.id),segments=v.segments,pickups=v.pickups,deliveries=v.deliveries) for v in s.vehs],
        cooldown_periods=s.t-s.T,wall_seconds=time.perf_counter()-started,
        solver_seconds=sum(r['plan']['solver'].get('runtime',0) for r in rows),
        physical_final_hash=physical_identity(s),fare_count_once=True,completed_commitments=True)

def paired_target(candidate_world,baseline_world):
    assert candidate_world['source_state_hash']==baseline_world['source_state_hash']
    assert candidate_world['future_hash']==baseline_world['future_hash']
    return (candidate_world['total']-candidate_world['current_stage'])-(baseline_world['total']-baseline_world['current_stage'])

def replay_world(state,orders,result):
    """No optimizer: replay saved domain actions and independently recheck cash."""
    s=state.clone();rows=[];origin=realized_ledger(s,orders,[],cutoff=s.t)['realized_profit']
    for index,record in enumerate(result['trace']):
        t=record['period']
        if record.get('drain'):
            s.pool={}
            assert_replay_snapshot(state_snapshot(s),record['before']);cooldown_tick(s)
        else:
            if index:s.begin_period([o for o in orders.values() if o.book_time==t])
            assert_replay_snapshot(state_snapshot(s),record['before'])
            serial=record['plan'];routes={int(a['vehicle_id']):{k:v for k,v in a.items() if k!='vehicle_id'} for a in serial['routes']}
            repositions={int(a['vehicle_id']):a['hub'] for a in serial['repositions']}
            ids={}
            for a in serial['assignments']:ids.setdefault(int(a['vehicle_id']),[]).append(a['order_id'])
            actions=tuple(VehicleAction(k,routes.get(k),tuple(ids.get(k,[])),repositions.get(k))
                for k in sorted(set(routes)|set(repositions)|set(ids)))
            info={};plan=compile_action(s,PeriodAction(t,actions),info)
            assert abs(plan[0]-serial['operating_profit'])<1e-6
            replay=serialize_plan(plan,t,digest(state_snapshot(s)),info)
            rows.append(dict(period=t,plan=replay));s.commit_plan(plan)
        assert_replay_snapshot(state_snapshot(s),record['after'])
    s.pool={}
    assert physical_identity(s)==result['physical_final_hash']
    ledger=realized_ledger(s,orders,rows)
    assert abs(ledger['realized_profit']-origin-result['total'])<1e-6
    return dict(status='PASSED',periods=len(result['trace']),total_difference=ledger['realized_profit']-origin-result['total'])
