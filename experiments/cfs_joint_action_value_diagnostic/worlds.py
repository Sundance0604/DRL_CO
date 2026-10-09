"""Force the complete frozen current action. No current-period reoptimization."""
import time
from experiment_core.storage import digest
from model_families.single.runner import state_snapshot, serialize_plan
from model_families.single.accounting import realized_ledger, cooldown_tick, completion_bound
from experiments.cfs_rollout_target_alignment.protocol import state_identity,physical_identity,replay_world
from .actions import decide, verify, physical_action

def evaluate(state,orders,policy,prior,action=None,timeout=600):
    started=time.perf_counter();s=state.clone();initial=s.t;rows=[];trace=[]
    origin=realized_ledger(s,orders,[],cutoff=initial)['realized_profit'];initialhash=state_identity(s)
    for t in range(initial,s.T):
        if time.perf_counter()-started>timeout:raise TimeoutError('Full-cooldown world budget exceeded')
        if t!=initial:s.begin_period([o for o in orders.values() if o.book_time==t])
        before=state_snapshot(s)
        serial=action if t==initial and action is not None else decide(s,policy,prior)
        plan,compiled=verify(s,serial)
        assert all(orders[a['order_id']].book_time<=t for a in compiled['assignments'])
        # Keep raw solver/cash diagnostics; independent compilation supplies executed money.
        compiled['solver']=serial['solver']
        if t==initial:
            current=physical_action(compiled)
            if action is not None:assert current==physical_action(action),'Current joint action changed'
        s.commit_plan(plan);rows.append(dict(period=t,plan=compiled))
        trace.append(dict(period=t,before=before,after=state_snapshot(s),plan=compiled))
    pending=len(s.pool);s.pool={};cap=completion_bound(s)
    while any(v.phase!='idle' for v in s.vehs):
        assert s.t<cap,'Unfinished commitment';before=state_snapshot(s);t=s.t;cooldown_tick(s)
        trace.append(dict(period=t,drain=True,before=before,after=state_snapshot(s)))
    ledger=realized_ledger(s,orders,rows)
    assigned=[a['order_id'] for r in rows for a in r['plan']['assignments']]
    oldassigned={oid for v in state.vehs for oid in v.orders}
    delivered={oid for v in s.vehs for oid in v.deliveries}
    assert len(assigned)==len(set(assigned)) and not set(assigned)&oldassigned
    assert set(assigned)|oldassigned <= delivered
    assert all(not v.orders and not v.onboard for v in s.vehs)
    result=dict(total=ledger['realized_profit']-origin,origin_realized_common=origin,
        ledger=ledger,trace=trace,physical_final_hash=physical_identity(s),source_state_hash=initialhash,
        current_physical_action=current,current_action_hash=digest(current),
        future_hash=digest([vars(orders[k]) for k in sorted(orders)]),continuation_hash=policy['policy_hash'],
        current_stage_USD=rows[0]['plan']['operating_profit'],pending_at_T=pending,
        cooldown_periods=max(0,s.t-s.T),solver_seconds=sum(r['plan']['solver'].get('runtime',0) for r in rows),
        independent_cash_max_residual=max(abs(r['plan']['accounting']['profit_identity_residual']) for r in rows))
    replay_started=time.perf_counter();result['replay']=replay_world(state,orders,result)
    result['replay_seconds']=time.perf_counter()-replay_started;result['wall_seconds']=time.perf_counter()-started
    return result
