"""M0 resource-world evaluation with exact integer-action cash, unchanged replay."""
import time
from model_families.single.accounting import realized_ledger
from experiments.cfs_resource_policy_iteration.protocol import state_identity,physical_identity,replay_world
from .simulation import simulate

def continuation(state,orders,policy,prior,trace=True,collect_times=()):
    origin=realized_ledger(state,orders,[],cutoff=state.t)['realized_profit']
    source=state.clone();s=state.clone();started=time.perf_counter()
    assert policy.get('table') is None, 'B target bank must evaluate only M0'
    provider=None
    m,rows,states=simulate(s,orders,'myopic',provider,prior,collect_times,trace,first_begun=True)
    assert m['limited_window_count']==0,'Nonoptimal continuation incumbent'
    assert m['unfinished_committed']==0 and all(not v.orders and not v.onboard for v in s.vehs)
    assert all(a['order_id'] in orders and orders[a['order_id']].book_time<=r['period']
        for r in rows if not r.get('drain') for a in r['plan']['assignments'])
    result=dict(total=m['realized_profit']-origin,trace=rows,physical_final_hash=physical_identity(s))
    replay_started=time.perf_counter();replay=replay_world(source,orders,result) if trace else None
    replay_seconds=time.perf_counter()-replay_started
    return dict(metrics=m,trace=rows,replay=replay,source_state_hash=state_identity(source),
        final_hash=physical_identity(s),policy_hash=policy['policy_hash'],table_hash=policy.get('table_hash'),
        value_calls=0 if provider is None else provider.calls,terminal_value_calls=0 if provider is None else provider.terminal_calls,
        seconds=time.perf_counter()-started,replay_seconds=replay_seconds,states=states)

