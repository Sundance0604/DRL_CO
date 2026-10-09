"""One-step base/bucket adapters, same physical control and full ledger."""
import time
from experiment_core.storage import digest
from model_families.single.accounting import realized_ledger
from model_families.single.value.providers import MCTableValueProvider
from experiments.cfs_resource_policy_iteration.protocol import (
    state_identity,physical_identity,replay_world,read_compressed,write_compressed,restore)
from .simulation import simulate
from .pressure import summary
from .provider import BucketValueProvider

class BaseProvider(MCTableValueProvider):
    def __init__(self,model):
        super().__init__(model);self.calls=0;self.lookup_seconds=0.
    def marginal_release_value(self,t,h,q,s):
        started=time.perf_counter();self.calls+=1
        v=super().marginal_release_value(t,h,q,s)
        self.lookup_seconds+=time.perf_counter()-started;return v

def provider(policy):
    if policy['table'] is None:return None
    return BucketValueProvider(policy['table']) if policy['table']['method']=='bucket_table' else BaseProvider(policy['table'])

def continuation(state,orders,policy,prior,collect_times=()):
    initial=state.clone();s=state.clone();started=time.perf_counter()
    origin=realized_ledger(s,orders,[],cutoff=s.t)['realized_profit']
    price=provider(policy)
    metrics,trace,states=simulate(s,orders,provider=price,prior=prior,collect_times=collect_times,trace=True,first_begun=True)
    assert metrics['limited_window_count']==0 and metrics['unfinished_committed']==0
    assert all(orders[a['order_id']].book_time<=row['period'] for row in trace if not row.get('drain') for a in row['plan']['assignments'])
    replay=replay_world(initial,orders,dict(trace=trace,total=metrics['realized_profit']-origin,physical_final_hash=physical_identity(s)))
    queries=[] if price is None else getattr(price,'query_log',[])
    return dict(metrics=metrics,trace=trace,states=states,replay=replay,
        final_hash=physical_identity(s),policy_hash=policy['policy_hash'],table_hash=policy.get('table_hash'),
        value_calls=0 if price is None else price.calls,lookup_queries=queries,
        fallback_calls=0 if price is None else getattr(price,'fallback_calls',0),seconds=time.perf_counter()-started)

def decide_same_state(state,policy,prior):
    from model_families.single.value.route_milp import ValueRouteMILPPlanner
    from model_families.single.rolling.contracts import WindowRequest
    from model_families.single.rolling.controller import observed_state
    from model_families.single.planning import compile_action,PeriodAction
    from model_families.single.runner import serialize_plan,state_snapshot
    before=state_identity(state);info={};price=provider(policy)
    if price is None:
        plan=state.solve_plan(state.parameters['cross_hub'],state.parameters['empty_cost'],state.parameters['reposition'],None,info)
    else:
        request=WindowRequest(observed_state(state),state.t,state.t+1,state.t+1,0,lambda:None)
        solution=ValueRouteMILPPlanner(price,summary(state,prior)).plan(request)
        plan=compile_action(state,solution.actions.get(state.t,PeriodAction(state.t)),info);info.update(solution.diagnostics)
    assert info['status']=='OPTIMAL' and state_identity(state)==before
    return serialize_plan(plan,state.t,digest(state_snapshot(state)),info)
