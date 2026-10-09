"""Pure slot policy adapter and exact paired-resource/cooldown audits."""
import time, copy
from experiment_core.storage import digest
from model_families.single.data import matching_state
from model_families.single.engine.mt_prototype import Veh
from model_families.single.value.providers import MCTableValueProvider
from model_families.single.accounting import realized_ledger
from .simulation import simulate,summary
from experiments.cfs_rollout_target_alignment.protocol import (
    state_identity,physical_identity,action_hash,write_compressed,read_compressed,replay_world)
from experiments.cfs_rollout_target_alignment.queries import pack_state

class TrackedProvider(MCTableValueProvider):
    """Call provenance is mandatory; no fallback to Zero or another model."""
    def __init__(self,model,table_hash,parent_policy_hash):
        super().__init__(model);self.table_hash=table_hash;self.parent_policy_hash=parent_policy_hash
        self.calls=0;self.terminal_calls=0;self.lookup_seconds=0.
    def marginal_release_value(self,t,h,q,s):
        self.calls+=1;self.terminal_calls+=int(t>=s['T'])
        started=time.perf_counter();value=super().marginal_release_value(t,h,q,s)
        self.lookup_seconds+=time.perf_counter()-started
        assert t<s['T'] or value==0
        return value

def resource_world(state,hub,q):
    assert q>=0 and int(q)==q
    c=state.clone()
    # Frozen engine addresses vehs[id]: common retained prefix is reindexed
    # identically in EVERY q world; added resources form an identical prefix.
    c.vehs=[v for v in c.vehs if not(v.phase=='idle' and v.loc==hub)]
    for i,v in enumerate(c.vehs):v.id=i
    for _ in range(q):c.vehs.append(Veh(len(c.vehs),hub))
    return c

def pair_audit(low,high,hub):
    a,b=pack_state(low),pack_state(high)
    assert {k:v for k,v in a.items() if k!='vehicles'}=={k:v for k,v in b.items() if k!='vehicles'}
    assert a['vehicles']==b['vehicles'][:-1],'Paired worlds changed existing resource/commitment'
    v=high.vehs[-1]
    assert v.phase=='idle' and v.loc==hub and not v.orders and not v.onboard and not v.segments
    return dict(status='PASSED',only_extra_idle=True,common_hash=digest(a),extra_id=v.id)

def restore(scenario,p,solver,packet):
    s,orders=matching_state(scenario,p,solver)
    for name in ['t','T','J','revision']:setattr(s,name,packet[name])
    s.S=packet['S'].copy();s.pool={oid:orders[oid] for oid in packet['pool']};s.vehs=[]
    for data in packet['vehicles']:
        v=Veh(data['id'],data['loc'])
        for name,val in data.items():setattr(v,name,copy.deepcopy(val))
        v.orders={oid:orders[oid] for oid in data['orders']};v.onboard=set(data['onboard']);s.vehs.append(v)
    return s,orders

def continuation(state,orders,policy,prior,trace=True,collect_times=()):
    origin=realized_ledger(state,orders,[],cutoff=state.t)['realized_profit']
    source=state.clone();s=state.clone();started=time.perf_counter()
    model=policy.get('table');provider=None if model is None else TrackedProvider(model,policy['table_hash'],policy['policy_hash'])
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

def decide_same_state(state,policy,prior):
    """No orders scheduler or future argument is exposed to the policy."""
    from model_families.single.value.route_milp import ValueRouteMILPPlanner
    from model_families.single.rolling.contracts import WindowRequest
    from model_families.single.rolling.controller import observed_state
    from model_families.single.planning import compile_action,PeriodAction
    from model_families.single.runner import serialize_plan,state_snapshot
    before=state_identity(state);info={}
    if policy.get('table') is None:
        plan=state.solve_plan(state.parameters['cross_hub'],state.parameters['empty_cost'],state.parameters['reposition'],None,info)
    else:
        provider=TrackedProvider(policy['table'],policy['table_hash'],policy['policy_hash'])
        request=WindowRequest(observed_state(state),state.t,state.t+1,state.t+1,0,lambda:None)
        solution=ValueRouteMILPPlanner(provider,summary(state,prior)).plan(request)
        plan=compile_action(state,solution.actions.get(state.t,PeriodAction(state.t)),info);info.update(solution.diagnostics)
    assert info.get('status')=='OPTIMAL' and state_identity(state)==before
    return serialize_plan(plan,state.t,digest(state_snapshot(state)),info)
