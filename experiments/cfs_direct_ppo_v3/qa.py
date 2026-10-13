"""Independent no-MC physical replay and explicit access guards."""
import copy
from contextlib import contextmanager
from dataclasses import asdict
import math
from model_families.single.strategic.contracts import identity
from model_families.single.data import matching_state
from model_families.single.parameters import MatchingParameters
from experiment_core.contracts import SolverParameters
from model_families.single.planning import compile_action
from model_families.single.runner import serialize_plan,state_snapshot
from model_families.single.accounting import realized_ledger,completion_bound,cooldown_tick
from model_families.single.rolling.controller import observed_state
from experiments.cfs_ppo_strategic_control.study_execution import action_from_plan,physical_hash,close
from .observation import DirectObserver,FeatureScales
from .contracts import DirectSchema,DirectSignal


@contextmanager
def no_mc_guard():
    """Fail any baseline construction/query or MC-table/old-checkpoint access."""
    import builtins,io
    from model_families.single.strategic.baseline import FrozenBaseline
    old_init,old_query=FrozenBaseline.__init__,FrozenBaseline.query
    old_open,old_io=builtins.open,io.open
    def forbidden(*a,**k):raise RuntimeError('MC access forbidden in Direct PPO')
    def wrap(original):
        def checked(path,*a,**k):
            p=str(path).replace('\\','/').lower()
            if ('/baselines/' in p or '/tables/' in p or '-scales.json' in p or
                '/cfs_laptop_ppo_v1/training/' in p or '/sealed/' in p):
                raise RuntimeError('forbidden Direct input: '+p)
            return original(path,*a,**k)
        return checked
    FrozenBaseline.__init__=forbidden;FrozenBaseline.query=forbidden
    builtins.open=wrap(old_open);io.open=wrap(old_io)
    try:yield
    finally:
        FrozenBaseline.__init__=old_init;FrozenBaseline.query=old_query
        builtins.open=old_open;io.open=old_io


def replay(scenario,cfg,record):
    sim,orders=matching_state(copy.deepcopy(scenario),MatchingParameters.model_validate(cfg['parameters']).model_dump(),
        SolverParameters.model_validate(cfg['solver']).model_dump())
    schema=DirectSchema(**cfg['action']);observer=DirectObserver(schema,FeatureScales(**cfg['feature_scales']))
    rows=[];history={};total=0.;coefficient_count=0
    for index,tr in enumerate(record['transitions']):
        arrivals=[o for o in orders.values() if o.book_time==sim.t];sim.begin_period(arrivals)
        for o in arrivals:history[o.departure,o.destination]=history.get((o.departure,o.destination),0)+1
        obs=observer.build(observed_state(sim),history)
        if obs.hash!=tr['observation_hash'] or identity(tr['observation'])!=obs.hash or identity(state_snapshot(sim))!=tr['before_state_hash']:
            raise RuntimeError('Direct replay observation/before mismatch')
        signal=DirectSignal(**tr['signal'])
        if (signal.period,signal.revision,signal.observation_hash)!=(obs.period,obs.revision,obs.hash):
            raise RuntimeError('Direct replay stale signal')
        for c in tr['coefficients']:
            i=schema.hubs.index(c['hub']);when=c['time'];ell=(when-sim.t)/(sim.T-sim.t)
            value=0. if when>=sim.T else schema.scales[i]*((1-ell)*signal.values[2*i]+ell*signal.values[2*i+1])
            if not close(c['value'],value) or c['value']!=c['direct_price'] or 'base' in c or 'residual' in c:
                raise RuntimeError('Direct independent price mismatch/MC field')
            if when>=sim.T:raise RuntimeError('terminal gate leaked provider query')
            coefficient_count+=1
        plan_data=tr['plan'];info=dict(plan_data['solver'])
        plan=compile_action(sim,action_from_plan(plan_data),info)
        serial=serialize_plan(plan,sim.t,obs.hash,info)
        if physical_hash(serial)!=tr['action_hash'] or info['accounting']!=plan_data['accounting'] or not close(plan[0],plan_data['operating_profit']):
            raise RuntimeError('Direct replay integer action/cash mismatch')
        if info['status']!='OPTIMAL':raise RuntimeError('Direct nonoptimal record')
        sim.commit_plan(plan);rows.append(dict(period=obs.period,plan=serial))
        if index==len(record['transitions'])-1:
            if sim.t!=sim.T:raise RuntimeError('incomplete Direct day')
            bound=completion_bound(sim);sim.pool={}
            while any(v.phase!='idle' for v in sim.vehs):
                if sim.t>=bound:raise RuntimeError('Direct cooldown failed')
                cooldown_tick(sim)
        if identity(state_snapshot(sim))!=tr['after_state_hash']:raise RuntimeError('Direct replay after mismatch')
        ledger=realized_ledger(sim,orders,rows)
        for k,v in tr['ledger'].items():
            if isinstance(v,(int,float)) and not close(ledger[k],v):raise RuntimeError('Direct ledger mismatch')
        if not close(tr['reward_USD'],ledger['realized_profit']-total):raise RuntimeError('Direct reward mismatch')
        total=ledger['realized_profit']
    if ledger!=record['ledger'] or ledger['unfinished_committed'] or not close(sum(t['reward_USD'] for t in record['transitions']),total):
        raise RuntimeError('Direct final ledger/telescope mismatch')
    for v in sim.vehs:
        if any(not 0<=s['load']<=cfg['parameters']['capacity'] for s in v.segments):raise RuntimeError('Direct capacity violation')
        if any(t<orders[o].start_time for o,t in v.pickups.items()):raise RuntimeError('Direct readiness violation')
        if any(t>orders[o].end_time for o,t in v.deliveries.items()):raise RuntimeError('Direct deadline violation')
    if 'learning' in record:
        # Independent recurrence, not calling the training GAE implementation.
        learning=record['learning'];carry=0.;next_value=0.
        for i in reversed(range(len(learning))):
            r=learning[i];scaled=record['transitions'][i]['reward_USD']/cfg['ppo']['reward_scale']
            carry=scaled+next_value-r['value']+cfg['ppo']['gae_lambda']*carry
            if not close(r['advantage'],carry) or not close(r['return'],carry+r['value']):raise RuntimeError('independent GAE mismatch')
            next_value=r['value']
    return dict(status='PASSED',periods=len(rows),coefficient_count=coefficient_count,profit_USD=total)
