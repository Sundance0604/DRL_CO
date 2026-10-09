"""Frozen A simulator with the independently expanded legal B summary.

Decision/physics/cooldown/ledger statements mirror cfs_classical_value.simulation;
Physical/control/cooldown/ledger statements remain identical. The summary adds
current pressure/bucket for B2's scalar provider; no MILP constraint is changed.
"""
import time
from experiment_core.storage import digest
from model_families.single.runner import state_snapshot,serialize_plan,make_step
from model_families.single.accounting import realized_ledger,completion_bound,cooldown_tick
from model_families.single.rolling.controller import RollingController,OraclePreviewController,observed_state
from model_families.single.rolling.contracts import WindowRequest
from model_families.single.planning import compile_action,PeriodAction
from model_families.single.value.route_milp import ValueRouteMILPPlanner
from .pressure import summary

def simulate(sim,orders,policy='myopic',provider=None,prior=None,collect_times=(),trace=False,first_begun=False,timeout=600):
    start=time.perf_counter();initial_t=sim.t;rows=[];snapshots={};arrivals={}
    for o in orders.values():
        if o.book_time>=initial_t:arrivals.setdefault(o.book_time,[]).append(o)
    def checkpoint():
        if time.perf_counter()-start>timeout:raise TimeoutError('daily wall-time budget exceeded')
    H=4 if policy.startswith('RH4') else 8 if policy in ('RH8','OracleH8') else 1
    rolling=None
    if policy in ('RH4','RH8','OracleH8'):
        p=dict(planning_horizon=H,commit_periods=1,completion_extension=0,planner_id='route_milp')
        rolling=(OraclePreviewController(p,checkpoint,lambda t,end:[o for o in orders.values() if t<o.book_time<=end])
            if policy=='OracleH8' else RollingController(p,checkpoint))
    online_eval=solver_time=build_time=0.;windows=[];limited=0;maxgap=0.;maxvars=0
    context_time=transition_time=postprocess_time=trace_time=accounting_time=cooldown_time=planning_wall=0.
    for t in range(initial_t,sim.T):
        checkpoint();tick=time.perf_counter()
        if not(first_begun and t==initial_t):sim.begin_period(arrivals.get(t,[]))
        transition_time+=time.perf_counter()-tick;tick=time.perf_counter()
        if t in collect_times:snapshots[t]=sim.clone()
        before=state_snapshot(sim);info={};context_time+=time.perf_counter()-tick
        if provider is not None:
            tick=time.perf_counter();s=summary(sim,prior);planner=ValueRouteMILPPlanner(provider,s)
            request=WindowRequest(observed_state(sim),t,min(t+H,sim.T),t+1,0,checkpoint)
            context_time+=time.perf_counter()-tick;tick=time.perf_counter()
            solution=planner.plan(request);meta=solution.diagnostics;planning_wall+=time.perf_counter()-tick
            tick=time.perf_counter();plan=compile_action(sim,solution.actions.get(t,PeriodAction(t)),info);info.update(meta)
            info['window_objective']=meta['window_objective'];windows.append(meta);postprocess_time+=time.perf_counter()-tick
            online_eval+=meta['online_value_eval_seconds'];build_time+=meta['build_seconds']
        elif rolling is not None:
            tick=time.perf_counter();plan=rolling.decide(sim,info);planning_wall+=time.perf_counter()-tick
            windows.append(rolling.windows[-1]);build_time+=rolling.windows[-1]['build_seconds']
        else:
            tick=time.perf_counter();plan=sim.solve_plan(sim.parameters['cross_hub'],sim.parameters['empty_cost'],sim.parameters['reposition'],None,info)
            planning_wall+=time.perf_counter()-tick
        solver_time+=info.get('runtime',0);limited+=int(info.get('status') not in ('OPTIMAL',None))
        maxgap=max(maxgap,info.get('gap',0) or 0);maxvars=max(maxvars,info.get('variable_count',0))
        tick=time.perf_counter();serialized=serialize_plan(plan,t,digest(before),info);postprocess_time+=time.perf_counter()-tick
        tick=time.perf_counter();sim.commit_plan(plan);transition_time+=time.perf_counter()-tick
        tick=time.perf_counter()
        rows.append(make_step(t,before,serialized,state_snapshot(sim),rows,orders,arrivals.get(t,[]),sim.parameters)
            if trace else dict(period=t,plan=serialized));trace_time+=time.perf_counter()-tick
    decision_rows=list(rows);pending=len(sim.pool);cap=completion_bound(sim);sim.pool={}
    while any(v.phase!='idle' for v in sim.vehs):
        checkpoint()
        if sim.t>=cap:raise RuntimeError('unfinished commitments')
        tick=time.perf_counter();before=state_snapshot(sim);period=sim.t;context_time+=time.perf_counter()-tick
        tick=time.perf_counter();cooldown_tick(sim);elapsed=time.perf_counter()-tick
        transition_time+=elapsed;cooldown_time+=elapsed
        tick=time.perf_counter()
        if trace:
            plan=dict(period=period,state_hash=digest(before),operating_profit=0,augmented_solver_objective=0,
                assignments=[],routes=[],repositions=[],expired=[],accounting={},solver={'runtime':0,'status':'COMMITTED_EXECUTION_ONLY'})
            rows.append(make_step(period,before,plan,state_snapshot(sim),rows,orders,[],sim.parameters)|{'drain':True})
        trace_time+=time.perf_counter()-tick
    tick=time.perf_counter();ledger=realized_ledger(sim,orders,decision_rows)
    if initial_t:ledger['realized_profit']+=initial_t*len(sim.vehs)*sim.parameters['waiting_cost']
    if initial_t==0 and(ledger['unfinished_committed'] or ledger['pending_unassigned']!=pending):raise RuntimeError('full-run accounting reconciliation failed')
    if trace:
        prev=0.
        for row in rows:
            l=realized_ledger(sim,orders,decision_rows,row['period']+1);row['realized_ledger']=l
            row['measurements'].update(realized_profit=l['realized_profit'],period_realized_profit=l['realized_profit']-prev);prev=l['realized_profit']
    accounting_time+=time.perf_counter()-tick
    pure_lookup=getattr(provider,'lookup_seconds',0.)
    return dict(**ledger,**sim.S,pending_at_T=pending,daily_wall_seconds=time.perf_counter()-start,
        online_value_eval_seconds=online_eval,daily_solver_seconds=solver_time,build_seconds=build_time,
        limited_window_count=limited,max_window_gap=maxgap,max_variable_count=maxvars,
        window_solve_count=len(windows),cooldown_periods=sim.t-sim.T,
        context_preparation_seconds=context_time,provider_lookup_seconds=pure_lookup,
        release_slot_assembly_seconds=online_eval,build_exclusive_lookup_seconds=max(0,build_time-pure_lookup),
        state_transition_seconds=transition_time,cooldown_transition_seconds=cooldown_time,
        plan_postprocessing_seconds=postprocess_time,trace_preparation_seconds=trace_time,
        realized_accounting_seconds=accounting_time,planning_wall_seconds=planning_wall),rows,snapshots

