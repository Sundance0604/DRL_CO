"""Uses frozen domain physics/accounting; only the choice of planner differs."""
import time
from experiment_core.storage import digest
from model_families.single.data import matching_state
from model_families.single.runner import state_snapshot, serialize_plan, make_step
from model_families.single.accounting import realized_ledger, completion_bound, cooldown_tick
from model_families.single.rolling.controller import RollingController, OraclePreviewController, observed_state
from model_families.single.rolling.contracts import WindowRequest
from model_families.single.rolling.route_milp import RouteMILPPlanner
from model_families.single.planning import compile_action
from model_families.single.local_value.route_milp import LocalRouteMILPPlanner


def summary(sim, prior):
    from model_families.single.accounting import completion_bound
    incoming=[]
    for v in sim.vehs:
        if v.phase!="idle":
            c=sim.clone();c.vehs=[v.copy()];c.pool={}
            cap=completion_bound(c)
            while c.vehs[0].phase!="idle":
                if c.t>=cap: raise RuntimeError("incoming projection failed")
                cooldown_tick(c)
            incoming.append(dict(hub=c.vehs[0].loc,time=c.vehs[0].deliveries and
                max(c.vehs[0].deliveries.values()) or c.t-1))
    return dict(T=sim.T,period=sim.t,hubs=sorted(sim.net.H),
        idle={h:sum(v.phase=="idle" and v.loc==h for v in sim.vehs) for h in sim.net.H},
        incoming=incoming,backlog={h:sum(o.passenger for o in sim.pool.values() if o.departure==h)/sim.parameters["capacity"] for h in sim.net.H},
        prior=prior)


from model_families.single.value.providers import feature_row


def simulate(sim,orders,policy="myopic",provider=None,prior=None,collect_times=(),trace=False,
             first_begun=False,timeout=600):
    """Future records belong ONLY to scheduler / explicit oracle; never to V."""
    start=time.perf_counter();initial_t=sim.t;rows=[];snapshots={}
    arrivals={}
    for o in orders.values():
        if o.book_time>=initial_t:arrivals.setdefault(o.book_time,[]).append(o)
    def checkpoint():
        if time.perf_counter()-start>timeout:raise TimeoutError("daily wall-time budget exceeded")
    H=4 if policy.startswith("RH4") else 8 if policy in ("RH8","OracleH8") else 1
    rolling=None
    if policy in ("RH4","RH8","OracleH8"):
        p=dict(planning_horizon=H,commit_periods=1,completion_extension=0,planner_id="route_milp")
        rolling=(OraclePreviewController(p,checkpoint,lambda t,end:[o for o in orders.values() if t<o.book_time<=end])
                 if policy=="OracleH8" else RollingController(p,checkpoint))
    online_eval=solver_time=build_time=0.;windows=[];limited=0;maxgap=0.;maxvars=0
    for t in range(initial_t,sim.T):
        checkpoint()
        if not (first_begun and t==initial_t):sim.begin_period(arrivals.get(t,[]))
        if t in collect_times:snapshots[t]=sim.clone()
        before=state_snapshot(sim);info={}
        if provider is not None:
            s=summary(sim,prior)
            planner=LocalRouteMILPPlanner(provider,s)
            request=WindowRequest(observed_state(sim),t,min(t+H,sim.T),t+1,0,checkpoint)
            solution=planner.plan(request);meta=solution.diagnostics
            from model_families.single.planning import PeriodAction
            plan=compile_action(sim,solution.actions.get(t,PeriodAction(t)),info)
            info.update(meta)
            # compile_action's money excludes V; solver's objective is separate.
            info["window_objective"]=meta["window_objective"]
            windows.append(meta)
            online_eval+=meta["online_value_eval_seconds"]
            build_time+=meta["build_seconds"]
        elif rolling is not None:
            plan=rolling.decide(sim,info);windows.append(rolling.windows[-1])
            build_time+=rolling.windows[-1]["build_seconds"]
        else:
            plan=sim.solve_plan(sim.parameters["cross_hub"],sim.parameters["empty_cost"],sim.parameters["reposition"],None,info)
        solver_time+=info.get("runtime",0)
        limited+=int(info.get("status") not in ("OPTIMAL",None))
        maxgap=max(maxgap,info.get("gap",0) or 0);maxvars=max(maxvars,info.get("variable_count",0))
        serialized=serialize_plan(plan,t,digest(before),info)
        sim.commit_plan(plan)
        if trace:
            post=summary(sim,prior or {"rate":{}})
            assigned=[a["order_id"] for a in serialized["assignments"]]
            serialized["post_decision_audit"]=dict(
                timestamp="after committed action and clock advance; before next booking revelation",
                remaining_backlog=post["backlog"],idle_supply=post["idle"],incoming=post["incoming"],
                committed_count=sum(v.phase!="idle" for v in sim.vehs),
                served_count=len(assigned),served_load=sum(orders[o].passenger for o in assigned),
                imbalance={h:post["backlog"][h]-post["idle"][h] for h in post["hubs"]},
                route_releases=[dict(hub=r["e"],time=r["t0"]+sim.net.tau[r["p"],r["e"]]) for r in serialized["routes"]])
        rows.append(make_step(t,before,serialized,state_snapshot(sim),rows,orders,arrivals.get(t,[]),sim.parameters)
                    if trace else dict(period=t,plan=serialized))
    decision_rows=list(rows);pending=len(sim.pool);cap=completion_bound(sim);sim.pool={}
    while any(v.phase!="idle" for v in sim.vehs):
        checkpoint()
        if sim.t>=cap:raise RuntimeError("unfinished commitments")
        before=state_snapshot(sim);period=sim.t;cooldown_tick(sim)
        if trace:
            plan=dict(period=period,state_hash=digest(before),operating_profit=0,augmented_solver_objective=0,
                assignments=[],routes=[],repositions=[],expired=[],accounting={},solver={"runtime":0,"status":"COMMITTED_EXECUTION_ONLY"})
            rows.append(make_step(period,before,plan,state_snapshot(sim),rows,orders,[],sim.parameters)|{"drain":True})
    ledger=realized_ledger(sim,orders,decision_rows)
    if initial_t:
        # Inserted free resources did not exist during [0, initial_t).
        # Removing this artificial historical idle charge yields the paired
        # future-profit increment (retained historical cash is a common constant).
        ledger["realized_profit"]+=initial_t*len(sim.vehs)*sim.parameters["waiting_cost"]
    # Midday continuation excludes old assignment rows: outstanding accounting
    # is not meaningful there; profit remains valid and common old money cancels.
    if initial_t==0 and (ledger["unfinished_committed"] or ledger["pending_unassigned"]!=pending):
        raise RuntimeError("full-run accounting reconciliation failed")
    if trace:
        prev=0.
        for row in rows:
            l=realized_ledger(sim,orders,decision_rows,row["period"]+1)
            row["realized_ledger"]=l;row["measurements"].update(realized_profit=l["realized_profit"],period_realized_profit=l["realized_profit"]-prev)
            prev=l["realized_profit"]
    return dict(**ledger,**sim.S,pending_at_T=pending,daily_wall_seconds=time.perf_counter()-start,
        online_value_eval_seconds=online_eval,daily_solver_seconds=solver_time,build_seconds=build_time,
        online_feature_seconds=sum(w.get("online_feature_seconds",0) for w in windows),
        online_provider_seconds=sum(w.get("online_provider_seconds",0) for w in windows),
        slot_value_seconds=sum(w.get("slot_value_seconds",0) for w in windows),
        exclusive_milp_build_seconds=sum(w.get("exclusive_milp_build_seconds",w["build_seconds"]) for w in windows),
        limited_window_count=limited,max_window_gap=maxgap,max_variable_count=maxvars,
        window_solve_count=len(windows),cooldown_periods=sim.t-sim.T),rows,snapshots
