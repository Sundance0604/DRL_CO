"""Joint, time-expanded route-macro-action MILP for a single planning window.

Per-vehicle flow connects waiting, repositioning, committed continuation and new
anchored backbone trips. Order assignment, link capacity and future vehicle reuse
are coupled in ONE MILP, not a sequence of myopic solves or sampled rollout.

Macro-action restriction: a planned trip is followed to its endpoint inside a
window; intermediate arrivals execute continuation, not additional planned route
extensions. At the next real replan, existing single prefix-compatible extension
rules are available again. This is a feasible restricted window formulation, not
an exact reformulation/global bound of every possible period-M_t action sequence.
"""
import math
import time
import gurobipy as gp
from gurobipy import GRB
from experiment_core.contracts import PlatformError
from ..planning import (PeriodAction,VehicleAction,route_candidates,
                        committed_arc_load,decision_vehicles)
from ..accounting import completion_bound
from ..engine.mt_prototype import Veh
from .contracts import WindowSolution


def first_handoff(sim, vehicle):
    """Project only physically locked movement to the first dispatch boundary.

    Deadhead/reposition links are indivisible. A currently held departure remains
    locked. begin_period handles delivery/boarding before residual loads are read.
    """
    v=vehicle.copy()
    if decision_vehicles(sim.t,[v]):
        return sim.t,v
    projected=sim.clone();projected.vehs=[v];projected.pool={}
    while v.phase != "idle":
        if v.arrive is None or v.arrive<=projected.t:
            raise RuntimeError("active vehicle has no future arrival event")
        projected.t=v.arrive
        projected.begin_period([])
        if decision_vehicles(projected.t,[v]):
            return projected.t,v
    return projected.t,v


class RouteMILPPlanner:
    def plan(self, request):
        request.checkpoint()
        m=gp.Model("single_rolling_window")
        try:
            return self._plan(request,m)
        finally:
            m.dispose()

    def _plan(self, request, m):
        started=time.monotonic()
        sim=request.state;p=sim.parameters;net=sim.net
        start,end=request.start,request.decision_end
        if start != sim.t or not start<end<=sim.T:
            raise RuntimeError("invalid rolling window boundary")
        pool=dict(sim.pool)
        for o in request.preview_orders:
            if o.id in pool or o.book_time<=start:
                raise RuntimeError("preview must contain distinct future bookings")
            pool[o.id]=o
        first={oid:max(start,o.book_time) for oid,o in pool.items()}
        expiry={oid:next((u for u in range(first[oid],end) if
                    max(u+1,o.start_time)+net.tau[o.departure,o.destination]>o.end_time),end)
                for oid,o in pool.items()}
        baseline={oid:o.penalty*max(0,min(end,expiry[oid]+1)-first[oid])+
                       p["expiry_cost"]*int(expiry[oid]<end) for oid,o in pool.items()}
        m.Params.OutputFlag=0
        for key,value in sim.solver_parameters.items():
            m.setParam({"time_limit_seconds":"TimeLimit","mip_gap":"MIPGap","threads":"Threads","seed":"Seed"}[key],value)
        if not sim.solver_parameters:
            m.Params.Threads=1;m.Params.MIPGap=0
        objective=[];all_x={oid:[] for oid in pool};chosen=[]
        candidate_count=0;max_finish=max(end+request.completion_extension,completion_bound(sim)-1)
        hubnode=lambda hub,when:("hub",hub,when)
        for vehicle in sim.vehs:
            request.checkpoint()
            handoff,v=first_handoff(sim,vehicle)
            sink=("sink",vehicle.id)
            source=hubnode(v.loc,handoff) if v.phase=="idle" and handoff<end else ("initial",vehicle.id)
            outgoing={};incoming={}
            def add_arc(a,b,cost,action=None,route=None,old_load=None):
                nonlocal candidate_count,max_finish
                y=m.addVar(vtype=GRB.BINARY)
                outgoing.setdefault(a,[]).append(y);incoming.setdefault(b,[]).append(y)
                objective.append(-cost*y)
                entry={"y":y,"action":action,"assignments":[],"vehicle_id":vehicle.id}
                chosen.append(entry)
                if route is None:
                    return
                candidate_count+=1
                available=[]
                when=action.period if action else end
                if action is not None:
                    for o in route["serve"]:
                        if first[o.id]<=when<=expiry[o.id]:
                            x=m.addVar(vtype=GRB.BINARY)
                            m.addConstr(x<=y)
                            all_x[o.id].append(x)
                            entry["assignments"].append((o.id,x))
                            available.append((o,x))
                            objective.append((o.revenue+baseline[o.id]-o.penalty*(when-first[o.id]))*x)
                loads=old_load or {}
                for arc in zip(route["B"],route["B"][1:]):
                    terms=[o.passenger*x for o,x in available if arc in net.arcs(o.departure,o.destination)]
                    m.addConstr(gp.quicksum(terms)<=p["capacity"]-loads.get(arc,0))
                if route["new"]:
                    if v.phase=="idle" or a[0]=="hub":
                        m.addConstr(y<=gp.quicksum(x for o,x in available if o.departure==route["p"]))
                    m.addConstr(y<=gp.quicksum(x for o,x in available if o.destination==route["e"]))
            def add_route(a,when,rv,initial=False):
                nonlocal max_finish
                for r in route_candidates(when,net,pool,rv,p["cross_hub"],p["empty_cost"],p):
                    if when>=end and r["new"]:
                        continue
                    finish=r["t0"]+net.tau[r["p"],r["e"]]
                    max_finish=max(max_finish,finish)
                    destination=sink if finish>=end else hubnode(r["e"],finish)
                    payload={k:value for k,value in r.items() if k!="serve"}
                    action=(PeriodAction(when,(VehicleAction(vehicle.id,payload),)) if when<end else None)
                    add_arc(a,destination,r["cost"]+p["tie_break"]*r["delay"],action,r,
                            committed_arc_load(net,rv) if initial else {})
            if source[0]=="initial":
                if v.phase=="trip":
                    add_route(source,handoff,v,initial=True)
                else:
                    add_arc(source,sink,0)  # locked arrival after the decision window
            for u in range(start,end):
                request.checkpoint()
                for hub in net.H:
                    node=hubnode(hub,u)
                    add_arc(node,sink if u+1==end else hubnode(hub,u+1),p["waiting_cost"])
                    idle=Veh(vehicle.id,hub)
                    add_route(node,u,idle)
                    if p["reposition"]:
                        for target in net.H:
                            if target==hub:
                                continue
                            finish=u+net.tau[hub,target]
                            max_finish=max(max_finish,finish)
                            add_arc(node,sink if finish>=end else hubnode(target,finish),
                                p["empty_cost"]*net.d[hub,target],
                                PeriodAction(u,(VehicleAction(vehicle.id,reposition_to=target),)))
            nodes=set(outgoing)|set(incoming)|{source,sink}
            for node in nodes:
                rhs=1 if node==source else -1 if node==sink else 0
                m.addConstr(gp.quicksum(outgoing.get(node,[]))-gp.quicksum(incoming.get(node,[]))==rhs)
        for variables in all_x.values():
            m.addConstr(gp.quicksum(variables)<=1)
        m.setObjective(gp.quicksum(objective)-sum(baseline.values()),GRB.MAXIMIZE)
        built=time.monotonic()-started
        interrupted=[]
        def control(model,where):
            if interrupted:
                return
            try:
                request.checkpoint()
            except Exception as exc:
                interrupted.append(exc);model.terminate()
        m.optimize(control)
        if interrupted:
            raise interrupted[0]
        request.checkpoint()
        if not m.SolCount:
            raise PlatformError("WINDOW_NO_INCUMBENT","window solve has no feasible incumbent; nothing committed",exit_code=4)
        actions={}
        for entry in chosen:
            if entry["y"].X<.5 or entry["action"] is None:
                continue
            action=entry["action"];a=action.vehicles[0]
            ids=tuple(oid for oid,x in entry["assignments"] if x.X>.5)
            actions.setdefault(action.period,[]).append(VehicleAction(a.vehicle_id,a.route,ids,a.reposition_to))
        diagnostics=dict(status="OPTIMAL" if m.Status==GRB.OPTIMAL else "FEASIBLE_LIMIT",
            runtime=m.Runtime,build_seconds=built,window_objective=m.ObjVal,
            bound=m.ObjBound if math.isfinite(m.ObjBound) else None,
            gap=(m.MIPGap if math.isfinite(m.MIPGap) else None) if m.IsMIP else 0,
            candidate_route_count=candidate_count,completion_end=max_finish,
            variable_count=m.NumVars,constraint_count=m.NumConstrs,
            information_mode="oracle_preview" if request.preview_orders else "known_supply",
            completion_auto_extension=max(0,max_finish-(end+request.completion_extension)),
            formulation="single-route-macro-action-window/v1",forecast="none",
            global_gap=None,global_gap_reason="window gap is not a full-run bound; route macro-action restriction applies")
        return WindowSolution({u:PeriodAction(u,tuple(a)) for u,a in actions.items()},diagnostics)
