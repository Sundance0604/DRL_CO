"""Solver-independent route feasibility and executable period-plan compilation.

The period MILP and window planners share these domain rules. Planned actions are
not solver variable names, and are revalidated against the live state at execution.
"""
from dataclasses import dataclass
from typing import Any
from .network import eligible_pickups
from .engine.demand import ready_time


def decision_vehicles(t, vehicles):
    return [v for v in vehicles if v.phase == "idle" or
            (v.phase == "trip" and v.dead == 0 and (v.arrive == t or v.hold is not None))]


def route_candidates(t, net, pool, vehicle, cross, empty_cost, parameters):
    c, candidates = vehicle.loc, []
    if vehicle.phase == "idle":
        pickups = eligible_pickups(net,c,cross,parameters.get("pickup_radius",0),
                                   parameters.get("pickup_rule","network_radius"))
        for p in pickups:
            for e in net.H:
                if e == p:
                    continue
                for delay in range(parameters.get("max_delay",3)+1):
                    candidates.append(dict(p=p,e=e,B=net.path[p,e],delay=delay,
                        t0=t+delay+net.tau[c,p],new=True,
                        cost=parameters.get("waiting_cost",0)*delay+empty_cost*net.d[c,p]+
                             parameters.get("loaded_cost",10)*net.d[p,e]))
    else:
        end=vehicle.route[-1]
        leave=t if vehicle.hold is None else vehicle.hold
        if [c]+vehicle.route != net.path[c,end]:
            raise RuntimeError("committed backbone is not a fixed path")
        candidates.append(dict(p=c,e=end,B=net.path[c,end],delay=0,t0=leave,cost=0,new=False))
        for e in net.H:
            if e not in (c,end) and net.path[c,e][:len(vehicle.route)+1] == [c]+vehicle.route:
                candidates.append(dict(p=c,e=e,B=net.path[c,e],delay=0,t0=leave,new=True,
                                       cost=parameters.get("loaded_cost",10)*net.d[end,e]))
    for r in candidates:
        serve=[]
        for o in pool.values():
            if o.departure not in r["B"] or o.destination not in r["B"]:
                continue
            i,j=r["B"].index(o.departure),r["B"].index(o.destination)
            if i<j and r["B"][i:j+1] == net.path[o.departure,o.destination]:
                pickup=r["t0"]+net.tau[r["p"],o.departure]
                delivery=r["t0"]+net.tau[r["p"],o.destination]
                if pickup>=ready_time(o,t) and delivery<=o.end_time:
                    serve.append(o)
        r["serve"]=serve
    return [r for r in candidates if r["serve"] or not r["new"]]


def committed_arc_load(net, vehicle):
    load={}
    if vehicle.phase == "trip":
        ahead=set(zip([vehicle.loc]+vehicle.route,vehicle.route))
        for o in vehicle.orders.values():
            for arc in net.arcs(o.departure,o.destination):
                if arc in ahead:
                    load[arc]=load.get(arc,0)+o.passenger
    return load


def expired_orders(t, net, pool):
    return {oid for oid,o in pool.items()
            if max(t+1,ready_time(o,t))+net.tau[o.departure,o.destination]>o.end_time}


@dataclass(frozen=True)
class VehicleAction:
    vehicle_id: int
    route: dict[str, Any] | None = None
    order_ids: tuple[str, ...] = ()
    reposition_to: str | None = None


@dataclass(frozen=True)
class PeriodAction:
    period: int
    vehicles: tuple[VehicleAction, ...] = ()


def route_key(route):
    return (route["p"],route["e"],tuple(route["B"]),route["delay"],route["t0"],route["new"])


def compile_action(sim, action, info):
    """Validate a domain action and produce the existing guarded DecisionPlan.

    A missing idle action means wait. A missing active action means continue the
    committed backbone, never abandon it. Fresh unplanned bookings are accounted
    for and expired each period, but not assigned inside a frozen prefix.
    """
    from .engine.mt_prototype import DecisionPlan
    if action.period != sim.t:
        raise RuntimeError("action period does not match the live clock")
    dec=decision_vehicles(sim.t,sim.vehs)
    by_id={v.id:v for v in dec}
    supplied={}
    for a in action.vehicles:
        if a.vehicle_id in supplied or a.vehicle_id not in by_id:
            raise RuntimeError("duplicate or unavailable decision vehicle")
        if a.route is not None and a.reposition_to is not None:
            raise RuntimeError("route and reposition are mutually exclusive")
        if a.order_ids and a.route is None:
            raise RuntimeError("assignment requires a route")
        supplied[a.vehicle_id]=a
    sol={"x":[],"y":[],"z":[]};routes={};assigned=set()
    p=sim.parameters
    for v in dec:
        candidates=route_candidates(sim.t,sim.net,sim.pool,v,p.get("cross_hub",True),p.get("empty_cost",10),p)
        routes[v.id]=candidates
        a=supplied.get(v.id)
        route_index=None
        if a and a.route is not None:
            route_index=next((i for i,r in enumerate(candidates) if route_key(r)==route_key(a.route)),None)
            if route_index is None:
                raise RuntimeError("planned route is no longer physically feasible")
        elif a and a.reposition_to is not None:
            if not p.get("reposition",False) or v.phase != "idle" or a.reposition_to not in sim.net.H or a.reposition_to==v.loc:
                raise RuntimeError("invalid reposition action")
            sol["z"].append((v.id,a.reposition_to))
        elif v.phase == "trip":
            route_index=next(i for i,r in enumerate(candidates) if not r["new"])
        if route_index is None:
            continue
        r=candidates[route_index]
        ids=a.order_ids if a else ()
        feasible={o.id for o in r["serve"]}
        load=committed_arc_load(sim.net,v)
        for oid in ids:
            if oid not in feasible or oid in assigned:
                raise RuntimeError("unavailable, infeasible or duplicate order assignment")
            assigned.add(oid);o=sim.pool[oid]
            for arc in sim.net.arcs(o.departure,o.destination):
                load[arc]=load.get(arc,0)+o.passenger
            sol["x"].append((oid,v.id))
        if any(q>p.get("capacity",7) for q in load.values()):
            raise RuntimeError("committed plus new load exceeds arc capacity")
        if r["new"]:
            if v.phase=="idle" and not any(sim.pool[o].departure==r["p"] for o in ids):
                raise RuntimeError("new route lacks pickup anchor")
            if not any(sim.pool[o].destination==r["e"] for o in ids):
                raise RuntimeError("new route lacks endpoint anchor")
        sol["y"].append((v.id,route_index))
    exp=expired_orders(sim.t,sim.net,sim.pool)
    moving={k for k,r in sol["y"]}|{k for k,j in sol["z"]}
    accounting=dict(assignment_revenue=sum(sim.pool[o].revenue for o in assigned),
        planned_route_cost=sum(routes[k][r]["cost"] for k,r in sol["y"]),
        planned_reposition_cost=sum(p.get("empty_cost",10)*sim.net.d[sim.vehs[k].loc,j] for k,j in sol["z"]),
        idle_waiting_cost=p.get("waiting_cost",0)*sum(v.phase=="idle" and v.id not in moving for v in dec),
        backlog_penalty=sum(o.penalty for oid,o in sim.pool.items() if oid not in assigned),
        expiry_penalty=p.get("expiry_cost",300)*len(exp-assigned))
    cost=sum(value for k,value in accounting.items() if k!="assignment_revenue")
    profit=accounting["assignment_revenue"]-cost
    accounting.update(business_cost=cost,profit_identity_residual=0)
    objective=profit-p.get("tie_break",1)*sum(routes[k][r]["delay"] for k,r in sol["y"])
    info["accounting"]=accounting
    return DecisionPlan((profit,objective,sol,routes,exp,len(dec)),sim.t,sim.revision)
