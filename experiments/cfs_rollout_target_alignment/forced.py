"""Isolated copy of the frozen original M_t; one added current-choice constraint.
No served subset is fixed. All x variables, other vehicles, anchors and capacity
remain in the original joint MILP. The intervention never enters simulator state.
"""
import math
import gurobipy as gp
from gurobipy import GRB
from experiment_core.storage import digest
from model_families.single.planning import decision_vehicles,route_candidates,expired_orders
from model_families.single.engine.mt_prototype import DecisionPlan

def route_payload(route):
    return {k:v for k,v in route.items() if k!='serve'}

def route_key(route):
    return digest(route_payload(route))

def forced_plan(state, candidate=None, info=None):
    info = info if info is not None else {}
    p=state.parameters
    if candidate is None or candidate.get('baseline'):
        # The canonical unforced optimum also solves the forced baseline choice.
        # Reuse it to avoid unrelated tie-breaking changes in other vehicles.
        return state.solve_plan(p['cross_hub'],p['empty_cost'],p['reposition'],None,info)
    result=_build_forced(state.t,state.net,state.pool,state.vehs,p['cross_hub'],
        p['empty_cost'],p['reposition'],parameters=p,solver_parameters=state.solver_parameters,
        solve_info=info,force=(candidate['vehicle_id'],candidate['kind'],candidate.get('route_key')))
    info['one_period_force']=dict(period=state.t,vehicle_id=candidate['vehicle_id'],kind=candidate['kind'],candidate_hash=candidate['candidate_hash'])
    return DecisionPlan(result,state.t,state.revision)

def _build_forced(t, net, pool, vehs, hops, c_empty, repos, flat_v=None, vfun=None, parameters=None,
                    solver_parameters=None, solve_info=None, force=None):
    """Build (M_t), or (M^V_t) when vfun(period, hub) is given.
    Returns (profit without value terms, objective, decisions, ...)."""
    parameters = parameters or {}
    Q = parameters.get("capacity", 7)
    C_LOADED = parameters.get("loaded_cost", 10.0)
    C_WAIT = parameters.get("waiting_cost", 0.0)
    PHI = parameters.get("expiry_cost", 300.0)
    DMAX = parameters.get("max_delay", 3)
    TIE = parameters.get("tie_break", 1.0)
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.Threads = 1
    # The flat-value invariant below compares two objective values exactly up to
    # numerical precision. Gurobi's default relative MIP gap can otherwise stop
    # a few objective units early after the large constant is added.
    m.Params.MIPGap = 0.0
    for key, value in (solver_parameters or {}).items():
        m.setParam({"time_limit_seconds": "TimeLimit", "mip_gap": "MIPGap", "threads": "Threads", "seed": "Seed"}[key], value)
    dec = decision_vehicles(t, vehs)
    routes, y, x, z, w = {}, {}, {}, {}, {}
    cover = {}                                          # (o, k) -> list of y vars whose route serves o
    for k in dec:
        c = k.loc
        cand = route_candidates(t, net, pool, k, hops, c_empty, parameters)
        routes[k.id] = cand
        for ri, r in enumerate(cand):
            y[k.id, ri] = m.addVar(vtype=GRB.BINARY)
            for o in r["serve"]:
                cover.setdefault((o.id, k.id), []).append(y[k.id, ri])
        if k.phase == "idle":
            w[k.id] = m.addVar(vtype=GRB.BINARY)
            if repos:
                for j in net.H:
                    if j != c:
                        z[k.id, j] = m.addVar(vtype=GRB.BINARY)
    for (oid, kid) in cover:
        x[oid, kid] = m.addVar(vtype=GRB.BINARY)
    s = {oid: m.addVar(vtype=GRB.BINARY) for oid in pool}
    m.update()
    for oid in pool:                                                            # (assign)
        m.addConstr(gp.quicksum(x[oid, k.id] for k in dec if (oid, k.id) in x) + s[oid] == 1)
    for k in dec:
        ys = gp.quicksum(y[k.id, ri] for ri in range(len(routes[k.id])))
        if k.phase == "idle":                                                   # (oneidle)
            m.addConstr(ys + gp.quicksum(z[k.id, j] for j in net.H if (k.id, j) in z) + w[k.id] == 1)
        else:                                                                   # (onetrip)
            m.addConstr(ys == 1)
        load = {}
        if k.phase == "trip":
            ahead = set(zip([k.loc] + k.route, k.route))
            for o in k.orders.values():
                for a in net.arcs(o.departure, o.destination):
                    if a in ahead:
                        load[a] = load.get(a, 0) + o.passenger
        seat = {}
        for o in pool.values():
            if (o.id, k.id) in x:
                for a in net.arcs(o.departure, o.destination):
                    seat.setdefault(a, []).append(o.passenger * x[o.id, k.id])
        for a, terms in seat.items():                                           # (seats)
            m.addConstr(gp.quicksum(terms) <= Q - load.get(a, 0))
        for ri, r in enumerate(routes[k.id]):
            if not r["new"]:
                continue
            if k.phase == "idle":                                               # (anchor-start)
                m.addConstr(y[k.id, ri] <= gp.quicksum(x[o.id, k.id] for o in r["serve"] if o.departure == r["p"]))
            m.addConstr(y[k.id, ri] <= gp.quicksum(x[o.id, k.id] for o in r["serve"] if o.destination == r["e"]))
    for (oid, kid), ys in cover.items():                                        # (cover)
        m.addConstr(x[oid, kid] <= gp.quicksum(ys))
    exp = expired_orders(t, net, pool)
    profit = (gp.quicksum(pool[oid].revenue * x[oid, kid] for (oid, kid) in x)
              - gp.quicksum(routes[kid][ri]["cost"] * y[kid, ri] for (kid, ri) in y)
              - gp.quicksum(c_empty * net.d[vehs[kid].loc, j] * z[kid, j] for (kid, j) in z)
              - C_WAIT * gp.quicksum(w.values())
              - gp.quicksum(pool[oid].penalty * s[oid] for oid in pool)
              - gp.quicksum(PHI * s[oid] for oid in exp))
    value = 0
    if flat_v is not None:
        value = flat_v * (gp.quicksum(y.values()) + gp.quicksum(z.values()) + gp.quicksum(w.values()))
    if vfun is not None:                       # value of the hub and period at which each vehicle is free again
        value = (gp.quicksum(vfun(routes[kid][ri]["t0"] + net.tau[routes[kid][ri]["p"], routes[kid][ri]["e"]],
                                  routes[kid][ri]["e"]) * var for (kid, ri), var in y.items())
                 + gp.quicksum(vfun(t + net.tau[vehs[kid].loc, j], j) * var for (kid, j), var in z.items())
                 + gp.quicksum(vfun(t + 1, vehs[kid].loc) * var for kid, var in w.items()))
    tie = TIE * gp.quicksum(routes[kid][ri]["delay"] * var for (kid, ri), var in y.items())
    m.setObjective(profit + value - tie, GRB.MAXIMIZE)
    if force is not None:
        kid, kind, candidate_key = force
        if kind == 'wait':
            if kid not in w: raise ValueError('wait requires an idle deciding vehicle')
            variable = w[kid]
        else:
            matches = [ri for ri,r in enumerate(routes.get(kid,[])) if route_key(r)==candidate_key]
            if len(matches)!=1: raise ValueError('candidate is not uniquely available in the observed state')
            variable = y[kid,matches[0]]
        m.addConstr(variable == 1, name='fifth_current_period_only_candidate')
    m.optimize()
    status = ("OPTIMAL" if m.Status == GRB.OPTIMAL else
              "INFEASIBLE" if m.Status == GRB.INFEASIBLE else
              "UNBOUNDED" if m.Status in (GRB.UNBOUNDED, GRB.INF_OR_UNBD) else
              "FEASIBLE_LIMIT" if m.SolCount > 0 else "NO_SOLUTION_LIMIT")
    if solve_info is not None:
        solve_info.update(status=status, runtime=m.Runtime, incumbent=m.ObjVal if m.SolCount else None,
                          bound=m.ObjBound if math.isfinite(m.ObjBound) else None,
                          gap=(m.MIPGap if math.isfinite(m.MIPGap) else None) if m.SolCount and m.IsMIP else 0 if m.SolCount else None)
        solve_info["candidate_route_count"] = sum(len(r) for r in routes.values())
    if not m.SolCount:
        m.dispose()
        raise RuntimeError(f"solver returned {status}; no incumbent to commit")
    sol = dict(x=[key for key, var in x.items() if var.X > 0.5],
               y=[key for key, var in y.items() if var.X > 0.5],
               z=[key for key, var in z.items() if var.X > 0.5])
    result = profit.getValue(), m.ObjVal, sol, routes, exp, len(dec)
    if solve_info is not None:
        assigned = {oid for oid, kid in sol["x"]}
        components = dict(
            assignment_revenue=sum(pool[oid].revenue for oid in assigned),
            planned_route_cost=sum(routes[kid][ri]["cost"] for kid, ri in sol["y"]),
            planned_reposition_cost=sum(c_empty * net.d[vehs[kid].loc, j] for kid, j in sol["z"]),
            idle_waiting_cost=C_WAIT * sum(var.X > 0.5 for var in w.values()),
            backlog_penalty=sum(o.penalty for oid, o in pool.items() if oid not in assigned),
            expiry_penalty=PHI * len(exp - assigned),
        )
        components["business_cost"] = sum(v for k, v in components.items() if k != "assignment_revenue")
        components["profit_identity_residual"] = components["assignment_revenue"] - components["business_cost"] - result[0]
        solve_info["accounting"] = components
    m.dispose()
    return result

