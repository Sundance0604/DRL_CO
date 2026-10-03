"""Perfect-information bound with individual vehicles (no seat pooling, no change of vehicle).

Same time-expanded network as (H) in bounds_check.py, but every vehicle has its own binary path
and every order boards one vehicle: sum_k beta[o,k,t] <= 1 and sum_o n_o beta[o,k,t'] <= Q xl[k,arc,t].
The route rules of (M_t) (fixed backbone, anchoring, pickup radius) are still not imposed, so this is
still an upper bound on every policy; it is tighter than (H) because groups cannot be split or
transferred between vehicles. Solved as a MILP with a time limit; the reported value is the solver's
dual bound, which is valid whether or not the search finished.

Run from the DRL_CO repo root:
    python -m model.pervehicle_bound <level> <seed> [time limit]
"""
import math, sys, time
import gurobipy as gp
from gurobipy import GRB
from .mt_prototype import load_case, run, Q, C_LOADED, C_WAIT, DMAX
from .bounds_check import bound, never_served_loss
from .online_lookahead import LEVELS, T, C_E


def pervehicle(net, hubs, orders, T, c_empty, time_limit=120, threads=8):
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.TimeLimit = time_limit; m.Params.Threads = threads
    arcs = [(u, v) for u in net.H for v in net.H.neighbors(u)]
    tmax = max(net.tau.values())
    Tbar = max(T + DMAX + 3 * tmax, math.ceil(max(o.end_time for o in orders.values())))
    K = range(len(hubs))
    xl = {(k, a, t): m.addVar(vtype=GRB.BINARY) for k in K for a in arcs for t in range(Tbar) if t + net.lt(*a) <= Tbar}
    xe = {(k, a, t): m.addVar(vtype=GRB.BINARY) for k in K for a in arcs for t in range(Tbar) if t + net.lt(*a) <= Tbar}
    eta = {(k, u, t): m.addVar(vtype=GRB.BINARY) for k in K for u in net.H for t in range(Tbar)}
    beta, obj, const = {}, [], 0.0
    for o in orders.values():
        tau = net.tau[o.departure, o.destination]
        L = never_served_loss(o, net, T)
        const -= L
        path = net.path[o.departure, o.destination]
        window = range(math.ceil(o.start_time), min(math.floor(o.end_time - tau), T - 1 + DMAX + 2 * tmax) + 1)
        for k in K:
            for t in window:
                beta[o.id, k, t] = m.addVar(vtype=GRB.BINARY)
                obj.append((o.revenue + L) * beta[o.id, k, t])
    m.update()
    for o in orders.values():
        vs = [v for (oid, k, t), v in beta.items() if oid == o.id]
        if vs:
            m.addConstr(gp.quicksum(vs) <= 1)
    seat = {}
    for (oid, k, t), b in beta.items():
        o = orders[oid]
        path = net.path[o.departure, o.destination]
        for a in zip(path, path[1:]):
            seat.setdefault((k, a, t + net.tau[o.departure, a[0]]), []).append(o.passenger * b)
    for key, terms in seat.items():
        m.addConstr(gp.quicksum(terms) <= Q * xl[key])
        # A movement is labelled loaded iff at least one assigned passenger
        # group uses the arc. This prevents a cost-free relabelling when loaded
        # and empty distance rates differ.
        m.addConstr(xl[key] <= gp.quicksum(terms))
    for key in xl.keys() - seat.keys():
        m.addConstr(xl[key] == 0)
    for k in K:
        for u in net.H:
            for t in range(Tbar):
                out = eta[k, u, t] + gp.quicksum(
                    xl[k, (u, v), t] + xe[k, (u, v), t]
                    for v in net.H.neighbors(u) if (k, (u, v), t) in xl)
                if t == 0:
                    m.addConstr(out == (1 if hubs[k] == u else 0))
                else:
                    inflow = eta[k, u, t - 1] + gp.quicksum(
                        xl[k, (v, u), t - net.lt(v, u)] + xe[k, (v, u), t - net.lt(v, u)]
                        for v in net.H.neighbors(u) if t - net.lt(v, u) >= 0)
                    m.addConstr(out == inflow)
    cost = gp.quicksum(
        net.H[a[0]][a[1]]["weight"] * (C_LOADED * xl[k, a, t] + c_empty * xe[k, a, t])
        for (k, a, t) in xl)
    wait = C_WAIT * gp.quicksum(eta[k, u, t] for k in K for u in net.H for t in range(min(T, Tbar)))
    m.setObjective(gp.quicksum(obj) + const - cost - wait, GRB.MAXIMIZE)
    m.optimize()
    return (m.ObjVal if m.SolCount else float("nan")), m.ObjBound


if __name__ == "__main__":
    level, seed = sys.argv[1], int(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    nv, ops = LEVELS[level]
    net, hubs, orders = load_case(seed, nv, ops, T)
    t0 = time.time()
    lp = bound(net, hubs, [(o, 1) for o in orders.values()], T, C_E, relax=True)[0]
    best, ub = pervehicle(net, hubs, orders, T, C_E, limit)
    own = run(net, hubs, orders, False, C_E)["J"]; cross = run(net, hubs, orders, True, C_E)["J"]
    print(f"{level} {seed}: myopic own={own:.0f} cross={cross:.0f} | H-LP={lp:.0f} | per-vehicle H: best solution={best:.0f} bound={ub:.0f} | {time.time()-t0:.0f}s")
