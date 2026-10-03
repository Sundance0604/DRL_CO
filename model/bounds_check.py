"""Numerical check of the two bounds in HV-AV-HV-operation/main_v2.tex.

(H)  perfect-information bound: time-expanded MILP, and its LP relaxation
(F)  fluid bound: the same LP with expected order counts per type

Checks
 1. J(policy, omega) <= J^H(omega) on every scenario, for the prototype policies.
 2. Jensen on a fixed network: mean over samples of the LP relaxation of (H)
    <= (F) evaluated at the empirical mean demand (exact inequality), and
    <= (F) evaluated at the true expected demand (up to sampling noise).

Run from the DRL_CO repo root:
    python -m model.bounds_check
Cost parameters are the placeholders of mt_prototype.py.
"""
import math, random, sys, time
from types import SimpleNamespace
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from drl_co.simulation.scenarios import generate_scenario
from .mt_prototype import Net, run, load_case, SPEED, Q, C_LOADED, C_WAIT, PHI, DMAX
from .demand import make_order, sample_batches as sample_stream, lead_pmf, SLACK


def never_served_loss(o, net, T, expiry_cost=PHI):
    """L_o: waiting penalties plus cancellation penalty of an order that is never assigned."""
    tau = net.tau[o.departure, o.destination]
    g = getattr(o, "book_time", o.start_time)
    tbar = max(g, math.floor(o.end_time - tau))                      # first period in which o expires
    return o.penalty * (min(tbar, T - 1) - g + 1) + (expiry_cost if tbar <= T - 1 else 0.0)


def bound(net, hubs, items, T, c_empty, relax, time_limit=60, barrier=False, penalties=None, parameters=None, solver_parameters=None, solve_info=None):
    """items: list of (order-like, multiplicity). relax=False gives (H) with multiplicity 1;
    relax=True gives the LP: the relaxation of (H), or (F) when multiplicities are expected counts.
    penalties: {("arc", (u, v), t): c} and {("wait", u, t): c} are subtracted from the objective per unit of
    flow on the movement arc entered in period t and per vehicle waiting at u during period t (information
    relaxation with a penalty)."""
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.TimeLimit = time_limit; m.Params.MIPGap = 1e-4
    parameters=parameters or {}
    Q,C_LOADED,C_WAIT,DMAX = [parameters.get(k,d) for k,d in (("capacity",7),("loaded_cost",10),("waiting_cost",0),("max_delay",3))]
    m.Params.Threads=1
    for key,value in (solver_parameters or {}).items():
        m.setParam({"time_limit_seconds":"TimeLimit","threads":"Threads","mip_gap":"MIPGap","seed":"Seed"}[key],value)
    if barrier:                     # large LP: interior point without crossover
        m.Params.Method = 2; m.Params.Crossover = 0
    arcs = [(u, v) for u in net.H for v in net.H.neighbors(u)]
    tmax = max(max(net.tau.values()), 1)
    Tbar = max(T + DMAX + 3 * tmax, math.ceil(max([T]+[o.end_time for o, _ in items])))
    vt = GRB.CONTINUOUS if relax else GRB.INTEGER
    xl = {(a, t): m.addVar(vtype=vt) for a in arcs for t in range(Tbar) if t + net.lt(*a) <= Tbar}
    xe = {(a, t): m.addVar(vtype=vt) for a in arcs for t in range(Tbar) if t + net.lt(*a) <= Tbar}
    eta = {(u, t): m.addVar(vtype=vt) for u in net.H for t in range(Tbar)}
    cap, obj, const = {}, [], 0.0
    for o, mult in items:
        tau = net.tau[o.departure, o.destination]
        L = never_served_loss(o, net, T, parameters.get("expiry_cost",PHI))
        const -= L * mult
        path = net.path[o.departure, o.destination]
        # boarding window W_o: ready time, deadline, and assignment before the horizon ends
        window = range(math.ceil(o.start_time), min(math.floor(o.end_time - tau), T - 1 + DMAX + 2 * tmax) + 1)
        bs = []
        for t in window:
            b = m.addVar(vtype=GRB.CONTINUOUS if relax else GRB.BINARY)
            bs.append(b); obj.append((o.revenue + L) * b)
            for a in zip(path, path[1:]):
                cap.setdefault((a, t + net.tau[o.departure, a[0]]), []).append(o.passenger * b)
        if bs:
            m.addConstr(gp.quicksum(bs) <= mult)                                   # (H1) / (F1)
    m.update()
    for key, terms in cap.items():                                                 # (H2)
        m.addConstr(gp.quicksum(terms) <= Q * xl[key])
    init = {u: 0 for u in net.H}
    for h in hubs:
        init[h] += 1
    for u in net.H:                                                                # (H3)
        for t in range(Tbar):
            out = eta[u, t] + gp.quicksum(xl[(u, v), t] + xe[(u, v), t] for v in net.H.neighbors(u) if ((u, v), t) in xl)
            if t == 0:
                m.addConstr(out == init[u])
            else:
                inflow = eta[u, t - 1] + gp.quicksum(
                    xl[(v, u), t - net.lt(v, u)] + xe[(v, u), t - net.lt(v, u)]
                    for v in net.H.neighbors(u) if t - net.lt(v, u) >= 0)
                m.addConstr(out == inflow)
    cost = gp.quicksum(net.H[a[0]][a[1]]["weight"] * (C_LOADED * xl[a, t] + c_empty * xe[a, t]) for (a, t) in xl)
    wait = C_WAIT * gp.quicksum(eta[u, t] for u in net.H for t in range(min(T, Tbar)))
    pen = 0
    if penalties:
        pen = (gp.quicksum(c * (xl[a, t] + xe[a, t]) for (kind, a, t), c in penalties.items()
                           if kind == "arc" and (a, t) in xl)
               + gp.quicksum(c * eta[u, t] for (kind, u, t), c in penalties.items() if kind == "wait" and (u, t) in eta))
    m.setObjective(gp.quicksum(obj) + const - cost - wait - pen, GRB.MAXIMIZE)
    m.optimize()
    if solve_info is not None:
        solve_info.update(status="OPTIMAL" if m.Status==GRB.OPTIMAL else "FEASIBLE_LIMIT" if m.SolCount else "NO_SOLUTION_LIMIT",runtime=m.Runtime)
    if relax:
        if m.Status != GRB.OPTIMAL:
            m.dispose();raise RuntimeError("LP bound requires optimal solution")
        result=m.ObjVal,m.ObjVal
    else:
        result=(m.ObjVal if m.SolCount else None), (m.ObjBound if math.isfinite(m.ObjBound) else None)
    m.dispose()
    return result


def expected_types(net, T, ops):
    """All order types with their expected number per day, for the demand of demand.py."""
    hubs, items = sorted(net.H), []
    lam = ops / (len(hubs) * (len(hubs) - 1) * Q * (SLACK[1] - SLACK[0] + 1))
    pmf = {u: lead_pmf(net.radii[u]) for u in hubs}
    for g in range(T):
        for dep in hubs:
            for des in hubs:
                if des == dep:
                    continue
                for n in range(1, Q + 1):
                    for slack in range(SLACK[0], SLACK[1] + 1):
                        for lead, prob in pmf[dep].items():
                            items.append((make_order(0, dep, des, n, g, slack, lead, net), lam * prob))
    return items


if __name__ == "__main__":
    T, c_e = 24, 10.0
    print("== check 1: J(policy) <= J^H on every scenario ==")
    for name, nv, ops in [("normal", 11, 5), ("high", 5, 8)]:
        rows, t0 = [], time.time()
        for seed in range(10_000, 10_030):
            net, hubs, all_orders = load_case(seed, nv, ops, T)
            items = [(o, 1) for o in all_orders.values()]
            milp = seed < 10_010            # the MILP is solved on the first 10 scenarios only (60 s limit each)
            lp, _ = bound(net, hubs, items, T, c_e, relax=True)
            best, ub = bound(net, hubs, items, T, c_e, relax=False) if milp else (float("nan"), lp)
            j_own = run(net, hubs, all_orders, False, c_e)["J"]
            j_cross = run(net, hubs, all_orders, True, c_e)["J"]
            assert max(j_own, j_cross) <= ub + 1e-6 and ub <= lp + 1e-6 * abs(lp)
            rows.append((j_own, j_cross, best, ub, lp, milp))
        r = np.array(rows)
        k = r[:, 5] > 0
        print(f"{name:7s} 30 scenarios: own-hub={r[:,0].mean():9.0f} cross-hub={r[:,1].mean():9.0f} H-LP={r[:,4].mean():9.0f}")
        print(f"{'':7s} 10 scenarios with MILP: own-hub={r[k,0].mean():9.0f} cross-hub={r[k,1].mean():9.0f} "
              f"H best solution={r[k,2].mean():9.0f} H bound={r[k,3].mean():9.0f} H-LP={r[k,4].mean():9.0f} | {time.time()-t0:.0f}s")

    print("== check 2: Jensen on a fixed network ==")
    for name, nv, ops in [("normal", 11, 5), ("high", 5, 8)]:
        net, hubs, _ = load_case(10_000, nv, ops, T)
        rng, N, lps, counts, reps = random.Random(7), 40, [], {}, {}
        for i in range(N):
            stream = sample_stream(net, net.radii, rng, T, ops, first_id=i * 10_000)
            orders = {o.id: o for batch in stream.values() for o in batch}
            lps.append(bound(net, hubs, [(o, 1) for o in orders.values()], T, c_e, relax=True)[0])
            for o in orders.values():          # orders with the same ready period and deadline are one type
                key = (o.departure, o.destination, o.passenger, o.book_time, math.ceil(o.start_time), o.end_time)
                counts[key] = counts.get(key, 0) + 1; reps[key] = o
        emp = bound(net, hubs, [(reps[k], c / N) for k, c in counts.items()], T, c_e, relax=True)[0]
        lps = np.array(lps)
        assert lps.mean() <= emp + 1e-6 * abs(emp), "Jensen violated at the empirical mean"
        print(f"{name:7s} mean H-LP={lps.mean():9.0f} (se {lps.std(ddof=1)/np.sqrt(N):.0f})  F(empirical mean)={emp:9.0f}"
              f"  [{len(counts)} types seen; with batch arrivals the true expected counts are not enumerated]")
    print("all checks passed")
