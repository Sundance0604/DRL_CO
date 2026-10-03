"""Prototype of the single-level matching model (M_t) in HV-AV-HV-operation/main_v2.tex.

Purpose: check that the formulation is implementable and internally consistent.
It is a standalone simulator; it does not modify or reuse the repo's lower layer.
Only the scenario generator of DRL_CO (commit 59935d3) is used for graphs and orders.

Run from the DRL_CO repo root:
    python -m model.mt_prototype

Cost parameters below are placeholders, not calibrated values.
"""
import math, random, sys
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from model_families.single.scenarios import generate_scenario
from .demand import city_times, convert, convert_batches, ready_time

SPEED, Q = 20, 7
C_LOADED, C_WAIT, PHI = 10.0, 0.0, 300.0      # placeholders
DMAX = 3                                      # longest departure delay of an idle vehicle, in periods
TIE = 1.0                                     # solver-only tie-break against needless delays (not part of the profit)


class Net:
    """Highway graph with a consistent family of shortest paths (Assumption 3)."""
    def __init__(self, G, seed, speed=SPEED):
        self.speed = speed
        rng = random.Random(seed)
        self.H = nx.Graph()
        for u, v, w in G.edges(data="weight"):
            self.H.add_edge(u, v, weight=w, pw=w + rng.random() * 1e-6)   # unique shortest paths
        self.path, self.d, self.tau = {}, {}, {}
        for u in self.H:
            for v, p in nx.single_source_dijkstra_path(self.H, u, weight="pw").items():
                self.path[u, v] = p
                self.d[u, v] = sum(self.H[a][b]["weight"] for a, b in zip(p, p[1:]))
                self.tau[u, v] = sum(self.lt(a, b) for a, b in zip(p, p[1:]))
        for (u, v), p in self.path.items():            # subpath consistency
            for i in range(len(p)):
                for j in range(i, len(p)):
                    assert p[i:j + 1] == self.path[p[i], p[j]]

    def lt(self, a, b):
        return max(1, math.ceil(self.H[a][b]["weight"] / self.speed))

    def arcs(self, u, v):
        p = self.path[u, v]
        return list(zip(p, p[1:]))


class Veh:
    def __init__(self, i, hub):
        self.id, self.loc, self.phase = i, hub, "idle"
        self.route, self.dead, self.arrive = [], 0, None   # remaining hubs, deadhead hubs left, next arrival
        self.hold = None                                   # departure period of a vehicle that waits at its hub with orders
        self.orders, self.onboard = {}, set()

    def copy(self):
        c = Veh(self.id, self.loc)
        c.phase, c.route, c.dead, c.arrive = self.phase, list(self.route), self.dead, self.arrive
        c.hold = self.hold
        c.orders, c.onboard = dict(self.orders), set(self.onboard)
        return c


class DecisionPlan(tuple):
    """Backward-compatible six-field result with a simulator revision guard."""
    def __new__(cls, result, period, revision):
        obj = super().__new__(cls, result)
        obj.period, obj.revision = period, revision
        return obj


def build_and_solve(t, net, pool, vehs, hops, c_empty, repos, flat_v=None, vfun=None, parameters=None,
                    solver_parameters=None, solve_info=None):
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
    dec = [k for k in vehs if k.phase == "idle"
           or (k.phase == "trip" and k.dead == 0 and (k.arrive == t or k.hold is not None))]
    routes, y, x, z, w = {}, {}, {}, {}, {}
    cover = {}                                          # (o, k) -> list of y vars whose route serves o
    for k in dec:
        c, cand = k.loc, []
        if k.phase == "idle":
            pickups = [c] + (list(net.H.neighbors(c)) if hops else [])
            for p in pickups:
                for e in net.H:
                    if e == p:
                        continue
                    for delay in range(DMAX + 1):          # wait `delay` periods at the own hub, then leave
                        cand.append(dict(p=p, e=e, B=net.path[p, e], delay=delay, t0=t + delay + net.tau[c, p],
                                         cost=C_WAIT * delay + c_empty * net.d[c, p] + C_LOADED * net.d[p, e],
                                         new=True))
        else:
            ebar = k.route[-1]
            leave = t if k.hold is None else k.hold        # departure period from the current hub
            assert [c] + k.route == net.path[c, ebar]
            cand.append(dict(p=c, e=ebar, B=net.path[c, ebar], delay=0, t0=leave, cost=0.0, new=False))
            for e in net.H:
                if e not in (c, ebar) and net.path[c, e][:len(k.route) + 1] == [c] + k.route:   # eq. (extension)
                    cand.append(dict(p=c, e=e, B=net.path[c, e], delay=0, t0=leave,
                                     cost=C_LOADED * net.d[ebar, e], new=True))
        for r in cand:
            B, serve = r["B"], []
            for o in pool.values():
                if o.departure in B and o.destination in B:
                    i, j = B.index(o.departure), B.index(o.destination)
                    if i < j and B[i:j + 1] == net.path[o.departure, o.destination]:
                        t_dep = r["t0"] + net.tau[r["p"], o.departure]
                        t_des = r["t0"] + net.tau[r["p"], o.destination]
                        if t_dep >= ready_time(o, t) and t_des <= o.end_time:  # eq. (okr)
                            serve.append(o)
            r["serve"] = serve
        cand = [r for r in cand if r["serve"] or not r["new"]]
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
    exp = {oid for oid, o in pool.items()
           if max(t + 1, ready_time(o, t)) + net.tau[o.departure, o.destination] > o.end_time}
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
    m.optimize()
    status = ("OPTIMAL" if m.Status == GRB.OPTIMAL else
              "INFEASIBLE" if m.Status == GRB.INFEASIBLE else
              "UNBOUNDED" if m.Status in (GRB.UNBOUNDED, GRB.INF_OR_UNBD) else
              "FEASIBLE_LIMIT" if m.SolCount > 0 else "NO_SOLUTION_LIMIT")
    if solve_info is not None:
        solve_info.update(status=status, runtime=m.Runtime, incumbent=m.ObjVal if m.SolCount else None,
                          bound=m.ObjBound if math.isfinite(m.ObjBound) else None,
                          gap=m.MIPGap if m.SolCount and math.isfinite(m.MIPGap) else None)
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


def depart(k, t, net, capacity=Q):
    if k.phase == "trip" and k.dead == 0:
        assert sum(k.orders[i].passenger for i in k.onboard) <= capacity, "seat limit violated on a link"
    k.arrive = t + net.lt(k.loc, k.route[0])


def load_case(seed, nv, ops, horizon=24, lead=True, batches=True):
    """Network, initial hubs and orders of a scenario.
    batches=True: HV tours bring the bookings of a collection period to the hub together (default).
    batches=False, lead=True: every order has its own direct first mile.
    lead=False: the old setting in which an order is at its hub the moment it is booked."""
    vehicles, all_orders, graph = generate_scenario(seed, horizon=horizon, num_vehicles=nv, orders_per_step=ops)
    net = Net(graph.G, seed)
    net.radii = city_times(net, seed)
    hubs = [v.intercity for v in vehicles.values()]
    if batches and lead:
        return net, hubs, convert_batches(all_orders, net, net.radii, seed)
    return net, hubs, convert(all_orders, net, net.radii, seed, lead)


def simulate(seed, nv, ops, hops, c_empty, repos=False, check_flat=False, horizon=24, lead=True, batches=True):
    net, hubs, orders = load_case(seed, nv, ops, horizon, lead, batches)
    return run(net, hubs, orders, hops, c_empty, repos, check_flat, horizon)


class Sim:
    """State of the rolling procedure (Algorithm 1): vehicles, order pool, clock and profit."""
    def __init__(self, net, hubs, horizon, parameters=None, solver_parameters=None):
        self.revision = 0
        self.parameters, self.solver_parameters = parameters or {}, solver_parameters or {}
        self.net, self.T, self.t, self.J = net, horizon, 0, 0.0
        self.vehs = [Veh(i, hub) for i, hub in enumerate(hubs)]
        self.pool = {}
        self.S = dict(assigned=0, cross=0, downstream=0, enroute=0, extended=0, delayed=0, repos=0, cancelled=0,
                      delivered=0)

    def clone(self):
        c = Sim.__new__(Sim)
        c.net, c.T, c.t, c.J = self.net, self.T, self.t, self.J
        c.revision = self.revision
        c.parameters, c.solver_parameters = dict(self.parameters), dict(self.solver_parameters)
        c.vehs, c.pool, c.S = [v.copy() for v in self.vehs], dict(self.pool), dict(self.S)
        return c

    def begin_period(self, new_orders):
        """Vehicles that reach a hub at the start of period t; then the newly booked orders join the pool."""
        self.revision += 1
        t, net, S = self.t, self.net, self.S
        for k in self.vehs:
            if k.arrive != t:
                continue
            k.loc = k.route.pop(0)
            if k.dead > 0:
                k.dead -= 1
            if k.phase == "repos":
                if k.route: depart(k, t, net, self.parameters.get("capacity", Q))
                else: k.phase, k.arrive = "idle", None
                continue
            if k.dead > 0:                                 # still driving empty to the pickup hub
                depart(k, t, net, self.parameters.get("capacity", Q)); continue
            for oid in [i for i in k.onboard if k.orders[i].destination == k.loc]:
                assert t <= k.orders[oid].end_time, "late delivery"
                k.onboard.discard(oid); del k.orders[oid]; S["delivered"] += 1
            for oid, o in k.orders.items():
                if o.departure == k.loc and oid not in k.onboard:
                    assert t >= o.start_time
                    k.onboard.add(oid)
            if not k.route:
                assert not k.orders, "trip ended with undelivered orders"
                k.phase, k.arrive = "idle", None
        for o in new_orders:
            self.pool[o.id] = o

    def decide(self, hops, c_empty, repos=False, vfun=None, check_flat=False):
        """Solve the period model, apply the decisions, cancel expired orders, move the clock."""
        t, net, pool, vehs, S = self.t, self.net, self.pool, self.vehs, self.S
        plan = self.solve_plan(hops, c_empty, repos, vfun)
        profit, obj, sol, routes, exp, ndec = plan
        if check_flat:                                     # flat value function must not change the optimum
            assert vfun is None
            _, obj_flat, _, _, _, _ = build_and_solve(t, net, pool, vehs, hops, c_empty, repos, flat_v=1234.5)
            assert abs(obj_flat - obj - 1234.5 * ndec) < 1e-4 * max(1.0, abs(obj))
        return self.commit_plan(plan)

    def solve_plan(self, hops, c_empty, repos=False, vfun=None, solve_info=None):
        """Pure decision step: no orders, vehicles, clock or accounting are changed."""
        result = build_and_solve(self.t, self.net, self.pool, self.vehs, hops, c_empty, repos,
                               vfun=vfun, parameters=self.parameters,
                               solver_parameters=self.solver_parameters, solve_info=solve_info)
        return DecisionPlan(result, self.t, self.revision)

    def commit_plan(self, plan):
        """Commit a solved incumbent once. Platform additionally records its state hash."""
        if not isinstance(plan, DecisionPlan) or plan.period != self.t or plan.revision != self.revision:
            raise RuntimeError("stale or already committed decision plan")
        self.revision += 1
        profit, obj, sol, routes, exp, ndec = plan
        t, net, pool, vehs, S = self.t, self.net, self.pool, self.vehs, self.S
        self.J += profit
        chosen = {kid: routes[kid][ri] for kid, ri in sol["y"]}
        for oid, kid in sol["x"]:
            k, o = vehs[kid], pool.pop(oid)
            S["assigned"] += 1
            if k.phase == "trip": S["enroute"] += 1
            elif chosen[kid]["p"] != k.loc: S["cross"] += 1          # served after an empty drive to another hub
            elif o.departure != k.loc: S["downstream"] += 1          # picked up later along the backbone
            k.orders[oid] = o
        for kid, r in chosen.items():
            k = vehs[kid]
            if k.phase == "idle":
                deadpath = net.path[k.loc, r["p"]][1:]
                k.phase, k.dead, k.route = "trip", len(deadpath), deadpath + r["B"][1:]
                if deadpath:                               # waits `delay` periods, then drives empty to the pickup hub
                    k.arrive = t + r["delay"] + net.lt(k.loc, k.route[0])
                    S["delayed"] += int(r["delay"] > 0)
                    continue
                k.hold = t + r["delay"]
                S["delayed"] += int(r["delay"] > 0)
            elif r["new"]:
                k.route = r["B"][1:]; S["extended"] += 1
            leave = t if k.hold is None else k.hold
            if leave > t:                                  # keeps waiting at the hub; decides again next period
                k.arrive = None
                continue
            k.hold = None
            for oid, o in k.orders.items():
                if o.departure == k.loc and oid not in k.onboard:
                    assert t >= o.start_time, "boarded before the order was ready"
                    k.onboard.add(oid)
            depart(k, t, net, self.parameters.get("capacity", Q))
        for kid, j in sol["z"]:
            k = vehs[kid]
            k.phase, k.route, k.dead = "repos", net.path[k.loc, j][1:], 0
            S["repos"] += 1
            depart(k, t, net, self.parameters.get("capacity", Q))
        for oid in exp:
            if oid in pool:
                del pool[oid]; S["cancelled"] += 1
        self.t += 1
        return profit


def run(net, hubs, all_orders, hops, c_empty, repos=False, check_flat=False, horizon=24, vfun=None, policy=None):
    """Rolling dispatch for a given network, initial hubs and order stream.
    policy(sim), if given, must call sim.decide(...) once; otherwise the fixed settings are used."""
    sim = Sim(net, hubs, horizon)
    by_t = {}
    for o in all_orders.values():
        by_t.setdefault(getattr(o, "book_time", o.start_time), []).append(o)
    for t in range(horizon):
        sim.begin_period(by_t.get(t, []))
        if policy is not None:
            policy(sim)
            assert sim.t == t + 1
        else:
            sim.decide(hops, c_empty, repos, vfun, check_flat)
    S = dict(sim.S)
    S["J"], S["arrivals"] = sim.J, len(all_orders)
    return S


if __name__ == "__main__":
    seeds = range(10_000, 10_030)
    for name, nv, ops in [("normal", 11, 5), ("high", 5, 8)]:
        for label, hops, c_empty, lead, batches in [
                ("no lead: own-hub", False, 10.0, False, False), ("no lead: cross-hub", True, 10.0, False, False),
                ("direct: own-hub", False, 10.0, True, False), ("direct: cross-hub", True, 10.0, True, False),
                ("own-hub only", False, 10.0, True, True), ("cross-hub, c_e=10", True, 10.0, True, True),
                ("cross-hub, c_e=50", True, 50.0, True, True)]:
            tot, js = None, []
            for i, seed in enumerate(seeds):
                s = simulate(seed, nv, ops, hops, c_empty, check_flat=(i < 3), lead=lead, batches=batches)
                js.append(s["J"])
                tot = s if tot is None else {key: tot[key] + val for key, val in s.items()}
            n = len(seeds)
            if label.endswith("own-hub") or label == "own-hub only":
                base = np.array(js)
            diff = np.array(js) - base
            half = 1.96 * diff.std(ddof=1) / np.sqrt(n) if hops else 0.0
            print(f"{name:7s} {label:20s} J/scn={tot['J']/n:9.0f}  vs own-hub: {diff.mean():+8.0f} +/- {half:6.0f}"
                  f"  assigned={tot['assigned']:5d}/{tot['arrivals']}  cross={tot['cross']:4d}"
                  f"  downstream={tot['downstream']:4d}  enroute={tot['enroute']:4d}  extended={tot['extended']:3d}"
                  f"  delayed={tot['delayed']:4d}"
                  f"  cancelled={tot['cancelled']:4d}  delivered={tot['delivered']:5d}")
    print("all invariants held: seat limit on every link, on-time delivery, pickup after ready time, flat-V equivalence")
