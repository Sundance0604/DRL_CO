"""Online look-ahead policies for the model in HV-AV-HV-operation/main_v2.tex.

All policies here are non-anticipative: they use the current state and the
distribution of future orders, never the realized future.

  own      (M_t) with pickup at the vehicle's own hub only
  cross    (M_t) with cross-hub pickup
  fluidS   (M^V_t), V = duals of a fluid LP solved once at the start of the day
  fluidR   (M^V_t), V = duals of a fluid LP re-solved every period from the current state
  rollout  each period, simulate K sampled futures for each candidate decision rule
           and apply the best one (base policy for the simulated future: one fixed rule)

The fluid LP used for V is a coarse one (group size fixed at its mean, deadline slack
at its mean); it only has to produce values, not a bound. The bound is (H) from bounds_check.py.

Run from the DRL_CO repo root:
    python -m model.online_lookahead phase1
    python -m model.online_lookahead rollout 10
Cost parameters are the placeholders of mt_prototype.py.
"""
import json, math, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from drl_co.simulation.scenarios import generate_scenario
from .mt_prototype import Net, Sim, run, load_case, SPEED, Q, C_LOADED, C_WAIT, PHI, DMAX
from .bounds_check import bound, never_served_loss
from .demand import sample_batches as sample_stream, ready_time, ready_bound, SLACK, H_BATCH, TAU_S, K_CA

T, C_E = 24, 10.0
CHOSEN = __file__.replace("online_lookahead.py", "online_lookahead_chosen.json")
LEVELS = {"normal": (11, 5), "high": (5, 8)}
TEST, TRAIN = list(range(10_000, 10_030)), list(range(20_000, 20_010))
MEAN_N, MEAN_SLACK = 4, 15
MEAN_DIST = 32 / (9 * math.pi)      # mean distance from a uniform point in a unit disc to a boundary point


def fluid_values(sim, ops, extra=None, from_period=None):
    """Coarse fluid LP from the current state; returns the duals of vehicle conservation, {(hub, period): value}.
    extra: {(hub, booking period): additional expected orders} perturbs the expected demand;
    from_period: first booking period of the expected demand (default: the period after the current one)."""
    net, t0 = sim.net, sim.t
    params = getattr(sim, "parameters", {})
    Q, DMAX = params.get("capacity", 7), params.get("max_delay", 3)
    PHI = params.get("expiry_cost", 300)
    C_LOADED, C_E = params.get("loaded_cost", 10), params.get("empty_cost", 10)
    hubs, tmax = list(net.H), max(net.tau.values())
    Tbar = sim.T + DMAX + 3 * tmax
    supply = {}
    for k in sim.vehs:                                   # where and when each vehicle is free again
        if k.phase == "idle":
            node = (k.loc, t0)
        else:
            if k.hold is not None:                       # waits at its hub with orders until period hold
                seq, when = [k.loc] + k.route, k.hold
            elif k.arrive == t0:                         # at a hub, about to leave
                seq, when = [k.loc] + k.route, t0
            else:                                        # on a link, or waiting before an empty drive
                seq, when = k.route, k.arrive
            when += sum(net.lt(a, b) for a, b in zip(seq, seq[1:]))
            node = (seq[-1], when)
        if node[1] < Tbar:
            supply[node] = supply.get(node, 0) + 1
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.Threads = 1
    for key, value in getattr(sim, "solver_parameters", {}).items():
        m.setParam({"time_limit_seconds": "TimeLimit", "mip_gap": "MIPGap", "threads": "Threads", "seed": "Seed"}[key], value)
    m.Params.Method = 2; m.Params.Crossover = 0          # interior point: central duals
    arcs = [(u, v) for u in hubs for v in net.H.neighbors(u)]
    periods = range(t0, Tbar)
    xl = {(a, t): m.addVar() for a in arcs for t in periods if t + net.lt(*a) <= Tbar}
    xe = {(a, t): m.addVar() for a in arcs for t in periods if t + net.lt(*a) <= Tbar}
    eta = {(u, t): m.addVar() for u in hubs for t in periods}
    cap, obj = {}, []

    def add(dep, des, n, first, last, value, mult):
        path, bs = net.path[dep, des], []
        for t in range(first, last + 1):
            b = m.addVar(); bs.append(b); obj.append(value * b)
            for a in zip(path, path[1:]):
                cap.setdefault((a, t + net.tau[dep, a[0]]), []).append(n * b)
        if bs:
            m.addConstr(gp.quicksum(bs) <= mult)

    cutoff = sim.T - 1 + DMAX + 2 * tmax
    for o in sim.pool.values():                          # orders already in the pool: known exactly
        tau = net.tau[o.departure, o.destination]
        tbar = max(t0, math.floor(o.end_time - tau))
        loss = o.penalty * (min(tbar, sim.T - 1) - t0 + 1) + (PHI if tbar <= sim.T - 1 else 0.0)
        add(o.departure, o.destination, o.passenger, max(t0, math.ceil(ready_time(o, t0))),
            min(math.floor(o.end_time - tau), cutoff), o.revenue + loss, 1.0)
    lam = ops / (len(hubs) * (len(hubs) - 1))
    extra = extra or {}
    for t in range(t0 + 1 if from_period is None else from_period, sim.T):   # orders still to come: expected counts
        for dep in hubs:
            lam_dep = lam + extra.get((dep, t), 0.0) / (len(hubs) - 1)
            for des in hubs:
                if des == dep:
                    continue
                d, tau = net.d[dep, des], net.tau[dep, des]
                # booking period, the rest of the collection period and a typical two-stop tour
                ready = t + math.ceil(H_BATCH / 2 + (2.0 + K_CA * math.sqrt(2 * math.pi)) * net.radii[dep] + 2 * TAU_S)
                o = SimpleNamespace(departure=dep, destination=des, book_time=t, start_time=ready,
                                    end_time=ready + tau + MEAN_SLACK, penalty=MEAN_N * 5)
                add(dep, des, MEAN_N, ready, min(math.floor(o.end_time - tau), cutoff),
                    d * 100 + MEAN_N * 50 + never_served_loss(o, net, sim.T, PHI), lam_dep)
    m.update()
    for key, terms in cap.items():
        m.addConstr(gp.quicksum(terms) <= Q * xl[key])
    flow = {}
    for u in hubs:
        for t in periods:
            out = eta[u, t] + gp.quicksum(xl[(u, v), t] + xe[(u, v), t] for v in net.H.neighbors(u) if ((u, v), t) in xl)
            inflow = gp.quicksum(xl[(v, u), t - net.lt(v, u)] + xe[(v, u), t - net.lt(v, u)]
                                 for v in net.H.neighbors(u) if t - net.lt(v, u) >= t0)
            if t > t0:
                inflow = inflow + eta[u, t - 1]
            flow[u, t] = m.addConstr(out - inflow == supply.get((u, t), 0))
    cost = gp.quicksum(net.H[a[0]][a[1]]["weight"] * (C_LOADED * xl[a, t] + C_E * xe[a, t]) for (a, t) in xl)
    m.setObjective(gp.quicksum(obj) - cost, GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        m.dispose()
        raise RuntimeError("fluid duals require an optimal LP")
    result = {key: c.Pi for key, c in flow.items()}
    m.dispose()
    return result


def rule(kind, ops, repos=False, theta=1.0, V=None):
    """A decision rule: a function that performs one sim.decide(...).
    For fluidS, V may be supplied; otherwise it is computed at the first call and kept."""
    if kind == "own":
        return lambda sim: sim.decide(False, C_E)
    if kind == "cross":
        return lambda sim: sim.decide(True, C_E)
    cache = {} if V is None else {"V": V}

    def fluid(sim):
        if kind == "fluidR" or "V" not in cache:
            cache["V"] = fluid_values(sim, ops)
        V = cache["V"]
        sim.decide(True, C_E, repos, vfun=lambda when, hub: theta * V.get((hub, when), 0.0))
    return fluid


def parse(name, ops, V=None):
    """'own', 'cross', 'fluidS-r0-th1.0', 'fluidR-r1-th0.5', ..."""
    parts = name.split("-")
    if parts[0] in ("own", "cross"):
        return rule(parts[0], ops)
    return rule(parts[0], ops, repos=parts[1] == "r1", theta=float(parts[2][2:]), V=V if parts[0] == "fluidS" else None)


def sample_future(net, rng, ops, after, counter):
    """Orders booked after the current period, drawn from the demand model."""
    out = sample_stream(net, net.radii, rng, T, ops, after=after, first_id=10 ** 6 + counter[0])
    counter[0] += sum(len(v) for v in out.values())
    return out


def rollout(ops, base_name, cand_names, K, seed, truth=None):
    """truth=None: futures are sampled from the order distribution (online policy).
    truth=orders by period: the realized future is used instead (not an online policy; diagnostic only)."""
    rng, counter = random.Random(seed), [0]
    picks, built = {}, {}

    def policy(sim):
        if not built:                                    # one fluid LP at the start of the day, shared by all rules
            V0 = fluid_values(sim, ops)
            built["base"] = parse(base_name, ops, V0)
            built["cands"] = [(c, parse(c, ops, V0)) for c in cand_names]
        base, cands = built["base"], built["cands"]
        if truth is None:
            futures = [sample_future(sim.net, rng, ops, sim.t, counter) for _ in range(K)]
        else:
            futures = [truth]
        best, best_val, best_name = None, -1e18, None
        for cname, fn in cands:
            total = 0.0
            for fut in futures:                          # same sampled futures for every candidate
                c = sim.clone(); fn(c)
                for t in range(c.t, T):
                    c.begin_period(fut.get(t, [])); base(c)
                total += c.J
            if total > best_val:
                best, best_val, best_name = fn, total, cname
        picks[best_name] = picks.get(best_name, 0) + 1
        best(sim)
    return policy, picks


def case(level, seed):
    nv, ops = LEVELS[level]
    net, hubs, orders = load_case(seed, nv, ops, T)
    return net, hubs, orders, ops


def task(args):
    level, seed, name = args
    net, hubs, orders, ops = case(level, seed)
    t0 = time.time()
    if name == "H-LP":
        val, extra = bound(net, hubs, [(o, 1) for o in orders.values()], T, C_E, relax=True)[0], {}
    elif name.startswith(("rollout:", "oracle:")):
        kind, base, cands, K = name.split(":")
        truth = None
        if kind == "oracle":
            truth = {}
            for o in orders.values():
                truth.setdefault(o.book_time, []).append(o)
        policy, picks = rollout(ops, base, cands.split(","), int(K), seed, truth)
        val, extra = run(net, hubs, orders, True, C_E, policy=policy)["J"], picks
    else:
        S = run(net, hubs, orders, True, C_E, policy=parse(name, ops))
        val, extra = S["J"], {k: S[k] for k in ("assigned", "cross", "delayed", "repos", "cancelled")}
    return level, seed, name, val, extra, time.time() - t0


def execute(tasks, workers=None):
    workers = workers or min(24, os.cpu_count() or 1)
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for level, seed, name, val, extra, secs in ex.map(task, tasks, chunksize=1):
            res[level, seed, name] = (val, extra, secs)
    return res


def report(res, level, seeds, names, ref):
    base = np.array([res[level, s, ref][0] for s in seeds])
    ub = np.array([res[level, s, "H-LP"][0] for s in seeds]) if (level, seeds[0], "H-LP") in res else None
    for name in names:
        v = np.array([res[level, s, name][0] for s in seeds])
        diff = v - base
        half = 1.96 * diff.std(ddof=1) / np.sqrt(len(seeds)) if name != ref else 0.0
        line = f"  {name[:34]:34s} J={v.mean():9.0f}  vs {ref}: {diff.mean():+8.0f} +/- {half:6.0f} ({diff.mean()/base.mean():+6.1%})"
        if ub is not None:
            line += f"  share of (H-LP - {ref}) gap: {diff.mean()/(ub.mean()-base.mean()):6.1%}"
        secs = np.mean([res[level, s, name][2] for s in seeds])
        print(line + f"  [{secs:.0f}s/run]")
    if ub is not None:
        print(f"  {'H-LP':34s} J={ub.mean():9.0f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "phase1":
        names = ["own", "cross"] + [f"{k}-r{r}-th{th}" for k in ("fluidS", "fluidR") for r in (0, 1) for th in (0.5, 1.0)]
        tasks = [(lv, s, n) for lv in LEVELS for s in TRAIN + TEST for n in names]
        tasks += [(lv, s, "H-LP") for lv in LEVELS for s in TEST]
        t0 = time.time(); res = execute(tasks)
        print(f"phase 1: {len(tasks)} runs in {time.time()-t0:.0f}s")
        chosen = {}
        for lv in LEVELS:
            means = {n: np.mean([res[lv, s, n][0] for s in TRAIN]) for n in names}
            myo = max(("own", "cross"), key=lambda n: means[n])
            best = max(names, key=lambda n: means[n])
            cheap = max([n for n in names if not n.startswith("fluidR")], key=lambda n: means[n])
            chosen[lv] = dict(myopic=myo, best=best, cheap=cheap)
            print(f"== {lv}: chosen on the 10 training scenarios: myopic benchmark={myo}, best rule={best}, best cheap rule={cheap}")
            print(" training scenarios:"); report(res, lv, TRAIN, names, myo)
            print(" test scenarios:"); report(res, lv, TEST, names, myo)
        json.dump(chosen, open(CHOSEN, "w"))
    elif mode == "rollout":
        chosen, K = json.load(open(CHOSEN)), int(sys.argv[2])
        tasks, names = [], {}
        for lv in LEVELS:
            c = chosen[lv]
            cands = ["own", "cross", "fluidS-r0-th1.0", "fluidS-r1-th1.0", "fluidS-r0-th0.5"]
            ro = f"rollout:{c['cheap']}:{','.join(cands)}:{K}"
            orc = f"oracle:{c['cheap']}:{','.join(cands)}:1"
            names[lv] = [c["myopic"], c["cheap"], ro, orc]
            tasks += [(lv, s, n) for s in TEST for n in set(names[lv]) | {"H-LP"}]
        t0 = time.time(); res = execute(tasks)
        print(f"rollout phase: {len(tasks)} runs in {time.time()-t0:.0f}s")
        for lv in LEVELS:
            print(f"== {lv}, 30 test scenarios")
            report(res, lv, TEST, list(dict.fromkeys(names[lv])), chosen[lv]["myopic"])
            picks = {}
            for s in TEST:
                for k, v in res[lv, s, names[lv][2]][1].items():
                    picks[k] = picks.get(k, 0) + v
            print("  rollout picks (rule applied, summed over periods and scenarios):", picks)
