"""Information-relaxation bound with a penalty (Brown, Smith & Sun 2010) for the model in main_v2.tex.

Penalty. Let W_t be the orders booked in period t (independent across periods, unknown before t).
For every vehicle movement that enters an arc in period s and reaches node (u, t''), and for every
vehicle waiting at u during period s (reaching node (u, s+1)), the hindsight problem pays

    C = sum_{t > s} [ G(u, t''; W_t) - E G(u, t''; W_t) ],

where G(u, t''; W_t) = sum over orders o booked in t of gamma[(dep_o, t)][(u, t'')] * n_o / mean n, and
gamma[(d, t)][(u, t'')] is the change in the fluid dual price of a vehicle at node (u, t'') when one more
order booked in period t at hub d is expected (a finite difference of the fluid LP of Section 6.3).
Along the trajectory of any non-anticipative policy the flow on an arc entered in period s is decided
by period s, and the bookings of periods t > s are independent of everything known by s, so the
expected penalty is zero; the penalized hindsight value is therefore an upper bound on the expected
profit of every such policy, for every scale theta >= 0 of gamma. theta = 0 gives the unpenalized bound.
theta is chosen on training scenarios and the bound is reported on the test scenarios.

Run from the DRL_CO repo root:
    python -m model.penalized_bound
"""
import json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .mt_prototype import Net, Sim, run, load_case
from .bounds_check import bound
from .online_lookahead import fluid_values, parse, LEVELS, TEST, TRAIN, T, C_E, MEAN_N, CHOSEN

THETAS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]


def gamma_table(net, hubs, ops):
    """gamma[(dep, t)] = {(u, t''): dual change per extra expected order booked in t at dep}."""
    sim = Sim(net, hubs, T)
    base = fluid_values(sim, ops, from_period=0)
    table = {}
    for dep in sorted(net.H):
        for t in range(1, T):
            pert = fluid_values(sim, ops, extra={(dep, t): 1.0}, from_period=0)
            table[dep, t] = {key: pert[key] - base[key] for key in base}
    return table


def penalties_for(orders, net, ops, table, theta):
    """Arc and waiting penalties for one realization of the order stream."""
    hubs = sorted(net.H)
    lam = ops / len(hubs)                                   # expected bookings per hub and period
    # innovation of the demand booked in period t, mapped onto nodes: realized minus expected
    innov = {}                                              # (t) -> {(u, t''): value}
    booked = {}
    for o in orders.values():
        booked.setdefault(o.book_time, []).append(o)
    for t in range(1, T):
        acc = {}
        for o in booked.get(t, []):
            for key, g in table[o.departure, t].items():
                acc[key] = acc.get(key, 0.0) + g * o.passenger / MEAN_N
        for dep in hubs:
            for key, g in table[dep, t].items():
                acc[key] = acc.get(key, 0.0) - lam * g
        innov[t] = acc
    # cumulative innovation of all bookings after period s, per node: S[s][(u, t'')] = sum_{t > s} innov[t]
    tail = {}
    running = {}
    for s in range(T - 1, -1, -1):
        for key, val in innov.get(s + 1, {}).items():
            running[key] = running.get(key, 0.0) + val
        tail[s] = dict(running)
    pen = {}
    arcs = [(u, v) for u in hubs for v in net.H.neighbors(u)]
    for s in range(T):
        for (u, v) in arcs:
            c = tail[s].get((v, s + net.lt(u, v)), 0.0)
            if c:
                pen["arc", (u, v), s] = theta * c
        for u in hubs:
            c = tail[s].get((u, s + 1), 0.0)
            if c:
                pen["wait", u, s] = theta * c
    return pen


def task(args):
    level, seed, thetas = args
    nv, ops = LEVELS[level]
    net, hubs, orders = load_case(seed, nv, ops, T)
    t0 = time.time()
    table = gamma_table(net, hubs, ops)
    items = [(o, 1) for o in orders.values()]
    vals = {}
    for theta in thetas:
        pen = penalties_for(orders, net, ops, table, theta) if theta > 0 else None
        vals[theta] = bound(net, hubs, items, T, C_E, relax=True, penalties=pen)[0]
    return level, seed, vals, time.time() - t0


if __name__ == "__main__":
    chosen = json.load(open(CHOSEN))
    tasks = [(lv, s, THETAS) for lv in LEVELS for s in TRAIN + TEST]
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(24, os.cpu_count() or 1)) as ex:
        for level, seed, vals, secs in ex.map(task, tasks, chunksize=1):
            res[level, seed] = vals
    print(f"{len(tasks)} scenarios in {time.time()-t0:.0f}s")
    for lv in LEVELS:
        train = {th: np.mean([res[lv, s][th] for s in TRAIN]) for th in THETAS}
        best = min(THETAS, key=lambda th: train[th])
        test = np.array([[res[lv, s][th] for th in THETAS] for s in TEST])
        print(f"== {lv}: training means by theta: " + ", ".join(f"{th}: {train[th]:.0f}" for th in THETAS) + f" -> theta*={best}")
        for i, th in enumerate(THETAS):
            print(f"   test theta={th:<5} bound={test[:, i].mean():9.0f}  (se {test[:, i].std(ddof=1)/np.sqrt(len(TEST)):.0f})")
        i0, ib = THETAS.index(0.0), THETAS.index(best)
        red = (test[:, i0] - test[:, ib])
        print(f"   penalized (theta*={best}) vs unpenalized: {red.mean():+.0f} +/- {1.96*red.std(ddof=1)/np.sqrt(len(TEST)):.0f}"
              f"  ({red.mean()/test[:, i0].mean():+.1%})")
    json.dump({f"{lv}|{s}": v for (lv, s), v in res.items()}, open(CHOSEN.replace("chosen", "penalized"), "w"))
