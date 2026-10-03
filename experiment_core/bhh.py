"""BHH v1: scalar steady theory and two-city time-expanded commodity MILP.

Local tours return to their own city. Direct HV and linehaul AV relocate.
Goods are divisible loads; a record is a commodity, NOT an indivisible order.
Finite cost is discrete-time fleet occupation + end-to-end load waiting.
It is deliberately not asserted identical to continuous steady unit cost.
"""

from __future__ import annotations

import math
from scipy.optimize import minimize_scalar
from .contracts import PlatformError


def inverse_f(theta, a, b):
    if theta <= 0:
        return 0.0
    if a == b == 0:
        return math.inf
    if a == 0:
        return (theta / b) ** 2
    if b == 0:
        return theta / a
    return (2 * theta / (math.sqrt(b * b + 4 * a * theta) + b)) ** 2


def city_parameters(p, city):
    return p | {
        key: p[key + "_by_city"][city]
        if p.get(key + "_by_city") is not None
        else p[key]
        for key in ("a", "b", "rho")
    }


def capacity(v, duration, p, city=0):
    p = city_parameters(p, city)
    duration *= p.get("period_duration", 1)
    if v == 0 or duration < 2 * p["rho"]:
        return 0.0
    return min(
        v * p["hv_capacity"], inverse_f(v * (duration - 2 * p["rho"]), p["a"], p["b"])
    )


def direct_capacity(v, duration, p, direction=0):
    origin, destination = (
        city_parameters(p, direction),
        city_parameters(p, 1 - direction),
    )
    duration *= p.get("period_duration", 1)
    tau = p["tau"] * p.get("period_duration", 1)
    if v == 0 or duration < tau + 2 * origin["rho"]:
        return 0.0
    lo, hi = 0.0, v * p["hv_capacity"]
    for _ in range(60):
        q = (lo + hi) / 2
        each = q / v
        used = (
            2 * origin["rho"]
            + tau
            + (origin["a"] * q + origin["b"] * math.sqrt(q)) / v
            + destination["a"] * each
            + destination["b"] * math.sqrt(each)
        )
        if used <= duration:
            lo = q
        else:
            hi = q
    return lo


def steady(p):
    if any(p.get(key + "_by_city") is not None for key in ("a", "b", "rho")):
        raise PlatformError(
            "STEADY_SYMMETRY", "steady v1 requires symmetric city parameters"
        )
    c, ca, w, rho, lam, tau = [
        p[k]
        for k in ("hv_cost", "av_cost", "waiting_cost", "rho", "demand_rate", "tau")
    ]
    M, MA = p["hv_capacity"], p["av_capacity"]

    def s(W):
        return p["a"] + p["b"] / math.sqrt(W)

    def hub(W, n1=None, n2=None):
        cap = min(M, W)
        n1 = (
            min(cap, max(1, math.sqrt(2 * c * rho / (w * s(W)))))
            if n1 is None and s(W)
            else (cap if n1 is None else n1)
        )
        n2 = (
            min(cap, max(1, math.sqrt(4 * c * rho / (w * s(W)))))
            if n2 is None and s(W)
            else (cap if n2 is None else n2)
        )
        cost = c * (2 * s(W) + 2 * rho / n1 + 2 * rho / n2) + ca * tau / min(W, MA)
        cost += (
            w * (W / (2 * lam) + 3 * rho + s(W) * n1 + tau + s(W) * n2 / 2)
            + p["resort_cost"]
        )
        return cost, n1, n2

    def direct(W, n):
        return c * (s(W) + p["a"] + p["b"] / math.sqrt(n) + (2 * rho + tau) / n) + w * (
            W / (2 * lam)
            + 2 * rho
            + s(W) * n
            + tau
            + (p["a"] * n + p["b"] * math.sqrt(n)) / 2
        )

    def nopt(W):
        if min(M, W) <= 1:
            return direct(W, 1), 1
        result = minimize_scalar(
            lambda n: direct(W, n), bounds=(1, min(M, W)), method="bounded"
        )
        options = [
            (result.fun, result.x),
            (direct(W, 1), 1),
            (direct(W, min(M, W)), min(M, W)),
        ]
        return min(options)

    def search(fn):
        maximum = max(1.00001, p["wave_max"])
        result = minimize_scalar(fn, bounds=(1, maximum), method="bounded")
        return min([(fn(1), 1), (result.fun, result.x), (fn(maximum), maximum)])

    gz, Wz = search(lambda W: hub(W)[0])
    gd, Wd = search(lambda W: nopt(W)[0])
    _, n1, n2 = hub(Wz)
    _, nd = nopt(Wd)
    linehaul = (c / nd - ca / min(Wd, MA)) * tau
    resort = (c / nd + w / 2) * (
        p["b"] * math.sqrt(nd) * (1 - math.sqrt(nd / Wd)) - 2 * rho
    )
    decouple = hub(Wd, nd, nd)[0] - gz
    # Integer policies are re-optimized on integer W, not rounded independently.
    rounding = []
    for W in range(1, min(10000, math.ceil(p["wave_max"])) + 1):
        ns = range(1, min(M, W) + 1)
        rounding.append(min((hub(W, i, j)[0], W, i, j) for i in ns for j in ns))
    integer = min(rounding)
    roundedW = max(1, round(Wz))
    rounded1 = max(1, min(M, roundedW, round(n1)))
    rounded2 = max(1, min(M, roundedW, round(n2)))
    return {
        "business_cost": gz,
        "direct_cost": gd,
        "net_value": gd - gz,
        "hub": {
            "wave": Wz,
            "collection_load": n1,
            "distribution_load": n2,
            "hv_per_city": lam * (2 * s(Wz) + 2 * rho / n1 + 2 * rho / n2),
            "av_total": 2 * lam * tau / min(Wz, MA),
        },
        "direct": {"wave": Wd, "load": nd},
        "decomposition": {
            "linehaul": linehaul,
            "resorting": resort,
            "decoupling": decouple,
            "overhead": p["resort_cost"],
            "identity_residual": linehaul
            + resort
            + decouple
            - p["resort_cost"]
            - (gd - gz),
        },
        "integer": {
            "cost": integer[0],
            "wave": integer[1],
            "collection_load": integer[2],
            "distribution_load": integer[3],
            "loss": integer[0] - gz,
        },
        "ordinary_rounding": {
            "cost": hub(roundedW, rounded1, rounded2)[0],
            "wave": roundedW,
            "collection_load": rounded1,
            "distribution_load": rounded2,
            "loss": hub(roundedW, rounded1, rounded2)[0] - gz,
        },
        "search_upper_bound": p["wave_max"],
        "boundary_optimum": bool(Wz > p["wave_max"] * 0.999),
        "assumptions": "balanced stationary two-city; continuous divisible loads; unconstrained fleets",
    }


def finite(
    scenario,
    p,
    solver,
    orders=None,
    fixed=None,
    frozen_before=0,
    start_until=None,
    complete_until=None,
):
    import gurobipy as gp
    from gurobipy import GRB

    orders = scenario["orders"] if orders is None else orders
    fixed = fixed or {}
    end = max([scenario["horizon"]] + [int(math.floor(o["end_time"])) for o in orders])
    end = min(end, complete_until) if complete_until is not None else end
    starts = range(min(end, start_until) if start_until is not None else end)
    m = gp.Model("bhh_v1")
    m.Params.OutputFlag = 0
    for key, val in solver.items():
        m.setParam(
            {
                "time_limit_seconds": "TimeLimit",
                "threads": "Threads",
                "mip_gap": "MIPGap",
                "seed": "Seed",
            }[key],
            val,
        )
    variables, wave, direct, av, cargo = {}, {}, {}, {}, {}

    def var(key, start=None, vtype=GRB.CONTINUOUS, ub=GRB.INFINITY):
        # All prior-start variables are frozen, including zeros. In-flight arcs
        # remain in every global resource/commodity balance; no fleet reset.
        pinned = (
            fixed.get(key, 0) if start is not None and start < frozen_before else None
        )
        v = m.addVar(
            lb=pinned if pinned is not None else 0,
            ub=pinned if pinned is not None else ub,
            vtype=vtype,
        )
        variables[key] = v
        return v

    for direction in (0, 1):
        for i in starts:
            for j in range(i + 1, end + 1):
                for v in range(1, sum(p["hv_fleet"]) + 1):
                    if p["mode"] != "direct":
                        for kind in ("C", "D"):
                            city = direction if kind == "C" else 1 - direction
                            local = city_parameters(p, city)
                            max_duration = math.ceil(
                                (
                                    2 * local["rho"]
                                    + local["a"] * p["hv_capacity"]
                                    + local["b"] * math.sqrt(p["hv_capacity"])
                                )
                                / p["period_duration"]
                            )
                            if (
                                j - i <= max_duration
                                and capacity(v, j - i, p, city) > 1e-8
                            ):
                                wave[kind, direction, i, j, v] = var(
                                    f"{kind}:{direction}:{i}:{j}:{v}", i, GRB.BINARY
                                )
                    direct_max = p["tau"] + math.ceil(
                        (
                            2 * max(city_parameters(p, k)["rho"] for k in (0, 1))
                            + sum(
                                city_parameters(p, k)["a"] * p["hv_capacity"]
                                + city_parameters(p, k)["b"]
                                * math.sqrt(p["hv_capacity"])
                                for k in (0, 1)
                            )
                        )
                        / p["period_duration"]
                    )
                    if (
                        p["mode"] != "hub"
                        and p["tau"] <= j - i <= direct_max
                        and (
                            j - i == p["tau"]
                            or direct_capacity(v, j - i, p, direction) > 1e-8
                        )
                    ):
                        direct[direction, i, j, v] = var(
                            f"H:{direction}:{i}:{j}:{v}", i, GRB.BINARY
                        )
            if i + p["tau"] <= end and p["mode"] != "direct":
                av[direction, i, i + p["tau"]] = var(
                    f"A:{direction}:{i}:{i + p['tau']}",
                    i,
                    GRB.INTEGER,
                    sum(p["av_fleet"]),
                )
    # At most one alternative v per wave on the same interval. Concurrent
    # waves are allowed only insofar as they fit the physical shared fleet.
    for kind in ("C", "D"):
        for direction in (0, 1):
            for i in starts:
                for j in range(i + 1, end + 1):
                    m.addConstr(
                        gp.quicksum(
                            x
                            for (k, d, a, b, v), x in wave.items()
                            if (k, d, a, b) == (kind, direction, i, j)
                        )
                        <= 1
                    )
    for direction in (0, 1):
        for i in starts:
            for j in range(i + 1, end + 1):
                m.addConstr(
                    gp.quicksum(
                        x
                        for (d, a, b, v), x in direct.items()
                        if (d, a, b) == (direction, i, j)
                    )
                    <= 1
                )
    unserved, obj = {}, []
    arcs = [("C", d, i, j) for k, d, i, j, v in wave if k == "C"] + [
        ("D", d, i, j) for k, d, i, j, v in wave if k == "D"
    ]
    arcs += [("A", d, i, j) for d, i, j in av] + [
        ("H", d, i, j) for d, i, j, v in direct
    ]
    arcs = sorted(set(arcs))
    for o in orders:
        oid, direction, load = o["id"], int(o["departure"]), o["passenger"]
        unserved[oid] = var(f"Z:{oid}", ub=load)
        obj.append(o["penalty"] * unserved[oid])
        local = {}
        for kind, d, i, j in arcs:
            if d != direction or i < o["start_time"] or j > o["end_time"]:
                continue
            x = var(f"g:{oid}:{kind}:{d}:{i}:{j}", i, ub=load)
            cargo[oid, kind, d, i, j] = x
            local[kind, i, j] = x
            if kind in ("D", "H"):
                obj.append(
                    p["finite_waiting_cost"]
                    * p["period_duration"]
                    * (j - o["book_time"])
                    * x
                )
        totals = {
            k: gp.quicksum(x for (kind, i, j), x in local.items() if kind == k)
            for k in ("C", "A", "D", "H")
        }
        m.addConstr(totals["C"] == totals["A"])
        m.addConstr(totals["A"] == totals["D"])
        m.addConstr(totals["D"] + totals["H"] + unserved[oid] == load)
        for t in range(end + 1):
            m.addConstr(
                gp.quicksum(x for (k, i, j), x in local.items() if k == "A" and i <= t)
                <= gp.quicksum(
                    x for (k, i, j), x in local.items() if k == "C" and j <= t
                )
            )
            m.addConstr(
                gp.quicksum(x for (k, i, j), x in local.items() if k == "D" and i <= t)
                <= gp.quicksum(
                    x for (k, i, j), x in local.items() if k == "A" and j <= t
                )
            )
    for kind, d, i, j in arcs:
        flow = gp.quicksum(
            x
            for (oid, k, dd, a, b), x in cargo.items()
            if (k, dd, a, b) == (kind, d, i, j)
        )
        if kind in ("C", "D"):
            cap = gp.quicksum(
                capacity(v, j - i, p, d if kind == "C" else 1 - d) * x
                for (k, dd, a, b, v), x in wave.items()
                if (k, dd, a, b) == (kind, d, i, j)
            )
        elif kind == "H":
            cap = gp.quicksum(
                (0 if j - i == p["tau"] else direct_capacity(v, j - i, p, d)) * x
                for (dd, a, b, v), x in direct.items()
                if (dd, a, b) == (d, i, j)
            )
        else:
            cap = p["av_capacity"] * av[d, i, j]
        m.addConstr(flow <= cap)
    for city in (0, 1):
        for t in range(end):
            local = gp.quicksum(
                v * x
                for (k, d, i, j, v), x in wave.items()
                if i <= t < j
                and ((k == "C" and d == city) or (k == "D" and 1 - d == city))
            )
            hv_out = gp.quicksum(
                v * x for (d, i, j, v), x in direct.items() if d == city and i <= t
            )
            hv_in = gp.quicksum(
                v * x for (d, i, j, v), x in direct.items() if 1 - d == city and j <= t
            )
            m.addConstr(local + hv_out - hv_in <= p["hv_fleet"][city])
            av_out = gp.quicksum(
                x for (d, i, j), x in av.items() if d == city and i <= t
            )
            av_in = gp.quicksum(
                x for (d, i, j), x in av.items() if 1 - d == city and j <= t
            )
            m.addConstr(av_out - av_in <= p["av_fleet"][city])
    hvcost = gp.quicksum(
        p["hv_cost"] * v * (j - i) * x for (k, d, i, j, v), x in wave.items()
    ) + gp.quicksum(
        p["hv_cost"] * v * (j - i) * x for (d, i, j, v), x in direct.items()
    )
    avcost = gp.quicksum(p["av_cost"] * (j - i) * x for (d, i, j), x in av.items())
    hvcost *= p["period_duration"]
    avcost *= p["period_duration"]
    m.setObjective(hvcost + avcost + gp.quicksum(obj), GRB.MINIMIZE)
    m.optimize()
    status = (
        "OPTIMAL"
        if m.Status == GRB.OPTIMAL
        else "INFEASIBLE"
        if m.Status == GRB.INFEASIBLE
        else "FEASIBLE_LIMIT"
        if m.SolCount
        else "NO_SOLUTION_LIMIT"
    )
    if not m.SolCount:
        m.dispose()
        raise PlatformError(status, "BHH solve has no incumbent", exit_code=4)
    solution = {k: float(v.X) for k, v in variables.items()}
    report = {
        "status": status,
        "business_cost": m.ObjVal,
        "bound": m.ObjBound if math.isfinite(m.ObjBound) else None,
        "gap": m.MIPGap,
        "runtime": m.Runtime,
        "hv_cost": hvcost.getValue(),
        "av_cost": avcost.getValue(),
        "delivered_load": sum(
            o["passenger"] - solution[f"Z:{o['id']}"] for o in orders
        ),
        "unserved_load": sum(solution[f"Z:{o['id']}"] for o in orders),
        "scope": "window",
        "objective_version": "bhh-finite-v1",
        "waiting_cost_included": p["finite_waiting_cost"] > 0,
    }
    m.dispose()
    return solution, report


def rolling(scenario, p, solver, controller, checkpoint):
    fixed, reports, frozen_before = {}, [], 0
    total_end = max(
        [scenario["horizon"]] + [math.floor(o["end_time"]) for o in scenario["orders"]]
    )
    last = {}
    for t in range(0, total_end, controller["commit_periods"]):
        checkpoint()
        known = [o for o in scenario["orders"] if o["book_time"] <= t]
        H, E = controller["planning_horizon"], controller["completion_extension"]
        # Exact commitments crossing a window edge must remain representable.
        committed_end = max(
            [0]
            + [
                int(k.split(":")[-2] if k[0] in "CDH" else k.split(":")[-1])
                for k, v in fixed.items()
                if v > 1e-8 and k[0] in "CDHA"
            ]
        )
        completion = max(t + H + E, committed_end)
        last, report = finite(
            scenario, p, solver, known, fixed, frozen_before, t + H, completion
        )
        until = min(t + controller["commit_periods"], total_end)
        for key, value in last.items():
            parts = key.split(":")
            if parts[0] in ("C", "D", "H", "A"):
                start = int(parts[2])
            elif parts[0] == "g":
                start = int(parts[-2])
            else:
                continue
            if start < until:
                fixed[key] = value
        frozen_before = until
        reports.append(
            {
                "period": t,
                "commit_until": until,
                "plan": {k: v for k, v in last.items() if v > 1e-8},
                **report,
            }
        )
    # Reconcile final state from committed deliveries, not the last planned z.
    delivered = sum(
        v
        for k, v in fixed.items()
        if k.startswith("g:") and k.split(":")[-4] in ("D", "H")
    )
    total = sum(o["passenger"] for o in scenario["orders"])
    costs = sum(
        (
            p["av_cost"] * (int(k.split(":")[3]) - int(k.split(":")[2]))
            if k.startswith("A:")
            else p["hv_cost"]
            * int(k.split(":")[4])
            * (int(k.split(":")[3]) - int(k.split(":")[2]))
        )
        * v
        for k, v in fixed.items()
        if k[0] in "CDHA"
    )
    by_id = {o["id"]: o for o in scenario["orders"]}
    costs *= p["period_duration"]
    for oid, o in by_id.items():
        served = sum(
            v
            for k, v in fixed.items()
            if k.startswith(f"g:{oid}:") and k.split(":")[-4] in ("D", "H")
        )
        costs += o["penalty"] * (o["passenger"] - served)
        costs += sum(
            p["finite_waiting_cost"]
            * p["period_duration"]
            * (int(k.split(":")[-1]) - o["book_time"])
            * v
            for k, v in fixed.items()
            if k.startswith(f"g:{oid}:") and k.split(":")[-4] in ("D", "H")
        )
    return (
        fixed,
        {
            "business_cost": costs,
            "delivered_load": delivered,
            "unserved_load": max(0, total - delivered),
            "global_gap": None,
            "global_gap_reason": "window gaps are not a full-run global gap",
        },
        reports,
    )
