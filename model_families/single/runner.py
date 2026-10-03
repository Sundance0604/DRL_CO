from __future__ import annotations
import copy
import hashlib
import math
import random
import time
import numpy as np
from experiment_core.contracts import PlatformError
from experiment_core.storage import atomic_json, digest, read_json, run_dir
from .data import matching_state

FEATURE_SCHEMA = "hub-time-value/v1"
FEATURES = [
    "intercept",
    "remaining_time",
    "degree",
    "mean_travel",
    "idle_supply",
    "backlog_load",
    "mean_radius",
]


def features(sim, hub, when):
    return [
        1,
        (sim.T - when) / max(1, sim.T),
        sim.net.H.degree(hub) / max(1, len(sim.net.H) - 1),
        np.mean([sim.net.tau[hub, j] for j in sim.net.H]) / max(1, sim.T),
        sum(v.phase == "idle" and v.loc == hub for v in sim.vehs)
        / max(1, len(sim.vehs)),
        sum(o.passenger for o in sim.pool.values() if o.departure == hub)
        / max(1, len(sim.vehs) * sim.parameters["capacity"]),
        np.mean(list(sim.net.radii.values())),
    ]


def state_snapshot(sim):
    return {
        "period": sim.t,
        "profit": sim.J,
        "pool": [vars(o) for o in sim.pool.values()],
        "vehicles": [
            dict(
                id=str(v.id),
                hub=v.loc,
                phase=v.phase,
                route=list(v.route),
                dead=v.dead,
                arrive=v.arrive,
                hold=v.hold,
                onboard=sorted(v.onboard),
                assigned_orders=sorted(v.orders),
            )
            for v in sim.vehs
        ],
        "counts": dict(sim.S),
    }


def serialize_plan(plan, period, before_hash, info):
    profit, obj, sol, routes, expired, ndec = plan
    return {
        "schema_version": "decision-plan/v1",
        "period": period,
        "state_hash": before_hash,
        "operating_profit": profit,
        "augmented_solver_objective": obj,
        "assignments": [
            {"order_id": str(o), "vehicle_id": str(v)} for o, v in sol["x"]
        ],
        "routes": [
            dict(
                vehicle_id=str(k),
                **{key: value for key, value in routes[k][r].items() if key != "serve"},
            )
            for k, r in sol["y"]
        ],
        "repositions": [{"vehicle_id": str(k), "hub": j} for k, j in sol["z"]],
        "expired": sorted(expired),
        "solver": info,
        "accounting": info.get("accounting", {}),
    }


def matching(spec, scenario, seed, checkpoint, stop_period=None, training=None):
    from .engine.online_lookahead import fluid_values
    from .engine.demand import sample_batches, sample_orders

    sim, all_orders = matching_state(
        scenario, spec.model.parameters, spec.solver.parameters.model_dump()
    )
    by_t = {}
    for o in all_orders.values():
        by_t.setdefault(o.book_time, []).append(o)
    params, vf = spec.model.parameters, spec.value_function
    cache, rng, trace, totals = (
        {},
        random.Random(seed),
        [],
        {
            "augmented_solver_objective": 0,
            "planned_loaded_distance": 0,
            "planned_deadhead_distance": 0,
            "planned_departure_delay": 0,
        },
    )
    if vf.id == "learned_hub_time":
        weights = read_json(run_dir(vf.parameters["checkpoint_run"]) / "weights.json")
        source = run_dir(vf.parameters["checkpoint_run"])
        meta = read_json(source / "checkpoint.json")
        if (
            hashlib.sha256((source / "weights.json").read_bytes()).hexdigest()
            != meta["weights_hash"]
        ):
            raise PlatformError("CHECKPOINT_HASH", "checkpoint content changed")
        if weights["feature_names"] != FEATURES:
            raise PlatformError("FEATURE_SCHEMA", "checkpoint feature order mismatch")

    def solve(c, cross=None, info=None):
        value = None
        if vf.id == "fluid_dual":
            if vf.parameters["recompute"] or "V" not in cache:
                cache["V"] = fluid_values(c, vf.parameters["expected_orders_per_step"])

            def value(when, hub):
                return vf.parameters["multiplier"] * cache["V"].get((hub, when), 0)

        if vf.id == "learned_hub_time":

            def value(when, hub):
                return vf.parameters["multiplier"] * float(
                    np.dot(features(c, hub, when), weights["weights"])
                )

        return c.solve_plan(
            params["cross_hub"] if cross is None else cross,
            params["empty_cost"],
            params["reposition"],
            value,
            info,
        )

    for t in range(scenario["horizon"]):
        checkpoint()
        sim.begin_period(by_t.get(t, []))
        before = state_snapshot(sim)
        if training is not None:
            V = fluid_values(
                sim,
                scenario.get(
                    "orders_per_step", len(scenario["orders"]) / scenario["horizon"]
                ),
            )
            for (hub, when), value in V.items():
                if when <= sim.T:
                    training.append((features(sim, hub, when), value))
        info = {}
        if spec.controller.id == "rollout":
            cp = spec.controller.parameters
            futures = []
            sampler = (
                sample_batches
                if any(hasattr(o, "cutoff") for o in all_orders.values())
                else sample_orders
            )
            for k in range(cp["samples"]):
                futures.append(
                    sampler(
                        sim.net,
                        sim.net.radii,
                        rng,
                        sim.T,
                        cp["expected_orders_per_step"],
                        after=t,
                        first_id=1_000_000 + k * 100_000,
                    )
                )
            # Conditional batch collection after cutoff is approximated by the
            # existing sampler; the limitation is recorded in every run.
            for fut in futures:
                for group in fut.values():
                    for o in group:
                        o.id = f"sample-{o.id}"
            candidates = []
            for cross in cp["candidate_cross_hub"]:
                scores = []
                first_info = {}
                first = solve(sim, cross, first_info)
                for fut in futures:
                    checkpoint()
                    clone = sim.clone()
                    clone.commit_plan(first)
                    for u in range(clone.t, clone.T):
                        clone.begin_period(fut.get(u, []))
                        clone.commit_plan(solve(clone, params["cross_hub"]))
                    scores.append(clone.J)
                candidates.append((float(np.mean(scores)), cross, first, first_info))
            _, cross, plan, info = max(candidates, key=lambda x: x[0])
            info = info | {
                "controller": "rollout",
                "cross_hub": cross,
                "candidate_scores": [
                    {"cross_hub": c, "score": v} for v, c, _, _ in candidates
                ],
            }
        else:
            plan = solve(sim, info=info)
        serialized = serialize_plan(plan, t, digest(before), info)
        if stop_period == t:
            return {
                "before": before,
                "plan": serialized,
                "committed": False,
                "network": {"nodes": scenario["nodes"], "edges": scenario["edges"]},
            }
        if digest(state_snapshot(sim)) != serialized["state_hash"]:
            raise PlatformError(
                "STALE_PLAN", "state changed before commit", exit_code=7
            )
        sim.commit_plan(plan)
        totals["augmented_solver_objective"] += plan[1]
        for r in serialized["routes"]:
            if r["new"]:
                previous = before["vehicles"][int(r["vehicle_id"])]
                start = previous["route"][-1] if previous["phase"] == "trip" else r["p"]
                totals["planned_loaded_distance"] += sim.net.d[start, r["e"]]
                if previous["phase"] == "idle":
                    totals["planned_deadhead_distance"] += sim.net.d[
                        previous["hub"], r["p"]
                    ]
                totals["planned_departure_delay"] += r["delay"]
        for r in serialized["repositions"]:
            v = before["vehicles"][int(r["vehicle_id"])]
            totals["planned_deadhead_distance"] += sim.net.d[v["hub"], r["hub"]]
        trace.append(make_step(t, before, serialized, state_snapshot(sim), trace, all_orders, by_t.get(t, []), params))
    if stop_period is not None:
        raise PlatformError("PERIOD_RANGE", "requested period outside scenario horizon")
    pending_unassigned = len(sim.pool)
    if spec.evaluation.terminal_policy == "drain_committed":
        sim.pool = {}
        cutoff = math.ceil(max([sim.T] + [o.end_time for o in all_orders.values()])) + 1
        while any(v.phase != "idle" for v in sim.vehs) and sim.t <= cutoff:
            checkpoint()
            sim.begin_period([])
            before = state_snapshot(sim)
            info = {}
            plan = solve(sim, info=info)
            sim.commit_plan(plan)
            totals["augmented_solver_objective"] += plan[1]
            trace.append(make_step(sim.t - 1, before, serialize_plan(plan, sim.t - 1, digest(before), info),
                                   state_snapshot(sim), trace, all_orders, [], params) | {"drain": True})
    pending_committed = sum(len(v.orders) for v in sim.vehs)
    assigned_ids = {a["order_id"] for row in trace for a in row["plan"]["assignments"]}
    outstanding = {oid for v in sim.vehs for oid in v.orders}
    total_load = sum(o.passenger for o in all_orders.values())
    metrics = dict(
        sim.S,
        operating_profit=sim.J,
        **totals,
        arrivals=len(all_orders),
        pending=pending_unassigned + pending_committed,
        pending_unassigned=pending_unassigned,
        pending_committed=pending_committed,
        assigned_order_rate=sim.S["assigned"] / len(all_orders) if all_orders else None,
        delivered_order_rate=sim.S["delivered"] / len(all_orders)
        if all_orders
        else None,
        assigned_load_rate=sum(all_orders[o].passenger for o in assigned_ids)
        / total_load
        if total_load
        else None,
        delivered_load_rate=sum(
            all_orders[o].passenger for o in assigned_ids - outstanding
        )
        / total_load
        if total_load
        else None,
        business_cost=sum(row["plan"]["accounting"].get("business_cost", 0) for row in trace),
        business_cost_reason="assignment-time planned route costs plus periodic penalties; not measured actual expenditure",
        actual_distance=None,
        actual_distance_reason="route distances are committed plans; final in-flight distance is not reconstructed",
    )
    return metrics, trace


def make_step(t, before, plan, after, history, orders, arrivals, parameters):
    from .engine.demand import ready_time
    previous = history[-1]["after"]["counts"] if history else {}
    visible = []
    for item in before["pool"]:
        observation = dict(item)
        observation["start_time"] = ready_time(orders[item["id"]], t)
        observation["ready_time_information"] = "upper-bound" if item.get("cutoff", t) > t else "revealed"
        visible.append(observation)
    cumulative_cost = sum(x["plan"]["accounting"].get("business_cost", 0) for x in history) + plan["accounting"].get("business_cost", 0)
    measurements = {
        "period_profit": plan["operating_profit"], "cumulative_profit": after["profit"],
        "period_objective": plan["augmented_solver_objective"],
        "cumulative_objective": sum(x["plan"]["augmented_solver_objective"] for x in history) + plan["augmented_solver_objective"],
        "period_business_cost": plan["accounting"].get("business_cost"), "cumulative_business_cost": cumulative_cost,
        "arrivals": len(arrivals), "pool_before": len(before["pool"]), "pool_after": len(after["pool"]),
        "assigned_delta": after["counts"].get("assigned", 0) - previous.get("assigned", 0),
        "delivered_delta": after["counts"].get("delivered", 0) - previous.get("delivered", 0),
        "cancelled_delta": after["counts"].get("cancelled", 0) - previous.get("cancelled", 0),
        "assigned_load": sum(orders[x["order_id"]].passenger for x in plan["assignments"]),
        "delivered_load": sum(orders[o].passenger for o in {
            oid for vehicle in (history[-1]["after"]["vehicles"] if history else []) for oid in vehicle["assigned_orders"]
        } - {oid for vehicle in after["vehicles"] for oid in vehicle["assigned_orders"]}),
        "idle_vehicles": sum(v["phase"] == "idle" for v in after["vehicles"]),
        "trip_vehicles": sum(v["phase"] == "trip" for v in after["vehicles"]),
        "reposition_vehicles": sum(v["phase"] == "repos" for v in after["vehicles"]),
        "onboard_load": sum(orders[o].passenger for v in after["vehicles"] for o in v["onboard"]),
    }
    invariants = {
        "state_hash_matches": digest(before) == plan["state_hash"], "clock_advanced_once": after["period"] == t+1,
        "unique_assignment": len({x["order_id"] for x in plan["assignments"]}) == len(plan["assignments"]),
        "capacity_valid": all(sum(orders[o].passenger for o in v["onboard"]) <= parameters["capacity"] for v in after["vehicles"]),
        "profit_reconciles": abs(plan["accounting"].get("profit_identity_residual", 0)) < 1e-5,
    }
    if not all(invariants.values()):
        raise PlatformError("TRACE_INVARIANT", "committed step failed accounting/state checks", exit_code=7)
    return {"schema_version":"single-trace-step/v2", "period":t, "before":before,
            "observation":{"period":t,"orders":visible,"vehicles":before["vehicles"]},
            "arrivals":[vars(o) for o in arrivals], "plan":plan, "after":after,
            "measurements":measurements, "invariants":invariants, "committed":True}


def execute(spec, data, scenarios, target, policy_seed, emit, cancelled):
    start = time.monotonic()
    random.seed(policy_seed)
    np.random.seed(policy_seed)

    def check():
        if cancelled():
            raise PlatformError("CANCELLED", "cancellation requested", exit_code=6)
        if time.monotonic() - start > spec.execution.timeout_seconds:
            raise PlatformError("TIMEOUT", "run timeout exceeded", exit_code=5)

    rows, traces, training = [], [], []
    component_training_seed = (read_json(run_dir(spec.value_function.parameters["checkpoint_run"])/"checkpoint.json").get("training_seed")
                               if spec.value_function.id == "learned_hub_time" else None)
    for i, scenario in enumerate(scenarios):
        check()
        emit(
            "scenario_started", scenario_id=scenario["id"], progress=i / len(scenarios)
        )
        if spec.controller.id in ("oracle_lp", "oracle_mip"):
            from .engine.bounds_check import bound

            sim, orders = matching_state(
                scenario, spec.model.parameters, spec.solver.parameters.model_dump()
            )
            info = {}
            incumbent, upper = bound(
                sim.net,
                scenario["hubs"],
                [(o, 1) for o in orders.values()],
                sim.T,
                spec.model.parameters["empty_cost"],
                relax=spec.controller.id == "oracle_lp",
                parameters=spec.model.parameters,
                solver_parameters=spec.solver.parameters.model_dump(),
                solve_info=info,
            )
            metrics = {
                "upper_bound": upper,
                "bound_incumbent": incumbent,
                "bound_direction": "maximize-upper",
                "bound_scope": "full horizon perfect-information pooled time-expanded network; relaxes indivisible vehicle assignment/backbone/anchors",
                "operating_profit": None,
                "operating_profit_reason": "a relaxation is not an executed policy",
                **info,
            }
            trace = []
        else:
            metrics, trace = matching(
                spec,
                scenario,
                policy_seed,
                check,
                training=training if spec.controller.id == "train_value" else None,
            )
        rows.append(
            {
                "scenario_id": scenario["id"],
                "policy_seed": None if spec.controller.id.startswith("train_") else policy_seed,
                "training_seed": policy_seed if spec.controller.id.startswith("train_") else component_training_seed,
                "scenario_seed": scenario.get("seed"),
                "family": spec.family,
                "framework_id": spec.framework_id,
                "metrics": metrics,
            }
        )
        if spec.execution.save_trace:
            traces.append(
                {
                    "scenario_id": scenario["id"],
                    "network": {"nodes": scenario["nodes"], "edges": scenario["edges"]},
                    "steps": trace,
                }
            )
        emit(
            "scenario_completed",
            scenario_id=scenario["id"],
            progress=(i + 1) / len(scenarios),
            metrics=metrics,
        )
    if spec.controller.id == "train_value":
        if not training:
            raise PlatformError(
                "NO_TRAINING_SAMPLES", "no fluid value labels available"
            )
        X = np.asarray([x for x, y in training])
        y = np.asarray([y for x, y in training])
        weights = np.zeros(X.shape[1])
        rate = spec.controller.parameters["learning_rate"] / (
            1 + float(np.linalg.norm(X, ord=2) ** 2 / len(X))
        )
        for epoch in range(spec.controller.parameters["epochs"]):
            check()
            error = X @ weights - y
            weights -= rate * (X.T @ error / len(X) + 1e-4 * weights)
            emit(
                "training",
                epoch=epoch,
                value_mse=float(np.mean(error**2)),
                method="supervised fluid-dual regression; not reinforcement learning",
            )
        atomic_json(
            target / "weights.json",
            {
                "feature_schema": FEATURE_SCHEMA,
                "feature_names": FEATURES,
                "weights": weights.tolist(),
            },
        )
        atomic_json(
            target / "checkpoint.json",
            {
                "feature_schema": FEATURE_SCHEMA,
                "weights_hash": hashlib.sha256(
                    (target / "weights.json").read_bytes()
                ).hexdigest(),
                "dataset_hash": spec.dataset.revision,
                "train_scenario_ids": [s["id"] for s in scenarios],
                "all_scenario_ids": [s["id"] for s in data["scenarios"]],
                "model_parameters": spec.model.parameters,
                "training_seed": policy_seed,
                "training_method": "supervised-fluid-dual-v1",
            },
        )
    atomic_json(target / "trace.json", traces)
    from .metrics import DEFINITIONS

    atomic_json(target / "metric-definitions.json", DEFINITIONS)
    atomic_json(
        target / "charts.json",
        {
            "schema_version": "chart-artifact/v1",
            "type": "scenario-bars",
            "data_ref": "metrics.json",
            "encodings": {
                "x": "scenario_id",
                "y": "operating_profit",
            },
            "units": {"y": "currency"},
            "metric_defs": "metric-definitions.json",
            "javascript_allowed": False,
        },
    )
    atomic_json(
        target / "metrics.json",
        {
            "schema_version": "run-metrics/v1",
            "rows": rows,
            "elapsed_seconds": time.monotonic() - start,
        },
    )
    return rows
