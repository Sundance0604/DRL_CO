from __future__ import annotations

import copy
import hashlib
import math
import random
import time
import numpy as np
from .contracts import PlatformError
from .datasets import matching_state
from .storage import atomic_json, digest, read_json, run_dir

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
    }


def matching(spec, scenario, seed, checkpoint, stop_period=None, training=None):
    from model.online_lookahead import fluid_values
    from model.demand import sample_batches, sample_orders

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
        trace.append(
            {
                "period": t,
                "before": before,
                "plan": serialized,
                "after": state_snapshot(sim),
                "committed": True,
            }
        )
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
            trace.append(
                {
                    "period": sim.t - 1,
                    "before": before,
                    "plan": serialize_plan(plan, sim.t - 1, digest(before), info),
                    "after": state_snapshot(sim),
                    "committed": True,
                    "drain": True,
                }
            )
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
        business_cost=None,
        business_cost_reason="legacy objective includes revenue and periodic penalties; separate cost breakdown not available",
        actual_distance=None,
        actual_distance_reason="route distances are committed plans; final in-flight distance is not reconstructed",
    )
    return metrics, trace


def legacy(spec, scenario, seed, checkpoint, agent=None, train=False):
    import networkx as nx
    from drl_co.domain.city_graph import CityGraph
    from drl_co.domain.vehicle import Vehicle
    from drl_co.domain.order import Order
    from drl_co.environment.dispatch import DispatchEnv
    from drl_co.simulation.tools import city_node_generator, city_update_without_drl
    from drl_co.rl.features import candidate_features
    from experiments.training.train_candidate import _supply_actions
    from experiments.training.train_fixed_id import solve_lower_layer

    mapping = {n: i for i, n in enumerate(scenario["nodes"])}
    graph = CityGraph.__new__(CityGraph)
    graph.num_nodes = len(mapping)
    graph.G = nx.Graph()
    graph.G.add_weighted_edges_from(
        [(mapping[a], mapping[b], w) for a, b, w in scenario["edges"]]
    )
    graph._calculate_shortest_paths()
    vehicles = {}
    for i, data in enumerate(scenario["legacy_vehicles"]):
        v = Vehicle.__new__(Vehicle)
        v.__dict__.update(copy.deepcopy(data))
        v.id = i
        vehicles[i] = v
    all_orders = {}
    for data in scenario["orders"]:
        o = Order.__new__(Order)
        o.__dict__.update(data)
        o.departure = mapping[o.departure]
        o.destination = mapping[o.destination]
        o.virtual_departure = o.departure
        all_orders[o.id] = o
    capacity = spec.model.parameters["capacity"]
    active = {}
    cities = city_node_generator(graph, active, vehicles, active)
    env = DispatchEnv(graph, vehicles, all_orders, cities, capacity)
    costs = np.tile(np.array([10, 1, 3, 10], dtype=float), (len(vehicles), 1))
    total = 0
    cancelled = 0
    trace = []
    for t in range(scenario["horizon"]):
        checkpoint()
        env.time = t
        active.update({o.id: o for o in all_orders.values() if o.book_time == t})
        city_update_without_drl(cities, vehicles, active, t)
        ids, states = candidate_features(
            cities, active, graph, t, capacity, scenario["horizon"], len(vehicles)
        )
        mask = env.get_mask(active)
        if ids:
            if agent is None:
                actions = _supply_actions(mask, cities, capacity)
                decisions = states
            else:
                actions, _, _, _, decisions = agent.take_action_candidates(
                    states,
                    mask,
                    explore=train
                    or spec.controller.parameters.get("stochastic", False),
                    greedy=not train
                    and not spec.controller.parameters.get("stochastic", False),
                )
            env.apply_actions(active, actions, strict=True)
        else:
            actions = []
            decisions = states
        references = {o.id: o for o in active.values()}
        info = {}
        objective, solved, ncancel = solve_lower_layer(
            graph,
            cities,
            vehicles,
            active,
            t,
            costs,
            spec.solver.parameters.model_dump(),
            info,
        )
        total += objective
        cancelled += ncancel
        if train and ids:
            city_update_without_drl(cities, vehicles, active, t)
            next_ids, next_states = candidate_features(
                cities,
                active,
                graph,
                t + 1,
                capacity,
                scenario["horizon"],
                len(vehicles),
            )
            next_mask = env.get_mask(active)
            rewards = {
                oid: (
                    references[oid].revenue
                    if references[oid].matched
                    else -references[oid].penalty
                )
                / 1000
                for oid in ids
            }
            agent.add_order_transitions(
                decisions,
                ids,
                mask,
                actions,
                rewards,
                next_states,
                next_ids,
                next_mask,
                done_global=t == scenario["horizon"] - 1,
            )
            agent.update_sac(1)
        trace.append(
            {
                "period": t,
                "objective": objective,
                "actions": actions,
                "order_ids": ids,
                "solver": info,
                "after": {
                    "vehicles": [
                        {
                            "id": str(v.id),
                            "hub": str(v.intercity),
                            "phase": "idle" if v.whether_city else "trip",
                            "orders": [str(o.id) for o in v.orders.values()],
                        }
                        for v in vehicles.values()
                    ],
                    "pool": [str(o) for o in active],
                },
                "committed": True,
            }
        )
    assigned = sum(o.matched for o in all_orders.values())
    pending = sum(len(v.orders) for v in vehicles.values())
    return {
        "operating_profit": total,
        "assigned": assigned,
        "cancelled": cancelled,
        "pending": pending + len(active),
        "delivered": None,
        "delivered_reason": "legacy transition does not expose verified delivery events",
        "arrivals": len(all_orders),
        "assigned_order_rate": assigned / len(all_orders) if all_orders else None,
    }, trace


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
    agent = None
    if spec.controller.id in ("candidate_sac", "train_sac"):
        import torch
        from drl_co.rl.candidate_sac import CandidateSAC

        torch.set_num_threads(1)
        torch.manual_seed(policy_seed)
        learning_rate = spec.controller.parameters.get("learning_rate", 0.0003)
        agent = CandidateSAC(
            "cpu",
            hidden_dim=32,
            batch_size=2,
            learning_starts=2,
            actor_lr=learning_rate,
            critic_lr=learning_rate,
            alpha_lr=learning_rate,
        )
        if spec.controller.id == "candidate_sac":
            source = run_dir(spec.controller.parameters["checkpoint_run"])
            meta = read_json(source / "checkpoint.json")
            if (
                hashlib.sha256((source / "weights.pt").read_bytes()).hexdigest()
                != meta["weights_hash"]
            ):
                raise PlatformError("CHECKPOINT_HASH", "checkpoint content changed")
            agent.load_state_dict(
                torch.load(source / "weights.pt", weights_only=True, map_location="cpu")
            )
    for i, scenario in enumerate(scenarios):
        check()
        emit(
            "scenario_started", scenario_id=scenario["id"], progress=i / len(scenarios)
        )
        if spec.model.id == "single_level_matching":
            if spec.controller.id in ("oracle_lp", "oracle_mip"):
                from model.bounds_check import bound

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
        elif spec.model.id == "legacy_dispatch":
            if spec.controller.id == "train_sac":
                for epoch in range(spec.controller.parameters["epochs"]):
                    metrics, trace = legacy(
                        spec, scenario, policy_seed, check, agent, train=True
                    )
                    emit(
                        "training",
                        scenario_id=scenario["id"],
                        epoch=epoch,
                        **agent.last_train_info,
                    )
            else:
                metrics, trace = legacy(spec, scenario, policy_seed, check, agent)
        elif spec.model.id == "bhh_steady":
            from .bhh import steady

            metrics = steady(spec.model.parameters)
            trace = []
        elif spec.model.id == "bhh_spatial":
            from .spatial import calibrate

            metrics = calibrate(spec.model.parameters)
            trace = []
        else:
            from .bhh import finite, rolling

            if spec.controller.id == "rolling_horizon":
                solution, metrics, trace = rolling(
                    scenario,
                    spec.model.parameters,
                    spec.solver.parameters.model_dump(),
                    spec.controller.parameters,
                    check,
                )
            else:
                solution, metrics = finite(
                    scenario, spec.model.parameters, spec.solver.parameters.model_dump()
                )
                trace = [
                    {
                        "period": 0,
                        "plan": {k: v for k, v in solution.items() if v > 1e-8},
                        "committed": False,
                        "oracle": True,
                    }
                ]
            metrics["order_rate_reason"] = (
                "BHH records represent divisible commodities, not indivisible orders"
            )
        rows.append(
            {
                "scenario_id": scenario["id"],
                "policy_seed": policy_seed,
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
    if spec.controller.id == "train_sac":
        import torch

        torch.save(agent.state_dict(), target / "weights.pt")
        atomic_json(
            target / "checkpoint.json",
            {
                "feature_schema": "candidate-sac/v1",
                "weights_hash": hashlib.sha256(
                    (target / "weights.pt").read_bytes()
                ).hexdigest(),
                "dataset_hash": spec.dataset.revision,
                "train_scenario_ids": [s["id"] for s in scenarios],
                "all_scenario_ids": [s["id"] for s in data["scenarios"]],
                "training_seed": policy_seed,
                "hidden_dim": 32,
                "model_parameters": spec.model.parameters,
            },
        )
    atomic_json(target / "trace.json", traces)
    if spec.model.id == "bhh_finite":
        from .bhh import capacity, direct_capacity

        p = spec.model.parameters
        table = {}
        for city in (0, 1):
            table[f"local_city_{city}"] = [
                {"vehicles": v, "duration": d, "capacity": capacity(v, d, p, city)}
                for v in range(sum(p["hv_fleet"]) + 1)
                for d in range(1, 13)
            ]
            table[f"direct_direction_{city}"] = [
                {
                    "vehicles": v,
                    "duration": d,
                    "capacity": 0 if d == p["tau"] else direct_capacity(v, d, p, city),
                }
                for v in range(sum(p["hv_fleet"]) + 1)
                for d in range(1, 13)
            ]
        atomic_json(target / "capacity-table.json", table)
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
                "y": "business_cost"
                if spec.model.id.startswith("bhh")
                else "operating_profit",
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
