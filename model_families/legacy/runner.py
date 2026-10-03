from __future__ import annotations
import copy
import hashlib
import math
import random
import time
import numpy as np
from experiment_core.contracts import PlatformError
from experiment_core.storage import atomic_json, digest, read_json, run_dir
def legacy(spec, scenario, seed, checkpoint, agent=None, train=False):
    import networkx as nx
    from .engine.domain.city_graph import CityGraph
    from .engine.domain.vehicle import Vehicle
    from .engine.domain.order import Order
    from .engine.environment.dispatch import DispatchEnv
    from .engine.simulation.tools import city_node_generator, city_update_without_drl
    from .engine.rl.features import candidate_features
    from .frameworks import _supply_actions
    from .frameworks import solve_lower_layer

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
    component_training_seed = None
    agent = None
    if spec.controller.id in ("candidate_sac", "train_sac"):
        import torch
        from .engine.rl.candidate_sac import CandidateSAC

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
            component_training_seed = meta.get("training_seed")
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
