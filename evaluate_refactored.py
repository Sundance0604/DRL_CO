"""Evaluate learned and non-learning dispatch policies on identical scenarios."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from gurobipy import GRB, Model, quicksum

from candidate_sac import CandidateSAC
from my_env import DispatchEnv
from rl_features import candidate_features, order_features, vehicle_features
from tool_func import city_node_generator, city_update_without_drl
from train_refactored import load_scenario, make_agent, solve_lower_layer


def fleet_pressure(orders, vehicles, capacity):
    """Backlogged passenger demand divided by total theoretical fleet capacity."""
    demand = sum(order.passenger for order in orders.values())
    return demand / max(1, len(vehicles) * capacity)


def standardize_action_values(values, mask):
    """Standardize learned action values within each order's feasible set."""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    result = np.zeros_like(values, dtype=float)
    for index, valid in enumerate(mask):
        selected = values[index, valid]
        if selected.size > 1 and selected.std() > 1e-8:
            result[index, valid] = np.clip(
                (selected - selected.mean()) / selected.std(), -3.0, 3.0
            )
    return result


def build_actor_candidate_mask(
    actor_values,
    legal_mask,
    departures,
    supply_by_city,
    eligible_mask=None,
    top_k=2,
    entropy_threshold=0.8,
    probability_margin=0.15,
    uncertain_top_k=4,
):
    """Build a safe actor-ranked subset of each order's legal city actions.

    The actor controls membership only.  The departure and highest-supply
    eligible city are structural anchors, while uncertain rows retain a wider
    set.  Absolute actor-logit magnitudes are never treated as economic value.
    """
    values = np.asarray(actor_values, dtype=float)
    legal = np.asarray(legal_mask, dtype=bool)
    eligible = legal if eligible_mask is None else legal & np.asarray(eligible_mask, dtype=bool)
    if values.shape != legal.shape or eligible.shape != legal.shape:
        raise ValueError("actor values and masks must have identical shapes")
    supply = np.asarray(supply_by_city, dtype=float)
    if supply.shape != (legal.shape[1],):
        raise ValueError("supply_by_city must contain one value per city")

    pruned = np.zeros_like(legal)
    uncertain_orders = 0
    for index, departure in enumerate(departures):
        valid = np.flatnonzero(legal[index])
        pool = np.flatnonzero(eligible[index])
        if pool.size == 0:
            pool = valid
        if valid.size == 0:
            continue

        row_values = values[index, pool]
        shifted = row_values - row_values.max()
        probability = np.exp(shifted)
        probability /= probability.sum()
        if pool.size > 1:
            entropy = -np.sum(probability * np.log(probability + 1e-12)) / np.log(pool.size)
            ordered_probability = np.sort(probability)[::-1]
            margin = ordered_probability[0] - ordered_probability[1]
        else:
            entropy = 0.0
            margin = 1.0
        uncertain = entropy >= entropy_threshold or margin <= probability_margin
        uncertain_orders += int(uncertain)
        keep_count = min(
            pool.size,
            max(1, uncertain_top_k if uncertain else top_k),
        )
        ranked = pool[np.argsort(values[index, pool])[-keep_count:]]
        pruned[index, ranked] = True

        departure = int(departure)
        if 0 <= departure < legal.shape[1] and legal[index, departure]:
            pruned[index, departure] = True
        if pool.size:
            supply_city = int(pool[np.argmax(supply[pool])])
            pruned[index, supply_city] = True

    pruned &= legal
    return pruned, {
        "candidate_total": int(pruned.sum()),
        "legal_total": int(legal.sum()),
        "uncertain_orders": int(uncertain_orders),
        "pruned_orders": int(legal.shape[0]),
    }


def _eligible_assignments(orders, vehicles, mask, capacity):
    order_list = list(orders.values())
    vehicle_list = [vehicle for vehicle in vehicles.values() if vehicle.whether_city]
    feasible_pairs = []
    eligible_city_mask = np.zeros_like(mask, dtype=bool)
    for order_index, order in enumerate(order_list):
        for vehicle in vehicle_list:
            city_id = vehicle.intercity
            if (
                mask[order_index, city_id]
                and vehicle.battery > order.battery
                and vehicle.get_capacity() + order.passenger <= capacity
                and order.end_time >= vehicle.time + order.least_time_consume
            ):
                feasible_pairs.append((order_index, vehicle.id))
                eligible_city_mask[order_index, city_id] = True
    return order_list, vehicle_list, feasible_pairs, eligible_city_mask


def _myopic_milp_actions(orders, vehicles, mask, capacity, learned_values=None,
                         learned_weight=0.0, learned_scale=1000.0,
                         candidate_city_mask=None, secondary_values=None,
                         return_stats=False):
    """Joint one-step assignment oracle; it has no access to future arrivals."""
    order_list, vehicle_list, feasible_pairs, _ = _eligible_assignments(
        orders, vehicles, mask, capacity
    )
    actions = [order.departure for order in order_list]
    if not order_list or not vehicle_list:
        stats = {"feasible_pairs": 0, "assigned_indices": []}
        return (actions, stats) if return_stats else actions

    if candidate_city_mask is not None:
        candidate_city_mask = np.asarray(candidate_city_mask, dtype=bool)
        if candidate_city_mask.shape != np.asarray(mask).shape:
            raise ValueError("candidate city mask must match the legal action mask")
        vehicle_city_lookup = {vehicle.id: vehicle.intercity for vehicle in vehicle_list}
        feasible_pairs = [
            pair for pair in feasible_pairs
            if candidate_city_mask[pair[0], vehicle_city_lookup[pair[1]]]
        ]

    model = Model("myopic_action_oracle")
    model.setParam("OutputFlag", 0)
    assignment = model.addVars(feasible_pairs, vtype=GRB.BINARY, name="assign")
    vehicle_city = {vehicle.id: vehicle.intercity for vehicle in vehicle_list}
    for order_index in range(len(order_list)):
        model.addConstr(
            quicksum(
                assignment[order_index, vehicle_id]
                for oi, vehicle_id in feasible_pairs if oi == order_index
            ) <= 1
        )
    for vehicle in vehicle_list:
        model.addConstr(
            quicksum(
                assignment[order_index, vehicle.id] * order_list[order_index].passenger
                for order_index, vehicle_id in feasible_pairs if vehicle_id == vehicle.id
            ) <= capacity - vehicle.get_capacity()
        )
    learned_bonus = (
        standardize_action_values(learned_values, mask)
        if learned_values is not None and learned_weight != 0 else np.zeros_like(mask, dtype=float)
    )
    primary_objective = (
        quicksum(
            assignment[order_index, vehicle_id] * (
                order_list[order_index].revenue
                + learned_weight * learned_scale
                * learned_bonus[order_index, vehicle_city[vehicle_id]]
            )
            for order_index, vehicle_id in feasible_pairs
        )
    )
    if secondary_values is None:
        model.setObjective(primary_objective, GRB.MAXIMIZE)
    else:
        secondary_bonus = standardize_action_values(secondary_values, mask)
        model.ModelSense = GRB.MAXIMIZE
        model.setObjectiveN(primary_objective, index=0, priority=2, name="current_value")
        model.setObjectiveN(
            quicksum(
                assignment[order_index, vehicle_id]
                * secondary_bonus[order_index, vehicle_city[vehicle_id]]
                for order_index, vehicle_id in feasible_pairs
            ),
            index=1,
            priority=1,
            name="actor_tiebreak",
        )
    model.optimize()
    assigned_indices = set()
    if model.status == GRB.OPTIMAL:
        by_id = {vehicle.id: vehicle for vehicle in vehicle_list}
        for order_index, vehicle_id in feasible_pairs:
            if assignment[order_index, vehicle_id].X > 0.5:
                actions[order_index] = by_id[vehicle_id].intercity
                assigned_indices.add(order_index)
    stats = {
        "feasible_pairs": len(feasible_pairs),
        "assigned_indices": sorted(assigned_indices),
    }
    return (actions, stats) if return_stats else actions


def _select_actions(policy, rng, agent, vehicle_state, order_state, mask, orders, cities,
                    vehicles, capacity, candidate_state=None, return_info=False,
                    pruning_diagnostics=True):
    def finish(actions, info=None):
        return (actions, info or {}) if return_info else actions

    if policy in {"sac", "sac_sample"}:
        if isinstance(agent, CandidateSAC):
            actions, _, _, _, _ = agent.take_action_candidates(
                candidate_state, mask, explore=policy == "sac_sample",
                greedy=policy == "sac", sequential=True,
            )
        else:
            actions, _, _, _ = agent.take_action_vehicle(
                vehicle_state, order_state, mask,
                explore=policy == "sac_sample", greedy=policy == "sac",
            )
        return finish(actions)
    if policy == "noop":
        return finish([order.departure for order in orders.values()])
    if policy == "random":
        return finish([rng.choice(np.flatnonzero(row).tolist()) for row in mask])
    if policy == "supply":
        result = []
        for row in mask:
            candidates = np.flatnonzero(row).tolist()
            result.append(
                max(
                    candidates,
                    key=lambda city_id: (
                        cities[city_id].city_seat_count(capacity), -city_id
                    ),
                )
            )
        return finish(result)
    if policy == "myopic_milp":
        return finish(_myopic_milp_actions(orders, vehicles, mask, capacity))
    if policy.startswith("actor_pruned_"):
        if not isinstance(agent, CandidateSAC):
            raise TypeError("actor-pruned policy requires CandidateSAC")
        top_k = int(policy.removeprefix("actor_pruned_"))
        actor_values = agent.candidate_actor_values(candidate_state).numpy()
        _, _, full_pairs, eligible_city_mask = _eligible_assignments(
            orders, vehicles, mask, capacity
        )
        supply_by_city = np.asarray([
            cities[city_id].city_seat_count(capacity) for city_id in range(len(cities))
        ])
        departures = [order.departure for order in orders.values()]
        candidate_mask, info = build_actor_candidate_mask(
            actor_values,
            mask,
            departures,
            supply_by_city,
            eligible_mask=eligible_city_mask,
            top_k=top_k,
        )
        actions, pruned_stats = _myopic_milp_actions(
            orders,
            vehicles,
            mask,
            capacity,
            candidate_city_mask=candidate_mask,
            secondary_values=actor_values,
            return_stats=True,
        )
        info.update({
            "full_pairs": len(full_pairs),
            "pruned_pairs": pruned_stats["feasible_pairs"],
        })
        if pruning_diagnostics:
            full_actions, full_stats = _myopic_milp_actions(
                orders, vehicles, mask, capacity, return_stats=True
            )
            oracle_indices = full_stats["assigned_indices"]
            info.update({
                "oracle_hits": sum(
                    bool(candidate_mask[index, full_actions[index]])
                    for index in oracle_indices
                ),
                "oracle_assigned": len(oracle_indices),
            })
        return finish(actions, info)
    if policy.startswith("hybrid_adaptive_"):
        if not isinstance(agent, CandidateSAC):
            raise TypeError("hybrid_adaptive policy requires CandidateSAC")
        threshold = float(policy.removeprefix("hybrid_adaptive_"))
        weight = 3.0 if fleet_pressure(orders, vehicles, capacity) >= threshold else 0.3
        learned_values = agent.candidate_q_values(candidate_state).numpy()
        return finish(_myopic_milp_actions(
            orders, vehicles, mask, capacity,
            learned_values=learned_values, learned_weight=weight,
        ))
    if policy.startswith("hybrid_shuffled_"):
        if not isinstance(agent, CandidateSAC):
            raise TypeError("hybrid_shuffled policy requires CandidateSAC")
        weight = float(policy.removeprefix("hybrid_shuffled_"))
        learned_values = agent.candidate_q_values(candidate_state).numpy()
        shuffled = learned_values.copy()
        for index, valid in enumerate(np.asarray(mask, dtype=bool)):
            valid_indices = np.flatnonzero(valid).tolist()
            valid_values = shuffled[index, valid_indices].tolist()
            rng.shuffle(valid_values)
            shuffled[index, valid_indices] = valid_values
        return finish(_myopic_milp_actions(
            orders, vehicles, mask, capacity,
            learned_values=shuffled, learned_weight=weight,
        ))
    if policy.startswith("hybrid_q_"):
        if not isinstance(agent, CandidateSAC):
            raise TypeError("hybrid_q policy requires CandidateSAC")
        weight = float(policy.removeprefix("hybrid_q_"))
        learned_values = agent.candidate_q_values(candidate_state).numpy()
        return finish(_myopic_milp_actions(
            orders, vehicles, mask, capacity,
            learned_values=learned_values, learned_weight=weight,
        ))
    raise ValueError(f"unknown policy: {policy}")


@torch.no_grad()
def evaluate_rollout(
    policy,
    original_vehicles,
    original_orders,
    graph,
    horizon,
    capacity=7,
    seed=0,
    agent=None,
    pruning_diagnostics=True,
):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    vehicles = copy.deepcopy(original_vehicles)
    all_orders = copy.deepcopy(original_orders)
    active_orders = {}
    cities = city_node_generator(graph, active_orders, vehicles, active_orders)
    env = DispatchEnv(graph, vehicles, all_orders, cities, capacity)
    costs = np.tile(np.asarray([10, 1, 3, 10], dtype=float), (len(vehicles), 1))
    objective_total = 0.0
    matched_total = 0
    cancelled_total = 0
    solve_failures = 0
    arrivals = 0
    pressure_values = []
    selection_totals = {
        "candidate_total": 0,
        "legal_total": 0,
        "uncertain_orders": 0,
        "pruned_orders": 0,
        "oracle_hits": 0,
        "oracle_assigned": 0,
        "full_pairs": 0,
        "pruned_pairs": 0,
    }

    if agent is not None:
        agent.eval()
    for time in range(horizon):
        env.time = time
        for order in all_orders.values():
            if order.start_time == time:
                active_orders[order.id] = order
                arrivals += 1
        city_update_without_drl(cities, vehicles, active_orders, time)
        vehicle_state = vehicle_features(cities, capacity, len(vehicles))
        _, order_state = order_features(active_orders, graph, time, capacity, horizon)
        _, candidate_state = candidate_features(
            cities, active_orders, graph, time, capacity, horizon, len(vehicles)
        )
        mask = env.get_mask(active_orders)
        pressure_values.append(fleet_pressure(active_orders, vehicles, capacity))
        acted_orders = list(active_orders.values())
        if acted_orders:
            actions, selection_info = _select_actions(
                policy, rng, agent, vehicle_state, order_state, mask,
                active_orders, cities, vehicles, capacity, candidate_state,
                return_info=True,
                pruning_diagnostics=pruning_diagnostics,
            )
            for key in selection_totals:
                selection_totals[key] += selection_info.get(key, 0)
            env.apply_actions(active_orders, actions, strict=True)
        objective, solved, cancelled = solve_lower_layer(
            graph, cities, vehicles, active_orders, time, costs
        )
        objective_total += objective
        matched_total += sum(order.matched for order in acted_orders)
        cancelled_total += cancelled
        solve_failures += int(solved is False)

    result = {
        "policy": policy,
        "seed": seed,
        "objective": objective_total,
        "matched": matched_total,
        "arrivals": arrivals,
        "match_rate": matched_total / max(1, arrivals),
        "cancelled": cancelled_total,
        "solve_failures": solve_failures,
        "pressure_mean": float(np.mean(pressure_values)),
        "pressure_max": float(np.max(pressure_values)),
    }
    if selection_totals["pruned_orders"]:
        result.update({
            "candidate_count_mean": (
                selection_totals["candidate_total"] / selection_totals["pruned_orders"]
            ),
            "candidate_fraction": (
                selection_totals["candidate_total"]
                / max(1, selection_totals["legal_total"])
            ),
            "uncertain_fraction": (
                selection_totals["uncertain_orders"] / selection_totals["pruned_orders"]
            ),
            "feasible_pair_reduction": 1.0 - (
                selection_totals["pruned_pairs"]
                / max(1, selection_totals["full_pairs"])
            ),
            "oracle_assigned": selection_totals["oracle_assigned"],
        })
        if selection_totals["oracle_assigned"]:
            result["oracle_recall"] = (
                selection_totals["oracle_hits"] / selection_totals["oracle_assigned"]
            )
    return result


def load_agent(checkpoint_path, scenario_path, horizon, capacity, batch_size, device):
    vehicles, orders, graph = load_scenario(scenario_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("agent_kind") == "candidate_sac":
        agent = CandidateSAC(
            device=device,
            feature_dim=int(config.get("feature_dim", 20)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            batch_size=batch_size,
            target_entropy_ratio=float(config.get("target_entropy_ratio", 0.2)),
            initial_alpha=float(config.get("initial_alpha", 0.1)),
        )
    else:
        first_orders = {order.id: order for order in orders.values() if order.start_time == 0}
        agent = make_agent(device, graph, vehicles, first_orders, capacity, horizon, batch_size)
    agent.load_state_dict(checkpoint["model"])
    return agent, vehicles, orders, graph


def evaluate_checkpoint(
    checkpoint_path,
    scenario_path=Path("sample_data.pkl"),
    horizon=72,
    capacity=7,
    batch_size=128,
    random_rollouts=10,
    device="cpu",
):
    agent, vehicles, orders, graph = load_agent(
        checkpoint_path, scenario_path, horizon, capacity, batch_size, device
    )
    rows = []
    for policy in ("noop", "supply", "myopic_milp", "sac"):
        rows.append(
            evaluate_rollout(policy, vehicles, orders, graph, horizon, capacity, 0, agent)
        )
    for seed in range(random_rollouts):
        rows.append(
            evaluate_rollout("random", vehicles, orders, graph, horizon, capacity, seed, agent)
        )
        rows.append(
            evaluate_rollout("sac_sample", vehicles, orders, graph, horizon, capacity, seed, agent)
        )
    return rows


def summarize(rows):
    summary = []
    for policy in sorted({row["policy"] for row in rows}):
        selected = [row for row in rows if row["policy"] == policy]
        summary.append({
            "policy": policy,
            "rollouts": len(selected),
            "objective_mean": float(np.mean([row["objective"] for row in selected])),
            "objective_std": float(np.std([row["objective"] for row in selected])),
            "matched_mean": float(np.mean([row["matched"] for row in selected])),
            "match_rate_mean": float(np.mean([row["match_rate"] for row in selected])),
            "cancelled_mean": float(np.mean([row["cancelled"] for row in selected])),
            "solve_failures": int(sum(row["solve_failures"] for row in selected)),
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scenario", type=Path, default=Path("sample_data.pkl"))
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--capacity", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-rollouts", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.csv"))
    args = parser.parse_args()
    rows = evaluate_checkpoint(
        args.checkpoint, args.scenario, args.horizon, args.capacity,
        args.batch_size, args.random_rollouts, args.device,
    )
    summary = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
