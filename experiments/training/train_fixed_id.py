"""Reproducible training entry point for the dispatch + Gurobi experiment.

Run a fast smoke experiment first:
    python -m experiments.training.train_fixed_id --episodes 3 --horizon 8 --batch-size 32
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from gurobipy import GRB

from drl_co.data_io import load_scenario
from drl_co.environment.dispatch import DispatchEnv
from drl_co.optimization.lower_layer import Lower_Layer
from drl_co.paths import DEFAULT_SCENARIO
from drl_co.rl.features import order_features, vehicle_features
from drl_co.rl.fixed_id_sac import MultiOrderSAC
from drl_co.simulation.tools import basic_cost, city_node_generator, city_update_without_drl
from drl_co.simulation.transitions import self_update, update_order, update_var, update_vehicle


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def solve_lower_layer(graph, cities, vehicles, orders, time, cost_matrix):
    """Solve one dispatch step and advance fleet/order state exactly once."""
    available = [vehicle.id for vehicle in vehicles.values() if vehicle.whether_city]
    travelling = [vehicle.id for vehicle in vehicles.values() if not vehicle.whether_city]
    objective = -float(basic_cost(vehicles, orders))
    solved = None

    if available:
        lower = Lower_Layer(graph, cities, vehicles, orders, "dispatch", [available, travelling], time)
        lower.get_decision()
        lower.constrain_1()
        lower.constrain_2()
        lower.constrain_3()
        lower.constrain_4()
        lower.constrain_5()
        lower.set_objective(cost_matrix)
        lower.model.setParam("OutputFlag", 0)
        lower.model.optimize()
        solved = lower.model.status == GRB.OPTIMAL
        if solved:
            objective = float(lower.model.objVal)
        else:
            self_update(vehicles, graph)
        # Also restores the real order ids when no optimum was found.
        update_var(lower, vehicles, orders)
    else:
        self_update(vehicles, graph)

    update_vehicle(vehicles, battery_consume=10, battery_add=300, speed=20, G=graph)
    cancelled = update_order(orders, time, speed=20)
    return objective, solved, cancelled


def make_agent(
    device, graph, vehicles, orders, capacity, horizon, batch_size,
    target_entropy_ratio=0.2, initial_alpha=0.1,
):
    cities = city_node_generator(graph, orders, vehicles, orders)
    vehicle_state = vehicle_features(cities, capacity, len(vehicles))
    _, order_state = order_features(orders, graph, 0, capacity, horizon)
    return MultiOrderSAC(
        device=device,
        VEHICLE_STATE_DIM=vehicle_state.size,
        ORDER_STATE_DIM=order_state.shape[1],
        ACTION_DIM=graph.num_nodes,
        HIDDEN_DIM=128,
        gamma=0.95,
        batch_size=batch_size,
        learning_starts=batch_size,
        target_entropy_ratio=target_entropy_ratio,
        initial_alpha=initial_alpha,
        dropout_p=0.0,
    )


@torch.no_grad()
def evaluate_agent(agent, original_vehicles, original_orders, graph, horizon, capacity, cost_matrix):
    """Run a deterministic greedy rollout without touching replay or optimizers."""
    was_training = agent.training
    agent.eval()
    vehicles = copy.deepcopy(original_vehicles)
    all_orders = copy.deepcopy(original_orders)
    active_orders = {}
    cities = city_node_generator(graph, active_orders, vehicles, active_orders)
    env = DispatchEnv(graph, vehicles, all_orders, cities, capacity)
    objective_total = 0.0
    matched_total = 0
    cancelled_total = 0
    solve_failures = 0

    for time in range(horizon):
        env.time = time
        for order in all_orders.values():
            if order.start_time == time:
                active_orders[order.id] = order
        city_update_without_drl(cities, vehicles, active_orders, time)
        vehicle_state = vehicle_features(cities, capacity, len(vehicles))
        order_ids, order_state = order_features(active_orders, graph, time, capacity, horizon)
        if order_ids:
            action_mask = env.get_mask(active_orders)
            actions, _, _, _ = agent.take_action_vehicle(
                vehicle_state, order_state, action_mask, explore=False, greedy=True
            )
            acted_orders = list(active_orders.values())
            env.apply_actions(active_orders, actions, strict=True)
        else:
            acted_orders = []
        objective, solved, cancelled = solve_lower_layer(
            graph, cities, vehicles, active_orders, time, cost_matrix
        )
        objective_total += objective
        matched_total += sum(order.matched for order in acted_orders)
        cancelled_total += cancelled
        solve_failures += int(solved is False)

    agent.train(was_training)
    return {
        "eval_objective": objective_total,
        "eval_matched": matched_total,
        "eval_cancelled": cancelled_total,
        "eval_solve_failures": solve_failures,
    }


def train(args):
    seed_everything(args.seed)
    original_vehicles, original_orders, graph = load_scenario(args.scenario)
    capacity = args.capacity
    horizon = min(args.horizon, max(order.start_time for order in original_orders.values()) + 1)
    first_orders = {
        order.id: order for order in original_orders.values() if order.start_time == 0
    }
    if not first_orders:
        raise ValueError("scenario has no orders at time zero")

    device = torch.device(args.device)
    agent = make_agent(
        device, graph, copy.deepcopy(original_vehicles), copy.deepcopy(first_orders),
        capacity, horizon, args.batch_size,
        args.target_entropy_ratio, args.initial_alpha,
    )
    cost_matrix = np.tile(np.asarray([10, 1, 3, 10], dtype=float), (len(original_vehicles), 1))
    metrics = []

    for episode in range(args.episodes):
        vehicles = copy.deepcopy(original_vehicles)
        all_orders = copy.deepcopy(original_orders)
        active_orders = {}
        cities = city_node_generator(graph, active_orders, vehicles, active_orders)
        env = DispatchEnv(graph, vehicles, all_orders, cities, capacity)
        episode_objective = 0.0
        matched_count = 0
        cancelled_count = 0
        solve_failures = 0

        for time in range(horizon):
            env.time = time
            for order in all_orders.values():
                if order.start_time == time:
                    active_orders[order.id] = order
            city_update_without_drl(cities, vehicles, active_orders, time)

            vehicle_state = vehicle_features(cities, capacity, len(vehicles))
            order_ids, order_state = order_features(active_orders, graph, time, capacity, horizon)
            action_mask = env.get_mask(active_orders)
            if order_ids:
                actions, _, _, _ = agent.take_action_vehicle(
                    vehicle_state, order_state, action_mask, explore=True
                )
                acted_orders = {order_id: active_orders[order_id] for order_id in order_ids}
                env.apply_actions(active_orders, actions, strict=True)
            else:
                actions = []
                acted_orders = {}

            objective, solved, cancelled = solve_lower_layer(
                graph, cities, vehicles, active_orders, time, cost_matrix
            )
            episode_objective += objective
            cancelled_count += cancelled
            solve_failures += int(solved is False)
            city_update_without_drl(cities, vehicles, active_orders, time)

            if order_ids:
                rewards = {}
                for order_id, order in acted_orders.items():
                    if order.matched:
                        rewards[order_id] = order.revenue / args.reward_scale
                        matched_count += 1
                    elif order_id not in active_orders:
                        rewards[order_id] = -args.cancel_penalty / args.reward_scale
                    else:
                        rewards[order_id] = -order.penalty / args.reward_scale

                next_vehicle_state = vehicle_features(cities, capacity, len(vehicles))
                next_order_ids, next_order_state = order_features(
                    active_orders, graph, time + 1, capacity, horizon
                )
                next_mask = env.get_mask(active_orders)
                agent.sac_add_order_transitions(
                    vehicle_state, order_ids, order_state, action_mask, actions, rewards,
                    next_vehicle_state, next_order_ids, next_order_state, next_mask,
                    done_global=time == horizon - 1,
                )
                agent.update_sac(args.updates_per_step)

        row = {
            "episode": episode,
            "objective": episode_objective,
            "matched": matched_count,
            "cancelled": cancelled_count,
            "solve_failures": solve_failures,
            **agent.last_train_info,
        }
        if (
            episode == 0
            or episode == args.episodes - 1
            or (args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0)
        ):
            row.update(
                evaluate_agent(
                    agent, original_vehicles, original_orders, graph,
                    horizon, capacity, cost_matrix,
                )
            )
        metrics.append(row)
        print(json.dumps(row, ensure_ascii=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": agent.state_dict(), "config": vars(args), "metrics": metrics},
        args.output_dir / "sac_checkpoint.pt",
    )
    columns = sorted({key for row in metrics for key in row})
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(metrics)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/refactored"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--capacity", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--target-entropy-ratio", type=float, default=0.2)
    parser.add_argument("--initial-alpha", type=float, default=0.1)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--cancel-penalty", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
