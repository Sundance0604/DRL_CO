"""Train the permutation-equivariant SAC with optimization demonstrations."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from candidate_sac import CandidateSAC
from evaluate_refactored import _myopic_milp_actions, evaluate_rollout
from my_env import DispatchEnv
from rl_features import candidate_features
from scenario_factory import generate_scenario
from tool_func import city_node_generator, city_update_without_drl
from train_refactored import load_scenario, seed_everything, solve_lower_layer


PRESSURE_CURRICULUM = (
    ("normal", 11, 5),
    ("moderate", 8, 6),
    ("high", 5, 8),
    ("extreme", 3, 10),
    ("high", 5, 8),
    ("extreme", 3, 10),
)


def curriculum_level(episode: int):
    """Deterministic mixed-pressure schedule with high/extreme oversampling."""
    return PRESSURE_CURRICULUM[episode % len(PRESSURE_CURRICULUM)]


def _supply_actions(mask, cities, capacity):
    """Deterministic supply baseline used by evaluation and reward control."""
    actions = []
    for row in np.asarray(mask, dtype=bool):
        candidates = np.flatnonzero(row).tolist()
        actions.append(max(
            candidates,
            key=lambda city_id: (cities[city_id].city_seat_count(capacity), -city_id),
        ))
    return actions


def _order_outcome_value(order, order_id, remaining_orders, cancel_penalty):
    if order.matched:
        return float(order.revenue)
    if order_id not in remaining_orders:
        return -float(cancel_penalty)
    return -float(order.penalty)


def solve_supply_counterfactual(
    graph,
    vehicles,
    active_orders,
    time,
    capacity,
    costs,
    cancel_penalty,
):
    """Solve a supply-policy control from an isolated copy of the current state."""
    control_vehicles = copy.deepcopy(vehicles)
    control_orders = copy.deepcopy(active_orders)
    order_references = dict(control_orders)
    control_cities = city_node_generator(
        graph, control_orders, control_vehicles, control_orders
    )
    control_env = DispatchEnv(
        graph, control_vehicles, control_orders, control_cities, capacity
    )
    control_env.time = time
    city_update_without_drl(control_cities, control_vehicles, control_orders, time)
    control_mask = control_env.get_mask(control_orders)
    if control_orders:
        actions = _supply_actions(control_mask, control_cities, capacity)
        control_env.apply_actions(control_orders, actions, strict=True)
    objective, solved, cancelled = solve_lower_layer(
        graph,
        control_cities,
        control_vehicles,
        control_orders,
        time,
        costs,
    )
    outcomes = {
        order_id: _order_outcome_value(
            order, order_id, control_orders, cancel_penalty
        )
        for order_id, order in order_references.items()
    }
    return {
        "objective": objective,
        "solved": solved,
        "cancelled": cancelled,
        "outcomes": outcomes,
    }


def _scenario_for_episode(args, original, episode: int, *, demonstration=False):
    if not args.random_scenarios:
        return tuple(copy.deepcopy(value) for value in original)
    seed_offset = 1_000_000 if demonstration else 10_000
    if getattr(args, "pressure_curriculum", False) and not demonstration:
        _, num_vehicles, orders_per_step = curriculum_level(episode)
    else:
        num_vehicles, orders_per_step = args.num_vehicles, args.orders_per_step
    return generate_scenario(
        args.seed * seed_offset + episode,
        horizon=args.horizon,
        num_cities=args.num_cities,
        num_vehicles=num_vehicles,
        orders_per_step=orders_per_step,
        capacity=args.capacity,
    )


def collect_demonstrations(agent, args, original):
    """Collect one-step MILP expert labels, then pretrain the actor."""
    states, masks, actions = [], [], []
    costs = None
    for episode in range(args.bc_episodes):
        vehicles, all_orders, graph = _scenario_for_episode(
            args, original, episode, demonstration=True
        )
        costs = np.tile(np.asarray([10, 1, 3, 10], dtype=float), (len(vehicles), 1))
        active_orders = {}
        cities = city_node_generator(graph, active_orders, vehicles, active_orders)
        env = DispatchEnv(graph, vehicles, all_orders, cities, args.capacity)
        for time in range(args.horizon):
            env.time = time
            for order in all_orders.values():
                if order.start_time == time:
                    active_orders[order.id] = order
            city_update_without_drl(cities, vehicles, active_orders, time)
            _, candidate_state = candidate_features(
                cities, active_orders, graph, time, args.capacity,
                args.horizon, len(vehicles),
            )
            mask = env.get_mask(active_orders)
            if active_orders:
                expert_actions = _myopic_milp_actions(
                    active_orders, vehicles, mask, args.capacity
                )
                states.extend(candidate_state)
                masks.extend(mask)
                actions.extend(expert_actions)
                env.apply_actions(active_orders, expert_actions, strict=True)
            solve_lower_layer(graph, cities, vehicles, active_orders, time, costs)
    return agent.behavior_clone(
        np.asarray(states, dtype=np.float32),
        np.asarray(masks, dtype=np.bool_),
        np.asarray(actions, dtype=np.int64),
        epochs=args.bc_epochs,
        batch_size=args.bc_batch_size,
    )


@torch.no_grad()
def evaluate_candidate(agent, original, horizon, capacity, seed=0):
    vehicles, orders, graph = original
    return evaluate_rollout(
        "sac", vehicles, orders, graph, horizon, capacity, seed, agent
    )


def train(args):
    seed_everything(args.seed)
    original = load_scenario(args.scenario)
    original_horizon = min(
        args.horizon, max(order.start_time for order in original[1].values()) + 1
    )
    agent = CandidateSAC(
        device=args.device,
        feature_dim=20,
        hidden_dim=args.hidden_dim,
        gamma=0.95,
        batch_size=args.batch_size,
        learning_starts=args.batch_size,
        target_entropy_ratio=args.target_entropy_ratio,
        initial_alpha=args.initial_alpha,
    )
    bc_info = collect_demonstrations(agent, args, original) if args.bc_episodes else {
        "bc_loss": 0.0, "bc_accuracy": 0.0, "bc_examples": 0,
    }
    print(json.dumps({"stage": "behavior_cloning", **bc_info}, ensure_ascii=False))

    metrics = []
    for episode in range(args.episodes):
        agent.train()
        vehicles, all_orders, graph = _scenario_for_episode(args, original, episode)
        horizon = args.horizon if args.random_scenarios else original_horizon
        active_orders = {}
        cities = city_node_generator(graph, active_orders, vehicles, active_orders)
        env = DispatchEnv(graph, vehicles, all_orders, cities, args.capacity)
        costs = np.tile(np.asarray([10, 1, 3, 10], dtype=float), (len(vehicles), 1))
        episode_objective = 0.0
        matched_count = 0
        cancelled_count = 0
        solve_failures = 0
        counterfactual_baseline_objective = 0.0
        counterfactual_advantage = 0.0
        counterfactual_failures = 0

        for time in range(horizon):
            env.time = time
            for order in all_orders.values():
                if order.start_time == time:
                    active_orders[order.id] = order
            city_update_without_drl(cities, vehicles, active_orders, time)
            order_ids, candidate_state = candidate_features(
                cities, active_orders, graph, time, args.capacity, horizon, len(vehicles)
            )
            action_mask = env.get_mask(active_orders)
            if order_ids:
                actions, _, _, _, decision_states = agent.take_action_candidates(
                    candidate_state, action_mask, explore=True, sequential=True
                )
                acted_orders = {order_id: active_orders[order_id] for order_id in order_ids}
                counterfactual = None
                if getattr(args, "counterfactual_reward", False):
                    counterfactual = solve_supply_counterfactual(
                        graph,
                        vehicles,
                        active_orders,
                        time,
                        args.capacity,
                        costs,
                        args.cancel_penalty,
                    )
                    if counterfactual["solved"] is False:
                        counterfactual_failures += 1
                        counterfactual = None
                env.apply_actions(active_orders, actions, strict=True)
            else:
                actions, decision_states, acted_orders, counterfactual = (
                    [], candidate_state, {}, None
                )

            objective, solved, cancelled = solve_lower_layer(
                graph, cities, vehicles, active_orders, time, costs
            )
            episode_objective += objective
            cancelled_count += cancelled
            solve_failures += int(solved is False)
            city_update_without_drl(cities, vehicles, active_orders, time)

            if order_ids:
                rewards = {}
                if counterfactual is not None:
                    step_advantage = objective - counterfactual["objective"]
                    counterfactual_baseline_objective += counterfactual["objective"]
                    counterfactual_advantage += step_advantage
                    team_component = (
                        args.team_reward_weight * step_advantage
                        / args.reward_scale / max(1, len(order_ids))
                    )
                    for order_id, order in acted_orders.items():
                        actual_value = _order_outcome_value(
                            order, order_id, active_orders, args.cancel_penalty
                        )
                        baseline_value = counterfactual["outcomes"][order_id]
                        individual_advantage = (
                            actual_value - baseline_value
                        ) / args.reward_scale
                        rewards[order_id] = (
                            args.counterfactual_individual_weight
                            * individual_advantage
                            + team_component
                        )
                        matched_count += int(order.matched)
                else:
                    team_component = (
                        args.team_reward_weight * objective
                        / args.reward_scale / max(1, len(order_ids))
                    )
                    for order_id, order in acted_orders.items():
                        individual = _order_outcome_value(
                            order, order_id, active_orders, args.cancel_penalty
                        ) / args.reward_scale
                        matched_count += int(order.matched)
                        rewards[order_id] = individual + team_component

                next_order_ids, next_candidate_state = candidate_features(
                    cities, active_orders, graph, time + 1, args.capacity,
                    horizon, len(vehicles),
                )
                next_mask = env.get_mask(active_orders)
                agent.add_order_transitions(
                    decision_states, order_ids, action_mask, actions, rewards,
                    next_candidate_state, next_order_ids, next_mask,
                    done_global=time == horizon - 1,
                )
                agent.update_sac(args.updates_per_step)

        row = {
            "episode": episode,
            "pressure_level": (
                curriculum_level(episode)[0]
                if getattr(args, "pressure_curriculum", False) else "fixed"
            ),
            "objective": episode_objective,
            "matched": matched_count,
            "cancelled": cancelled_count,
            "solve_failures": solve_failures,
            "counterfactual_baseline_objective": counterfactual_baseline_objective,
            "counterfactual_advantage": counterfactual_advantage,
            "counterfactual_failures": counterfactual_failures,
            **bc_info,
            **agent.last_train_info,
        }
        if (
            episode == 0 or episode == args.episodes - 1
            or (args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0)
        ):
            evaluation = evaluate_candidate(
                agent, original, original_horizon, args.capacity, args.seed
            )
            row.update({f"eval_{key}": value for key, value in evaluation.items()
                        if key not in {"policy", "seed"}})
        metrics.append(row)
        print(json.dumps(row, ensure_ascii=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"agent_kind": "candidate_sac", "feature_dim": 20})
    torch.save(
        {"model": agent.state_dict(), "config": config, "metrics": metrics},
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
    parser.add_argument("--scenario", type=Path, default=Path("sample_data.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/candidate-sac"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--capacity", type=int, default=7)
    parser.add_argument("--num-cities", type=int, default=8)
    parser.add_argument("--num-vehicles", type=int, default=11)
    parser.add_argument("--orders-per-step", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--bc-episodes", type=int, default=0)
    parser.add_argument("--bc-epochs", type=int, default=15)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--target-entropy-ratio", type=float, default=0.2)
    parser.add_argument("--initial-alpha", type=float, default=0.1)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--cancel-penalty", type=float, default=300.0)
    parser.add_argument("--team-reward-weight", type=float, default=0.25)
    parser.add_argument(
        "--counterfactual-reward", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--counterfactual-individual-weight", type=float, default=1.0)
    parser.add_argument("--random-scenarios", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pressure-curriculum", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
