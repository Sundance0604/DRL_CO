"""Check that changing policy actions changes the lower-layer input/output."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from drl_co.data_io import load_scenario
from drl_co.environment.dispatch import DispatchEnv
from drl_co.paths import DEFAULT_SCENARIO
from drl_co.simulation.tools import city_node_generator, city_update_without_drl
from experiments.training.train_fixed_id import solve_lower_layer


def run_policy(scenario, choose_alternative: bool, horizon: int):
    original_vehicles, original_orders, graph = load_scenario(scenario)
    vehicles = copy.deepcopy(original_vehicles)
    all_orders = copy.deepcopy(original_orders)
    orders = {}
    cities = city_node_generator(graph, orders, vehicles, orders)
    env = DispatchEnv(graph, vehicles, all_orders, cities, capacity=7)
    costs = np.tile(np.asarray([10, 1, 3, 10], dtype=float), (len(vehicles), 1))
    actions_by_time = []
    buckets_by_time = []
    objective = 0.0
    matched = 0
    cancelled = 0
    solve_failures = 0
    for time in range(horizon):
        env.time = time
        for order in all_orders.values():
            if order.start_time == time:
                orders[order.id] = order
        city_update_without_drl(cities, vehicles, orders, time)
        mask = env.get_mask(orders)
        actions = []
        acted_orders = list(orders.values())
        for order, row in zip(acted_orders, mask):
            candidates = np.flatnonzero(row).tolist()
            alternatives = [candidate for candidate in candidates if candidate != order.departure]
            actions.append(alternatives[0] if choose_alternative and alternatives else order.departure)
        env.apply_actions(orders, actions)
        actions_by_time.append(actions)
        buckets_by_time.append({city_id: sorted(city.virtual_departure) for city_id, city in cities.items()})
        step_objective, solved, step_cancelled = solve_lower_layer(
            graph, cities, vehicles, orders, time, costs
        )
        objective += step_objective
        matched += sum(order.matched for order in acted_orders)
        cancelled += step_cancelled
        solve_failures += int(solved is False)
    return {
        "actions_by_time": actions_by_time,
        "city_buckets_by_time": buckets_by_time,
        "objective": objective,
        "matched": matched,
        "cancelled": cancelled,
        "solve_failures": solve_failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    result = {
        "noop": run_policy(args.scenario, False, args.horizon),
        "alternative": run_policy(args.scenario, True, args.horizon),
    }
    result["objective_changed"] = result["noop"]["objective"] != result["alternative"]["objective"]
    result["buckets_changed"] = (
        result["noop"]["city_buckets_by_time"]
        != result["alternative"]["city_buckets_by_time"]
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
