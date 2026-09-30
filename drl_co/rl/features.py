"""Deterministic, scaled features for the dispatch policy."""

from __future__ import annotations

import numpy as np


def vehicle_features(cities, capacity: int, num_vehicles: int) -> np.ndarray:
    """Aggregate fleet supply per city without depending on dictionary order."""
    city_ids = sorted(cities)
    normalizer = max(1, num_vehicles)
    seats = [
        sum(capacity - vehicle.get_capacity() for vehicle in cities[city_id].vehicle_available.values())
        / max(1, normalizer * capacity)
        for city_id in city_ids
    ]
    counts = [len(cities[city_id].vehicle_available) / normalizer for city_id in city_ids]
    return np.asarray(seats + counts, dtype=np.float32)


def order_features(orders, graph, now: int, capacity: int, horizon: int):
    """Return stable order ids and normalized per-order features.

    The legacy SAC fed only the binary feasibility mask as the order state, so
    orders with different demand, value and deadlines were indistinguishable.
    """
    order_list = list(orders.values())
    num_cities = graph.num_nodes
    if not order_list:
        return [], np.empty((0, 6 + 3 * num_cities), dtype=np.float32)

    max_distance = max((entry["distance"] for entry in graph.get_dijkstra_results()), default=1.0)
    max_battery = max_distance * 10.0 + 10.0
    max_revenue = max_distance * 100.0 + capacity * 50.0
    rows = []
    ids = []
    for order in order_list:
        departure = np.zeros(num_cities, dtype=np.float32)
        destination = np.zeros(num_cities, dtype=np.float32)
        virtual_departure = np.zeros(num_cities, dtype=np.float32)
        departure[order.departure] = 1.0
        destination[order.destination] = 1.0
        virtual_departure[order.virtual_departure] = 1.0
        scalar = np.asarray(
            [
                order.passenger / max(1, capacity),
                np.clip((order.end_time - now) / max(1, horizon), -1.0, 1.0),
                order.battery / max(1.0, max_battery),
                order.distance / max(1.0, max_distance),
                order.revenue / max(1.0, max_revenue),
                order.penalty / max(1.0, capacity * 5.0),
            ],
            dtype=np.float32,
        )
        rows.append(np.concatenate([scalar, departure, destination, virtual_departure]))
        ids.append(order.id)
    return ids, np.vstack(rows)


def _distance_matrix(graph) -> np.ndarray:
    """Return a dense, normalized all-pairs distance matrix."""
    num_cities = graph.num_nodes
    distances = np.zeros((num_cities, num_cities), dtype=np.float32)
    for source in range(num_cities):
        for target in range(num_cities):
            if source == target:
                continue
            result = graph.get_intercity_path(source, target)
            if result is None:
                distances[source, target] = 1.0
            else:
                distances[source, target] = float(result[0])
    maximum = float(distances.max())
    if maximum > 0:
        distances /= maximum
    return distances


def candidate_features(cities, orders, graph, now: int, capacity: int, horizon: int,
                       num_vehicles: int) -> tuple[list[int], np.ndarray]:
    """Build one feature vector for every (order, candidate-city) pair.

    No absolute city id or city-position one-hot is included.  Consequently a
    permutation of city labels only permutes the action axis, which is the
    inductive bias needed for evaluation on unseen graphs.

    Feature columns are intentionally documented because the sequential policy
    updates ``available_seats`` (column 6) after each provisional assignment.
    """
    order_list = list(orders.values())
    num_cities = graph.num_nodes
    feature_dim = 20
    if not order_list:
        return [], np.empty((0, num_cities, feature_dim), dtype=np.float32)

    distances = _distance_matrix(graph)
    max_distance = max((entry["distance"] for entry in graph.get_dijkstra_results()), default=1.0)
    max_battery = max_distance * 10.0 + 10.0
    max_revenue = max_distance * 100.0 + capacity * 50.0
    fleet_normalizer = max(1, num_vehicles)

    available_seats = np.asarray([
        sum(capacity - vehicle.get_capacity() for vehicle in cities[city_id].vehicle_available.values())
        / max(1, fleet_normalizer * capacity)
        for city_id in range(num_cities)
    ], dtype=np.float32)
    available_counts = np.asarray([
        len(cities[city_id].vehicle_available) / fleet_normalizer
        for city_id in range(num_cities)
    ], dtype=np.float32)
    real_demand = np.asarray([
        sum(order.passenger for order in cities[city_id].real_departure.values())
        / max(1, fleet_normalizer * capacity)
        for city_id in range(num_cities)
    ], dtype=np.float32)
    virtual_demand = np.asarray([
        sum(order.passenger for order in cities[city_id].virtual_departure.values())
        / max(1, fleet_normalizer * capacity)
        for city_id in range(num_cities)
    ], dtype=np.float32)
    degrees = np.asarray([
        len(graph.get_neighbors(city_id)) / max(1, num_cities - 1)
        for city_id in range(num_cities)
    ], dtype=np.float32)
    charging = np.asarray([
        float(cities[city_id].charging_capacity)
        for city_id in range(num_cities)
    ], dtype=np.float32)
    charging /= max(1.0, float(charging.max()))
    global_seats = float(available_seats.sum()) / max(1, num_cities)
    global_count = float(available_counts.sum()) / max(1, num_cities)

    rows = []
    ids = []
    for order in order_list:
        scalar = np.asarray([
            order.passenger / max(1, capacity),
            np.clip((order.end_time - now) / max(1, horizon), -1.0, 1.0),
            order.battery / max(1.0, max_battery),
            order.distance / max(1.0, max_distance),
            order.revenue / max(1.0, max_revenue),
            order.penalty / max(1.0, capacity * 5.0),
        ], dtype=np.float32)
        candidates = np.zeros((num_cities, feature_dim), dtype=np.float32)
        candidates[:, :6] = scalar
        candidates[:, 6] = available_seats
        candidates[:, 7] = available_counts
        candidates[:, 8] = real_demand
        candidates[:, 9] = virtual_demand
        candidates[order.departure, 10] = 1.0
        candidates[order.destination, 11] = 1.0
        candidates[order.virtual_departure, 12] = 1.0
        candidates[:, 13] = distances[:, order.departure]
        candidates[:, 14] = distances[:, order.destination]
        candidates[:, 15] = distances[:, order.virtual_departure]
        candidates[:, 16] = degrees
        candidates[:, 17] = charging
        candidates[:, 18] = global_seats
        candidates[:, 19] = global_count
        rows.append(candidates)
        ids.append(order.id)
    return ids, np.stack(rows)
