"""Generate deterministic held-out dispatch scenarios for evaluation."""

from __future__ import annotations

import random

from drl_co.domain.city_graph import CityGraph
from drl_co.domain.order import Order
from drl_co.domain.vehicle import Vehicle


def generate_scenario(
    seed: int,
    horizon: int = 24,
    num_cities: int = 8,
    num_vehicles: int = 11,
    orders_per_step: int = 5,
    capacity: int = 7,
    speed: int = 20,
):
    """Create a scenario without the legacy generator's city off-by-one bug."""
    random.seed(seed)
    graph = CityGraph(num_cities, 0.3, (10, 30))
    vehicles = {
        vehicle_id: Vehicle(
            vehicle_id, 0, 0, random.randrange(num_cities), 2,
            random.uniform(2_000.0, 20_000.0), {},
        )
        for vehicle_id in range(num_vehicles)
    }
    orders = {}
    for time in range(horizon):
        for offset in range(orders_per_step):
            order_id = time * orders_per_step + offset
            departure = random.randrange(num_cities)
            destination = random.randrange(num_cities)
            while destination == departure:
                destination = random.randrange(num_cities)
            distance, _ = graph.get_intercity_path(departure, destination)
            passenger = random.randint(1, capacity)
            least_time = distance / speed
            orders[order_id] = Order(
                order_id,
                passenger,
                departure,
                destination,
                time,
                time + least_time + random.randint(10, 20),
                departure,
                random.uniform(0, 10) + distance * 10,
                distance,
                distance * 100 + passenger * 50,
                passenger * 5,
                least_time,
            )
    return vehicles, orders, graph
