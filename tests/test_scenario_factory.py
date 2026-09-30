from scenario_factory import generate_scenario


def _signature(seed):
    vehicles, orders, graph = generate_scenario(seed, horizon=3)
    edges = sorted(
        (min(left, right), max(left, right), data["weight"])
        for left, right, data in graph.G.edges(data=True)
    )
    order_rows = [
        (order.departure, order.destination, order.passenger, order.distance)
        for order in orders.values()
    ]
    vehicle_rows = [(vehicle.intercity, round(vehicle.battery, 6)) for vehicle in vehicles.values()]
    return edges, order_rows, vehicle_rows


def test_heldout_scenarios_are_deterministic_and_valid():
    assert _signature(101) == _signature(101)
    assert _signature(101) != _signature(102)
    _, orders, graph = generate_scenario(101, horizon=3)
    assert all(0 <= order.departure < graph.num_nodes for order in orders.values())
    assert all(0 <= order.destination < graph.num_nodes for order in orders.values())
    assert all(order.departure != order.destination for order in orders.values())
