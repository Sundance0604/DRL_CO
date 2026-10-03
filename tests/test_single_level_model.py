from types import SimpleNamespace

import networkx as nx

from model.bounds_check import bound
from model.mt_prototype import Net, load_case, run, simulate
from model.pervehicle_bound import pervehicle


def test_flat_vehicle_value_is_decision_invariant_on_regression_case():
    """Regression for the default-MIP-gap failure at seed 10000, period 15."""
    result = simulate(
        seed=10_000,
        nv=11,
        ops=5,
        hops=True,
        c_empty=10.0,
        check_flat=True,
        horizon=24,
        lead=True,
        batches=True,
    )
    assert result["assigned"] > 0


def test_hindsight_bounds_dominate_rolling_policies():
    net, hubs, orders = load_case(10_000, 4, 2, horizon=8)
    own = run(net, hubs, orders, False, 10.0, horizon=8)["J"]
    cross = run(net, hubs, orders, True, 10.0, horizon=8)["J"]
    items = [(order, 1) for order in orders.values()]
    lp, _ = bound(net, hubs, items, 8, 10.0, relax=True)
    _, milp_upper = bound(net, hubs, items, 8, 10.0, relax=False, time_limit=20)

    assert max(own, cross) <= milp_upper + 1e-6
    assert milp_upper <= lp + 1e-6 * abs(lp)


def test_per_vehicle_bound_charges_empty_distance_rate():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=10)
    net = Net(graph, seed=1)
    order = SimpleNamespace(
        id=0,
        departure=1,
        destination=0,
        passenger=1,
        book_time=0,
        start_time=1,
        end_time=5,
        revenue=1_000.0,
        penalty=0.0,
    )

    cheap_empty, _ = pervehicle(net, [0], {0: order}, 2, c_empty=1.0, time_limit=20, threads=1)
    expensive_empty, _ = pervehicle(net, [0], {0: order}, 2, c_empty=50.0, time_limit=20, threads=1)

    assert cheap_empty > expensive_empty
