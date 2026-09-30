from drl_co.data_io import load_scenario
from drl_co.domain.city_graph import CityGraph
from drl_co.domain.order import Order
from drl_co.domain.vehicle import Vehicle
from drl_co.paths import DEFAULT_SCENARIO


def test_legacy_sample_scenario_loads_from_data_directory():
    vehicles, orders, graph = load_scenario(DEFAULT_SCENARIO)

    assert DEFAULT_SCENARIO.parent.name == "data"
    assert isinstance(graph, CityGraph)
    assert vehicles and all(isinstance(vehicle, Vehicle) for vehicle in vehicles.values())
    assert orders and all(isinstance(order, Order) for order in orders.values())
