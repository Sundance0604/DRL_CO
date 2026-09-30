from drl_co.domain.city import City
from drl_co.domain.order import Order
from drl_co.environment.dispatch import DispatchEnv


class TinyGraph:
    num_nodes = 4

    def get_intercity_path(self, source, target):
        assert (source, target) == (0, 3)
        return 2, [0, 1, 3]

    def get_neighbors(self, city_id):
        return {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}[city_id]


def make_order():
    return Order(9, 1, 0, 3, 0, 10, 0, 20.0, 2, 250, 5, 0.1)


def make_cities():
    graph = TinyGraph()
    return {
        city_id: City(city_id, graph.get_neighbors(city_id), {}, 10, {}, {})
        for city_id in range(4)
    }


def test_action_mask_has_noop_and_never_empty():
    order = make_order()
    env = DispatchEnv(TinyGraph(), {}, {order.id: order}, make_cities(), 7)
    mask = env.get_mask({order.id: order})
    assert mask.tolist() == [[True, False, True, False]]


def test_apply_actions_refreshes_solver_city_buckets():
    order = make_order()
    cities = make_cities()
    env = DispatchEnv(TinyGraph(), {}, {order.id: order}, cities, 7)
    env.apply_actions({order.id: order}, [2])
    assert order.virtual_departure == 2
    assert order.id in cities[2].virtual_departure
    assert order.id not in cities[0].virtual_departure
