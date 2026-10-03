"""Trace what the DRL_CO simulator does under the supply policy (commit 59935d3).

Run from the DRL_CO repo root:
    python -m checks_on_original.trace_check
30 scenarios per pressure level (seeds 10000-10029), horizon 24.
"""
import logging, sys
import numpy as np
from drl_co.simulation.scenarios import generate_scenario
from drl_co.simulation.tools import city_node_generator, city_update_without_drl
from drl_co.environment.dispatch import DispatchEnv
from experiments.training.train_candidate import _supply_actions
from experiments.training.train_fixed_id import solve_lower_layer

def run(seed, nv, ops, horizon=24, cap=7):
    vehicles, all_orders, graph = generate_scenario(seed, horizon=horizon, num_vehicles=nv, orders_per_step=ops)
    costs = np.tile(np.asarray([10, 1, 3, 10], dtype=float), (len(vehicles), 1))
    active = {}
    cities = city_node_generator(graph, active, vehicles, active)
    env = DispatchEnv(graph, vehicles, all_orders, cities, cap)
    S = dict(arrivals=0, matched=0, reloc=0, reloc_pass_real=0, reloc_closer=0, enroute=0,
             late=0, delivered=0, charge=0, idle=0, avail_steps=0, cancelled=0,
             extra_steps=[], min_batt=1e18)
    open_trips = {}   # order_id -> dict
    for t in range(horizon):
        env.time = t
        for o in all_orders.values():
            if o.start_time == t:
                active[o.id] = o; S['arrivals'] += 1
        city_update_without_drl(cities, vehicles, active, t)
        if active:
            mask = env.get_mask(active)
            env.apply_actions(active, _supply_actions(mask, cities, cap), strict=True)
        had = {v.id: set(v.orders) for v in vehicles.values()}
        was_matched = {oid for oid, o in all_orders.items() if o.matched}
        avail = [v for v in vehicles.values() if v.whether_city]
        for v in avail:
            S['avail_steps'] += 1
            S['min_batt'] = min(S['min_batt'], v.battery)
        _, solved, cancelled = solve_lower_layer(graph, cities, vehicles, active, t, costs)
        S['cancelled'] += cancelled
        # decisions of vehicles that stayed in city this step
        for v in vehicles.values():
            if v.whether_city and v.decision == 1: S['charge'] += 1
            if v.whether_city and v.decision == 2: S['idle'] += 1
        for oid, o in all_orders.items():
            if o.matched and oid not in was_matched:
                S['matched'] += 1
                v = vehicles[o.matched_vehicle_id]
                if had[v.id]: S['enroute'] += 1
                rel = o.virtual_departure != o.departure
                if rel:
                    S['reloc'] += 1
                    d_real, _ = graph.get_intercity_path(o.departure, o.destination)
                    d_virt, p_virt = graph.get_intercity_path(o.virtual_departure, o.destination)
                    if d_virt < d_real: S['reloc_closer'] += 1
                open_trips[oid] = dict(v=v.id, t=t, rel=rel, visited=[o.virtual_departure], o=o)
        for oid, tr in list(open_trips.items()):
            v = vehicles[tr['v']]
            if v.intercity not in tr['visited'][-1:]:
                tr['visited'].append(v.intercity)
            if oid not in v.orders:          # delivered
                o = tr['o']; S['delivered'] += 1; S['reloc_delivered'] = S.get('reloc_delivered', 0) + int(tr['rel'])
                if t + 1 > o.end_time: S['late'] += 1   # vehicle.time after update == t+1
                S['extra_steps'].append((t + 1 - tr['t']) - o.least_time_consume)
                if tr['rel'] and o.departure in tr['visited']: S['reloc_pass_real'] += 1
                del open_trips[oid]
        city_update_without_drl(cities, vehicles, active, t)
    S['undelivered_at_end'] = len(open_trips)
    return S

def main():
    logging.disable(logging.CRITICAL)
    for name, nv, ops in [("normal", 11, 5), ("high", 5, 8)]:
        tot = None
        for seed in range(30):
            s = run(10_000 + seed, nv, ops)
            if tot is None: tot = s
            else:
                for k, val in s.items():
                    if k == 'extra_steps': tot[k] += val
                    elif k == 'min_batt': tot[k] = min(tot[k], val)
                    else: tot[k] += val
        ex = np.array(tot.pop('extra_steps'))
        print(name, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in tot.items()})
        print("   trip steps minus continuous least time: mean %.2f, max %.2f" % (ex.mean(), ex.max()))


if __name__ == "__main__":
    main()
