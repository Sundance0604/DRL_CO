"""Field-whitelisted observation; no scenario, future orders, seed or booked J."""
from dataclasses import dataclass
import math
from model_families.single.planning import committed_arc_load
from model_families.single.strategic.contracts import EncodedObservation, VehicleObservation
from .contracts import OBSERVATION_VERSION


@dataclass(frozen=True)
class FeatureScales:
    money: float = 1000.
    distance: float = 100.
    time: float = 24.
    load: float = 7.

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.money, self.distance, self.time, self.load)):
            raise ValueError("feature scales must be frozen positive finite constants")


def free_release(state, vehicle):
    """Project the already committed route, after this boundary's arrival events.

    Re-running begin_period here would process a mid-route arrival twice.
    """
    if vehicle.phase == "idle":
        return vehicle.loc, state.t
    if not vehicle.route:
        raise RuntimeError("active vehicle has no committed route")
    first = (vehicle.arrive if vehicle.arrive is not None and vehicle.arrive > state.t else
             max(state.t, vehicle.hold or state.t)+state.net.lt(vehicle.loc, vehicle.route[0]))
    finish = first+sum(state.net.lt(a, b) for a, b in zip(vehicle.route, vehicle.route[1:]))
    return vehicle.route[-1], finish


class DirectObserver:
    def __init__(self, schema, scales):
        self.schema, self.scales = schema, scales

    def build(self, state, history, baseline=None):
        if baseline is not None:
            raise ValueError("Direct observation must not receive a baseline")
        hubs, scale = self.schema.hubs, self.scales
        if set(hubs) != set(state.net.H):
            raise ValueError("observation hub set mismatch")
        onehot = lambda h: tuple(float(h == x) for x in hubs)
        t, T, p = state.t, state.T, state.parameters
        n = max(1, len(state.vehs))

        def order_row(o, assigned=False, onboard=False):
            return (*onehot(o.departure), *onehot(o.destination),
                    o.passenger/scale.load, o.revenue/scale.money, o.penalty/scale.money,
                    o.book_time/scale.time, o.start_time/scale.time, o.end_time/scale.time,
                    (t-o.book_time)/scale.time, (o.start_time-t)/scale.time,
                    (o.end_time-t-state.net.tau[o.departure, o.destination])/scale.time,
                    float(assigned), float(onboard))

        releases, vehicles = [], []
        for v in state.vehs:
            hub, when = free_release(state, v)
            releases.append((v.phase, hub, when))
            loads = committed_arc_load(state.net, v)
            route = [v.loc, *v.route]
            arcs = tuple((*onehot(a), *onehot(b), i/max(1, len(route)-1),
                          state.net.d[a, b]/scale.distance,
                          state.net.lt(a, b)/scale.time, loads.get((a, b), 0)/scale.load,
                          (p["capacity"]-loads.get((a, b), 0))/scale.load)
                         for i, (a, b) in enumerate(zip(route, route[1:])))
            commitments = tuple(sorted(order_row(o, True, oid in v.onboard)
                                       for oid, o in v.orders.items()))
            row = (*onehot(v.loc), *(float(v.phase == x) for x in ("idle", "trip", "repos")),
                   (v.arrive-t)/scale.time if v.arrive is not None else 0.,
                   float(v.arrive is not None), (v.hold-t)/scale.time if v.hold is not None else 0.,
                   float(v.hold is not None), float(v.dead), *onehot(hub), (when-t)/scale.time)
            vehicles.append(VehicleObservation(row, arcs, commitments))
        hub_rows = []
        for h in hubs:
            pending = [o for o in state.pool.values() if o.departure == h]
            near = t+(T-t)//2
            hub_rows.append((*onehot(h),
                sum(v.phase == "idle" and v.loc == h for v in state.vehs)/n,
                sum(ph != "idle" and rh == h and rt <= near for ph, rh, rt in releases)/n,
                sum(ph != "idle" and rh == h and rt > near for ph, rh, rt in releases)/n,
                len(pending)/n, sum(o.passenger for o in pending)/(n*scale.load),
                sum(o.end_time-t-state.net.tau[o.departure, o.destination] <= 1 for o in pending)/n,
                sum(o.passenger for o in state.pool.values() if o.destination == h)/(n*scale.load)))
        global_row = (t/scale.time, (T-t)/scale.time, T/scale.time, float(len(state.vehs)),
                      p["capacity"]/scale.load, p["speed"]/scale.distance,
                      *(p[k]/scale.money for k in ("loaded_cost", "empty_cost", "waiting_cost", "expiry_cost", "tie_break")),
                      p["max_delay"]/scale.time, float(p["cross_hub"]), float(p["reposition"]),
                      p["pickup_radius"]/scale.distance, float(p["pickup_rule"] == "legacy_adjacency"),
                      *(state.net.d[a, b]/scale.distance for a in hubs for b in hubs),
                      *(state.net.tau[a, b]/scale.time for a in hubs for b in hubs),
                      *(history.get((a, b), 0)/n for a in hubs for b in hubs))
        result = EncodedObservation(t, state.revision, hubs, tuple(global_row), tuple(hub_rows),
                                   tuple(sorted(order_row(o) for o in state.pool.values())),
                                   tuple(sorted(vehicles, key=lambda x: (x.features, x.arcs, x.commitments))), version=OBSERVATION_VERSION)
        rows = [result.global_features, *result.hub_features, *result.orders]
        for vehicle in result.vehicles:
            rows.extend([vehicle.features, *vehicle.arcs, *vehicle.commitments])
        if any(not math.isfinite(x) for row in rows for x in row):
            raise FloatingPointError("nonfinite public observation")
        return result


def dimensions(hub_count):
    return dict(global_dim=16+3*hub_count**2, hub_dim=hub_count+7,
                order_dim=2*hub_count+11, vehicle_dim=2*hub_count+9,
                arc_dim=2*hub_count+5)
