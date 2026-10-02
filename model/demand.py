"""Demand with city geometry, as in Section 1.3 of HV-AV-HV-operation/main_v2.tex.

An order is booked in period g at a point inside its origin city. The first mile takes the
travel time from that point to the hub, so the order is ready at the hub at a = g + first mile.
The deadline b at the destination hub is stated at booking and is never earlier than the
physically earliest arrival, ceil(a) + tau(dep, des).

Placeholders (not calibrated): each city is a disc, demand points are uniform in the disc,
the hub lies on the boundary, and the radius divided by the city speed is between 0.5 and
1.5 periods. OD pairs, group sizes, booking periods and the slack of the deadline follow
drl_co/simulation/scenarios.py.
"""
import math, random
from types import SimpleNamespace

import itertools

SPEED, Q = 20, 7
CITY_TIME = (0.5, 1.5)        # city radius / city speed, in periods
SLACK = (10, 20)              # deadline slack beyond the earliest possible arrival, in periods
H_BATCH, Q_HV, K_CA, TAU_S = 2, 7, 0.9, 0.1   # collection period, HV seats, CA constant, stop time (periods)


def city_times(net, seed):
    rng = random.Random(seed * 7919 + 1)
    return {u: rng.uniform(*CITY_TIME) for u in sorted(net.H)}


def first_mile(rng, radius_time):
    """Travel time from a uniform point in the disc to a hub on its boundary."""
    r, th = math.sqrt(rng.random()), rng.uniform(0.0, 2.0 * math.pi)
    return radius_time * math.sqrt(r * r + 1.0 - 2.0 * r * math.cos(th))


def make_order(oid, dep, des, n, g, slack, lead, net):
    a = g + lead
    d = net.d[dep, des]
    return SimpleNamespace(id=oid, departure=dep, destination=des, passenger=n, book_time=g, start_time=a,
                           end_time=math.ceil(a) + net.tau[dep, des] + slack,
                           revenue=d * 100 + n * 50, penalty=n * 5)


def convert(all_orders, net, radii, seed, lead=True):
    """Keep OD, group size, booking period and slack of the repo scenario; add the first mile."""
    rng, out = random.Random(seed + 555), {}
    for oid in sorted(all_orders):
        o = all_orders[oid]
        slack = int(round(o.end_time - o.start_time - o.distance / SPEED))
        delta = first_mile(rng, radii[o.departure]) if lead else 0.0
        out[oid] = make_order(oid, o.departure, o.destination, o.passenger, o.start_time, slack, delta, net)
    return out


def sample_orders(net, radii, rng, T, ops, after=-1, first_id=0):
    """Order stream for booking periods after < g < T; returns {g: [orders]}."""
    hubs, out, oid = sorted(net.H), {}, first_id
    for g in range(after + 1, T):
        for _ in range(ops):
            dep = rng.choice(hubs)
            des = rng.choice([h for h in hubs if h != dep])
            n = rng.randint(1, Q)
            oid += 1
            out.setdefault(g, []).append(
                make_order(oid, dep, des, n, g, rng.randint(*SLACK), first_mile(rng, radii[dep]), net))
    return out


def point(rng):
    """Uniform point in the unit disc; the hub is at (1, 0)."""
    r, th = math.sqrt(rng.random()), rng.uniform(0.0, 2.0 * math.pi)
    return (r * math.cos(th), r * math.sin(th))


def tour_time(points, radius_time, n_stops=None):
    """Collection tour hub -> points -> hub, in periods. Exact for up to 5 stops, CA formula beyond
    (Section 1.3 of main_v2.tex: T(n) = [2r + k sqrt(nA)] / v + n tau_s with r = radius, A = pi r^2)."""
    n = len(points) if n_stops is None else n_stops
    if n == 0:
        return 0.0
    if points is not None and n <= 5:
        hub = (1.0, 0.0)

        def dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])
        best = min(dist(hub, seq[0]) + sum(dist(x, y) for x, y in zip(seq, seq[1:])) + dist(seq[-1], hub)
                   for seq in itertools.permutations(points))
        return best * radius_time + n * TAU_S
    return (2.0 + K_CA * math.sqrt(n * math.pi)) * radius_time + n * TAU_S


def ready_bound(radius_time):
    """Latest possible hub arrival of a batch, measured from the end of its collection period."""
    return tour_time(None, radius_time, Q_HV)


def batch(orders_by_hub_period, net, radii, rng):
    """Turn the bookings of one hub and one collection period into HV tours.
    Sets book_time, cutoff, start_time (actual hub arrival) and bound on every order."""
    for (dep, block), group in orders_by_hub_period.items():
        cutoff = (block + 1) * H_BATCH                      # end of the collection period
        group.sort(key=lambda o: o.id)
        tours, load = [[]], 0
        for o in group:                                     # first fit by seats
            if load + o.passenger > Q_HV and tours[-1]:
                tours.append([]); load = 0
            tours[-1].append(o); load += o.passenger
        for tour in tours:
            pts = [point(rng) for _ in tour]
            a = cutoff + tour_time(pts, radii[dep])
            for o in tour:
                o.cutoff, o.start_time = cutoff, a
                o.bound = cutoff + ready_bound(radii[dep])
                o.end_time = math.ceil(o.bound) + net.tau[o.departure, o.destination] + o.slack


def ready_time(o, t):
    """Ready time the operator can plan with in period t: the bound until the batch is formed."""
    c = getattr(o, "cutoff", None)
    if c is None or t >= c:
        return o.start_time
    return o.bound


def convert_batches(all_orders, net, radii, seed):
    """Batch version of convert(): same OD, group size, booking period and slack as the repo scenario."""
    rng, groups, out = random.Random(seed + 777), {}, {}
    for oid in sorted(all_orders):
        o = all_orders[oid]
        slack = int(round(o.end_time - o.start_time - o.distance / SPEED))
        d = net.d[o.departure, o.destination]
        new = SimpleNamespace(id=oid, departure=o.departure, destination=o.destination, passenger=o.passenger,
                              book_time=o.start_time, slack=slack, revenue=d * 100 + o.passenger * 50,
                              penalty=o.passenger * 5)
        groups.setdefault((o.departure, o.start_time // H_BATCH), []).append(new)
        out[oid] = new
    batch(groups, net, radii, rng)
    return out


def sample_batches(net, radii, rng, T, ops, after=-1, first_id=0):
    """Sampled order stream with batch arrivals, {booking period: [orders]}. Bookings before `after`
    are not sampled, so the batches of the first affected collection period may be incomplete."""
    hubs, groups, out, oid = sorted(net.H), {}, {}, first_id
    for g in range(after + 1, T):
        for _ in range(ops):
            dep = rng.choice(hubs)
            des = rng.choice([h for h in hubs if h != dep])
            n, d = rng.randint(1, Q), 0
            oid += 1
            o = SimpleNamespace(id=oid, departure=dep, destination=des, passenger=n, book_time=g,
                                slack=rng.randint(*SLACK), revenue=net.d[dep, des] * 100 + n * 50, penalty=n * 5)
            groups.setdefault((dep, g // H_BATCH), []).append(o)
            out.setdefault(g, []).append(o)
    batch(groups, net, radii, rng)
    return out


def lead_pmf(radius_time, n=200_000, seed=1):
    """Distribution of ceil(first mile) in periods, by simulation."""
    rng, cnt = random.Random(seed), {}
    for _ in range(n):
        j = math.ceil(first_mile(rng, radius_time))
        cnt[j] = cnt.get(j, 0) + 1
    return {j: c / n for j, c in sorted(cnt.items())}
