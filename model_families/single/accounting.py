"""Realized execution accounting; never used in the optimization objective."""
import math


def completion_bound(sim):
    finishes = [sim.t]
    for v in sim.vehs:
        if v.phase == "idle":
            continue
        if not v.route:
            raise RuntimeError("active vehicle has no committed route")
        start = max(sim.t, v.hold or sim.t)
        finish = (max(start, v.arrive) if v.arrive is not None else
                  start + sim.net.lt(v.loc, v.route[0]))
        finish += sum(sim.net.lt(a, b) for a, b in zip(v.route, v.route[1:]))
        finishes.append(finish)
    return math.ceil(max(finishes)) + 1


def cooldown_tick(sim):
    """No model solve, assignment, route selection, pool processing or new supply."""
    from .engine.mt_prototype import depart
    if sim.pool:
        raise RuntimeError("cooldown must not contain discretionary orders")
    sim.begin_period([])
    for v in sim.vehs:
        if v.phase == "trip" and (v.arrive == sim.t or
                                  (v.arrive is None and v.hold is not None and v.hold <= sim.t)):
            v.hold = None
            for oid, o in v.orders.items():
                if o.departure == v.loc and oid not in v.onboard:
                    if sim.t < o.start_time:
                        raise RuntimeError("committed pickup before ready time")
                    v.onboard.add(oid)
                    v.pickups[oid] = sim.t
            depart(v, sim.t, sim.net, sim.parameters.get("capacity", 7))
    sim.t += 1
    sim.revision += 1


def realized_ledger(sim, orders, decision_rows, cutoff=None):
    """Cutoff is exclusive: events at cutoff have not yet been processed.

    In-flight distance is proportional to elapsed link travel time. Vehicle wait
    includes all nontravel decision periods and committed holds during cooldown,
    never idle time after a vehicle finishes. Order wait is booking to pickup,
    cancellation, or T for unassigned pending orders. It is reported, not an
    additional invented cost: the existing stage backlog penalty remains the cost.
    """
    cutoff = sim.t if cutoff is None else cutoff
    params = sim.parameters
    loaded = empty = pickup_empty = load_miles = vehicle_wait = 0.0
    delivered, picked = {}, {}
    for v in sim.vehs:
        travel = decision_travel = 0.0
        end = sim.T
        for s in v.segments:
            elapsed = max(0, min(cutoff, s["end"]) - s["start"])
            elapsed = min(elapsed, s["end"] - s["start"])
            miles = s["distance"] * elapsed / (s["end"] - s["start"])
            travel += elapsed
            decision_travel += max(0, min(cutoff, sim.T, s["end"]) - s["start"])
            end = max(end, min(cutoff, s["end"]))
            if s["load"] > 0:
                loaded += miles
                load_miles += miles * s["load"]
            else:
                empty += miles
                if s["kind"] == "pickup_empty":
                    pickup_empty += miles
        vehicle_wait += min(sim.T, cutoff) - decision_travel + max(0, end-sim.T) - (travel-decision_travel)
        delivered.update({o: t for o, t in v.deliveries.items() if t < cutoff})
        picked.update({o: t for o, t in v.pickups.items() if t < cutoff})
    cancelled_times = {}
    backlog_cost = cancellation_cost = booked_cost = assigned_fare = 0.0
    assigned = set()
    for row in decision_rows:
        if row["period"] >= min(cutoff, sim.T):
            continue
        plan = row["plan"]
        accounting = plan["accounting"]
        backlog_cost += accounting.get("backlog_penalty", 0)
        cancellation_cost += accounting.get("expiry_penalty", 0)
        booked_cost += accounting.get("business_cost", 0)
        assigned_fare += accounting.get("assignment_revenue", 0)
        now = {a["order_id"] for a in plan["assignments"]}
        assigned.update(now)
        for oid in plan["expired"]:
            if oid not in now:
                cancelled_times[oid] = row["period"] + 1
    visible = {o: obj for o, obj in orders.items() if obj.book_time < min(cutoff, sim.T)}
    order_wait = sum(max(0, picked.get(oid, cancelled_times.get(oid, min(cutoff, sim.T))) - o.book_time)
                     if oid not in assigned or oid in picked else max(0, cutoff-o.book_time)
                     for oid, o in visible.items())
    fare = sum(orders[o].revenue for o in delivered)
    cost = (loaded*params.get("loaded_cost", 10) + empty*params.get("empty_cost", 10) +
            vehicle_wait*params.get("waiting_cost", 0) + backlog_cost + cancellation_cost)
    outstanding = assigned-set(delivered)
    pending_unassigned = len(visible)-len(assigned)-len(cancelled_times)
    if pending_unassigned < 0:
        raise RuntimeError("order accounting partition does not reconcile")
    return dict(schema_version="single-realized-ledger/v1", cutoff_exclusive=cutoff,
                completed_fare=fare, assigned_fare=assigned_fare, booked_cost=booked_cost,
                booked_profit=assigned_fare-booked_cost, realized_profit=fare-cost,
                realized_cost=cost, actual_loaded_distance=loaded, actual_empty_distance=empty,
                empty_pickup_distance=pickup_empty, actual_vehicle_waiting=vehicle_wait,
                actual_order_waiting=order_wait, backlog_penalty=backlog_cost,
                cancellation_penalty=cancellation_cost, completed_orders=len(delivered),
                cancelled_orders=len(cancelled_times), unfinished_orders=len(outstanding)+pending_unassigned,
                unfinished_committed=len(outstanding), pending_unassigned=pending_unassigned,
                credited_unrealized_fare=sum(orders[o].revenue for o in outstanding),
                load_factor=load_miles/(params.get("capacity", 7)*(loaded+empty)) if loaded+empty else 0,
                credit_on_delivery_diagnostic=fare-booked_cost)
