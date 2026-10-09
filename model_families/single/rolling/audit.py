"""Read-only execution audit, independent of objectives and information preview."""
import itertools
from ..planning import decision_vehicles,route_candidates,committed_arc_load
from ..engine.demand import ready_time
from ..network import eligible_pickups


def incoming_supply(sim):
    events=[]
    for v in sim.vehs:
        if v.phase=='idle' or not v.route:
            continue
        eta=v.arrive if v.arrive is not None and v.arrive>sim.t else max(sim.t,v.hold or sim.t)+sim.net.lt(v.loc,v.route[0])
        previous=v.loc
        for i,hub in enumerate(v.route):
            if i:eta+=sim.net.lt(previous,hub)
            if eta>sim.t:
                events.append(dict(vehicle_id=str(v.id),hub=hub,eta=eta,
                                   idle_release=i==len(v.route)-1,committed_order_ids=sorted(v.orders)))
            previous=hub
    return sorted(events,key=lambda e:(e['eta'],e['vehicle_id'],e['hub']))


def feasible_now(sim):
    """Existence of a capacity-feasible single-vehicle coalition with required anchors.

    Other vehicles can wait/continue. Enumerate only the at most two missing
    anchors, not an optimizer or counterfactual reward. No simulation mutation.
    """
    feasible=set();cross=[];p=sim.parameters
    for v in decision_vehicles(sim.t,sim.vehs):
        for r in route_candidates(sim.t,sim.net,sim.pool,v,p['cross_hub'],p['empty_cost'],p):
            if v.phase=='idle' and r['p']!=v.loc:
                cross.append(dict(vehicle_id=str(v.id),pickup=r['p'],endpoint=r['e'],
                                  departure_delay=r['delay'],empty_miles=sim.net.d[v.loc,r['p']],
                                  route_cost=r['cost'],eligible_orders=sorted(o.id for o in r['serve'])))
            for o in r['serve']:
                if o.id in feasible:continue
                groups=[]
                if r['new']:
                    if v.phase=='idle' and o.departure!=r['p']:
                        groups.append([a for a in r['serve'] if a.departure==r['p']])
                    if o.destination!=r['e']:
                        groups.append([a for a in r['serve'] if a.destination==r['e']])
                for anchors in itertools.product(*groups):
                    load=committed_arc_load(sim.net,v)
                    coalition={a.id:a for a in (o,*anchors)}
                    for a in coalition.values():
                        for arc in sim.net.arcs(a.departure,a.destination):
                            load[arc]=load.get(arc,0)+a.passenger
                    if max(load.values(),default=0)<=p['capacity']:
                        feasible.add(o.id);break
    return feasible,cross


class DeferralAudit:
    def __init__(self):
        self.records={}

    def observe(self,sim,plan):
        assigned={a['order_id'] for a in plan['assignments']}
        expired=set(plan['expired'])-assigned
        feasible,cross=feasible_now(sim)
        incoming=incoming_supply(sim);records=[];p=sim.parameters
        for oid,o in sim.pool.items():
            if oid in assigned:continue
            future=[]
            for event in incoming:
                if not event['idle_release']:continue
                hub=event['hub']
                if o.departure not in eligible_pickups(sim.net,hub,p['cross_hub'],p['pickup_radius'],p['pickup_rule']):continue
                pickup=max(event['eta']+sim.net.tau[hub,o.departure],ready_time(o,sim.t))
                if o.passenger<=p['capacity'] and pickup+sim.net.tau[o.departure,o.destination]<=o.end_time:
                    future.append(event)
            reason=('cancellation' if oid in expired else 'deliberate_deferral' if oid in feasible else
                    'no_vehicle' if not decision_vehicles(sim.t,sim.vehs) else 'infeasible_unassignment')
            history=self.records.setdefault(oid,[])
            row=dict(order_id=oid,period=sim.t,deferral_reason=reason,
                     deferral_periods=1+sum(a['deferral_reason']=='deliberate_deferral' for a in history)
                        if reason=='deliberate_deferral' else sum(a['deferral_reason']=='deliberate_deferral' for a in history),
                     future_supply_available=bool(future),future_supply_eta=min((e['eta'] for e in future),default=None),
                     future_supply_vehicle_ids=sorted({e['vehicle_id'] for e in future}),eventual_serving_vehicle=None)
            history.append(row);records.append(row)
        # Detailed candidates are reproducible from the preserved pre-action state.
        return dict(orders=records,incoming_committed_supply=incoming,
                    feasible_order_ids=sorted(feasible),candidate_cross_hub_count=len(cross),
                    candidate_cross_hub_actions=cross[:12],candidate_display_truncated=len(cross)>12)

    def finish(self,trace,orders,ledger):
        vehicles={a['order_id']:a['vehicle_id'] for r in trace for a in r['plan']['assignments']}
        # Delivery events are already evaluated by the authoritative ledger;
        # all assigned commitments complete under the study's cooldown profile.
        rows=[]
        for oid,history in self.records.items():
            for row in history:row['eventual_serving_vehicle']=vehicles.get(oid)
            count=sum(r['deferral_reason']=='deliberate_deferral' for r in history)
            if count:
                rows.append(dict(order_id=oid,deferral_periods=count,eventual_serving_vehicle=vehicles.get(oid),
                                 assigned_after_deferral=oid in vehicles,
                                 first_deferral_period=next(r['period'] for r in history if r['deferral_reason']=='deliberate_deferral')))
        n=len(rows);served=sum(r['assigned_after_deferral'] for r in rows)
        return dict(deferral_records=rows,deliberate_deferral_count=n,
                    deliberate_deferral_period_events=sum(r['deferral_periods'] for r in rows),
                    deliberate_deferral_rate=n/len(orders) if orders else 0,
                    mean_deliberate_deferral_periods=sum(r['deferral_periods'] for r in rows)/n if n else 0,
                    orders_eventually_served_after_deferral=served if ledger['unfinished_committed']==0 else None,
                    orders_lost_after_deferral=n-served if ledger['unfinished_committed']==0 else None,
                    deferral_success_rate=served/n if n and ledger['unfinished_committed']==0 else None,
                    deferral_definition='unassigned despite a currently feasible anchored capacity coalition, excluding cancellation; unique orders / arrivals',
                    future_supply_definition='all committed hub arrivals recorded; per-order future availability uses guaranteed idle release only')
