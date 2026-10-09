"""Pressure uses the visible pool, locked supply, and a construction-only prior.

No orders scheduler or future realization can be passed to online pressure().
Demand and backlog are capacity-normalized vehicle equivalents. Known bookings
not yet ready enter Dnear, not Bload; the prior covers ONLY unbooked arrivals.
"""
import numpy as np
from experiment_core.storage import digest
from experiments.cfs_classical_value.simulation import summary as frozen_summary
from model_families.single.rolling.audit import incoming_supply
from model_families.single.engine.demand import ready_time

HB=4
ETA=1.

def expected_prior(scenarios,capacity=40):
    assert scenarios
    hubs=sorted(scenarios[0]['nodes']);T=scenarios[0]['horizon'];N=len(scenarios)
    values={h:{} for h in hubs}
    for h in hubs:
        for t in range(T):
            end=min(T-1,t+HB)
            values[h][str(t)]=sum(o['passenger']/capacity for s in scenarios for o in s['orders']
                if o['departure']==h and t<o['book_time']<=end and o['start_time']<=end)/N
    return dict(expected_unbooked_vehicle_equivalents=values,capacity=capacity,Hb=HB,eta=ETA,
        source_seeds=[s['seed'] for s in scenarios],source_hashes=[digest(s) for s in scenarios],
        rule='historical construction mean of future unbooked near-ready loads; '
            'known bookings counted separately; horizon clipped before last dispatch cutoff')

def bucket(ratio,cuts):
    a,b=cuts['c33'],cuts['c67']
    assert a<=b and np.isfinite(ratio)
    return 0 if ratio<=a else 1 if ratio<=b else 2

def pressure(state,prior,online=None):
    """The ORIGINAL pre-injection state is required; q worlds never relabel it."""
    online=frozen_summary(state,dict(rate={})) if online is None else online
    assert online['period']==state.t
    end=min(state.T-1,state.t+HB);capacity=state.parameters['capacity'];out={}
    assert capacity==prior['capacity'] and prior['Hb']==HB and prior['eta']==ETA
    assert all(o.book_time<=state.t for o in state.pool.values()),'Unbooked future present in visible pool'
    # The frozen summary's historical last-delivery timestamp is NOT an idle
    # release guarantee (and repeating an already-processed arrival can be early).
    # Reuse the existing readonly rolling audit's committed-route events instead;
    # no future demand, model solve, vehicle cloning/ticking or physical change.
    committed=[e for e in incoming_supply(state) if e['idle_release']]
    visible=sorted(state.pool.values(),key=lambda o:o.id)
    for h in sorted(state.net.H):
        backlog=[o for o in visible if o.departure==h and ready_time(o,state.t)<=state.t]
        bookings=[o for o in visible if o.departure==h and state.t<ready_time(o,state.t)<=end]
        B=sum(o.passenger for o in backlog)/capacity
        known=sum(o.passenger for o in bookings)/capacity
        expected=prior['expected_unbooked_vehicle_equivalents'][h][str(state.t)]
        M=online['idle'][h]
        near=[e for e in committed if e['hub']==h and state.t<e['eta']<=end]
        I=len(near)
        R=(B+ETA*(known+expected))/(M+I+1)
        out[h]=dict(R=float(R),Bload=float(B),Dnear_known_bookings=float(known),Dnear_prior=float(expected),
            Dnear=float(known+expected),Midle=int(M),Inear=int(I),window_end=end,
            visible_backlog_ids=sorted(o.id for o in backlog),visible_near_booking_ids=sorted(o.id for o in bookings),
            near_incoming_vehicle_ids=sorted(e['vehicle_id'] for e in near),
            near_incoming_release_events=near,
            information='observed pool + locked committed supply + frozen construction mean; no current unbooked realization')
    return out

def thresholds(records):
    rr=np.asarray([r['R'] for r in records],float)
    assert len(rr)>0 and np.isfinite(rr).all()
    c1,c2=np.quantile(rr,[1/3,2/3],method='linear')
    return dict(c33=float(c1),c67=float(c2),count=len(rr),quantiles=[1/3,2/3],
        tied_thresholds=bool(c1==c2),boundary='surplus R<=c33; balanced c33<R<=c67; scarce R>c67',
        aggregation='construction M0 states only; equal native time/hub/scenario weight; no formal tuning')

def summary(state,prior):
    """Preserve A's summary and add current-state bucket, never candidate post-state."""
    online=frozen_summary(state,prior['base_prior'])
    pp=pressure(state,prior['pressure_prior'],online)
    online['pressure']=pp
    online['bucket']={h:bucket(row['R'],prior['thresholds']) for h,row in pp.items()}
    online['bucket_semantics']='current observation period/hub, not future release forecast or candidate post-state'
    return online
