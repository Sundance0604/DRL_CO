import copy
import pytest
from experiments.cfs_resource_value_state_bucket_exact_accounting.pressure import bucket,thresholds,expected_prior,pressure
from experiments.cfs_resource_value_state_bucket_exact_accounting.provider import BucketValueProvider
from experiments.cfs_rollout_target_alignment.toys import fixture,order
from experiments.cfs_resource_policy_iteration.protocol import resource_world

def test_threshold_ties_and_boundaries():
    cuts=thresholds([dict(R=0.) for _ in range(60)])
    assert cuts['tied_thresholds'] and bucket(0,cuts)==0 and bucket(1,cuts)==2
    cuts=dict(c33=1.,c67=2.)
    assert [bucket(x,cuts) for x in [0,1,1.1,2,2.1]]==[0,0,1,1,2]

def test_prior_only_historical_future_and_no_booking_double_count():
    scenario=dict(nodes=['A','B','C'],horizon=2,seed=8300,orders=[dict(
        departure='A',book_time=1,start_time=1,passenger=40)])
    prior=expected_prior([scenario],capacity=40)
    assert prior['expected_unbooked_vehicle_equivalents']['A']['0']==1.
    assert prior['expected_unbooked_vehicle_equivalents']['A']['1']==0.
    s,orders=fixture([order('booked','A','B',10,deadline=2)],horizon=2)
    s.parameters['capacity']=40
    r=pressure(s,prior)['A']
    assert r['Bload']>0 and r['Dnear_known_bookings']==0.
    # Scheduler-only future changes do not enter pressure; no orders argument exists.
    snapshot=copy.deepcopy(r);orders['future_unbooked']=order('future_unbooked','A','B',100,book=1,deadline=2)
    assert pressure(s,prior)['A']==snapshot

def model():
    surf=lambda v:dict(method='table',values={'A':{'1':dict(times=[0,3,24],values=[v,v,0.])}})
    return dict(base=surf(7),buckets={str(b):surf(-20+b) for b in range(3)},
        support={'A':{str(b):{'1':{'0':12,'3':12}} for b in range(3)}},measured_q=[1],observed_max_q=1)

def test_supported_negative_bucket_and_explicit_fallback():
    m=model();p=BucketValueProvider(m);s=dict(T=24,period=0,bucket={'A':2},pressure={'A':dict(R=3.)})
    assert p.marginal_release_value(1,'A',1,s)==-18.
    assert not p.query_log[-1]['fallback'] and p.query_log[-1]['bucket']==2
    m['support']['A']['2']['1']['3']=9
    assert p.marginal_release_value(1,'A',1,s)==7.
    assert p.query_log[-1]['fallback_reason']=='sample_count_below_10'
    assert p.fallback_calls==1
    assert p.marginal_release_value(24,'A',1,s)==0.
    assert p.marginal_release_value(24,'unknown',999,dict(T=24,period=24))==0.

def test_pre_injection_pressure_is_not_recomputed_for_q_pair():
    s,orders=fixture([order('now','A','B',6,deadline=2)],horizon=2);s.parameters['capacity']=40
    prior=expected_prior([dict(nodes=['A','B','C'],horizon=2,seed=8300,orders=[])],capacity=40)
    original=pressure(s,prior)['A'];lo,hi=resource_world(s,'A',0),resource_world(s,'A',1)
    assert pressure(lo,prior)['A']['R']!=pressure(hi,prior)['A']['R']
    # Both targets must carry this SAME original record, never the altered denominator.
    paired_labels=[dict(q=q,pre_injection_R=original['R']) for q in [0,1]]
    assert paired_labels[0]['pre_injection_R']==paired_labels[1]['pre_injection_R']

def test_missing_bucket_fails_and_unobserved_q_is_explicit_fallback():
    m=model()
    for surface in [m['base']]+list(m['buckets'].values()):surface['values']['A']['2']=copy.deepcopy(surface['values']['A']['1'])
    p=BucketValueProvider(m)
    with pytest.raises(AssertionError,match='Missing current-state bucket'):
        p.marginal_release_value(0,'A',1,dict(T=24,period=0))
    assert p.marginal_release_value(0,'A',2,dict(T=24,period=0,bucket={'A':0}))==7
    assert p.query_log[-1]['fallback_reason']=='unsupported_q_endpoint'

def test_pressure_rejects_unbooked_future_in_visible_pool():
    s,_=fixture([order('now','A','B',6,deadline=2)],horizon=2);s.parameters['capacity']=40
    prior=expected_prior([dict(nodes=['A','B','C'],horizon=2,seed=8300,orders=[])],capacity=40)
    next(iter(s.pool.values())).book_time=1
    with pytest.raises(AssertionError,match='Unbooked future'):pressure(s,prior)

def test_incoming_uses_final_release_not_past_delivery_or_repeated_arrival():
    from experiments.cfs_classical_value.simulation import summary as historical_summary
    s,orders=fixture([order('earlier_at_B','A','B',10),order('remaining','A','C',30)],horizon=4)
    s.parameters['capacity']=40
    # Legal post-begin-period boundary: already arrived/delivered at B in t=1,
    # still carries the locked C order. Departure B->C has not been committed
    # for the next link yet. Re-running begin_period(1) would teleport to C.
    s.decide(False,0);s.begin_period([]);v=s.vehs[0]
    assert s.t==1 and v.loc=='B' and v.phase=='trip' and v.route==['C'] and v.arrive==1
    pp=expected_prior([dict(nodes=['A','B','C'],horizon=4,seed=8300,orders=[])],capacity=40)
    old=historical_summary(s,dict(rate={}))
    assert old['incoming'][0]['time']==1
    before=copy.deepcopy(vars(v));r=pressure(s,pp,old)['C']
    assert r['Inear']==1 and r['near_incoming_release_events'][0]['eta']==2
    assert vars(v)==before and s.t==1

def test_pressure_does_not_use_unrevealed_batch_ready_realization():
    s,_=fixture([order('booked','A','B',6,deadline=10)],horizon=6);s.parameters['capacity']=40
    prior=expected_prior([dict(nodes=['A','B','C'],horizon=6,seed=8300,orders=[])],capacity=40)
    o=next(iter(s.pool.values()));o.cutoff=2;o.bound=5;o.start_time=1
    a=pressure(s,prior)['A'];o.start_time=3;b=pressure(s,prior)['A']
    assert a==b and a['Bload']==0 and a['Dnear_known_bookings']==0
    s.t=2;r=pressure(s,prior)['A']
    assert r['Dnear_known_bookings']==o.passenger/40

