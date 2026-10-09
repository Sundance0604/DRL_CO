"""Small exact transport fixtures, not CFS observations."""
from experiment_core.storage import atomic_json
from model_families.single.data import matching_state
from .common import OUTPUT
from .queries import choose_candidates
from .protocol import world,paired_target,replay_world

def order(oid,a,b,fare,book=0,deadline=10):
    return dict(id=oid,departure=a,destination=b,passenger=1,revenue=fare,
        book_time=book,start_time=book,end_time=deadline,penalty=0)

def fixture(orders,hubs=None,horizon=4,max_delay=0):
    data=dict(id='fifth-toy',seed=0,horizon=horizon,nodes=['A','B','C'],
        edges=[['A','B',1],['B','C',1]],hubs=hubs or ['A'],orders=orders)
    p=dict(speed=1,capacity=1,loaded_cost=0,empty_cost=0,waiting_cost=0,expiry_cost=0,
        cross_hub=False,reposition=False,pickup_radius=0,pickup_rule='network_radius',max_delay=max_delay,tie_break=1)
    s,o=matching_state(data,p,dict(threads=1,seed=0,mip_gap=0,time_limit_seconds=30))
    s.begin_period([x for x in o.values() if x.book_time==0])
    return s,o

def find(candidates,hub,delay=0):
    return next(c for c in candidates if c['kind']=='route' and c['release_hub']==hub and c['route']['delay']==delay and c['vehicle_id']==0)

def audit():
    # A: same physical action and future opportunities, only owned immediate fare differs.
    s,a=fixture([order('current','A','B',10)],horizon=2)
    s2,b=fixture([order('current','A','B',20)],horizon=2)
    low=world(s,a);high=world(s2,b)
    deltaJ=high['total']-low['total'];deltaG=high['current_stage']-low['current_stage']
    assert deltaJ==deltaG==10 and high['future_component']==low['future_component']==0
    # B: equal current money; distinct release creates exactly 100 in future opportunity.
    s,o=fixture([order('short','A','B',10,deadline=1),order('long','A','C',10,deadline=2),
        order('future','B','A',100,book=1,deadline=2)])
    candidates,_,_=choose_candidates(s)
    short=world(s,o,find(candidates,'B'));long=world(s,o,find(candidates,'C'))
    assert short['current_stage']==long['current_stage']==10
    assert short['future_component']-long['future_component']==100
    # C: M0 prefers 50 now to 40. Short leaves two B resources next period,
    # earning 170 rather than 100, plus the shared 100 opportunity at C at t=2.
    s,o=fixture([order('short','A','B',40,deadline=1),order('long','A','C',50,deadline=3),
        order('return','B','A',100,book=1,deadline=2),order('out','B','C',70,book=1,deadline=2),
        order('later','C','B',100,book=2,deadline=3)],hubs=['A','B'],max_delay=1)
    candidates,_,_=choose_candidates(s)
    base=world(s,o);r2=find(candidates,'B');delayed=find(candidates,'C',1)
    sr=world(s,o,r2);dr=world(s,o,delayed)
    bf=world(s,o,freeze_release=delayed['release_time']);df=world(s,o,delayed,freeze_release=delayed['release_time'])
    assert base['current_stage']==50 and sr['current_stage']==40
    assert paired_target(sr,base)==70
    assert paired_target(dr,base)==-100 and paired_target(df,bf)==0
    replay=[replay_world(s,o,w) for w in [base,sr,dr,bf,df]]
    result=dict(status='PASSED',A=dict(delta_total=deltaJ,delta_stage=deltaG,target=deltaJ-deltaG,
        scope='same physical release and future opportunity; immediate owned-fare perturbation, not an empirical paired state'),
        B=dict(short_g=short['current_stage'],long_g=long['current_stage'],future_difference=100),
        C=dict(baseline_total=base['total'],short_total=sr['total'],baseline_g=50,short_g=40,
            short_Y_roll=paired_target(sr,base),delayed_Y_roll=paired_target(dr,base),delayed_Y_freeze=paired_target(df,bf),
            note='Freeze masks the delayed route loss; exact primary comparisons share the same candidate-dependent freeze threshold in both worlds.'),
        replay=replay)
    atomic_json(OUTPUT/'qa/toy_accounting_audit.json',result)
    return result

if __name__=='__main__':print(audit())
