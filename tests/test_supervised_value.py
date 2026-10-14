import math
from types import SimpleNamespace
import pytest
from experiments.cfs_supervised_value.common import decision_time,features,STATE,metric,check_packet
from experiments.cfs_supervised_value.analyse import concordance
from experiments.cfs_supervised_value.run import Provider,business


def test_release_time_uses_earlier_public_node():
    assert [decision_time(t) for t in [0,3,7,12,16,20,23]]==[0,0,3,7,12,16,20]
    with pytest.raises(ValueError):decision_time(24)


def test_packet_guard_separates_roundoff_diagnostic_from_physics():
    assert check_packet({'J':1.,'loc':'A'},{'J':1.+1e-12,'loc':'A'})<1e-7
    with pytest.raises(ValueError):check_packet({'J':1.,'loc':'A'},{'J':1.,'loc':'B'})
    with pytest.raises(ValueError):check_packet({'J':1.,'loc':'A'},{'J':2.,'loc':'A'})


def test_metrics_zero_and_nonzero_targets():
    result=metric([0,1,-2],[0,2,-1])
    assert result['RMSE']==pytest.approx(math.sqrt(2/3)) and result['sign_error']==0
    assert metric([1,1],[1,2])['R2'] is None


def test_ranking_excludes_label_ties_and_counts_predict_ties():
    r=concordance([1,2,2],[0,0,1],['g']*3)
    assert r['pairs']==2 and r['mean_group_concordance']==.75


def test_features_reject_hidden_orders():
    with pytest.raises(ValueError,match='hidden'):
        features(SimpleNamespace(t=2),{'hidden':SimpleNamespace(book_time=3)},'A',4,1,{'A':0})


def test_terminal_gate_does_not_call_ml():
    provider=Provider('State',SimpleNamespace(t=23,T=24,vehs=[]),{}, {},None,None,1)
    assert provider.marginal_release_value(24,'A',1,{'T':24})==0
    assert provider.batch_rows==0


def test_business_equivalence_requires_homogeneous_before_state():
    def plan(v):return dict(period=0,assignments=[dict(vehicle_id=v,order_id='o')],routes=[dict(vehicle_id=v,B=['A','B'],e='B')],repositions=[],expired=[])
    def vehicle(i,h):return dict(id=i,hub=h,phase='idle',route=[],dead=0,arrive=None,hold=None,assigned_orders=[],onboard=[])
    before={'vehicles':[vehicle(0,'A'),vehicle(1,'A')]}
    assert business(plan('0'),before)==business(plan('1'),before)
    before['vehicles'][1]['hub']='B'
    assert business(plan('0'),before)!=business(plan('1'),before)


def test_h1_full_day_replay_and_prefix_occupancy(tmp_path):
    from experiments.cfs_supervised_value.common import write
    from experiments.cfs_supervised_value.run import one_day,Budget
    p=tmp_path/'table.json'
    write(p,dict(method='table',values={h:{'1':dict(times=[0,3],values=[0,0])} for h in ['A','B']}))
    scene=dict(id='synthetic-qa',seed=1,split='test',horizon=3,nodes=['A','B'],edges=[['A','B',1.]],hubs=['A'],
        orders=[dict(id='o',departure='A',destination='B',passenger=1,book_time=0,start_time=0,end_time=5,revenue=10.,penalty=1.)])
    cfg=dict(parameters=dict(speed=1.,capacity=2,loaded_cost=2.,empty_cost=2.2,waiting_cost=0.,expiry_cost=3.,max_delay=0),
        solver=dict(threads=1,mip_gap=0.,seed=0),baseline=str(p))
    budget=Budget();r,rt,q,z=one_day(scene,cfg,'M0',{'A':0,'B':0},None,budget,100)
    assert budget.solves==3 and r['ledger']['realized_profit']==8
    assert r['ledger']['unfinished_committed']==0 and z and q
    assert all(isinstance(v,(int,float)) for v in r['ledger'].values())
