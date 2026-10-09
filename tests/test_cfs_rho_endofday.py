from types import SimpleNamespace
import networkx as nx
import pytest
from model_families.single.network import geometric_network, eligible_pickups
from model_families.single.engine.mt_prototype import Net, Sim
from model_families.single.accounting import cooldown_tick, completion_bound
from model_families.single.runner import matching
from experiment_core.plugins import resolve


def test_physical_eligibility():
    g = nx.Graph()
    g.add_weighted_edges_from([(a,b,3) for a,b in zip('ABCD','BCDE')])
    net = Net(g,1)
    assert net.d['A','E'] == 12
    assert 'E' in eligible_pickups(net,'A',radius=20)
    assert 'E' not in eligible_pickups(net,'A',rule='legacy_adjacency')
    assert eligible_pickups(net,'A',radius=0) == ['A']
    for a,b in [(0,3),(3,6),(6,20)]:
        assert set(eligible_pickups(net,'A',radius=a)) <= set(eligible_pickups(net,'A',radius=b))
    g = nx.Graph(); g.add_edge('F','G',weight=40)
    assert eligible_pickups(Net(g,0),'F',radius=20) == ['F']


def test_geometry_and_fixed_paths():
    import math
    g, xy = geometric_network({'A':[34,-118],'B':[35,-119],'C':[36,-120],'D':[37,-121]})
    assert nx.is_connected(g)
    assert all(w >= math.dist(xy[a],xy[b]) for a,b,w in g.edges(data='weight'))
    diamond = [('A','B',1),('B','D',1),('A','C',1),('C','D',1),('A','D',2.00000001)]
    a=nx.Graph(); a.add_weighted_edges_from(diamond)
    b=nx.Graph(); b.add_weighted_edges_from(reversed(diamond))
    n,m=Net(a,1),Net(b,999)
    assert n.path == m.path and n.d['A','D']==2
    for (u,v),path in n.path.items():
        assert path == list(reversed(n.path[v,u]))
        for i in range(len(path)):
            for j in range(i,len(path)):
                assert path[i:j+1] == n.path[path[i],path[j]]


def one_order(booking, distance=80, delay=0, waiting=0):
    horizon=12
    scenario={'id':'timing','seed':1,'split':'test','horizon':horizon,'nodes':['A','B'],
              'edges':[['A','B',distance]],'hubs':['A'],
              'orders':[dict(id='o',departure='A',destination='B',passenger=1,book_time=booking,
                             start_time=booking+delay,end_time=booking+delay+30,revenue=1000,penalty=0)]}
    spec=resolve({'name':'timing','dataset':{'dataset_id':'x','revision':'x'},
                  'model':{'id':'single_level_matching','parameters':{'speed':10,'loaded_cost':2,
                        'empty_cost':3,'capacity':2,'waiting_cost':waiting,'pickup_radius':100,'max_delay':delay}},
                  'evaluation':{'accounting_version':'single-realized-v1','terminal_policy':'cooldown_realized'}})
    return spec,scenario


def test_early_late_and_convergence():
    early=matching(*one_order(2),0,lambda:None)[0]
    late,trace=matching(*one_order(11),0,lambda:None)
    assert early['realized_profit']==late['realized_profit']==840
    assert early['booked_profit']==late['booked_profit']==840
    assert early['ledger_at_T']['unfinished_orders']==0
    assert late['ledger_at_T']['unfinished_orders']==1
    assert late['ledger_at_T']['credited_unrealized_fare']==1000
    assert late['ledger_at_T']['completed_fare']==0
    assert late['ledger_at_T']['credit_on_delivery_diagnostic']==-160
    assert late['cooldown_scan']['10']['realized_profit']==late['cooldown_scan']['20']['realized_profit']==840
    assert all(r['plan']['assignments']==[] and r['plan']['solver']['status']=='COMMITTED_EXECUTION_ONLY'
               for r in trace if r.get('drain'))
    assert late['unfinished_orders']==0 and late['pending_committed']==0
    assert sum(s['distance'] for v in late['execution_events'] for s in v['segments'])==80
    assert sum(len(v['delivery_times']) for v in late['execution_events'])==late['completed_orders']
    assert all(r['invariants']['clock_advanced_once'] for r in trace)


def test_hold_and_multihop_cooldown_without_solver(monkeypatch):
    spec,scenario=one_order(11,delay=3)
    original=Sim.solve_plan
    def checked(self,*args,**kwargs):
        assert self.t < self.T, 'solver called during cooldown'
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Sim,'solve_plan',checked)
    m,trace=matching(spec,scenario,0,lambda:None)
    assert m['delivered']==1 and m['actual_loaded_distance']==80
    assert m['actual_vehicle_waiting']==14  # 11 idle + 3 committed hold
    assert m['actual_order_waiting']==3
    assert m['cooldown_periods']<=m['cooldown_cap']


def test_cross_pickup_committed_multihop_and_capacity(monkeypatch):
    spec,s=one_order(11,distance=10)
    s['nodes']=['A','B','C','D']; s['edges']=[['A','B',10],['B','C',10],['C','D',10]]
    s['orders'][0].update(departure='C',destination='D')
    original=Sim.solve_plan
    def checked(self,*args,**kwargs):
        assert self.t<self.T
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Sim,'solve_plan',checked)
    m,trace=matching(spec,s,0,lambda:None)
    assert m['cross']==1 and m['delivered']==1
    assert m['empty_pickup_distance']==20 and m['actual_loaded_distance']==10
    assert m['realized_profit']==920
    assert all(r['invariants']['capacity_valid'] for r in trace)


def test_incomplete_commitment_is_error():
    graph=nx.Graph();graph.add_edge('A','B',weight=10)
    sim=Sim(Net(graph,0),['A'],1)
    sim.vehs[0].phase='trip'
    with pytest.raises(RuntimeError,match='no committed route'):
        completion_bound(sim)
    sim.pool={'o':SimpleNamespace()}
    with pytest.raises(RuntimeError,match='discretionary'):
        cooldown_tick(sim)


def test_cap_failure_and_order_partition(monkeypatch):
    import model_families.single.accounting as accounting
    monkeypatch.setattr(accounting,'completion_bound',lambda sim:sim.T)
    with pytest.raises(RuntimeError,match='cap reached'):
        matching(*one_order(11),0,lambda:None)


def test_unassigned_orders_not_erased():
    spec,s=one_order(11)
    s['orders'][0]['revenue']=0
    m,trace=matching(spec,s,0,lambda:None)
    assert m['unfinished_committed']==0 and m['pending_unassigned']==1 and m['unfinished_orders']==1
    assert m['completed_orders']+m['cancelled_orders']+m['unfinished_orders']==m['arrivals']


def test_realized_selection_contract():
    from model_families.single.plugin import validate_selection
    from experiment_core.contracts import PlatformError
    spec,s=one_order(11)
    validate_selection(spec,{'family':'single_level_matching'},[s])
    spec.evaluation.terminal_policy='report_pending'
    with pytest.raises(PlatformError,match='requires'):
        validate_selection(spec,{'family':'single_level_matching'},[s])


def test_adapter_weighted_sampling_and_no_fallback(tmp_path):
    import random
    from model_families.single.cfs_adapter import normalize_area,weighted_reservoir,read_pools
    assert normalize_area('6-348')=='06-348'
    rows=[{'SHIPMT_ID':str(i),'WGT_FACTOR':i+1} for i in range(30)]
    assert weighted_reservoir(rows,8,random.Random(12))==weighted_reservoir(rows,8,random.Random(12))
    assert len({r['SHIPMT_ID'] for r in weighted_reservoir(rows,8,random.Random(12))})==8
    assert weighted_reservoir(rows,0,random.Random(12))==[]
    with pytest.raises(ValueError,match='insufficient'):
        weighted_reservoir(rows,31,random.Random(12))
    with pytest.raises(FileNotFoundError):
        read_pools(tmp_path/'missing.sqlite',['06-348','06-488'])


def test_cfs_generated_time_not_horizon_clipped():
    from model_families.single.cfs_adapter import make_scenario
    region={'id':'adapter-synthetic-fixture','coordinates':{'06-348':[34,-118],'06-488':[38,-122]}}
    # Explicit synthetic adapter fixture, not formal CFS data.
    def row(i,a,b):return {'SHIPMT_ID':str(i),'ORIG_CFS_AREA':a,'DEST_CFS_AREA':b,'WGT_FACTOR':1,'SHIPMT_WGHT':2000}
    pools={'06-348__06-488':[[row(i,'06-348','06-488') for i in range(10)],
                            [row(i+20,'06-488','06-348') for i in range(10)]]}
    cfg={'knn':1,'stretch':1.2,'speed':50,'orders':10,'horizon':2,'ready_delay_max':2,
         'deadline_slack':[4,8],'pounds_per_unit':1000,'fare_per_network_mile':4,'fare_per_pound':.03,
         'order_backlog_penalty':5,'vehicles_per_hub':1}
    a,p=make_scenario(region,pools,100,cfg)
    b,q=make_scenario(region,pools,100,cfg)
    assert a==b and p==q
    assert all(o['end_time']>a['horizon'] for o in a['orders'])
    assert all(o['passenger']==2 for o in a['orders'])


def test_mismatched_resume_preserves_archive(tmp_path):
    import json
    from experiments.cfs_rho_endofday.run_experiment import run
    source=tmp_path/'fixture.sqlite';source.write_bytes(b'explicit synthetic guard fixture')
    output=tmp_path/'output';(output/'raw').mkdir(parents=True)
    archive=output/'raw/source.snapshot.zip';archive.write_bytes(b'original archive')
    (output/'manifest.json').write_text(json.dumps({'config':{'old':'config'},'source_code_digest':'old',
                                                 'source':{}}))
    with pytest.raises(RuntimeError,match='another source/config/code'):
        run({'data_source':str(source)},output)
    assert archive.read_bytes()==b'original archive'
