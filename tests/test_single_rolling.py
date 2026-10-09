"""Synthetic physics/control tests. No claim that these fixtures are CFS data."""
import copy
import pytest
from pydantic import ValidationError
from experiment_core.plugins import resolve
from experiment_core.storage import digest
from model_families.single.data import matching_state
from model_families.single.runner import matching,state_snapshot
from model_families.single.parameters import SingleRollingParameters
from model_families.single.rolling import RollingController,WindowSolution
from model_families.single.planning import PeriodAction,VehicleAction,route_candidates,compile_action


def order(oid,a,b,ready=0,deadline=10,fare=10,book=0,load=1,penalty=0):
    return dict(id=oid,departure=a,destination=b,book_time=book,start_time=ready,
                end_time=deadline,revenue=fare,passenger=load,penalty=penalty)


def scenario(orders,horizon=4,hubs=None):
    return dict(id='rolling-synthetic',seed=1,split='test',horizon=horizon,
                nodes=['A','B','C'],edges=[['A','B',1],['B','C',1]],hubs=hubs or ['A'],orders=orders)


def spec(H=3,k=1,capacity=1,**parameters):
    return resolve({'name':'single rolling test','dataset':{'dataset_id':'x','revision':'x'},
                    'model':{'id':'single_level_matching','parameters':{
                        'speed':1,'capacity':capacity,'loaded_cost':0,'empty_cost':0,
                        'waiting_cost':0,'expiry_cost':0,'max_delay':0,**parameters}},
                    'controller':{'id':'rolling_horizon','parameters':{
                        'planning_horizon':H,'commit_periods':k,'completion_extension':0}},
                    'evaluation':{'accounting_version':'single-realized-v1','terminal_policy':'cooldown_realized'}})


def execute(config,data):
    return matching(config,data,0,lambda:None)


def test_joint_planning_changes_current_action():
    data=scenario([order('small','A','B',deadline=2,fare=5),
                   order('long','A','C',deadline=2,fare=8),
                   order('return','B','A',ready=1,deadline=2,fare=100)],horizon=3)
    rolling,trace=execute(spec(H=3,k=2),data)
    one_step,_=execute(spec(H=1),data)
    myopic=spec(H=1);myopic.controller.id='myopic';myopic.controller.parameters={}
    short,_=execute(myopic,data)
    assert rolling['booked_profit']==rolling['realized_profit']==105
    assert short['realized_profit']==one_step['realized_profit']==8
    assert trace[0]['plan']['assignments']==[{'order_id':'small','vehicle_id':'0'}]
    assert trace[1]['plan']['assignments']==[{'order_id':'return','vehicle_id':'0'}]
    assert not trace[1]['plan']['solver']['replanned']
    assert rolling['planning_windows'][0]['window_objective']==105
    assert rolling['planning_windows'][0]['global_gap'] is None


def test_window_and_prefix_boundaries():
    metrics,trace=execute(spec(H=4,k=2),scenario([],horizon=5))
    assert [(w['window_start'],w['decision_end'],w['commit_end']) for w in metrics['planning_windows']]==[(0,4,2),(2,5,4),(4,5,5)]
    assert metrics['window_solve_count']==3 and len(trace)==5
    assert all(r['invariants']['clock_advanced_once'] for r in trace)


def test_long_trip_not_truncated_at_window_or_T(monkeypatch):
    from model_families.single.rolling.route_milp import RouteMILPPlanner
    original=RouteMILPPlanner.plan
    def checked(self,request):
        assert request.start<request.state.T
        return original(self,request)
    monkeypatch.setattr(RouteMILPPlanner,'plan',checked)
    data=scenario([order('late','A','C',ready=2,book=2,deadline=30,fare=100)],horizon=3)
    data['edges']=[['A','B',4],['B','C',4]]
    metrics,trace=execute(spec(H=1,loaded_cost=2),data)
    assert metrics['assigned']==metrics['delivered']==1
    assert metrics['unfinished_committed']==0 and metrics['realized_profit']==84
    assert metrics['planning_windows'][-1]['decision_end']==3
    assert metrics['planning_windows'][-1]['completion_end']>=10
    assert metrics['planning_windows'][-1]['completion_auto_extension']>0
    assert all(r['plan']['solver']['status']=='COMMITTED_EXECUTION_ONLY' for r in trace if r.get('drain'))


@pytest.mark.parametrize('capacity,expected',[(1,1),(2,2)])
def test_committed_backbone_and_residual_link_capacity(capacity,expected):
    data=scenario([order('old','A','C',fare=50),order('new','B','C',ready=1,book=1,deadline=2,fare=20)],horizon=3)
    metrics,trace=execute(spec(H=1,capacity=capacity),data)
    assert metrics['assigned']==metrics['delivered']==expected
    assert trace[1]['plan']['routes'][0]['B']==['B','C']
    assert trace[1]['plan']['routes'][0]['new'] is False
    assert metrics['realized_profit']==50+(20 if expected==2 else 0)
    assert all(segment['load']<=capacity for v in metrics['execution_events'] for segment in v['segments'])


def test_future_bookings_invisible_and_new_orders_wait_for_replan():
    a=scenario([order('future','A','B',ready=1,book=1,deadline=2,fare=100)],horizon=3)
    b=copy.deepcopy(a);b['orders'][0]['revenue']=10000
    ma,ta=execute(spec(H=3,k=2),a);mb,tb=execute(spec(H=3,k=2),b)
    assert ta[0]['plan']['assignments']==tb[0]['plan']['assignments']==[]
    assert ma['planning_windows'][0]['visible_orders']==mb['planning_windows'][0]['visible_orders']==[]
    assert ma['cancelled']==1 and ma['delivered']==0
    adaptive,_=execute(spec(H=3,k=1),a)
    assert adaptive['delivered']==1


def test_batch_ready_information_frozen_at_current_bound():
    from model_families.single.rolling.controller import observed_state
    base=order('batch','A','B',ready=1,deadline=10,fare=20)
    base.update(cutoff=2,bound=3)
    cfg=spec();sim,orders=matching_state(scenario([base]),cfg.model.parameters,cfg.solver.parameters.model_dump())
    sim.begin_period(list(orders.values()))
    observed=observed_state(sim)
    assert observed.pool['batch'].start_time==3 and not hasattr(observed.pool['batch'],'cutoff')
    sim.pool['batch'].start_time=2
    other=observed_state(sim)
    assert vars(other.pool['batch'])==vars(observed.pool['batch'])
    observed.pool['batch'].revenue=999
    observed.net.H['A']['B']['weight']=999
    assert sim.pool['batch'].revenue==20 and sim.net.H['A']['B']['weight']==1


def test_replace_planner_without_changing_simulator():
    class WaitPlanner:
        def plan(self,request):
            request.state.J=999
            return WindowSolution({}, {'status':'TEST_WAIT','runtime':0})
    cfg=spec(H=2,k=2);sim,orders=matching_state(scenario([]),cfg.model.parameters,cfg.solver.parameters.model_dump())
    sim.begin_period([]);before=digest(state_snapshot(sim))
    controller=RollingController(cfg.controller.parameters,lambda:None,WaitPlanner())
    info={};plan=controller.decide(sim,info)
    assert digest(state_snapshot(sim))==before
    sim.commit_plan(plan)
    assert sim.t==1 and sim.J==0 and info['status']=='TEST_WAIT'


def test_invalid_plan_and_stale_commit_are_rejected():
    cfg=spec();sim,orders=matching_state(scenario([order('o','A','B')]),cfg.model.parameters,cfg.solver.parameters.model_dump())
    sim.begin_period(list(orders.values()))
    r=route_candidates(0,sim.net,sim.pool,sim.vehs[0],True,0,cfg.model.parameters)[0]
    with pytest.raises(RuntimeError,match='duplicate'):
        compile_action(sim,PeriodAction(0,(VehicleAction(0,r,('o','o')),)),{})
    assert sim.S['assigned']==0 and sim.t==0
    p=compile_action(sim,PeriodAction(0,(VehicleAction(0,r,('o',)),)),{})
    sim.commit_plan(p)
    with pytest.raises(RuntimeError,match='stale'):
        sim.commit_plan(p)


def test_parameter_and_algorithm_validation():
    for params in [{'planning_horizon':2,'commit_periods':3},{'planning_horizon':0},{'completion_extension':-1}]:
        with pytest.raises(ValidationError):SingleRollingParameters(**params)
    from model_families.single.plugin import validate_selection
    from experiment_core.contracts import PlatformError
    cfg=spec();cfg.controller.parameters['planner_id']='not_registered'
    with pytest.raises(PlatformError,match='unregistered'):
        validate_selection(cfg,{'family':'single_level_matching'},[])


def test_cross_pickup_and_delayed_commitment_finish():
    data=scenario([order('cross','B','C',ready=2,deadline=15,fare=50)],horizon=1)
    metrics,_=execute(spec(H=1,pickup_radius=2,max_delay=3),data)
    assert metrics['cross']==1 and metrics['delivered']==1 and metrics['unfinished_committed']==0
    assert metrics['empty_pickup_distance']==1


def test_hold_survives_replans_without_double_charging():
    data=scenario([order('held','A','C',ready=3,deadline=5,fare=100,penalty=20)],horizon=5)
    metrics,trace=execute(spec(H=2,k=1,max_delay=3,waiting_cost=2,loaded_cost=3),data)
    assert trace[0]['plan']['assignments']==[{'order_id':'held','vehicle_id':'0'}]
    assert trace[0]['plan']['routes'][0]['delay']==3
    assert trace[1]['before']['vehicles'][0]['hold']==3
    assert trace[1]['plan']['routes'][0]['t0']==3
    assert metrics['delivered']==1 and metrics['unfinished_committed']==0
    assert metrics['booked_profit']==metrics['realized_profit']==88
    assert sum(r['plan']['accounting'].get('planned_route_cost',0) for r in trace)==12


def test_two_vehicles_cannot_duplicate_one_order():
    data=scenario([order('shared','A','B',fare=100)],horizon=2,hubs=['A','A'])
    metrics,trace=execute(spec(H=2,k=2),data)
    assert metrics['assigned']==metrics['delivered']==1
    assert len(trace[0]['plan']['assignments'])==1
    assert metrics['booked_profit']==100


def test_invalid_or_interrupted_planner_does_not_touch_live_state():
    cfg=spec();sim,_=matching_state(scenario([]),cfg.model.parameters,cfg.solver.parameters.model_dump())
    sim.begin_period([]);before=digest(state_snapshot(sim))
    class Invalid:
        def plan(self,request):
            return WindowSolution({request.decision_end:PeriodAction(request.decision_end)}, {})
    with pytest.raises(RuntimeError,match='outside decision window'):
        RollingController(cfg.controller.parameters,lambda:None,Invalid()).decide(sim,{})
    from model_families.single.rolling.route_milp import RouteMILPPlanner
    class Interrupted:
        def plan(self,request):
            def cancel():raise RuntimeError('cancel during window construction')
            from dataclasses import replace
            # Allow creation, but interrupt at the first vehicle-building checkpoint.
            calls=0
            def checked():
                nonlocal calls
                calls+=1
                if calls>=2:cancel()
            return RouteMILPPlanner().plan(replace(request,checkpoint=checked))
    with pytest.raises(RuntimeError,match='cancel during window construction'):
        RollingController(cfg.controller.parameters,lambda:None,Interrupted()).decide(sim,{})
    assert digest(state_snapshot(sim))==before and sim.t==0


def test_catalog_and_saved_result_integration(tmp_path):
    import json
    from model_families.single.plugin import manifest,validate_selection
    from model_families.single.runner import execute as execute_run
    catalog=manifest();framework=next(f for f in catalog['frameworks'] if f['id']=='rolling')
    assert not framework['training_required'] and framework['value_function']=='zero'
    assert framework['accounting_version']=='single-realized-v1'
    assert any(p['id']=='cooldown_realized' for p in catalog['evaluation_profiles'])
    cfg=spec(H=2,k=1)
    data={'family':'single_level_matching','scenarios':[scenario([order('o','A','B',fare=50)])]}
    validate_selection(cfg,data,data['scenarios'])
    rows=execute_run(cfg,data,data['scenarios'],tmp_path,0,lambda *a,**k:None,lambda:False)
    assert rows[0]['framework_id']=='rolling'
    metrics=json.loads((tmp_path/'metrics.json').read_text())['rows'][0]['metrics']
    trace=json.loads((tmp_path/'trace.json').read_text())[0]['steps']
    assert metrics['planning_algorithm']=='route_milp' and metrics['delivered']==1
    assert len(metrics['planning_windows'])==metrics['window_solve_count']
    assert all('window' in r['plan']['solver'] for r in trace if not r.get('drain'))
    assert metrics['realized_profit']==50


def test_generic_batch_protocol_scans_H_and_k(monkeypatch):
    from experiment_core import service
    cfg=spec(H=2,k=1)
    data={'family':'single_level_matching','scenarios':[scenario([])]}
    monkeypatch.setattr(service,'load_dataset',lambda ref:(data,data['scenarios']))
    batch={'base_spec':cfg.model_dump(),
           'ranges':{'controller.parameters.planning_horizon':{'start':2,'stop':4,'step':2}},
           'sweeps':{'controller.parameters.commit_periods':[1,2]}}
    plan=service.plan(batch)
    assert plan['count']==4
    assert {(r['spec']['controller']['parameters']['planning_horizon'],
             r['spec']['controller']['parameters']['commit_periods']) for r in plan['runs']}=={(2,1),(2,2),(4,1),(4,2)}
    assert all(r['spec']['framework_id']=='rolling' and r['training_seed'] is None for r in plan['runs'])
    batch['sweeps']['controller.parameters.commit_periods']=[3]
    with pytest.raises(ValidationError,match='must not exceed'):
        service.plan(batch)
