"""Canonical synthetic tests A--E; fixtures are NOT CFS observations."""
from types import SimpleNamespace
import pytest
from tests.test_single_rolling import scenario,order,spec,execute
from model_families.single.data import matching_state
from model_families.single.rolling.controller import RollingController
from model_families.single.rolling.audit import DeferralAudit,incoming_supply


def incoming_case(deadline=8,with_incoming=True):
    cfg=spec(H=4,k=1,capacity=2,loaded_cost=10)
    data=scenario([order('o','A','B',deadline=deadline,fare=50)],horizon=4,hubs=['C','A'] if with_incoming else ['A'])
    data['edges']=[['C','A',2],['A','B',1]]
    sim,orders=matching_state(data,cfg.model.parameters,cfg.solver.parameters.model_dump())
    sim.begin_period(list(orders.values()))
    if with_incoming:
        v=sim.vehs[0];v.phase='trip';v.route=['A','B'];v.arrive=2;v.dead=0
        old=SimpleNamespace(**order('committed','C','B',deadline=8,fare=100))
        v.orders={'committed':old};v.onboard={'committed'}
    return cfg,sim


def test_A_known_future_supply_and_auditable_deferral():
    cfg,sim=incoming_case()
    myopic=sim.solve_plan(True,0,False,None,{})
    assert myopic[2]['x']==[('o',1)]
    controller=RollingController(cfg.controller.parameters,lambda:None)
    info={};plan=controller.decide(sim,info)
    assert plan[2]['x']==[]
    assert controller.solution.actions[2].vehicles[0].order_ids==('o',)
    assert controller.solution.actions[2].vehicles[0].vehicle_id==0
    audit=DeferralAudit().observe(sim,{'assignments':[],'expired':[]})
    assert audit['orders'][0]['deferral_reason']=='deliberate_deferral'
    assert any(e['hub']=='A' and e['eta']==2 for e in audit['incoming_committed_supply'])
    # Execute/replan each period; the original commitment still reaches B.
    for t in range(3):
        if t:sim.begin_period([]);plan=controller.decide(sim,{})
        sim.commit_plan(plan)
    assert sim.S['assigned']==1 and 'o' in sim.vehs[0].orders
    assert sim.vehs[0].route==['B']


def test_B_tight_deadline_requires_immediate_service():
    cfg,sim=incoming_case(deadline=1)
    plan=RollingController(cfg.controller.parameters,lambda:None).decide(sim,{})
    assert plan[2]['x']==[('o',1)]


def test_C_no_incoming_supply_cannot_invent_future_saving():
    cfg,sim=incoming_case(with_incoming=False)
    plan=RollingController(cfg.controller.parameters,lambda:None).decide(sim,{})
    assert plan[0]==40 and incoming_supply(sim)==[]
    # Waiting is a tied optimum with zero backlog/wait costs, not extra profit.
    assert plan[0]<=sim.solve_plan(True,0,False,None,{})[0]


def test_D_preview_differs_but_does_not_commit_unbooked_orders():
    data=scenario([order('low','A','C',fare=5),order('future','A','B',book=1,ready=1,deadline=2,fare=100)],horizon=3)
    supply_cfg=spec(H=2,k=1)
    ms,ts=execute(supply_cfg,data)
    preview_cfg=spec(H=2,k=1);preview_cfg.controller.id='oracle_preview';preview_cfg.framework_id='oracle_preview';preview_cfg.evaluation.information_set='oracle'
    mp,tp=execute(preview_cfg,data)
    assert ts[0]['plan']['assignments']==[{'order_id':'low','vehicle_id':'0'}]
    assert tp[0]['plan']['assignments']==[]
    assert mp['realized_profit']>ms['realized_profit']
    assert tp[1]['plan']['assignments']==[{'order_id':'future','vehicle_id':'0'}]
    assert mp['planning_windows'][0]['information_mode']=='oracle_preview'
    assert mp['planning_windows'][0]['preview_order_ids']==['future']
    assert ms['planning_windows'][0]['preview_order_ids']==[]
    assert all(a['order_id']!='future' for a in tp[0]['plan']['assignments'])
    assert all(o['id']!='future' for o in tp[0]['observation']['orders'])


def test_E_H1_agrees_with_myopic_current_stage():
    cfg,sim=incoming_case()
    cfg.controller.parameters['planning_horizon']=1
    plan=RollingController(cfg.controller.parameters,lambda:None).decide(sim,{})
    myopic=sim.solve_plan(True,0,False,None,{})
    assert plan[0]==myopic[0] and plan[1]==myopic[1] and plan[2]['x']==myopic[2]['x']


def test_preview_contract_must_be_oracle_and_one_period():
    from model_families.single.plugin import validate_selection
    from experiment_core.contracts import PlatformError
    cfg=spec();cfg.controller.id='oracle_preview'
    with pytest.raises(PlatformError,match='requires oracle'):
        validate_selection(cfg,{'family':'single_level_matching'},[])
    cfg.evaluation.information_set='oracle';cfg.controller.parameters['commit_periods']=2
    with pytest.raises(PlatformError,match='one-period'):
        validate_selection(cfg,{'family':'single_level_matching'},[])


def test_placement_is_deterministic_and_not_randomized():
    from experiments.cfs_dynamic_lookahead.common import placement
    hubs=['A','B','C','D','E'];mass={'A':1,'B':5,'C':3,'D':5,'E':2}
    assert placement(hubs,mass,3)==['B','D','C']
    assert placement(hubs,mass,5)==hubs
    assert placement(hubs[::-1],mass,8)==hubs+['B','D','C']


def test_incoming_eta_after_processing_current_arrival():
    cfg,sim=incoming_case();sim.t=2;sim.begin_period([])
    events=incoming_supply(sim)
    assert events[0]['hub']=='B' and events[0]['eta']==3 and events[0]['idle_release']


def test_preview_not_costed_or_cancelled_before_booking():
    from model_families.single.rolling.controller import OraclePreviewController
    cfg,sim=incoming_case(with_incoming=False)
    cfg.controller.parameters['planning_horizon']=1
    future=SimpleNamespace(**order('boundary','A','B',book=1,ready=1,deadline=1,fare=999,penalty=1000))
    oracle=OraclePreviewController(cfg.controller.parameters,lambda:None,lambda t,end:[future])
    plan=oracle.decide(sim,{})
    assert plan[0]==sim.solve_plan(True,0,False,None,{})[0]
    assert 'boundary' not in sim.pool
    assert oracle.solution.diagnostics['window_objective']==plan[1]
