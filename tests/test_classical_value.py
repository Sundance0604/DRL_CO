"""Classical value contract, boundary, slots, paired worlds, fit and leak guards."""
from types import SimpleNamespace
import numpy as np
import pytest
import gurobipy as gp
from gurobipy import GRB
from model_families.single.value.contracts import add_release_slots,ZeroValueProvider
from model_families.single.value.providers import design,feature_row,RegressionValueProvider
from model_families.single.value.route_milp import ValueRouteMILPPlanner
from model_families.single.rolling.contracts import WindowRequest
from model_families.single.rolling.controller import observed_state
from model_families.single.planning import compile_action,PeriodAction
from model_families.single.data import matching_state
from experiments.cfs_classical_value.common import nodes
from experiments.cfs_classical_value.fit_models import fit,prediction_metrics
from experiments.cfs_classical_value.build_mc_targets import resource_world,future_orders
from experiments.cfs_classical_value.simulation import simulate,summary
from tests.test_single_rolling import scenario,order,spec


def test_projected_nodes_are_legal_unique_actual_integers():
    ts=nodes(24,7)
    assert ts==sorted(set(ts)) and len(ts)==7 and min(ts)==0 and max(ts)==23


@pytest.mark.parametrize("values,expected", [([10,3,1],13),([1,2,100],3),([-10,-3,100],-13)])
def test_collision_slots_are_prefix_not_repeat_or_cherry_pick(values,expected):
    with gp.Model() as m:
        m.Params.OutputFlag=0
        a=m.addVar(vtype=GRB.BINARY,lb=1);b=m.addVar(vtype=GRB.BINARY,lb=1)
        provider=SimpleNamespace(marginal_release_value=lambda t,h,q,s:values[q-1])
        terms,_=add_release_slots(m,{(1,"A"):[a,b]},provider,{},5,3)
        m.setObjective(gp.quicksum(terms),GRB.MAXIMIZE);m.optimize()
        assert m.ObjVal==pytest.approx(expected)


def test_free_resource_terminal_skips_provider_but_not_accounting():
    with gp.Model() as m:
        m.Params.OutputFlag=0
        provider=SimpleNamespace(marginal_release_value=lambda *a:(_ for _ in ()).throw(AssertionError()))
        terms,coeff=add_release_slots(m,{(5,"A"):[m.addVar()]},provider,{},5,1)
        assert terms==coeff==[]


def test_nonfinite_coefficient_is_not_silently_zero():
    with gp.Model() as m:
        m.Params.OutputFlag=0
        with pytest.raises(ValueError,match="nonfinite"):
            add_release_slots(m,{(1,"A"):[m.addVar()]},SimpleNamespace(marginal_release_value=lambda *a:float("nan")),{},5,1)


def test_zero_value_h1_matches_original_current_objective_and_is_executable():
    data=scenario([order("a","A","B",fare=40),order("b","A","C",fare=15)],horizon=4)
    cfg=spec();sim,orders=matching_state(data,cfg.model.parameters,cfg.solver.parameters.model_dump())
    sim.begin_period(list(orders.values()))
    old=sim.solve_plan(True,0,False,None,{})
    request=WindowRequest(observed_state(sim),0,1,1,0,lambda:None)
    result=ValueRouteMILPPlanner(ZeroValueProvider(),{}).plan(request)
    assert result.diagnostics["window_objective"]==pytest.approx(old[1])
    info={};plan=compile_action(sim,result.actions.get(0,PeriodAction(0)),info)
    assert plan[0]==pytest.approx(old[0])
    sim.commit_plan(plan)


def test_counterfactual_removes_only_free_query_supply_and_reindexes_safely():
    data=scenario([],horizon=4,hubs=["A","B","A"])
    cfg=spec();sim,_=matching_state(data,cfg.model.parameters,cfg.solver.parameters.model_dump())
    sim.vehs[2].phase="trip";sim.vehs[2].route=["B"];sim.vehs[2].arrive=1
    c=resource_world(sim,"A",2)
    assert [v.id for v in c.vehs]==list(range(4))
    assert sum(v.phase=="idle" and v.loc=="A" for v in c.vehs)==2
    assert len(sim.vehs)==3 and sim.vehs[0].loc=="A"
    assert any(v.phase=="trip" for v in c.vehs)


def test_mc_future_keeps_current_observation_and_renames_future_only():
    original={o.id:o for o in [SimpleNamespace(**order("known","A","B")),SimpleNamespace(**order("oldfuture","A","B",book=2))]}
    donor={o.id:o for o in [SimpleNamespace(**order("known","A","B")),SimpleNamespace(**order("newfuture","A","B",book=2))]}
    f=future_orders(original,donor,0,1)
    assert set(f)=={"known","mc1-newfuture"} and f["known"] is original["known"]
    assert donor["newfuture"].id=="newfuture"


def test_online_summary_does_not_accept_future_orders():
    data=scenario([order("future","A","B",book=2)],horizon=4)
    cfg=spec();sim,orders=matching_state(data,cfg.model.parameters,cfg.solver.parameters.model_dump())
    s=summary(sim,{"rate":{"A":1}})
    assert s["backlog"]["A"]==0 and "orders" not in s
    row=feature_row(3,"A",1,s)
    assert row["expected"]==1 and "future" not in str(row)


def toy_rows():
    return [dict(remaining=t/10,other_idle=k%3,incoming=k%2,backlog=k%4,expected=t/5,
        q=k%3+1.,hub="A" if k%2 else "B",time=10-t,scenario_seed=k,
        target=20*(t/10)**2+2*(k%3)) for k in range(20) for t in range(11)]


@pytest.mark.parametrize("method",["linear","chebyshev","spline"])
def test_fits_finite_regularized_and_boundary_gate(method):
    rows=toy_rows();model=fit(rows,method,1,degree=4,knots=4,boundary="natural")
    pred=design(rows,model)@np.asarray(model["coefficients"])
    assert np.isfinite(pred).all() and model["regularized_condition_number"]>0
    provider=RegressionValueProvider(model)
    assert provider.marginal_release_value(10,"A",1,{"T":10})==0


def test_natural_spline_second_derivatives_zero():
    from scipy.interpolate import BSpline
    model=fit(toy_rows(),"spline",10,knots=4,boundary="natural")
    for b in model["bases"].values():
        if b["kind"]=="spline":
            derivative=BSpline(b["knots"],np.eye(len(b["knots"])-4),3).derivative(2)
            assert np.max(abs(derivative([b["lower"],b["upper"]])@np.asarray(b["transform"])))<1e-7


def test_prediction_pairs_group_within_seed_and_time():
    rows=[dict(target=y,hub=h,q=q,time=1,scenario_seed=1) for y,h,q in [(3,"A",1),(1,"B",1),(2,"A",2)]]
    m=prediction_metrics(rows,[3,1,2])
    assert m["pairwise_ranking_accuracy"]==m["spatial_same_q_ranking_accuracy"]==1


def test_cooldown_fare_not_lost_when_free_terminal_value_zero():
    data=scenario([order("late","A","B",book=1,ready=1,deadline=10,fare=40)],horizon=2)
    cfg=spec();sim,orders=matching_state(data,cfg.model.parameters,cfg.solver.parameters.model_dump())
    m,trace,_=simulate(sim,orders,trace=True)
    assert m["delivered"]==1 and m["unfinished_committed"]==0
    assert any(r.get("drain") for r in trace) and m["completed_fare"]==40
