"""Fourth-study pre-solve information, affine objective and Gaussian inference."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import gurobipy as gp
from gurobipy import GRB
from experiment_core.storage import digest
from model_families.single.local_value.providers import LegacyAdapter,context,features,design,LocalValueProvider
from model_families.single.local_value.route_milp import LocalRouteMILPPlanner
from model_families.single.value.contracts import ZeroValueProvider,add_release_slots
from model_families.single.rolling.contracts import WindowRequest
from model_families.single.rolling.controller import observed_state
from model_families.single.data import matching_state
from experiments.cfs_classical_value.simulation import summary
from experiments.cfs_local_post_bayesian_value.fit_models import gaussian_posterior,meta
from experiments.cfs_local_post_bayesian_value.targets import interventions,project_intervention
from tests.test_single_rolling import scenario,order,spec

def state():
    cfg=spec();s,o=matching_state(scenario([order('a','A','B',fare=40),order('future','A','C',book=2)],horizon=4),cfg.model.parameters,cfg.solver.parameters.model_dump())
    s.begin_period([o['a']]);return s,o

def test_legacy_none_exact_compatibility():
    legacy=SimpleNamespace(marginal_release_value=lambda t,h,q,s:float(t+q))
    p=LegacyAdapter(legacy)
    assert p.marginal_release_value(2,'A',3,{})==5.
    assert p.marginal_release_value(2,'A',3,{}, {'ignored':True})==5.

def test_context_is_solver_independent_and_blocks_extra_solution_fields():
    s,o=state();c,_=interventions(s,1,0)[0];z=summary(s,{'rate':{}})
    assert 'future' not in str(c) and 'assignment' not in str(c)
    with pytest.raises(AssertionError):features(c|{'actual_served':1},z)
    c2=copy.deepcopy(c); s.pool.clear(); assert c2==c

def test_unbooked_order_cannot_enter_context():
    s,o=state()
    r=dict(serve=[o['future']],p='A',e='C',delay=0)
    with pytest.raises(AssertionError,match='unbooked'):context(s,s.vehs[0],r,0,3,'C','x')

def test_planner_seals_context_once_and_remains_linear():
    s,o=state();z=summary(s,{'rate':{}});p=LegacyAdapter(ZeroValueProvider())
    result=LocalRouteMILPPlanner(p,z).plan(WindowRequest(observed_state(s),0,1,1,0,lambda:None))
    d=result.diagnostics
    assert d['context_hash_before']==d['context_hash_after']
    assert d['quadratic_terms']==d['quadratic_constraints']==d['general_constraints']==0
    assert d['window_objective']==pytest.approx(s.solve_plan(True,0,False,None,{})[1])

def test_oracle_inputs_rejected_by_deployable_planner():
    s,o=state();p=LegacyAdapter(ZeroValueProvider())
    with pytest.raises(AssertionError,match='Oracle'):
        LocalRouteMILPPlanner(p,{}).plan(WindowRequest(s,0,1,1,0,lambda:None,(o['future'],)))

@pytest.mark.parametrize('slots,corr,expected',[([10,3,100],[-2,5],16),([-10,-3,100],[2,-5],-16)])
def test_correction_does_not_reassign_slots(slots,corr,expected):
    with gp.Model() as m:
        m.Params.OutputFlag=0
        ys=[m.addVar(vtype=GRB.BINARY,lb=1) for _ in range(2)]
        p=SimpleNamespace(marginal_release_value=lambda t,h,q,s:slots[q-1])
        terms,_=add_release_slots(m,{(1,'A'):ys},p,{},4,3)
        m.setObjective(gp.quicksum(terms)+gp.quicksum(c*y for c,y in zip(corr,ys)),GRB.MAXIMIZE)
        m.optimize();assert m.ObjVal==pytest.approx(expected)

def test_posterior_psd_nonnegative_variance_deterministic():
    rng=np.random.default_rng(7);X=rng.normal(size=(40,8));y=rng.normal(size=40)
    a,C=gaussian_posterior(X,y,2,np.ones(8));b,D=gaussian_posterior(X,y,2,np.ones(8))
    assert np.array_equal(a,b) and np.array_equal(C,D)
    assert np.linalg.eigvalsh(C).min()>=0 and np.all(np.einsum('ij,jk,ik->i',X,C,X)+2>=0)

def test_sparse_hub_shrinks_more_than_dense_hub():
    X=np.zeros((101,2));X[:100,0]=1;X[100,1]=1;y=np.r_[np.repeat(2.,100),10.]
    b,_=gaussian_posterior(X,y,4,np.ones(2))
    assert abs(b[0]-2)<abs(b[1]-10) and abs(b[1])<10

def test_action_targets_change_real_state_not_decorate_old_labels():
    s,o=state();items=interventions(s,7,0)
    routes=[x for x in items if x[0]['kind']=='route'];assert routes
    ctx,act=routes[0];p,a=project_intervention(s,act,o,ctx['release_time'])
    assert a['served_count']>0 and digest(vars(s.pool['a']))
    assert p.S['assigned']>s.S['assigned'] and s.t==0

def test_deployed_q_is_fixed_mixture_not_selected_slot():
    rows=[dict(hub='A',q=q,**{f:float(i) for f in ['remaining','duration','deadhead','loaded','compatible_load','expiring_load','incoming','release_idle','origin_idle','expected','wait','same_hub']}) for i,q in enumerate([1,2,3])]
    m,_=meta(rows,'linear')
    a=design([rows[0]|{'q':1}],m,True);b=design([rows[0]|{'q':3}],m,True)
    assert np.array_equal(a,b)
