"""Independent regression, manual two-step policy toy and terminal gate."""
from experiment_core.storage import read_json,atomic_json,digest
from model_families.single.data import matching_state
from .simulation import simulate
from experiments.cfs_classical_value.build_mc_targets import future_orders
from experiments.cfs_local_post_bayesian_value.baseline_regression import physical
from experiments.cfs_rollout_target_alignment.toys import fixture,order
from .common import OUTPUT,FIFTH,key,parameters,rho_levels,map_jobs,file_hash
from .protocol import resource_world,pair_audit,continuation,read_compressed

def toy_policy(model,n):
    return dict(policy_id=f'pi{n}',policy_hash=digest(dict(n=n,model=model)),table=model,
        table_hash=None if model is None else digest(model))

def toy_table(a1,a2):
    values={h:{str(q):dict(times=[0,1,2],values=[a1 if q==1 else a2,10 if q==1 else 0,0])
        for q in range(1,5)} for h in ['A','B','C']}
    for h in ['B','C']:
        for r in values[h].values():r['values']=[0,0,0]
    return dict(method='table',values=values)

def toy_audit():
    s,orders=fixture([order('now','A','B',6,deadline=1),order('later','A','B',10,book=1,deadline=2)],horizon=2)
    prior=dict(rate={});p0=toy_policy(None,0);v0=toy_table(6,10);p1=toy_policy(v0,1)
    v1=toy_table(10,6);p2=toy_policy(v1,2)
    r0=continuation(s,orders,p0,prior);r1=continuation(s,orders,p1,prior);r2=continuation(s,orders,p2,prior)
    assert [r['metrics']['realized_profit'] for r in [r0,r1,r2]]==[6,10,10]
    marginal=[]
    for n,p in enumerate([p0,p1,p2]):
        worlds=[resource_world(s,'A',q) for q in range(3)]
        results=[continuation(w,orders,p,prior) for w in worlds]
        y=[r['metrics']['realized_profit'] for r in results]
        for q in [1,2]:pair_audit(worlds[q-1],worlds[q],'A')
        assert y==([0,6,16] if n==0 else [0,10,16])
        marginal.append(dict(policy=f'pi{n}',profits=y,V_A_q1=y[1]-y[0],V_A_q2=y[2]-y[1],
            continuation_policy_hash=p['policy_hash'],continuation_table_hash=p.get('table_hash'),calls=sum(r['value_calls'] for r in results)))
    # One still-committed trip passes T, while only new FREE supply changes.
    terminal,to=fixture([order('commit','A','C',20,deadline=3)],horizon=1)
    info={};plan=terminal.solve_plan(False,0,False,None,info);terminal.commit_plan(plan)
    assert terminal.t==terminal.T and any(v.phase!='idle' for v in terminal.vehs)
    low,high=resource_world(terminal,'A',0),resource_world(terminal,'A',1);pair_audit(low,high,'A')
    a,b=continuation(low,to,p1,prior),continuation(high,to,p1,prior)
    assert a['metrics']['realized_profit']==b['metrics']['realized_profit'] and not any(not r.get('drain') for r in b['trace'])
    assert a['metrics']['unfinished_committed']==b['metrics']['unfinished_committed']==0
    result=dict(status='PASSED',manual=dict(pi0_profit=6,V0_t0_A=[6,10],V0_t1_A=[10,0],
        pi1_profit=10,V1_t0_A=[10,6],pi2_profit=10,explanation='M0 dispatches fare6 now; slot value10 at t1 preserves A car for fare10. Second evaluation changes resource marginals, next action stays stable.'),
        evaluations=marginal,terminal=dict(status='PASSED',profit_difference=0,committed_cooldown=True,new_dispatches=0,
            steps=a['replay']['periods']+b['replay']['periods']),claim='exact toy enumeration only; no general exact PI theorem')
    atomic_json(OUTPUT/'qa/policy_iteration_toy.json',result)
    atomic_json(OUTPUT/'qa/terminal_resource_test.json',result['terminal']);return result

def baseline_job(c):
    cid=key(c);out=[]
    for label,policy in [('P0','myopic'),('P2','myopic+V'),('R0','RH4'),('R1','RH8')]:
        p=FIFTH/'raw/runs'/f'formal-{cid}-7000-{label}';j=read_json(p/'spec.json')
        s,orders=matching_state(j['scenario'],j['parameters'],j['solver'])
        from model_families.single.value.providers import load_provider
        provider=load_provider(FIFTH/j['model']) if label=='P2' else None
        metrics,trace,_=simulate(s,orders,policy,provider,j['prior'],trace=True)
        old=read_compressed(p/'observations.json.gz')
        assert abs(metrics['realized_profit']-old['metrics']['realized_profit'])<1e-6 and physical(trace)==physical(old['trace'])
        out.append(dict(condition=cid,policy=label,seed=7000,profit_difference=metrics['realized_profit']-old['metrics']['realized_profit'],
            physical_trace_identical=True,pure_slot_planner=label=='P2',historical_trace_hash=file_hash(p/'observations.json.gz')))
    # Reconstruct an old q pair with its actual own/donor future, not a new label.
    rows=read_json(FIFTH/'targets/construction_mc.json');r=next(r for r in rows if r['condition']==cid and not r.get('terminal_anchor'))
    scenario=read_json(FIFTH/'raw/datasets'/f'{cid}-s{r["scenario_seed"]}.dataset.json')['scenarios'][0]
    base=read_json(OUTPUT/'manifest.json')['baseline_config'];pa=parameters(base,rho_levels(scenario,base)['rho_max'])
    sim,orders=matching_state(scenario,pa,base['solver']);_,_,states=simulate(sim,orders,collect_times=[r['time']])
    donor=read_json(FIFTH/'raw/datasets'/f'{cid}-s{r["donor_seed"]}.dataset.json')['scenarios'][0]
    _,donororders=matching_state(donor,pa,base['solver']);future=future_orders(orders,donororders,r['time'],r['mc_replication'])
    q=int(r['q']);state=states[r['time']];a,b=resource_world(state,r['hub'],q-1),resource_world(state,r['hub'],q)
    pair_audit(a,b,r['hub']);x,_,_=simulate(a,future,first_begun=True);y,_,_=simulate(b,future,first_begun=True)
    assert abs(y['realized_profit']-x['realized_profit']-r['target'])<1e-6
    return dict(runs=out,target=dict(condition=cid,time=r['time'],q=q,seed=r['scenario_seed'],
        old_target=r['target'],reproduced_target=y['realized_profit']-x['realized_profit'],difference=0))

def run(cfg):
    p=OUTPUT/'qa/baseline_regression.json'
    if p.exists():assert read_json(p)['status']=='PASSED';return
    toy_audit();results,wall=map_jobs(baseline_job,cfg['conditions'],'Sixth phase0',cfg['max_parallel'])
    report=dict(status='PASSED',runs=16,rows=[r for item in results for r in item['runs']],
        resource_target_checks=[item['target'] for item in results],wall_seconds=wall)
    atomic_json(p,report);atomic_json(OUTPUT/'manifests/baseline_manifest.json',report)
