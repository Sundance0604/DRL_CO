"""Complete period actions, common verification and frozen historical scores."""
from collections import Counter
from experiment_core.storage import digest, read_json
from model_families.single.planning import PeriodAction, VehicleAction, compile_action, decision_vehicles, route_candidates
from model_families.single.runner import serialize_plan, state_snapshot
from model_families.single.value.route_milp import ValueRouteMILPPlanner, first_handoff
from model_families.single.rolling.contracts import WindowRequest
from model_families.single.rolling.controller import observed_state
from experiments.cfs_resource_value_state_bucket_exact_accounting.protocol import provider
from experiments.cfs_resource_value_state_bucket_exact_accounting.pressure import summary
from experiments.cfs_resource_value_state_bucket_exact_accounting.exact_accounting import solve_myopic_exact, reconcile_integer_action
from experiments.cfs_resource_policy_iteration_damping.policies import load as load_a2
from experiments.cfs_resource_value_state_bucket_exact_accounting.policies import load as load_b
from experiments.cfs_rollout_target_alignment.forced import forced_plan, route_key as force_key
from experiments.cfs_resource_policy_iteration.action_analysis import canonical
from .common import B, sha

def historical(cid):
    policies=dict(M0=load_b('B0',cid),pi1=load_a2('U1',cid),D4=load_a2('D4',cid,'025'),B2=load_b('B2',cid),
                  pi3=load_a2('U3',cid))
    prior=read_json(B/'tables/priors.json')[cid]
    return policies,prior

def period_action(serial):
    routes={int(r['vehicle_id']):{k:v for k,v in r.items() if k!='vehicle_id'} for r in serial['routes']}
    repos={int(r['vehicle_id']):r['hub'] for r in serial['repositions']};ids={}
    for a in serial['assignments']:ids.setdefault(int(a['vehicle_id']),[]).append(a['order_id'])
    return PeriodAction(serial['period'],tuple(VehicleAction(v,routes.get(v),tuple(sorted(ids.get(v,[]))),repos.get(v))
        for v in sorted(set(routes)|set(repos)|set(ids))))

def physical_action(serial):
    return canonical(serial)

def verify(state,serial):
    before=digest(state_snapshot(state));info={}
    p=compile_action(state,period_action(serial),info)
    checked=serialize_plan(p,state.t,before,info)
    assert physical_action(serial)==physical_action(checked),'Whole action compilation changed assignments/routes/expiration'
    assert abs(float(serial['operating_profit'])-p[0])<=1e-6,'Integer cash mismatch'
    assert digest(state_snapshot(state))==before
    return p,checked

def decide(state,policy,prior):
    before=digest(state_snapshot(state));info={};price=provider(policy)
    if price is None:plan=solve_myopic_exact(state,info)
    else:
        request=WindowRequest(observed_state(state),state.t,state.t+1,state.t+1,0,lambda:None)
        solution=ValueRouteMILPPlanner(price,summary(state,prior)).plan(request)
        plan=compile_action(state,solution.actions.get(state.t,PeriodAction(state.t)),info);info.update(solution.diagnostics)
    assert info['status']=='OPTIMAL','Nonoptimal solve must stop, not become a sample'
    serial=serialize_plan(plan,state.t,before,info);verify(state,serial)
    assert digest(state_snapshot(state))==before
    return serial

def releases(state,serial):
    """Mirror H1 sink semantics, including locked vehicles and exact terminal exits."""
    byid={int(r['vehicle_id']):r for r in serial['routes']}
    repos={int(r['vehicle_id']):r['hub'] for r in serial['repositions']}
    deciding={v.id for v in decision_vehicles(state.t,state.vehs)};rows=[]
    for v in state.vehs:
        if v.id in byid:
            r=byid[v.id];when=r['t0']+state.net.tau[r['p'],r['e']];hub=r['e'];kind='selected_route'
        elif v.id in repos:
            hub=repos[v.id];when=state.t+state.net.tau[v.loc,hub];kind='selected_reposition'
        elif v.id in deciding:
            assert v.phase=='idle','Active continuation must appear as a full compiled route'
            when=state.t+1;hub=v.loc;kind='idle_wait'
        else:
            handoff,w=first_handoff(state,v)
            if w.phase=='trip':
                r=next(r for r in route_candidates(handoff,state.net,state.pool,w,state.parameters['cross_hub'],
                    state.parameters['empty_cost'],state.parameters) if not r['new'])
                when=r['t0']+state.net.tau[r['p'],r['e']];hub=r['e']
            else:when=handoff;hub=w.loc
            kind='locked_supply'
        rows.append(dict(vehicle_id=v.id,time=int(when),hub=hub,kind=kind,terminal=when>=state.T))
    assert len(rows)==len(state.vehs) and len({r['vehicle_id'] for r in rows})==len(state.vehs)
    assert all(r['time']>state.t for r in rows)
    return rows

def score(state,serial,policy,prior):
    price=provider(policy);online=summary(state,prior);events=releases(state,serial)
    grouped=Counter((r['time'],r['hub']) for r in events);coeff=[]
    for (when,hub),n in sorted(grouped.items()):
        for q in range(1,n+1):
            value=0. if price is None or when>=state.T else price.marginal_release_value(when,hub,q,online)
            coeff.append(dict(time=when,hub=hub,q=q,value_USD=value,terminal=when>=state.T))
    g=serial['operating_profit'];bonus=sum(c['value_USD'] for c in coeff)
    tie=state.parameters.get('tie_break',1)*sum(r['delay'] for r in serial['routes'])
    return dict(g_booked_USD=g,value_bonus_USD=bonus,L_V_USD=g+bonus,tie_break_USD=tie,
        exact_optimizer_score_USD=g+bonus-tie,release_events=events,occupied_coefficients=coeff,
        lookup_queries=[] if price is None else getattr(price,'query_log',[]),
        policy_hash=policy['policy_hash'],table_hash=policy.get('table_hash'),
        past_J_ignored=True,J_minus_g_is_not_pure_release_cash=True)

def candidates(state,cid):
    policies,prior=historical(cid);out=[];missing=[]
    def add(label,serial):
        verify(state,serial);physical=physical_action(serial);ah=digest(physical)
        scores={k:score(state,serial,p,prior) for k,p in policies.items() if k!='pi3'}
        if label in ['pi1','D4','B2']:
            # Frozen solver objective contains the delay tie-break, not the reporting L_V.
            actual=serial['solver']['window_objective'];expected=scores[label]['exact_optimizer_score_USD']
            assert abs(actual-expected)<=1e-5,(label,actual,expected,'H1 sink score mismatch')
        old=next((a for a in out if a['action_hash']==ah),None)
        if old is not None:
            old['labels'].append(label);old['generating_solver'][label]=serial['solver'];return
        out.append(dict(action_hash=ah,labels=[label],serial=serial,physical=physical,scores=scores,
            generating_solver={label:serial['solver']}))
    for label in ['M0','pi1','D4','B2']:add(label,decide(state,policies[label],prior))
    # Alternate full joint solutions come from the SAME original period feasible set.
    # Only generation constrains one route. The resulting ENTIRE solution is frozen.
    baseline=next(a for a in out if 'M0' in a['labels']);poss=[]
    for v in decision_vehicles(state.t,state.vehs):
        if v.phase=='idle':poss.append(dict(vehicle_id=v.id,kind='wait',route_key=None))
        for r in route_candidates(state.t,state.net,state.pool,v,state.parameters['cross_hub'],state.parameters['empty_cost'],state.parameters):
            poss.append(dict(vehicle_id=v.id,kind='route',route_key=force_key(r)))
    # Pre-feasibility: impossible anchor/capacity choices are documented, never rolled out.
    from experiments.cfs_local_post_bayesian_value.targets import coalition
    legal=[]
    for p in poss:
        if p['kind']=='wait':legal.append(p);continue
        v=state.vehs[p['vehicle_id']]
        r=next(r for r in route_candidates(state.t,state.net,state.pool,v,state.parameters['cross_hub'],state.parameters['empty_cost'],state.parameters)
               if force_key(r)==p['route_key'])
        if coalition(state,v,r) is not None:legal.append(p)
    added=0
    for p in sorted(legal,key=digest):
        if added==2:break
        p=p|dict(candidate_hash=digest(p));info={};raw=forced_plan(state,p,info)
        checked=reconcile_integer_action(state,raw,info)
        serial=serialize_plan(checked,state.t,digest(state_snapshot(state)),info)
        n=len(out);add('alternative'+str(added+1),serial)
        if len(out)>n:added+=1
    if not decision_vehicles(state.t,state.vehs):missing.append('no_deciding_vehicle')
    if len(out)<2:missing.append('no_distinct_feasible_alternative')
    assert len(out)<=6
    return out,missing

def choose(actions,values):
    best=max(values.values());ties=sorted(a['action_hash'] for a in actions if best-values[a['action_hash']]<=1e-8)
    base=next(a['action_hash'] for a in actions if 'pi1' in a['labels'])
    return dict(action_hash=base if base in ties else ties[0],tie_set=ties,best_score=best)
