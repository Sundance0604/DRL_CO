"""Feasibility witnesses preserve original constraints; probes never mutate state."""
import copy
import hashlib
import sys
import gurobipy as gp
from gurobipy import GRB
from model_families.single.value.route_milp import ValueRouteMILPPlanner
from model_families.single.planning import PeriodAction,VehicleAction,compile_action
from model_families.single.runner import state_snapshot,serialize_plan
from model_families.single.rolling.contracts import WindowRequest
from model_families.single.rolling.controller import observed_state
from experiments.cfs_structured_ppo_q.control import QProvider,QSignal,upgrade,replay
from experiments.cfs_structured_ppo_q.training import Policy
from experiments.cfs_direct_ppo_v3.network import DirectPolicy
from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
from experiments.cfs_direct_ppo_diagnosis.common import restore,canonical,finish
from .common import *


def constraints_hash(m):
    h=hashlib.sha256();a=m.getA()
    for v in (a.indptr,a.indices,a.data):h.update(v.tobytes())
    h.update(str([(c.Sense,c.RHS) for c in m.getConstrs()]).encode())
    h.update(str([(v.LB,v.UB,v.VType) for v in m.getVars()]).encode())
    return h.hexdigest()


class FeasibilityPlanner(ValueRouteMILPPlanner):
    def __init__(self,provider,summary,budget):super().__init__(provider,summary);self.budget=budget;self.certificate=None
    def _plan(self,request,m):
        captured={}
        def trace(frame,event,arg):
            if frame.f_code is not ValueRouteMILPPlanner._plan.__code__:return None
            if event=='return':captured['chosen']=frame.f_locals['chosen']
            return trace
        previous=sys.gettrace()
        if previous is not None:raise RuntimeError('do not replace another trace')
        sys.settrace(trace)
        try:base=super()._plan(request,m)
        finally:sys.settrace(previous)
        if base.diagnostics['status']!='OPTIMAL':raise RuntimeError('original solve not optimal')
        cs=base.diagnostics['release_coefficients'];vs=m.getVars();offset=len(vs)-len(cs)
        if any(abs(vs[offset+i].Obj-c['value'])>1e-7 for i,c in enumerate(cs)):raise RuntimeError('slot map')
        original=constraints_hash(m);aux=m.copy()
        try:
            if constraints_hash(aux)!=original:raise RuntimeError('copy changed feasible set')
            av=aux.getVars();q2=[av[offset+i] for i,c in enumerate(cs) if c['supply_slot']==2]
            aux.setObjective(gp.quicksum(q2),GRB.MAXIMIZE)
            self.budget.claim();aux.optimize()
            if aux.Status!=GRB.OPTIMAL:raise RuntimeError('q feasibility not certified OPTIMAL')
            if constraints_hash(aux)!=original:raise RuntimeError('aux changed constraints')
            selected=[]
            for e in captured['chosen']:
                if av[e['y'].index].X<.5 or e['action'] is None:continue
                a=e['action'].vehicles[0]
                ids=tuple(oid for oid,x in e['assignments'] if av[x.index].X>.5)
                selected.append(VehicleAction(a.vehicle_id,a.route,ids,a.reposition_to))
            info=copy.deepcopy(base.diagnostics);info.update(diagnostic_only=True,feasibility_q2_groups=round(aux.ObjVal))
            plan=compile_action(request.state,PeriodAction(request.start,tuple(selected)),info)
            serial=serialize_plan(plan,request.start,'feasibility-public-state',info)
            slots=[dict(time=c['time'],hub=c['hub'],q=c['supply_slot'],occupied=int(av[offset+i].X>.5)) for i,c in enumerate(cs)]
            self.certificate=dict(status='OPTIMAL',constraint_sha256=original,copy_constraint_sha256=constraints_hash(aux),
                max_q2_groups=round(aux.ObjVal),q_active=aux.ObjVal>.5,slot_occupancy=slots,witness_plan=serial,
                original_coefficients=cs,scope='original H1 route macro-action MILP, not all possible future controls')
        finally:aux.dispose()
        return base


def public_episode(spec,budget=None):
    scene=next(s for s in read_json(P10/'data'/W40/'validation.json')['scenarios'] if s['seed']==spec['scenario_seed'])
    record=read_json(spec['source']);ep,o=restore(scene,cfg(W40),record,spec['period'],budget)
    return upgrade(ep),o,scene,record


def select():
    load();budget=Budget();pool=[]
    if (TARGET/'qa/selection.json').exists():return
    try:
        with no_mc_guard():
            policy=Policy(P20/'training'/W40/'30001/inference-0008')
            for seed in (21008,21009,21010,21011):
                source=P20/'training'/W40/'30001'/f'validation-s{seed}.json'
                for t in (4,8,12,16,20):
                    budget();spec=dict(scenario_seed=seed,period=t,source=str(source),source_sha256=sha(source))
                    ep,o,scene,record=public_episode(spec,budget);before=state_snapshot(ep._sim)
                    signal=QSignal(tuple(policy.act(o)),o.period,o.revision,o.hash)
                    provider=QProvider(ep.schema,signal,ep._sim.T,len(ep._sim.vehs))
                    planner=FeasibilityPlanner(provider,dict(T=24,period=t),budget)
                    budget.claim();planner.plan(WindowRequest(observed_state(ep._sim),t,t+1,t+1,0,ep._check))
                    if state_snapshot(ep._sim)!=before:raise RuntimeError('selection mutated state')
                    cert=planner.certificate
                    occupied=[r for r in cert['slot_occupancy'] if r['occupied'] and r['q']==2]
                    hub=sorted((r['time'],r['hub']) for r in occupied)[0][1] if occupied else ep.schema.hubs[0]
                    spec.update(observation_hash=o.hash,state_hash=__import__('model_families.single.strategic.contracts',fromlist=['identity']).identity(before),
                        q_active=cert['q_active'],max_q2_groups=cert['max_q2_groups'],fixed_hub=hub,
                        idle_vehicles=sum(v['phase']=='idle' for v in before['vehicles']),pending_orders=len(o.orders))
                    write_json(TARGET/'raw/selection'/f's{seed}-t{t}.json',dict(spec=spec,state=before,certificate=cert));pool.append(spec)
        active=[s for s in pool if s['q_active']][:8];controls=[s for s in pool if not s['q_active']][:2]
        chosen=sorted([*active,*controls],key=lambda s:(s['scenario_seed'],s['period']))
        write_json(TARGET/'states-frozen.json',dict(frozen_at=now(),selected=chosen,candidate_pool=pool,
            requested=dict(q_active=8,q_only=2),actual=dict(q_active=len(active),q_only=len(controls)),
            rule='first feasible-active and first infeasible within fixed seed/time pool, never by profit'))
        write_json(TARGET/'qa/selection.json',dict(status='PASSED',candidates=20,states=len(chosen),q_active=len(active),q_only=len(controls),
            exact_original_constraint_hashes=True,new_calls=40))
        print('selected',len(active),'q-active',len(controls),'q-only',flush=True)
    finally:budget.save()


def probes():
    load();budget=Budget();chosen=read_json(TARGET/'states-frozen.json')['selected'];cases=[];count=0
    if (TARGET/'qa/probes.json').exists():return
    from model_families.single.strategic.contracts import identity
    try:
        with no_mc_guard():
            continuation=DirectPolicy(P10/'training'/W40/'B/30001/inference-0008')
            class Common:
                def act(self,o):return list(continuation.act(o))+[0.]*10
            for state_index,spec in enumerate(chosen):
                first=None
                for seed in SEEDS:
                    qpol=Policy(P20/'training'/W40/str(seed)/'inference-0008')
                    old=DirectPolicy(P10/'training'/W40/'B'/str(seed)/'inference-0008')
                    ep,o,scene,record=public_episode(spec,budget);before=state_snapshot(ep._sim)
                    if identity(before)!=spec['state_hash'] or o.hash!=spec['observation_hash']:raise RuntimeError('frozen state differs')
                    a=qpol.act(o);zero=a[:10]+[0.]*10
                    controls=[('A_learned',a),('B_beta0',zero),('D_old10',list(old.act(o))+[0.]*10)]
                    i=ep.schema.hubs.index(spec['fixed_hub'])
                    for direction in ('single','all'):
                        for slope in (-.6,-.2,.2,.6):
                            beta=[0.]*10
                            if direction=='single':beta[2*i:2*i+2]=[slope,slope]
                            else:beta=[slope]*10
                            controls.append((f'C_{direction}_{slope}',a[:10]+beta))
                    rows=[]
                    for label,signal in controls:
                        p=ep.probe(signal);count+=1
                        out=any(abs(c['value'])>ep.schema.scales[ep.schema.hubs.index(c['hub'])]+1e-7 for c in p['coefficients'])
                        rows.append(dict(control=label,signal=signal,probe=p,canonical=canonical(p['plan'],before),
                            any_constructed_price_outside_old_envelope=out))
                    base=rows[1]['canonical']
                    for r in rows:r['changed_vs_beta0']=r['canonical']!=base
                    if state_snapshot(ep._sim)!=before:raise RuntimeError('same-state probe mutated state')
                    write_json(TARGET/'raw/probes'/f's{spec["scenario_seed"]}-t{spec["period"]}-m{seed}.json',
                        dict(spec=spec,training_seed=seed,state=before,rows=rows))
                    eligible=next((r for r in rows if r['control']!='D_old10' and r['changed_vs_beta0']),None)
                    if first is None and eligible is not None:first=(seed,eligible,rows[1])
                # At most2 distinct states; selection before any cash evaluation.
                if first is not None and len(cases)<2:
                    seed,change,base=first;profits=[];cash=[]
                    for label,row in (('nonzero',change),('beta0',base)):
                        ep,o,scene,record=public_episode(spec,budget)
                        r=finish(ep,row['signal'],Common());r['replay']=replay(scene,cfg(W40),r)
                        profits.append(r['ledger']['realized_profit']);cash.append(r['transitions'][spec['period']]['reward_USD'])
                        write_json(TARGET/'raw/counterfactual'/f'case{len(cases)+1}-{label}.json',r)
                    cases.append(dict(case=len(cases)+1,scenario_seed=spec['scenario_seed'],period=spec['period'],training_seed=seed,
                        control=change['control'],profit_difference_USD=profits[0]-profits[1],current_cash_difference_USD=cash[0]-cash[1],
                        later_cash_difference_USD=(profits[0]-profits[1])-(cash[0]-cash[1]),
                        original_envelope=not change['any_constructed_price_outside_old_envelope'],
                        selection='first2 distinct switch states, then first trainingseed/control, no profit selection'))
        write_json(TARGET/'qa/probes.json',dict(status='PASSED',probe_count=count,state_model_pairs=len(chosen)*3,cases=cases,
            counterfactual_missing_reason=None if cases else 'no eligible canonical switches'))
        print('probes',count,'CF cases',len(cases),flush=True)
    finally:budget.save()
