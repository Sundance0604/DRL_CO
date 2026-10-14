"""Non-solving delivery audit. Frozen numerical sources are never rewritten."""
import copy
from collections import Counter
import torch
from model_families.single.planning import compile_action
from model_families.single.runner import serialize_plan,state_snapshot
from model_families.single.strategic.contracts import identity
from model_families.single.value.route_milp import first_handoff
from experiments.cfs_ppo_strategic_control.study_execution import action_from_plan,physical_hash,close
from experiments.cfs_direct_ppo_diagnosis.common import canonical,observation
from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
from experiments.cfs_direct_ppo_v3.network import DirectPolicy
from experiments.cfs_structured_ppo_q.control import replay
from .states import public_episode
from .kl import fingerprint
from .common import *


def release_counts(sim,plan):
    """Project the compiled current actions to their release-group boundaries.

    H1 locked movement beyond the current boundary follows its committed route.
    This checks witness/selected slots, not a global multi-period feasible hull.
    """
    routes={int(r['vehicle_id']):r for r in plan['routes']}
    repos={int(r['vehicle_id']):r['hub'] for r in plan['repositions']}
    counts=Counter()
    for v in sim.vehs:
        if v.id in routes:
            r=routes[v.id];key=(r['t0']+sim.net.tau[r['p'],r['e']],r['e'])
        elif v.id in repos:
            hub=repos[v.id];key=(sim.t+sim.net.tau[v.loc,hub],hub)
        else:
            t,w=first_handoff(sim,v)
            if w.phase=='idle':key=(max(sim.t+1,t),w.loc)
            else:
                hub=w.route[-1];leave=t if w.hold is None else w.hold
                key=(leave+sim.net.tau[w.loc,hub],hub)
        if key[0]<sim.T:counts[key]+=1
    return counts


def validate_plan(sim,data,slots,coefficients,signal=None,objective=True):
    before=state_snapshot(sim);info=copy.deepcopy(data['solver'])
    compiled=compile_action(sim,action_from_plan(data),info)
    serial=serialize_plan(compiled,sim.t,data['state_hash'],info)
    if (physical_hash(serial)!=physical_hash(data) or info['accounting']!=data['accounting'] or
            not close(compiled[0],data['operating_profit'])):raise RuntimeError('compiled action/cash differs')
    counts=release_counts(sim,serial);occupied=Counter();seen=set()
    for slot in slots:
        key=(slot['time'],slot['hub']);q=slot['q'];tag=(*key,q)
        if tag in seen or slot['occupied'] not in (0,1):raise RuntimeError('invalid duplicate slot')
        seen.add(tag)
        if slot['occupied']!=int(q<=counts[key]):raise RuntimeError(f'physical release prefix differs t={sim.t} key={tag} slot={slot["occupied"]} physical_count={counts[key]}')
        occupied[key]+=slot['occupied']
    if +occupied!=counts:raise RuntimeError('release count/terminal gate mismatch')
    if len(slots)!=len(coefficients):raise RuntimeError('slot/coefficient length differs')
    for slot,c in zip(slots,coefficients):
        if (slot['time'],slot['hub'],slot['q'])!=(c['time'],c['hub'],c['supply_slot']):raise RuntimeError('slot order mismatch')
        if signal is not None:
            schema=cfg(W40)['action'];i=schema['hubs'].index(c['hub'])
            ell=(c['time']-sim.t)/(sim.T-sim.t);psi=(c['supply_slot']-1)/(len(sim.vehs)-1)
            alpha=(1-ell)*signal[2*i]+ell*signal[2*i+1]
            beta=(1-ell)*signal[10+2*i]+ell*signal[11+2*i]
            value=schema['scales'][i]*(alpha+beta*psi)
            if c['time']>=sim.T or abs(value-c['value'])>1e-7:raise RuntimeError('independent USD coefficient differs')
            if c['direct_price']!=c['value'] or any(k in c for k in ('base','residual')):raise RuntimeError('MC field/price mismatch')
    if objective:
        expected=compiled[1]+sum(c['value']*s['occupied'] for s,c in zip(slots,coefficients))
        if not close(expected,data['solver']['window_objective']):raise RuntimeError('cash/tie/strategic objective decomposition differs')
    if state_snapshot(sim)!=before:raise RuntimeError('audit changed restored state')
    return canonical(serial,before)


def audit():
    load();torch.set_num_threads(1);selected=read_json(TARGET/'states-frozen.json');certificates={}
    count=0;trigger_count=0;periods=0
    with no_mc_guard():
        for p in sorted((TARGET/'raw/selection').glob('*.json')):
            d=read_json(p);spec=d['spec'];ep,o,scene,record=public_episode(spec)
            if (identity(state_snapshot(ep._sim))!=spec['state_hash'] or o.hash!=spec['observation_hash'] or
                    sha(spec['source'])!=spec['source_sha256']):raise RuntimeError('selection source/state mismatch')
            c=d['certificate']
            if c['status']!='OPTIMAL' or c['constraint_sha256']!=c['copy_constraint_sha256']:raise RuntimeError('constraint certificate differs')
            try:validate_plan(ep._sim,c['witness_plan'],c['slot_occupancy'],c['original_coefficients'],objective=False)
            except Exception as exc:raise RuntimeError(f'{p}: {exc}') from exc
            q2=sum(s['q']==2 and s['occupied'] for s in c['slot_occupancy'])
            if q2!=c['max_q2_groups'] or (q2>0)!=c['q_active'] or q2!=spec['max_q2_groups']:raise RuntimeError('q-active certificate differs')
            certificates[(spec['scenario_seed'],spec['period'])]=spec
        pool=selected['candidate_pool']
        if len(pool)!=20 or any(s!=certificates[(s['scenario_seed'],s['period'])] for s in pool):raise RuntimeError('frozen pool differs')
        expected=sorted([*[s for s in pool if s['q_active']][:8],*[s for s in pool if not s['q_active']][:2]],key=lambda s:(s['scenario_seed'],s['period']))
        if expected!=selected['selected']:raise RuntimeError('prespecified public selection differs')
        eligible=[]
        for p in sorted((TARGET/'raw/probes').glob('*.json')):
            d=read_json(p);spec=d['spec'];ep,o,scene,record=public_episode(spec)
            if d['state']!=state_snapshot(ep._sim) or spec not in expected:raise RuntimeError('probe public state differs')
            base=next(r['canonical'] for r in d['rows'] if r['control']=='B_beta0')
            learned=next(r['signal'] for r in d['rows'] if r['control']=='A_learned')
            for r in d['rows']:
                probe=r['probe'];plan=probe['plan']
                if plan['solver']['status']!='OPTIMAL' or plan['state_hash']!=o.hash:raise RuntimeError('nonoptimal/stale probe')
                can=validate_plan(ep._sim,plan,plan['solver']['slot_occupancy'],probe['coefficients'],r['signal'])
                if can!=r['canonical'] or (can!=base)!=r['changed_vs_beta0'] or probe['action_hash']!=physical_hash(plan):raise RuntimeError('canonical/raw action differs')
                if r['control']!='D_old10' and r['signal'][:10]!=learned[:10]:raise RuntimeError('alpha changed in q contrast')
                inside=all(abs(c['value'])<=ep.schema.scales[ep.schema.hubs.index(c['hub'])]+1e-7 for c in probe['coefficients'])
                if inside==r['any_constructed_price_outside_old_envelope']:raise RuntimeError('price envelope differs')
                count+=1
            first=next((r for r in d['rows'] if r['control']!='D_old10' and r['changed_vs_beta0']),None)
            if first:eligible.append((spec,d['training_seed'],first))
        eligible.sort(key=lambda v:(v[0]['scenario_seed'],v[0]['period'],v[1]))
        expected_cases=[];seen=set()
        for spec,seed,row in eligible:
            key=(spec['scenario_seed'],spec['period'])
            if key not in seen:expected_cases.append((spec,seed,row));seen.add(key)
            if len(expected_cases)==2:break
        cases=read_json(TARGET/'qa/probes.json')['cases']
        if len(cases)!=len(expected_cases):raise RuntimeError('CF prespecified selection differs')
        continuation=DirectPolicy(P10/'training'/W40/'B/30001/inference-0008')
        for c,(spec,seed,row) in zip(cases,expected_cases):
            if (c['scenario_seed'],c['period'],c['training_seed'],c['control'])!=(spec['scenario_seed'],spec['period'],seed,row['control']):raise RuntimeError('CF was reselected')
            ep,o,scene,original=public_episode(spec);profits=[];cash=[]
            for label in ('nonzero','beta0'):
                r=read_json(TARGET/'raw/counterfactual'/f'case{c["case"]}-{label}.json')
                if r['scenario_seed']!=spec['scenario_seed'] or r['transitions'][:spec['period']]!=original['transitions'][:spec['period']]:raise RuntimeError('CF changed original prefix')
                for tr in r['transitions'][spec['period']+1:]:
                    signal=list(continuation.act(observation(tr['observation'])))+[0.]*10
                    if list(tr['signal']['values'])!=signal:raise RuntimeError('CF continuation policy differs')
                result=replay(scene,cfg(W40),r)
                if result!=r['replay']:raise RuntimeError('CF replay result differs')
                tr=r['transitions'][spec['period']]
                expected_signal=row['signal'] if label=='nonzero' else row['signal'][:10]+[0.]*10
                if list(tr['signal']['values'])!=expected_signal:raise RuntimeError('CF first signal differs')
                profits.append(r['ledger']['realized_profit']);cash.append(tr['reward_USD']);periods+=result['periods']
            differences=(profits[0]-profits[1],cash[0]-cash[1],profits[0]-profits[1]-cash[0]+cash[1])
            if any(not close(a,c[k]) for a,k in zip(differences,('profit_difference_USD','current_cash_difference_USD','later_cash_difference_USD'))):raise RuntimeError('CF cash decomposition differs')
    # Actual early-stop checkpoints, independently re-evaluate trigger parameters.
    from experiments.cfs_structured_ppo_q.network import build as build20
    from experiments.cfs_direct_ppo_training_ablation_v6.network import build as build10
    for p in sorted((TARGET/'raw/kl').glob('*.json')):
        d=read_json(p);s=d['summary']
        for c in d['checks']:
            if not c['stopping_trigger']:continue
            path=Path(c['trigger_snapshot'])
            if sha(path)!=c['trigger_snapshot_sha256']:raise RuntimeError('trigger snapshot hash differs')
            snapshot=torch.load(path,map_location='cpu',weights_only=True)
            net=build20(s['training_seed']) if s['method']=='PPO20' else build10('B',s['training_seed'])
            net.load_state_dict(snapshot['network']);net.eval()
            if fingerprint(net)!=c['parameter_sha256'] or snapshot['parameter_sha256']!=c['parameter_sha256']:raise RuntimeError('actual trigger parameters differ')
            source=P20/'training'/s['condition']/str(s['training_seed']) if s['method']=='PPO20' else P10/'training'/s['condition']/'B'/str(s['training_seed'])
            obs=[]
            for i in range((s['update']-1)*4+1,s['update']*4+1):
                f=source/(f'training-{i:04d}.json' if s['method']=='PPO20' else f'training-episode-{i:06d}.json')
                obs.extend(observation(tr['observation']) for tr in read_json(f)['transitions'])
            with torch.no_grad():mu,sigma,_=net([obs[i] for i in c['indices']])
            if mu.tolist()!=c['new_mu'] or sigma[0].tolist()!=c['new_sigma'] or (snapshot['epoch'],snapshot['batch_index'])!=(c['epoch'],c['batch_index']):raise RuntimeError('trigger forward evaluation differs')
            trigger_count+=1
    budget=read_json(TARGET/'qa/budget.json');expected_calls=40+count+sum(2*(24-c['period']) for c in cases)
    if expected_calls!=budget['calls']:raise RuntimeError('new solver-call reconciliation differs')
    write_json(TARGET/'qa/physical.json',dict(status='PASSED',at=now(),public_certificates=len(certificates),
        compiled_probe_actions=count,CF_full_days=len(cases)*2,CF_replayed_periods=periods,actual_trigger_snapshots=trigger_count,
        constraints='archived before/copy digest equality + unchanged numeric source + actual toy-copy regression; no new solves',
        hidden_future='both branches use identical frozen scenario and prefix; future only in simulator',new_calls_reconciled=expected_calls,
        source_sha256=sha(__file__)))
    print('physical QA PASSED',count,'probes',trigger_count,'triggers',periods,'CF periods',flush=True)


if __name__=='__main__':audit()
