"""Bounded post-hoc diagnostics. Never modifies or retrains parent models."""
import argparse,copy,time,math
from pathlib import Path
import numpy as np
import torch
from .common import *
from experiments.cfs_direct_ppo_v3.network import DirectPolicy,DirectNetwork
from experiments.cfs_direct_ppo_v3.training import run_record
from experiments.cfs_direct_ppo_v3.qa import no_mc_guard,replay
from model_families.single.runner import state_snapshot

class Budget:
    def __init__(self):self.started=time.monotonic();self.calls=0
    def __call__(self):
        import psutil
        if time.monotonic()-self.started>1200 or self.calls>6000:raise TimeoutError('registered 1200s/6000calls limit')
        if psutil.Process().memory_info().rss>2*2**30:raise MemoryError('2GiB limit')
        return False
    def claim(self,count=1):self.calls+=count;self()
    def result(self):return dict(calls=self.calls,seconds=time.monotonic()-self.started)

def check():
    m=read_json(TARGET/'manifest.json')
    if m['status'] in ('COMPLETED','FAILED','BUDGET_LIMITED'):raise RuntimeError('immutable terminal attempt')
    if m['source_files']!=source_files():raise RuntimeError('numeric source changed')
    verify_hashes(TARGET,m['inputs']);verify_parent();return m

def prepare():
    if TARGET.exists():raise FileExistsError('new attempt required')
    parent=verify_parent(True);selected=[]
    # Selection uses observed order counts and idle counts only, no profit/hidden future.
    for name in NAMES:
        for seed in SEEDS:
            ch=choice(name,seed);folder=PARENT/'training'/name/str(seed)
            for period in (4,12,20):
                candidates=[]
                for s in scenes(name,'validation')[:4]:
                    path=folder/f'validation-{ch["selected_update"]:04d}-s{s["seed"]}.json'
                    r=read_json(path);o=observation(r['transitions'][period]['observation'])
                    idle=sum(v.features[5] for v in o.vehicles)/len(o.vehicles)
                    candidates.append(dict(condition=name,seed=seed,scenario_seed=s['seed'],period=period,
                        pending=len(o.orders),idle_fraction=idle,source=path.relative_to(PARENT).as_posix(),observation_hash=o.hash))
                ordered=sorted(candidates,key=lambda x:(x['pending'],x['idle_fraction'],x['scenario_seed']))
                for r in (ordered[0],ordered[-1]):
                    if r not in selected:selected.append(r)
    TARGET.mkdir(parents=True)
    for f in ('protocol','m0','zero','pilot','diagnostics','sensitivity','counterfactual','analysis','figures','qa'):(TARGET/f).mkdir()
    protocol=dict(version='direct-failure-diagnosis/v1',post_hoc=True,no_training=True,selected_states=selected,
        pilot_states=[dict(condition=n,seed=30001,scenario_seed=21008,period=12,
            source=f'training/{n}/30001/validation-{choice(n,30001)["selected_update"]:04d}-s21008.json') for n in NAMES],
        pilot_offline_episodes_per_model=4,pilot_actor_states_per_model=12,
        full_actor_states_per_model=96,controls_per_state=15,maximum_counterfactual_cases=9,
        wall_seconds=1200,maximum_calls=6000,maximum_output_bytes=2*2**30,workers=1,solver_threads=1,
        bootstrap_replicates=20000,bootstrap_seed=53000,descriptive_family=9,
        CI='95 and Bonferroni99.444%; exploratory not new confirmatory evidence',
        selection='phase t4/12/20 public pending min/max within four validation scenarios; seed breaks ties',
        counterfactual='first canonically changed control in frozen order, at most one per model; same Direct continuation',
        gradient='new frozen final-recovery loss gradients on last4 training episodes, not historical gradients',
        historical_missing=['per-update actor/critic gradient split','per-update critic weights','per-update log_std snapshots except8/16/24/32'],
        budget_policy='pilot projection only determines proceed/stop; no subset adaptation on profit')
    write_json(TARGET/'protocol/preregistration.json',protocol)
    (TARGET/'protocol/task.md').write_text(Path('D:/download/writing-block (2).md').read_text(encoding='utf-8'),encoding='utf-8')
    (TARGET/'experiment_protocol.md').write_text((PACKAGE/'README.md').read_text(encoding='utf-8'),encoding='utf-8')
    write_json(TARGET/'manifest.json',dict(status='PREPARED',created_at=now(),source_files=source_files(),
        parent_manifest_sha256=sha(PARENT/'final_manifest.json'),parent_artifacts=parent['final_artifact_hashes'],
        inputs={p.relative_to(TARGET).as_posix():sha(p) for p in (TARGET/'protocol').glob('*')},core=source_identity()))
    write_json(TARGET/'qa/prepare.json',dict(status='PASSED',parent_files=len(parent['final_artifact_hashes']),states=len(selected)))
    print('PREPARED',len(selected),'states',flush=True)

def m0(budget):
    from experiments.cfs_ppo_strategic_control.study_execution import run_record as old_run,replay as old_replay
    check()
    if (TARGET/'qa/m0.json').exists():raise RuntimeError('M0 must not repeat')
    for name in NAMES:
        cfg=configuration(name);legacy=copy.deepcopy(cfg);legacy['action'].pop('version');legacy['baseline']=dict(method='zero')
        for s in scenes(name,'test'):
            budget.claim(24);r=old_run(s,legacy,lambda o:(0.,)*10,'M0');old_replay(s,legacy,r)
            with no_mc_guard():z=run_record(s,cfg,lambda o:(0.,)*10,'Direct-zero',budget)
            if r['ledger']!=z['ledger']:raise RuntimeError('M0/zero ledger mismatch')
            ep,_=restore(s,cfg,z,0)
            # Both histories must match every public state, and equivalent integer plan.
            for a,b in zip(r['transitions'],z['transitions'],strict=True):
                before=state_snapshot(ep._sim)
                if a['before_state_hash']!=b['before_state_hash'] or canonical(a['plan'],before)!=canonical(b['plan'],before):raise RuntimeError('M0/zero action mismatch')
                if any(q['value']!=0 for q in b['coefficients']):raise RuntimeError('zero coefficients nonzero')
                from experiments.cfs_ppo_strategic_control.study_execution import action_from_plan
                from model_families.single.planning import compile_action
                ep._sim.commit_plan(compile_action(ep._sim,action_from_plan(b['plan']),{}))
                if ep._sim.t<24:ep._publish()
            write_json(TARGET/'m0'/f'{name}-s{s["seed"]}.json',r);write_json(TARGET/'zero'/f'{name}-s{s["seed"]}.json',z)
        print('M0+zero 24 paired days',name,flush=True)
    write_json(TARGET/'qa/m0.json',dict(status='PASSED',M0_episodes=72,zero_episodes=72,all_replayed=True,all_zero_equivalent=True,**budget.result()))

def critic_statistics(records):
    predicted=[];mc=[];gae=[];adv=[];audit=0
    for r in records:
        rewards=[t['reward_USD']/1000 for t in r['transitions']];future=np.cumsum(rewards[::-1])[::-1]
        values=[v['value'] for v in r['learning']];carry=0.;next_value=0.
        for i in reversed(range(24)):
            carry=rewards[i]+next_value-values[i]+.95*carry
            if not math.isclose(carry,r['learning'][i]['advantage'],abs_tol=1e-6,rel_tol=1e-10):raise RuntimeError('GAE mismatch')
            next_value=values[i];audit+=1
        predicted.extend(values);mc.extend(future);gae.extend(v['return'] for v in r['learning']);adv.extend(v['advantage'] for v in r['learning'])
    x,y,g,a=map(np.asarray,(predicted,mc,gae,adv))
    ev=lambda target:float(1-np.var(target-x)/np.var(target)) if np.var(target)>0 else None
    return dict(episodes=len(records),states=len(x),critic_MC_RMSE_reward_units=float(np.sqrt(np.mean((x-y)**2))),
        critic_MC_EV=ev(y),critic_GAE_RMSE_reward_units=float(np.sqrt(np.mean((x-g)**2))),critic_GAE_EV=ev(g),
        value_mean=float(x.mean()),realized_RTG_mean=float(y.mean()),advantage_mean=float(a.mean()),advantage_std=float(a.std()),
        advantage_positive_fraction=float((a>0).mean()),GAE_recurrences_checked=audit)

def actor_diagnostic(name,seed,pilot=False):
    cfg=configuration(name);ch=choice(name,seed);folder=PARENT/'training'/name/str(seed)
    states=[]
    for s in scenes(name,'validation')[:4]:
        r=read_json(folder/f'validation-{ch["selected_update"]:04d}-s{s["seed"]}.json')
        for tr in r['transitions']:
            if not pilot or tr['period'] in (4,12,20):states.append(dict(scenario_seed=s['seed'],observation=tr['observation']))
    actor=[]
    with no_mc_guard():
        for update in (8,16,24,32):
            policy=DirectPolicy(folder/f'inference-{update:04d}',verify_parent()['source']);signals=np.array([policy.act(observation(s['observation'])) for s in states])
            average=signals.mean(0);points=[]
            for s,a in zip(states,signals):
                o=observation(s['observation']);points.append(dict(scenario_seed=s['scenario_seed'],period=o.period,pending=len(o.orders),
                    idle_fraction=float(sum(v.features[5] for v in o.vehicles)/len(o.vehicles)),observation_hash=o.hash,
                    signal=a.tolist(),near_USD=(a[::2]*np.asarray(cfg['action']['scales'])).tolist(),far_USD=(a[1::2]*np.asarray(cfg['action']['scales'])).tolist()))
            actor.append(dict(update=update,selected=update==ch['selected_update'],points=points,
                mean=average.tolist(),std=signals.std(0).tolist(),positive_fraction=(signals>0).mean(0).tolist(),
                negative_fraction=(signals<0).mean(0).tolist(),D_pi=float(np.abs(signals-average).mean()),
                near_far_difference_mean=float((signals[:,::2]-signals[:,1::2]).mean()),
                log_std=policy.network.log_std.detach().tolist(),clipped_std=policy.network.log_std.clamp(-5,2).exp().detach().tolist()))
    paths=list(sorted(folder.glob('training-episode-*.json')))
    if pilot:paths=paths[:4]
    records=[read_json(p) for p in paths]
    grouped=[]
    for update in sorted({r['update'] for r in records}):
        g=[r for r in records if r['update']==update];log=read_json(folder/f'update-{update:04d}.json')
        grouped.append(dict(update=update,**critic_statistics(g),KL=log['approximate_kl'],clip_fraction=log['clip_fraction'],
            gradient_norm_total_preclip=log['gradient_norm'],kl_early_stop=log['kl_early_stop'],training_mean_USD=log['training_mean_USD'],validation_mean_USD=log['validation_mean_USD']))
    gradients=None
    if not pilot:
        recovery=torch.load(folder/'recovery.pt',weights_only=False,map_location='cpu')
        with torch.random.fork_rng():network=DirectNetwork(5,64,True)
        network.load_state_dict(recovery['network']);network.eval()
        chosen=records[-4:];obs=[observation(t['observation']) for r in chosen for t in r['transitions']]
        latent=torch.tensor([v['latent'] for r in chosen for v in r['learning']]);old=torch.tensor([v['logp'] for r in chosen for v in r['learning']])
        advantage=torch.tensor([v['advantage'] for r in chosen for v in r['learning']]);advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
        target=torch.tensor([v['return'] for r in chosen for v in r['learning']])
        from model_families.single.learning.ppo.networks import squashed_log_probability
        mean,std,value=network(obs);ratio=(squashed_log_probability(mean,std,latent)-old).exp()
        actor_loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean();critic_loss=.5*(value-target).square().mean()
        params=list(network.parameters());ga=torch.autograd.grad(actor_loss,params,retain_graph=True,allow_unused=True);gc=torch.autograd.grad(critic_loss,params,allow_unused=True)
        a=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,ga)])
        c=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,gc)])
        gradients=dict(type='post_training_frozen_gradients_not_historical',states=len(obs),actor_norm=float(a.norm()),
            weighted_critic_norm=float(c.norm()),weighted_critic_actor_ratio=float(c.norm()/a.norm().clamp_min(1e-12)),
            full_parameter_cosine=float(torch.dot(a,c)/(a.norm()*c.norm()).clamp_min(1e-12)),optimizer_steps=0,
            note='post-update32 weights on pre-update32 rollout; diagnostic, not original training gradient')
    return dict(condition=name,seed=seed,pilot=pilot,actor=actor,critic=grouped,pooled_critic=critic_statistics(records),gradient_diagnostic=gradients)

def state_probes(spec,index,budget,output):
    name,seed=spec['condition'],spec['seed'];cfg=configuration(name);s=next(s for s in scenes(name,'validation') if s['seed']==spec['scenario_seed'])
    record=read_json(PARENT/spec['source']);ep,o=restore(s,cfg,record,spec['period'],budget)
    before=state_snapshot(ep._sim)
    with no_mc_guard():
        policy=DirectPolicy(PARENT/choice(name,seed)['checkpoint'],verify_parent()['source']);det=policy.act(o);rows=[]
        for control in controls(det,policy.network,o,index):
            before_hash=identity(state_snapshot(ep._sim));start=time.perf_counter();p=ep.probe(control['signal'])
            if before_hash!=identity(state_snapshot(ep._sim)):raise RuntimeError('probe mutated state')
            rows.append(dict(**control,probe=p,canonical=canonical(p['plan'],before),seconds=time.perf_counter()-start))
    result=dict(spec=spec,state=before,observation=asdict(o),rows=rows,source_SHA=sha(PARENT/spec['source']))
    write_json(output,result);return result

def counterfactual(result,index,budget,folder):
    spec=result['spec'];name,seed=spec['condition'],spec['seed'];reference=result['rows'][1]
    changed=next((r for r in result['rows'] if r['canonical']!=reference['canonical']),None)
    if changed is None:return None
    cfg=configuration(name);s=next(s for s in scenes(name,'validation') if s['seed']==spec['scenario_seed']);record=read_json(PARENT/spec['source'])
    outputs=[]
    with no_mc_guard():
        policy=DirectPolicy(PARENT/choice(name,seed)['checkpoint'],verify_parent()['source'])
        for branch,control in [('reference',reference),('variant',changed)]:
            ep,o=restore(s,cfg,record,spec['period'],budget);first=ep._reward.previous
            r=finish(ep,control['signal'],policy);r['replay']=replay(s,cfg,r)
            if canonical(r['transitions'][spec['period']]['plan'],result['state'])!=control['canonical']:raise RuntimeError('counterfactual forced action mismatch')
            if branch=='reference' and r['ledger']!=record['ledger']:raise RuntimeError('deterministic continuation regression mismatch')
            r.update(branch=branch,start_period=spec['period'],before_realized_USD=first,forced_control=control['kind'])
            write_json(folder/f'case{index:03d}-{branch}.json',r);outputs.append(r)
    a,b=outputs
    return dict(condition=name,seed=seed,scenario_seed=s['seed'],period=spec['period'],variant=changed['kind'],
        reference_profit_USD=a['ledger']['realized_profit'],variant_profit_USD=b['ledger']['realized_profit'],
        difference_USD=b['ledger']['realized_profit']-a['ledger']['realized_profit'],
        current_realized_difference_USD=b['transitions'][spec['period']]['reward_USD']-a['transitions'][spec['period']]['reward_USD'],
        current_booked_difference_USD=changed['probe']['plan']['operating_profit']-reference['probe']['plan']['operating_profit'],
        hidden_future_SHA=identity([o for o in s['orders'] if o['book_time']>spec['period']]),state_SHA=identity(result['state']),
        reference=(folder/f'case{index:03d}-reference.json').relative_to(TARGET).as_posix(),
        variant_record=(folder/f'case{index:03d}-variant.json').relative_to(TARGET).as_posix())

def pilot(budget):
    check();p=read_json(TARGET/'protocol/preregistration.json');started=time.perf_counter();cases=[]
    if (TARGET/'qa/pilot.json').exists():raise RuntimeError('pilot must not repeat')
    for name in NAMES:
        for seed in SEEDS:write_json(TARGET/'pilot'/f'actor-{name}-{seed}.json',actor_diagnostic(name,seed,True))
    offline=time.perf_counter()-started;probe_start=time.perf_counter();calls=budget.calls
    for index,spec in enumerate(p['pilot_states']):
        r=state_probes(spec,index,budget,TARGET/'pilot'/f'probes-{index}.json')
        case=counterfactual(r,index,budget,TARGET/'pilot')
        if case:cases.append(case)
    used=budget.calls-calls;probe_seconds=time.perf_counter()-probe_start
    # 32x pilot training records; up to54x15 probes and9x48 continuation calls.
    projection=offline*32+probe_seconds/max(used,1)*(54*15+9*48)
    write_json(TARGET/'qa/pilot.json',dict(status='PASSED' if projection*2<1200 else 'BUDGET_LIMITED',
        offline_seconds=offline,probe_seconds=probe_seconds,pilot_calls=used,projected_remaining_seconds=projection,
        conservative_remaining_seconds=projection*2,planned_remaining_call_bound=1242,cases=cases,**budget.result()))
    if projection*2>1200:raise TimeoutError('pilot projected bound')
    print('PILOT PASSED',projection,'projected seconds',used,'pilot calls',flush=True)

def diagnose(budget):
    m=check();p=read_json(TARGET/'protocol/preregistration.json')
    if read_json(TARGET/'qa/pilot.json')['status']!='PASSED' or (TARGET/'qa/diagnostics.json').exists():raise RuntimeError('stage order/repetition')
    for name in NAMES:
        for seed in SEEDS:
            write_json(TARGET/'diagnostics'/f'{name}-{seed}.json',actor_diagnostic(name,seed))
            print('OFFLINE',name,seed,flush=True)
    used_models=set();cases=[]
    for i,spec in enumerate(p['selected_states']):
        r=state_probes(spec,i,budget,TARGET/'sensitivity'/f'state{i:03d}.json')
        model=(spec['condition'],spec['seed'])
        if model not in used_models:
            case=counterfactual(r,i,budget,TARGET/'counterfactual')
            if case:cases.append(case);used_models.add(model)
        print('PROBES',i+1,'/',len(p['selected_states']),flush=True)
    write_json(TARGET/'counterfactual/results.json',cases)
    budget();size=sum(p.stat().st_size for p in TARGET.rglob('*') if p.is_file())
    if size>2*2**30:raise MemoryError('2GiB result bound')
    write_json(TARGET/'qa/diagnostics.json',dict(status='PASSED',models=9,states=len(p['selected_states']),counterfactual_cases=len(cases),
        output_bytes=size,no_training=True,all_counterfactual_replayed=True,**budget.result()))
    m.update(status='NUMERICAL_COMPLETE_QA_PENDING',numerical_completed_at=now());write_json(TARGET/'manifest.json',m)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','all'));args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import psutil,os
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    try:
        if args.stage=='prepare':prepare()
        else:
            budget=Budget();m0(budget);pilot(budget);diagnose(budget)
    except BaseException as exc:
        if TARGET.exists() and args.stage!='prepare':
            write_json(TARGET/'qa/stop.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc)))
            m=read_json(TARGET/'manifest.json');m.update(status='FAILED',stopped_at=now());write_json(TARGET/'manifest.json',m)
        raise

if __name__=='__main__':main()
