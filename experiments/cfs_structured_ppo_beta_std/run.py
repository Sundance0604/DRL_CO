import argparse
import random
import time
import numpy as np
import torch
from model_families.single.learning.ppo.config import PPOConfig
from model_families.single.learning.ppo.environment import PPOEnvironment
from model_families.single.learning.ppo.backend import advantages
from model_families.single.learning.ppo.networks import sample_action
from model_families.single.runner import state_snapshot
from experiments.cfs_direct_ppo_training_ablation_v6.backend import update,full_returns
from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
from experiments.cfs_structured_ppo_q.control import episode,replay
from experiments.cfs_structured_ppo_q.network import build
from experiments.cfs_structured_ppo_q.training import checkpoint,Policy,run_record
from experiments.cfs_structured_ppo_q_mechanism.kl import Capture,fingerprint
from experiments.cfs_structured_ppo_q_mechanism.states import public_episode
from experiments.cfs_direct_ppo_diagnosis.common import canonical
from .common import *


def modified(seed=30001):
    net=build(seed)
    with torch.no_grad():net.log_std[10:].fill_(-.5)
    return net


def train():
    m=load()
    if (TARGET/'qa/training.json').exists():return
    if (TARGET/'training/recovery.pt').exists():raise RuntimeError('partial training must not be silently restarted')
    c=configuration();config=PPOConfig.model_validate(c['ppo']);net=modified();old=build(30001)
    for key,value in net.state_dict().items():
        ref=old.state_dict()[key]
        if key=='log_std':
            if not torch.equal(value[:10],ref[:10]) or not torch.equal(value[10:],torch.full((10,),-.5)):raise RuntimeError('initial scale intervention differs')
        elif not torch.equal(value,ref):raise RuntimeError('another initial weight changed')
    torch.save(dict(old=old.state_dict(),new=net.state_dict()),TARGET/'snapshots/initial.pt')
    torch.save(net.state_dict(),TARGET/'training/initial.pt')
    write_json(TARGET/'qa/initialization.json',dict(status='PASSED',only_beta_log_std_differs=True,
        old_fingerprint=fingerprint(old),new_fingerprint=fingerprint(net),snapshot_sha256=sha(TARGET/'snapshots/initial.pt')))
    optim=(torch.optim.Adam(net.parameters(),lr=config.learning_rate),)
    sampling=torch.Generator().manual_seed(31002);batches=torch.Generator().manual_seed(31003);data=random.Random(31004)
    scenes=read_json(ABLATION/'data'/W40/'train.json')['scenarios'];schedule=read_json(TARGET/'protocol.json')['training_schedule']
    budget=Budget();count=0
    try:
        with no_mc_guard():
            for u in range(1,9):
                started=time.monotonic();rows=[];profits=[];all_learning=[]
                for _ in range(4):
                    scene=scenes[data.randrange(len(scenes))]
                    if scene['seed']!=schedule[count]:raise RuntimeError('historical scenario sequence differs')
                    ep=episode(scene,c,budget);env=PPOEnvironment(ep,config.reward_scale);obs,_=env.reset();daily=[];learning=[];rewards=[];values=[]
                    while True:
                        noise=torch.Generator().set_state(sampling.get_state());eps=torch.randn((1,20),generator=noise)[0]
                        a,z,lp,value=sample_action(net,obs,sampling)
                        nxt,r,done,truncated,_=env.step(a)
                        if truncated:raise RuntimeError('partial episode')
                        daily.append(dict(observation=obs,latent=z,logp=lp.item()))
                        learning.append(dict(latent=z.tolist(),action=a.tolist(),logp=lp.item(),value=value.item(),standard_epsilon=eps.tolist()))
                        rewards.append(r);values.append(value.item())
                        if done:break
                        obs=nxt
                    if len(daily)!=24:raise RuntimeError('full active horizon differs')
                    adv,returns=advantages(rewards,values,config.gae_lambda,config.gamma);rtg=full_returns(rewards)
                    for row,l,av,ret,future in zip(daily,learning,adv,returns,rtg):row.update(advantage=av,**{'return':ret});l.update(advantage=av,RTG=future,**{'return':ret})
                    count+=1
                    record=dict(scenario_id=scene['id'],scenario_seed=scene['seed'],training_seed=30001,update=u,
                        transitions=ep.records,ledger=ep.final_ledger,metrics=ep.metrics,learning=learning)
                    write_json(TARGET/'training'/f'training-{count:04d}.json',record)
                    rows.extend(daily);all_learning.extend(learning);profits.append(ep.final_ledger['realized_profit'])
                global_rng=torch.get_rng_state().clone()
                with Capture(net,rows,TARGET/'snapshots',u) as capture:diag=update(net,optim,rows,config,batches,'B')
                if not torch.equal(global_rng,torch.get_rng_state()):raise RuntimeError('KL capture consumed global RNG')
                if len(capture.checks)!=diag['minibatches']+int(diag['kl_early_stop']):raise RuntimeError('true check count differs')
                predicted=np.array([r['value']*10000 for r in all_learning]);targets=np.array([r['RTG']*10000 for r in all_learning])
                diag.update(update=u,training_mean_USD=float(np.mean(profits)),log_std=net.log_std.detach().tolist(),
                    critic_RMSE_USD=float(np.sqrt(np.mean((targets-predicted)**2))),
                    critic_EV=float(1-np.var(targets-predicted)/np.var(targets)) if np.var(targets)>0 else None,
                    action_mean=np.array([r['action'] for r in all_learning]).mean(0).tolist(),update_wall_seconds=time.monotonic()-started)
                write_json(TARGET/'training'/f'update-{u:04d}.json',diag)
                write_json(TARGET/'training/kl'/f'update-{u:04d}.json',dict(checks=capture.checks,final_fingerprint=fingerprint(net)))
                recovery=dict(network=net.state_dict(),optimizer=optim[0].state_dict(),sampling_rng=sampling.get_state(),
                    minibatch_rng=batches.get_state(),data_rng=data.getstate(),update=u,config=c,source=m['source'])
                torch.save(recovery,TARGET/'training/recovery.pt');torch.save(recovery,TARGET/'snapshots'/f'update-{u:04d}.pt')
                budget.save();print('BETA STD update',u,'steps',diag['minibatches'],'stop',diag['kl_early_stop'],flush=True)
            net.eval();checkpoint(TARGET/'training/inference-0008',net,c,m['source'])
        write_json(TARGET/'qa/training.json',dict(status='PASSED',episodes=count,updates=8,periods=768,
            checkpoint_sha256=sha(TARGET/'training/inference-0008/actor.pt'),selection='fixed endpoint; no validation winner'))
        m['status']='MODEL_FROZEN';write_json(TARGET/'manifest.json',m)
    finally:budget.save()


def probes():
    load()
    if (TARGET/'qa/probes.json').exists():return
    policy=Policy(TARGET/'training/inference-0008');old=Policy(OLD_TRAIN/'inference-0008');budget=Budget();count=0
    try:
        with no_mc_guard():
            for spec in read_json(MECHANISM/'states-frozen.json')['selected']:
                ep,o,scene,record=public_episode(spec,budget);before=state_snapshot(ep._sim)
                frozen=read_json(MECHANISM/'raw/probes'/f's{spec["scenario_seed"]}-t{spec["period"]}-m30001.json')
                a=next(r for r in frozen['rows'] if r['control']=='A_learned');a0=next(r for r in frozen['rows'] if r['control']=='B_beta0')
                if a['signal']!=old.act(o) or identity(before)!=spec['state_hash']:raise RuntimeError('old deterministic probe identity differs')
                signal=policy.act(o);rows=[]
                for label,values in (('new',signal),('new_beta0',signal[:10]+[0.]*10)):
                    p=ep.probe(values);rows.append(dict(control=label,signal=values,probe=p,canonical=canonical(p['plan'],before)));count+=1
                if state_snapshot(ep._sim)!=before:raise RuntimeError('probe changed original state')
                write_json(TARGET/'probes'/f's{spec["scenario_seed"]}-t{spec["period"]}.json',dict(spec=spec,state=before,
                    old_source=str(MECHANISM/'raw/probes'/f's{spec["scenario_seed"]}-t{spec["period"]}-m30001.json'),
                    old=a,old_beta0=a0,rows=rows))
        write_json(TARGET/'qa/probes.json',dict(status='PASSED',states=9,new_solves=count,observed_development_set=True))
    finally:budget.save()


def evaluate():
    m=load()
    if (TARGET/'qa/evaluation.json').exists():return
    if m['status']!='MODEL_FROZEN':raise RuntimeError('fixed update8 frozen before development')
    c=configuration();old=Policy(OLD_TRAIN/'inference-0008');new=Policy(TARGET/'training/inference-0008');budget=Budget();days=0
    try:
        with no_mc_guard():
            for scene in read_json(TARGET/'data/development.json')['scenarios']:
                for method,policy in (('old',old),('new',new)):
                    r=run_record(scene,c,policy.act,method,budget)
                    write_json(TARGET/'evaluation'/f'{method}-s{scene["seed"]}.json',r);days+=1
                print('DEVELOPMENT paired',scene['seed'],flush=True)
        write_json(TARGET/'qa/evaluation.json',dict(status='PASSED',days=days,paired_scenarios=4,statistical_test=False))
        m['status']='NUMERICAL_COMPLETE_QA_PENDING';write_json(TARGET/'manifest.json',m)
    finally:budget.save()


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','train','probes','evaluate','all'));stage=p.parse_args().stage
    torch.set_num_threads(1)
    try:
        if stage=='prepare':prepare()
        elif stage=='all':train();probes();evaluate()
        else:globals()[stage]()
    except Exception as exc:
        if TARGET.exists():
            write_json(TARGET/'qa/stop.json',dict(error=type(exc).__name__,message=str(exc),stage=stage,at=now()))
            m=read_json(TARGET/'manifest.json');m['status']='BUDGET_LIMITED' if isinstance(exc,TimeoutError) else 'FAILED';write_json(TARGET/'manifest.json',m)
        raise


if __name__=='__main__':main()
