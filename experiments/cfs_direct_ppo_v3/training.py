"""Independent Direct trainer; unchanged existing PPO backend and sampling math."""
import random
import time
from pathlib import Path
from dataclasses import asdict
import torch
from model_families.single.learning.ppo.backend import advantages,update
from model_families.single.learning.ppo.environment import PPOEnvironment
from model_families.single.learning.ppo.networks import sample_action
from model_families.single.learning.ppo.config import PPOConfig
from model_families.single.strategic.artifacts import write_json
from experiments.cfs_ppo_strategic_control.study_common import sha
from .network import DirectNetwork
from .control import episode
from .qa import replay,no_mc_guard


def run_record(scenario,cfg,policy,method,cancelled=None):
    ep=episode(scenario,cfg,cancelled);obs=ep.reset();seconds=0.
    while True:
        started=time.perf_counter();a=policy(obs);seconds+=time.perf_counter()-started
        tr=ep.step(a)
        if tr.terminated:break
        obs=tr.observation
    r=dict(scenario_id=scenario['id'],scenario_seed=scenario['seed'],method=method,
        ledger=ep.final_ledger,metrics=ep.metrics,policy_seconds=seconds,transitions=ep.records)
    r['replay']=replay(scenario,cfg,r)
    return r


def checkpoint(path,weights,cfg,source,update_index,validation):
    path.mkdir(exist_ok=False)
    torch.save(weights,path/'actor.pt')
    write_json(path/'manifest.json',dict(kind='direct-no-mc-inference/v1',source=source,
        action=cfg['action'],feature_scales=cfg['feature_scales'],parameters=cfg['parameters'],solver=cfg['solver'],
        hidden_size=cfg['ppo']['hidden_size'],update=update_index,validation_USD=validation,
        actor_sha256=sha(path/'actor.pt'),no_baseline=True))


def train(train_scenes,validation,cfg,output,source,cancelled=None):
    target=Path(output);target.mkdir(parents=True,exist_ok=False)
    config=PPOConfig.model_validate(cfg['ppo']);cfg=dict(cfg,ppo=config.model_dump());started=time.monotonic()
    write_json(target/'status.json',dict(status='RUNNING'))
    with torch.random.fork_rng():
        torch.manual_seed(config.training_seed);net=DirectNetwork(len(cfg['action']['hubs']),config.hidden_size)
    initial={k:v.detach().clone() for k,v in net.state_dict().items()}
    optimizer=torch.optim.Adam(net.parameters(),lr=config.learning_rate)
    sampling=torch.Generator().manual_seed(config.training_seed+1001)
    batches=torch.Generator().manual_seed(config.training_seed+1002)
    rng=random.Random(config.training_seed+1003)
    completed=0;best_score=None;best_weights=None;best_update=None
    try:
        with no_mc_guard():
            for iteration in range(1,config.updates+1):
                rows=[];profits=[];outputs=[];net.train()
                for _ in range(config.episodes_per_update):
                    s=train_scenes[rng.randrange(len(train_scenes))];ep=episode(s,cfg,cancelled)
                    env=PPOEnvironment(ep,config.reward_scale);obs,_=env.reset()
                    daily=[];rewards=[];values=[];learning=[]
                    while True:
                        action,latent,logp,value=sample_action(net,obs,sampling)
                        next_obs,r,terminated,truncated,_=env.step(action)
                        if truncated:raise RuntimeError('incomplete Direct rollout')
                        daily.append(dict(observation=obs,latent=latent,logp=logp.item()))
                        rewards.append(r);values.append(value.item());outputs.append(action.tolist())
                        learning.append(dict(latent=latent.tolist(),action=action.tolist(),logp=logp.item(),value=value.item()))
                        if terminated:break
                        obs=next_obs
                    adv,returns=advantages(rewards,values,config.gae_lambda,config.gamma)
                    for row,l,a,ret in zip(daily,learning,adv,returns):row.update(advantage=a,**{'return':ret});l.update(advantage=a,**{'return':ret})
                    rows.extend(daily);completed+=1
                    record=dict(scenario_id=s['id'],scenario_seed=s['seed'],training_seed=config.training_seed,
                        update=iteration,ledger=ep.final_ledger,metrics=ep.metrics,transitions=ep.records,learning=learning)
                    record['replay']=replay(s,cfg,record)
                    write_json(target/f'training-episode-{completed:06d}.json',record)
                    profits.append(ep.final_ledger['realized_profit'])
                diagnostics=update(net,optimizer,rows,config,batches)
                score=None;net.eval()
                if iteration%config.validation_every==0 or iteration==config.updates:
                    vals=[]
                    for s in validation:
                        r=run_record(s,cfg,lambda o:sample_action(net,o,sampling,True)[0].tolist(),'Direct-validation',cancelled)
                        write_json(target/f'validation-{iteration:04d}-s{s["seed"]}.json',r)
                        vals.append(r['ledger']['realized_profit'])
                    score=sum(vals)/len(vals)
                    write_json(target/f'validation-{iteration:04d}.json',dict(update=iteration,mean_profit_USD=score,profits=vals))
                    if best_score is None or score>best_score:
                        best_score=score;best_update=iteration
                        best_weights={k:v.detach().clone() for k,v in net.state_dict().items() if not k.startswith('critic.')}
                    checkpoint(target/f'inference-{iteration:04d}',{k:v.detach().clone() for k,v in net.state_dict().items() if not k.startswith('critic.')},cfg,source,iteration,score)
                actions=torch.tensor(outputs)
                write_json(target/f'update-{iteration:04d}.json',dict(**diagnostics,updates=iteration,episodes=completed,
                    training_seed=config.training_seed,training_mean_USD=sum(profits)/len(profits),validation_mean_USD=score,
                    action_mean=actions.mean(0).tolist(),action_std=actions.std(0,unbiased=False).tolist(),
                    elapsed_seconds=time.monotonic()-started,transitions=len(rows)))
                # Recovery is separate from inference, with all RNG and best-selection state.
                torch.save(dict(network=net.state_dict(),optimizer=optimizer.state_dict(),sampling_rng=sampling.get_state(),
                    minibatch_rng=batches.get_state(),data_rng=rng.getstate(),best_weights=best_weights,best_update=best_update,
                    best_score=best_score,update=iteration,episodes=completed,config=cfg,source=source),target/'recovery.pt')
                print('Direct training',config.training_seed,'update',iteration,'validation',score,flush=True)
            checkpoint(target/'best-inference',best_weights,cfg,source,best_update,best_score)
        changed=[k for k,v in net.state_dict().items() if not torch.equal(initial[k],v)]
        progress=dict(status='COMPLETED',updates=config.updates,episodes=completed,selected_update=best_update,
            best_validation_USD=best_score,changed_parameter_tensors=changed,
            critic_changed=any(k.startswith('critic.') for k in changed),actor_changed=any(k.startswith('actor.') for k in changed),
            no_mc_guard_passed=True,wall_seconds=time.monotonic()-started)
        if not progress['actor_changed'] or not progress['critic_changed']:raise RuntimeError('Direct actor/critic did not update')
        write_json(target/'status.json',progress);return progress
    except BaseException as exc:
        write_json(target/'status.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),episodes=completed))
        raise
