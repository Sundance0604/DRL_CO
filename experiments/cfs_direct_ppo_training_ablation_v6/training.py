"""Full-day on-policy training, same scenario and sampling schedules as A."""
import random
import time
from pathlib import Path
import numpy as np
import torch
from model_families.single.learning.ppo.config import PPOConfig
from model_families.single.learning.ppo.backend import advantages
from model_families.single.learning.ppo.environment import PPOEnvironment
from model_families.single.learning.ppo.networks import sample_action
from model_families.single.strategic.artifacts import read_json
from .io_retry import write_json
from experiments.cfs_direct_ppo_v3.control import episode
from experiments.cfs_direct_ppo_v3.training import run_record, checkpoint
from experiments.cfs_direct_ppo_v3.qa import no_mc_guard, replay as physical_replay
from experiments.cfs_direct_ppo_diagnosis.common import observation
from .network import build, optimizers, actor_weights, actor_network
from .backend import full_returns, TimeBaseline, update


def learning_replay(scene, cfg, record, variant):
    physical_replay(scene, cfg, {k: v for k, v in record.items() if k != 'learning'})
    learn = record['learning']
    if len(learn) != 24 or len(record['transitions']) != 24:
        raise RuntimeError('complete 24-period training required')
    rewards = [t['reward_USD'] / cfg['ppo']['reward_scale'] for t in record['transitions']]
    future, carry, next_value = full_returns(rewards), 0., 0.
    for i in reversed(range(24)):
        row = learn[i]
        if abs(row['RTG'] - future[i]) > 1e-8:
            raise RuntimeError('full return mismatch')
        if variant == 'C':
            if row['value'] is not None or row['baseline_episodes'] != cfg['ppo']['episodes_per_update'] * (record['update'] - 1):
                raise RuntimeError('C has critic/current batch in baseline')
            expected = future[i] - row['time_baseline']
            if abs(row['advantage'] - expected) > 1e-8 or row['return'] != future[i]:
                raise RuntimeError('historical-baseline advantage mismatch')
        else:
            carry = rewards[i] + next_value - row['value'] + .95 * carry
            if abs(row['advantage'] - carry) > 1e-7 or abs(row['return'] - carry - row['value']) > 1e-7:
                raise RuntimeError('GAE units/recurrence mismatch')
            next_value = row['value']
    return dict(status='PASSED',periods=24,reward_divisor=cfg['ppo']['reward_scale'],
                physical_cash_price_cooldown=True,learning_returns=True)


def errors(predicted, target):
    p, t = np.asarray(predicted, float), np.asarray(target, float)
    return dict(predicted_mean_USD=float(p.mean()), target_mean_USD=float(t.mean()),
                RMSE_USD=float(np.sqrt(np.mean((p - t) ** 2))), MAE_USD=float(np.abs(p - t).mean()),
                EV=float(1 - np.var(t - p) / np.var(t)) if np.var(t) > 0 else None)


def sensitivity(network, references, scales):
    model = actor_network(network)
    gen = torch.Generator().manual_seed(0)
    signals = np.array([sample_action(model, observation(r['observation']), gen, True)[0].tolist()
                        for r in references])
    prices = signals * np.repeat(np.asarray(scales), 2)
    return dict(D_pi=float(np.abs(signals - signals.mean(0)).mean()),
                action_mean=signals.mean(0).tolist(), action_std=signals.std(0).tolist(),
                price_mean_USD=prices.mean(0).tolist(), price_std_USD=prices.std(0).tolist(),
                log_std=model.log_std.detach().tolist(), points=[dict(period=r['period'],
                    scenario_seed=r['scenario_seed'],pending=r['pending'],observation_hash=r['observation_hash'],
                    action=a.tolist(),price_USD=p.tolist()) for r,a,p in zip(references,signals,prices)])


def train(scenes, validation, cfg, folder, source, variant, references, budget, resume=None):
    folder = Path(folder)
    if resume is None:folder.mkdir(parents=True,exist_ok=False)
    elif not folder.is_dir():raise RuntimeError('explicit copied update prefix required')
    config = PPOConfig.model_validate(cfg['ppo'])
    write_json(folder/'status.json',dict(status='RUNNING',variant=variant))
    started = time.monotonic()
    net = build(variant,config.training_seed,len(cfg['action']['hubs']),config.hidden_size)
    optim = optimizers(net,config.learning_rate)
    sampling = torch.Generator().manual_seed(config.training_seed+1001)
    batches = torch.Generator().manual_seed(config.training_seed+1002)
    rng = random.Random(config.training_seed+1003)
    baseline = TimeBaseline()
    best_score,best_weights,best_update = None,None,None
    completed=0
    first_update=1
    elapsed_offset=0.
    if resume is not None:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['config']!=cfg or not 0<saved['update']<config.updates:raise RuntimeError('resume config/update mismatch')
        net.load_state_dict(saved['network'])
        if len(optim)!=len(saved['optimizers']):raise RuntimeError('resume optimizer count')
        for optimizer,state in zip(optim,saved['optimizers']):optimizer.load_state_dict(state)
        sampling.set_state(saved['sampling_rng']);batches.set_state(saved['minibatch_rng']);rng.setstate(saved['data_rng'])
        baseline.sums=list(saved['baseline_sums']);baseline.episodes=saved['baseline_episodes']
        best_score,best_weights,best_update=saved['best_score'],saved['best_weights'],saved['best_update']
        first_update=saved['update']+1;completed=saved['update']*config.episodes_per_update
        if len(list(folder.glob('training-episode-*.json')))!=completed:raise RuntimeError('resume saved prefix coverage')
        elapsed_offset=read_json(folder/f'update-{saved["update"]:04d}.json')['elapsed_seconds']
        if variant=='C':
            check=TimeBaseline()
            for i in range(1,completed+1):
                record=read_json(folder/f'training-episode-{i:06d}.json')
                learning_replay(next(s for s in scenes if s['seed']==record['scenario_seed']),cfg,record,variant)
                check.observe_after_optimization([[r['RTG'] for r in record['learning']]])
            if check.sums!=baseline.sums or check.episodes!=baseline.episodes:raise RuntimeError('resume historical baseline mismatch')
    else:
        write_json(folder/'initialization.json',dict(variant=variant,seed=config.training_seed,
        total_parameters=sum(p.numel() for p in net.parameters()),
        actor_parameters=sum(p.numel() for p in actor_network(net).parameters()),
        critic_separate=variant=='D',actor_initialization='exact original Direct seed; remove value head after init'))
    try:
        with no_mc_guard():
            for iteration in range(first_update,config.updates+1):
                update_started=time.monotonic()
                net.train(); rows=[];profits=[];outputs=[];episode_returns=[];learning_records=[]
                historical=baseline.values(); historical_count=baseline.episodes
                for _ in range(config.episodes_per_update):
                    scene=scenes[rng.randrange(len(scenes))]
                    ep=episode(scene,cfg,budget);env=PPOEnvironment(ep,config.reward_scale)
                    obs,_=env.reset();daily=[];learning=[];rewards=[];values=[]
                    while True:
                        action,latent,logp,value=sample_action(net,obs,sampling)
                        next_obs,r,done,truncated,_=env.step(action)
                        if truncated:raise RuntimeError('incomplete ablation rollout')
                        daily.append(dict(observation=obs,latent=latent,logp=logp.item()))
                        learning.append(dict(latent=latent.tolist(),action=action.tolist(),logp=logp.item(),
                                             value=value.item() if value is not None else None))
                        outputs.append(action.tolist());rewards.append(r)
                        if value is not None:values.append(value.item())
                        if done:break
                        obs=next_obs
                    future=full_returns(rewards);episode_returns.append(future)
                    if variant=='C':adv=[v-b for v,b in zip(future,historical)];returns=future
                    else:adv,returns=advantages(rewards,values,.95,1.)
                    for i,(row,l,a,ret) in enumerate(zip(daily,learning,adv,returns)):
                        row.update(advantage=a,**{'return':ret});l.update(advantage=a,RTG=future[i],**{'return':ret})
                        if variant=='C':l.update(time_baseline=historical[i],baseline_episodes=historical_count)
                    completed+=1;rows.extend(daily)
                    record=dict(scenario_id=scene['id'],scenario_seed=scene['seed'],training_seed=config.training_seed,
                        variant=variant,update=iteration,ledger=ep.final_ledger,metrics=ep.metrics,
                        transitions=ep.records,learning=learning)
                    record['replay']=learning_replay(scene,cfg,record,variant)
                    write_json(folder/f'training-episode-{completed:06d}.json',record)
                    profits.append(ep.final_ledger['realized_profit']);learning_records.extend(learning)
                rollout_seconds=time.monotonic()-update_started
                optimize_started=time.monotonic()
                diag=update(net,optim,rows,config,batches,variant)
                optimize_seconds=time.monotonic()-optimize_started
                if variant=='C':baseline.observe_after_optimization(episode_returns)
                critic=None
                if variant!='C':
                    p=[r['value']*config.reward_scale for r in learning_records]
                    critic=dict(MC=errors(p,[r['RTG']*config.reward_scale for r in learning_records]),
                        GAE=errors(p,[r['return']*config.reward_scale for r in learning_records]),
                        periods=[dict(period=i,**errors(p[i::24],
                            [r['RTG']*config.reward_scale for r in learning_records[i::24]])) for i in range(24)],
                        timing='pre-update on-policy predictions; MC and lambda-return distinct')
                score=None;net.eval();validation_started=time.monotonic()
                if iteration%config.validation_every==0 or iteration==config.updates:
                    vals=[]
                    for scene in validation:
                        rec=run_record(scene,cfg,lambda o:sample_action(net,o,sampling,True)[0].tolist(),
                                       variant+'-deterministic-validation',budget)
                        write_json(folder/f'validation-{iteration:04d}-s{scene["seed"]}.json',rec)
                        vals.append(rec['ledger']['realized_profit'])
                    score=sum(vals)/len(vals)
                    write_json(folder/f'validation-{iteration:04d}.json',dict(update=iteration,mean_profit_USD=score,profits=vals))
                    checkpoint(folder/f'inference-{iteration:04d}',actor_weights(net),cfg,source,iteration,score)
                    torch.save(net.state_dict(),folder/f'network-{iteration:04d}.pt')
                    write_json(folder/f'sensitivity-{iteration:04d}.json',sensitivity(net,references,cfg['action']['scales']))
                    if best_score is None or score>best_score:
                        best_score,best_update,best_weights=score,iteration,actor_weights(net)
                validation_seconds=time.monotonic()-validation_started
                actions=torch.tensor(outputs)
                write_json(folder/f'update-{iteration:04d}.json',dict(**diag,updates=iteration,episodes=completed,
                    variant=variant,training_seed=config.training_seed,training_mean_USD=sum(profits)/len(profits),
                    validation_mean_USD=score,critic=critic,action_mean=actions.mean(0).tolist(),
                    action_std=actions.std(0,unbiased=False).tolist(),log_std=actor_network(net).log_std.detach().tolist(),
                    elapsed_seconds=elapsed_offset+time.monotonic()-started,transitions=len(rows),
                    rollout_with_replay_write_seconds=rollout_seconds,optimization_seconds=optimize_seconds,
                    validation_with_replay_write_seconds=validation_seconds,
                    baseline_before=historical if variant=='C' else None,
                    baseline_episodes_before=historical_count if variant=='C' else None,
                    baseline_after=baseline.values() if variant=='C' else None))
                torch.save(dict(network=net.state_dict(),optimizers=[o.state_dict() for o in optim],
                    sampling_rng=sampling.get_state(),minibatch_rng=batches.get_state(),data_rng=rng.getstate(),
                    baseline_sums=baseline.sums,baseline_episodes=baseline.episodes,best_weights=best_weights,
                    best_update=best_update,best_score=best_score,update=iteration,config=cfg),folder/'recovery.pt')
                print(variant,config.training_seed,'update',iteration,'validation',score,flush=True)
            checkpoint(folder/'best-inference',best_weights,cfg,source,best_update,best_score)
        status=dict(status='COMPLETED',variant=variant,episodes=completed,selected_update=best_update,
                    best_validation_USD=best_score,wall_seconds=elapsed_offset+time.monotonic()-started)
        write_json(folder/'status.json',status);return status
    except BaseException as exc:
        write_json(folder/'status.json',dict(status='FAILED',variant=variant,episodes=completed,
                                           error=type(exc).__name__,message=str(exc)))
        raise

