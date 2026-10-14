"""Existing reward-scaled PPO math; only local episode/action head differ."""
import random
from pathlib import Path
import torch
from model_families.single.learning.ppo.config import PPOConfig
from model_families.single.learning.ppo.backend import advantages
from model_families.single.learning.ppo.environment import PPOEnvironment
from model_families.single.learning.ppo.networks import sample_action
from experiments.cfs_direct_ppo_training_ablation_v6.backend import update, full_returns
from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
from .control import episode, replay, VERSION
from .network import build, QNetwork
from .common import write_json, read_json, sha, OLD


def run_record(scene,cfg,policy,method,budget=None):
    ep=episode(scene,cfg,budget);o=ep.reset()
    while True:
        tr=ep.step(policy(o))
        if tr.terminated:break
        o=tr.observation
    r=dict(scenario_id=scene['id'],scenario_seed=scene['seed'],method=method,
        ledger=ep.final_ledger,metrics=ep.metrics,transitions=ep.records)
    r['replay']=replay(scene,cfg,r);return r


def checkpoint(folder,net,cfg,source):
    folder.mkdir(parents=True,exist_ok=False)
    weights={k:v.detach().clone() for k,v in net.state_dict().items() if not k.startswith('critic.')}
    torch.save(weights,folder/'actor.pt')
    write_json(folder/'manifest.json',dict(version=VERSION,source=source,config=cfg,
        actor_sha256=sha(folder/'actor.pt'),update=8,layout='alpha near/far interleaved; beta near/far interleaved'))


class Policy:
    def __init__(self,path):
        path=Path(path);m=read_json(path/'manifest.json')
        if m['version']!=VERSION or sha(path/'actor.pt')!=m['actor_sha256']:raise RuntimeError('q checkpoint identity')
        with torch.random.fork_rng():self.network=QNetwork(with_critic=False)
        self.network.load_state_dict(torch.load(path/'actor.pt',weights_only=True,map_location='cpu'),strict=True)
        self.network.eval();self.network.requires_grad_(False);self.generator=torch.Generator().manual_seed(0)
    def act(self,o):return sample_action(self.network,o,self.generator,True)[0].tolist()


def fit(name,seed,scenes,validation,cfg,folder,source,budget):
    folder.mkdir(parents=True,exist_ok=False);config=PPOConfig.model_validate(cfg['ppo'])
    net=build(seed);optim=(torch.optim.Adam(net.parameters(),lr=config.learning_rate),)
    sampling=torch.Generator().manual_seed(seed+1001);batches=torch.Generator().manual_seed(seed+1002)
    rng=random.Random(seed+1003);count=0
    with no_mc_guard():
        for u in range(1,9):
            rows=[];profits=[];all_learning=[]
            for _ in range(4):
                scene=scenes[rng.randrange(len(scenes))];ep=episode(scene,cfg,budget)
                env=PPOEnvironment(ep,config.reward_scale);obs,_=env.reset()
                daily=[];learn=[];rewards=[];values=[]
                while True:
                    a,z,logp,value=sample_action(net,obs,sampling)
                    nxt,r,done,truncated,_=env.step(a)
                    if truncated:raise RuntimeError('incomplete q rollout')
                    daily.append(dict(observation=obs,latent=z,logp=logp.item()))
                    learn.append(dict(latent=z.tolist(),action=a.tolist(),logp=logp.item(),value=value.item()))
                    rewards.append(r);values.append(value.item())
                    if done:break
                    obs=nxt
                adv,returns=advantages(rewards,values,.95,1.);rtg=full_returns(rewards)
                for row,l,av,ret,future in zip(daily,learn,adv,returns,rtg):
                    row.update(advantage=av,**{'return':ret});l.update(advantage=av,RTG=future,**{'return':ret})
                count+=1
                old=read_json(OLD/'training'/name/'B'/str(seed)/f'training-episode-{count:06d}.json')
                if scene['seed']!=old['scenario_seed']:raise RuntimeError('matched data sampling differs')
                record=dict(scenario_id=scene['id'],scenario_seed=scene['seed'],training_seed=seed,update=u,
                    transitions=ep.records,ledger=ep.final_ledger,metrics=ep.metrics,learning=learn)
                record['replay']=replay(scene,cfg,record);write_json(folder/f'training-{count:04d}.json',record)
                rows.extend(daily);all_learning.extend(learn);profits.append(ep.final_ledger['realized_profit'])
            diag=update(net,optim,rows,config,batches,'B')
            import numpy as np
            p=np.array([r['value']*10000 for r in all_learning]);target=np.array([r['RTG']*10000 for r in all_learning])
            diag.update(update=u,training_mean_USD=float(np.mean(profits)),log_std=net.log_std.detach().tolist(),
                critic_RMSE_USD=float(np.sqrt(np.mean((target-p)**2))),
                critic_EV=float(1-np.var(target-p)/np.var(target)) if np.var(target)>0 else None,
                action_mean=np.array([r['action'] for r in all_learning]).mean(0).tolist())
            write_json(folder/f'update-{u:04d}.json',diag)
            torch.save(dict(network=net.state_dict(),optimizer=optim[0].state_dict(),sampling_rng=sampling.get_state(),
                minibatch_rng=batches.get_state(),data_rng=rng.getstate(),update=u,config=cfg,source=source),folder/'recovery.pt')
            print('Q20',name,seed,'update',u,flush=True)
        net.eval();vals=[]
        for scene in validation:
            r=run_record(scene,cfg,lambda o:sample_action(net,o,sampling,True)[0].tolist(),'q-validation',budget)
            write_json(folder/f'validation-s{scene["seed"]}.json',r);vals.append(r['ledger']['realized_profit'])
        checkpoint(folder/'inference-0008',net,cfg,source)
    result=dict(status='COMPLETED',updates=8,episodes=32,validation_mean_USD=sum(vals)/len(vals),seed=seed,condition=name)
    write_json(folder/'status.json',result);return result
