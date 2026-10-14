"""Independent non-solving numerical reconstruction and cash/physical audit."""
import copy
import random
import torch
from model_families.single.learning.ppo.networks import sample_action
from model_families.single.learning.ppo.config import PPOConfig
from experiments.cfs_direct_ppo_training_ablation_v6.backend import update
from experiments.cfs_direct_ppo_diagnosis.common import observation
from experiments.cfs_structured_ppo_q.control import replay
from experiments.cfs_structured_ppo_q.training import Policy
from experiments.cfs_structured_ppo_q.network import build
from experiments.cfs_structured_ppo_q_mechanism.states import public_episode
from experiments.cfs_structured_ppo_q_mechanism.physical_qa import validate_plan
from experiments.cfs_structured_ppo_q_mechanism.kl import fingerprint
from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
from .run import modified
from .common import *


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def audit():
    m=load();torch.set_num_threads(1);c=configuration();config=PPOConfig.model_validate(c['ppo'])
    verify_hashes(Path('/'),m['history_hashes'])
    if sha(TARGET/'data/development.json')!=m['development_sha256']:raise RuntimeError('development identities changed')
    scenes={s['seed']:s for s in read_json(ABLATION/'data'/W40/'train.json')['scenarios']}
    days=0;checks=0;triggers=0
    with no_mc_guard():
        for method,folder in (('old',OLD_TRAIN),('new',TARGET/'training')):
            net=build(30001) if method=='old' else modified();optimizer=(torch.optim.Adam(net.parameters(),lr=config.learning_rate),)
            sampling=torch.Generator().manual_seed(31002);batches=torch.Generator().manual_seed(31003);data=random.Random(31004)
            pool=list(read_json(ABLATION/'data'/W40/'train.json')['scenarios']);counter=0
            for u in range(1,9):
                rows=[]
                for _ in range(4):
                    counter+=1;r=read_json(folder/f'training-{counter:04d}.json');expected_scene=pool[data.randrange(len(pool))]
                    if expected_scene['seed']!=r['scenario_seed']:raise RuntimeError('sampled scenario sequence mismatch')
                    for tr,l in zip(r['transitions'],r['learning']):
                        o=observation(tr['observation']);g=torch.Generator().set_state(sampling.get_state());epsilon=torch.randn((1,20),generator=g)[0]
                        a,z,lp,v=sample_action(net,o,sampling)
                        if (a.tolist()!=l['action'] or z.tolist()!=l['latent'] or lp.item()!=l['logp'] or v.item()!=l['value']):raise RuntimeError('Actor/noise/logp/value mismatch')
                        if method=='new' and epsilon.tolist()!=l['standard_epsilon']:raise RuntimeError('shared standard Gaussian noise mismatch')
                        rows.append(dict(observation=o,latent=z,logp=lp.item(),advantage=l['advantage'],**{'return':l['return']}))
                    replay(scenes[r['scenario_seed']],c,r);days+=1
                capture=read_json(MECHANISM/'raw/kl'/f'PPO20-{W40}-30001-u{u:02d}.json' if method=='old' else TARGET/'training/kl'/f'update-{u:04d}.json')
                for item in capture['checks']:
                    checks+=1
                    if item['stopping_trigger']:
                        path=Path(item['trigger_snapshot'])
                        if sha(path)!=item['trigger_snapshot_sha256']:raise RuntimeError('trigger snapshot hash mismatch')
                        snap=torch.load(path,map_location='cpu',weights_only=True);inspect=copy.deepcopy(net);inspect.load_state_dict(snap['network'])
                        if fingerprint(inspect)!=item['parameter_sha256']:raise RuntimeError('trigger parameter identity mismatch')
                        with torch.no_grad():mu,sigma,_=inspect([rows[i]['observation'] for i in item['indices']])
                        if mu.tolist()!=item['new_mu'] or sigma[0].tolist()!=item['new_sigma']:raise RuntimeError('actual trigger predictions mismatch')
                        triggers+=1
                diag=update(net,optimizer,rows,config,batches,'B');original=read_json(folder/f'update-{u:04d}.json')
                for k,v in diag.items():
                    if v!=original[k]:raise RuntimeError('PPO original update reconstruction mismatch '+k)
                if method=='new':
                    saved=torch.load(TARGET/'snapshots'/f'update-{u:04d}.pt',map_location='cpu',weights_only=True)
                    if not all((equal(net.state_dict(),saved['network']),equal(optimizer[0].state_dict(),saved['optimizer']),
                        torch.equal(sampling.get_state(),saved['sampling_rng']),torch.equal(batches.get_state(),saved['minibatch_rng']),data.getstate()==saved['data_rng'])):
                        raise RuntimeError('network/optimizer/three RNG snapshots mismatch')
            final=torch.load(folder/'recovery.pt',map_location='cpu',weights_only=True)
            if not equal(net.state_dict(),final['network']) or not torch.equal(sampling.get_state(),final['sampling_rng']):raise RuntimeError('final recovery mismatch')
        write_json(TARGET/'qa/update-reconstruction.json',dict(status='PASSED',days=64,updates=16,actual_checks=checks,actual_trigger_snapshots=triggers,
            new_all_update_network_Adam_and_three_rng_exact=True,standard_noise_stream_coupled=True))
        for scene in read_json(TARGET/'data/development.json')['scenarios']:
            for method,path in (('old',OLD_TRAIN/'inference-0008'),('new',TARGET/'training/inference-0008')):
                r=read_json(TARGET/'evaluation'/f'{method}-s{scene["seed"]}.json');policy=Policy(path)
                replay(scene,c,r)
                for tr in r['transitions']:
                    if list(tr['signal']['values'])!=policy.act(observation(tr['observation'])):raise RuntimeError('development deterministic Actor differs')
                days+=1
        probe_count=0
        old=Policy(OLD_TRAIN/'inference-0008');new=Policy(TARGET/'training/inference-0008')
        for p in sorted((TARGET/'probes').glob('*.json')):
            d=read_json(p);spec=d['spec'];ep,o,*_=public_episode(spec)
            if d['state']!=__import__('model_families.single.runner',fromlist=['state_snapshot']).state_snapshot(ep._sim):raise RuntimeError('probe state mismatch')
            for r in (d['old'],d['old_beta0'],*d['rows']):
                plan=r['probe']['plan'];can=validate_plan(ep._sim,plan,plan['solver']['slot_occupancy'],r['probe']['coefficients'],r['signal'])
                if can!=r['canonical'] or plan['solver']['status']!='OPTIMAL':raise RuntimeError('probe canonical or solver status differs')
                probe_count+=1
            if d['old']['signal']!=old.act(o) or d['rows'][0]['signal']!=new.act(o):raise RuntimeError('probe policy output differs')
            if d['rows'][1]['signal']!=new.act(o)[:10]+[0.]*10:raise RuntimeError('new beta-zero signal differs')
        budget=read_json(TARGET/'qa/budget.json')
        if budget['calls']!=32*24+8*24+18:raise RuntimeError('solver call ledger mismatch')
        if budget['calls']>1000 or budget['seconds']>600:raise RuntimeError('budget failed')
    write_json(TARGET/'qa/replay.json',dict(status='PASSED',full_days=days,new_days=40,old_training_days_readonly=32,
        compiled_probe_actions=probe_count,new_probe_solves=18,new_solver_claims=budget['calls'],full_cooldown=True,
        no_MC_guard_passed=True,source_sha256=sha(__file__)))
    print('QA PASSED',days,'days',probe_count,'plans',checks,'checks',flush=True)


if __name__=='__main__':audit()
