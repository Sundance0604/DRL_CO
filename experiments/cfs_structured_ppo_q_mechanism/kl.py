"""Read-only capture at actual original-backend KL check, with bit-exact replay."""
import copy
import hashlib
import inspect
import sys
import numpy as np
import torch
from experiments.cfs_direct_ppo_training_ablation_v6.backend import update
from experiments.cfs_direct_ppo_training_ablation_v6.network import build as build10
from experiments.cfs_structured_ppo_q.network import build as build20
from experiments.cfs_direct_ppo_diagnosis.common import observation
from model_families.single.learning.ppo.networks import sample_action,squashed_log_probability
from model_families.single.learning.ppo.config import PPOConfig
from .common import *


def gaussian_parts(old_mu,old_sigma,new_mu,new_sigma):
    a,s,b,t=[v.detach().double() for v in (old_mu,old_sigma,new_mu,new_sigma)]
    mean=(a-b).square()/(2*t.square())
    scale=(t/s).log()+s.square()/(2*t.square())-.5
    total=mean+scale
    independent=torch.distributions.kl_divergence(torch.distributions.Normal(a,s),torch.distributions.Normal(b,t))
    if not torch.allclose(total,independent,atol=1e-10,rtol=1e-9) or total.min() < -1e-10:
        raise RuntimeError('Gaussian KL independent mismatch')
    return mean,scale,total


def fingerprint(net):
    h=hashlib.sha256()
    for k,v in sorted(net.state_dict().items()):h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


class Capture:
    def __init__(self,net,rows,folder,iteration):
        self.net=net;self.old=copy.deepcopy(net).eval();self.old.requires_grad_(False)
        self.rows=rows;self.folder=folder;self.iteration=iteration;self.checks=[]
        lines,start=inspect.getsourcelines(update)
        self.line=start+next(i for i,s in enumerate(lines) if s.strip()=='if kl.item() > config.target_kl:')
        self.previous=None
    def __enter__(self):
        self.previous=sys.gettrace()
        if self.previous is not None:raise RuntimeError('do not replace another active trace')
        sys.settrace(self.trace);return self
    def __exit__(self,*args):sys.settrace(self.previous)
    def trace(self,frame,event,arg):
        if frame.f_code is not update.__code__:return None
        if event=='line' and frame.f_lineno==self.line:
            f=frame.f_locals;ids=f['ids'].tolist();mu,sigma=f['mean'].detach(),f['std'].detach()
            with torch.no_grad():old_mu,old_sigma,_=self.old([self.rows[i]['observation'] for i in ids])
            mean,scale,total=gaussian_parts(old_mu,old_sigma,mu,sigma)
            dim=mu.shape[1];stop=f['kl'].item()>f['config'].target_kl
            def group(sl):
                return dict(mean_KL=mean[:,sl].sum(1).mean().item(),std_KL=scale[:,sl].sum(1).mean().item(),
                    exact_KL=total[:,sl].sum(1).mean().item(),mu_shift_RMS=(mu[:,sl]-old_mu[:,sl]).square().mean().sqrt().item(),
                    log_sigma_shift_RMS=(sigma[:,sl].log()-old_sigma[:,sl].log()).square().mean().sqrt().item())
            alpha=group(slice(0,10));beta=group(slice(10,20)) if dim==20 else None
            n=len(self.checks);state=fingerprint(self.net)
            item=dict(check=n,epoch=f['epoch'],batch_index=f['batch_index'],indices=ids,
                approximate_KL=f['kl'].item(),exact_KL=total.sum(1).mean().item(),alpha=alpha,beta=beta,
                stopping_trigger=stop,accepted=not stop,completed_steps_before_check=n,
                parameter_sha256=state,old_mu=old_mu.tolist(),new_mu=mu.tolist(),old_sigma=old_sigma[0].tolist(),
                new_sigma=sigma[0].tolist(),latent=f['latent'][f['ids']].tolist(),old_logp=f['old'][f['ids']].tolist(),
                log_std_before=self.old.log_std.detach().tolist(),log_std_at_check=self.net.log_std.detach().tolist())
            if dim==20:
                paired=old_mu[:,10:].double()-mu[:,10:].double()
                ref=(paired.square()/(2*sigma[:,:10].double().square())).sum(1).mean().item()
                item.update(beta_mean_KL_with_paired_alpha_sigma=ref,
                    beta_mean_denominator_sensitivity=(sigma[:,:10].double().square()/sigma[:,10:].double().square()).mean().item())
            if abs(item['exact_KL']-alpha['exact_KL']-(beta['exact_KL'] if beta else 0.))>1e-10:
                raise RuntimeError('KL groups do not sum')
            if stop:
                path=self.folder/f'trigger-u{self.iteration:02d}.pt'
                torch.save(dict(network={k:v.detach().clone() for k,v in self.net.state_dict().items()},
                    optimizer=f['optim'][0].state_dict(),generator=f['generator'].get_state(),
                    epoch=f['epoch'],batch_index=f['batch_index'],parameter_sha256=state),path)
                item['trigger_snapshot']=str(path);item['trigger_snapshot_sha256']=sha(path)
            self.checks.append(item)
        return self.trace


def run():
    load();budget=Budget();summary=[]
    if (TARGET/'qa/kl.json').exists():return
    torch.set_num_threads(1)
    try:
        for method in ('PPO20','PPO10'):
            for name in NAMES:
                for seed in SEEDS:
                    budget();config=PPOConfig.model_validate(cfg(name,seed)['ppo'])
                    net=build20(seed) if method=='PPO20' else build10('B',seed)
                    optim=(torch.optim.Adam(net.parameters(),lr=.0003),)
                    sampling=torch.Generator().manual_seed(seed+1001);batches=torch.Generator().manual_seed(seed+1002)
                    source=P20/'training'/name/str(seed) if method=='PPO20' else P10/'training'/name/'B'/str(seed)
                    folder=TARGET/'snapshots'/method/name/str(seed);folder.mkdir(parents=True,exist_ok=True)
                    for u in range(1,9):
                        budget();rows=[]
                        for i in range((u-1)*4+1,u*4+1):
                            p=source/(f'training-{i:04d}.json' if method=='PPO20' else f'training-episode-{i:06d}.json')
                            r=read_json(p)
                            for tr,l in zip(r['transitions'],r['learning']):
                                obs=observation(tr['observation']);a,z,lp,v=sample_action(net,obs,sampling)
                                if a.tolist()!=l['action'] or z.tolist()!=l['latent'] or lp.item()!=l['logp'] or v.item()!=l['value']:
                                    raise RuntimeError('saved rollout/actor/RNG reconstruction differs')
                                rows.append(dict(observation=obs,latent=z,logp=lp.item(),advantage=l['advantage'],**{'return':l['return']}))
                        rng=torch.get_rng_state().clone();before_b=batches.get_state().clone()
                        with Capture(net,rows,folder,u) as capture:diag=update(net,optim,rows,config,batches,'B')
                        if not torch.equal(rng,torch.get_rng_state()):raise RuntimeError('instrumentation consumed global RNG')
                        original=read_json(source/f'update-{u:04d}.json')
                        for k in ('policy_loss','value_loss','approximate_kl','gradient_norm','clip_fraction','minibatches','trigger'):
                            if diag[k]!=original[k]:raise RuntimeError('original backend trajectory/log differs '+k)
                        checks=capture.checks;last=checks[-1]
                        if len(checks)!=diag['minibatches']+int(diag['kl_early_stop']):raise RuntimeError('check/optimization counts')
                        if diag['trigger'] and (last['epoch'],last['batch_index'],last['approximate_KL'])!= (
                            diag['trigger']['epoch'],diag['trigger']['batch_index'],diag['trigger']['KL']):raise RuntimeError('trigger identity')
                        s=dict(method=method,condition=name,training_seed=seed,update=u,dimension=20 if method=='PPO20' else 10,
                            planned_steps=8,effective_steps=diag['minibatches'],skipped_steps=8-diag['minibatches'],
                            early_stop=diag['kl_early_stop'],completed_epochs=last['epoch'] if diag['kl_early_stop'] else 4,
                            accepted_mean_approximate_KL=diag['approximate_kl'],last_checked_approximate_KL=last['approximate_KL'],
                            last_checked_exact_KL=last['exact_KL'],alpha_exact_KL=last['alpha']['exact_KL'],
                            beta_exact_KL=last['beta']['exact_KL'] if last['beta'] else None,policy_loss=diag['policy_loss'],
                            final_parameter_sha256=fingerprint(net),log_std_after=net.log_std.detach().tolist())
                        summary.append(s)
                        write_json(TARGET/'raw/kl'/f'{method}-{name}-{seed}-u{u:02d}.json',dict(summary=s,checks=checks,original_log=original,
                            minibatch_rng_before=before_b.tolist(),minibatch_rng_after=batches.get_state().tolist()))
                    expected=torch.load(source/('recovery.pt' if method=='PPO20' else 'network-0008.pt'),map_location='cpu',weights_only=method=='PPO10')
                    weights=expected['network'] if method=='PPO20' else expected
                    if any(not torch.equal(v,weights[k]) for k,v in net.state_dict().items()):raise RuntimeError('final frozen network differs')
                    if method=='PPO20' and (not torch.equal(expected['sampling_rng'],sampling.get_state()) or not torch.equal(expected['minibatch_rng'],batches.get_state())):
                        raise RuntimeError('frozen final RNG differs')
                    print('KL reconstructed',method,name,seed,flush=True)
        write_json(TARGET/'qa/kl.json',dict(status='PASSED',models=18,updates=len(summary),new_rollouts=0,new_solver_calls=0,
            original_logs_and_final_tensors_exact=True,global_RNG_unchanged=True,rows=summary))
    finally:budget.save()
