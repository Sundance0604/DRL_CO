"""PPO-Clip interventions with real pre-clipping shared-gradient diagnostics."""
import torch
from model_families.single.learning.ppo.networks import squashed_log_probability
from .network import DecoupledNetwork


def full_returns(rewards):
    total = 0.
    result = []
    for reward in reversed(rewards):
        total += reward
        result.append(total)
    return result[::-1]


class TimeBaseline:
    def __init__(self, horizon=24):
        self.sums = [0.] * horizon
        self.episodes = 0

    def values(self):
        return [s / self.episodes if self.episodes else 0. for s in self.sums]

    def observe_after_optimization(self, episode_returns):
        for values in episode_returns:
            if len(values) != len(self.sums):
                raise ValueError('incomplete baseline episode')
            self.sums = [s + v for s, v in zip(self.sums, values)]
            self.episodes += 1


def vector(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, grads)])


def update(network, optim, rows, config, generator, variant):
    observations = [r['observation'] for r in rows]
    latent = torch.stack([r['latent'] for r in rows])
    old = torch.tensor([r['logp'] for r in rows])
    advantage = torch.tensor([r['advantage'] for r in rows])
    target = torch.tensor([r['return'] for r in rows])
    if not all(torch.isfinite(x).all() for x in (latent, old, advantage, target)):
        raise FloatingPointError('nonfinite rollout')
    normalized = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    detailed, minibatches, trigger = None, [], None
    for epoch in range(config.epochs):
        for batch_index, ids in enumerate(torch.randperm(len(rows), generator=generator).split(config.minibatch_size)):
            mean, std, value = network([observations[i] for i in ids.tolist()])
            logp = squashed_log_probability(mean, std, latent[ids])
            logratio = logp - old[ids]
            ratio = logratio.exp()
            kl = ((ratio - 1) - logratio).mean()
            if not all(torch.isfinite(x).all() for x in (mean, std, logp, ratio, kl)):
                raise FloatingPointError('nonfinite prediction/KL')
            if kl.item() > config.target_kl:
                trigger = dict(epoch=epoch, batch_index=batch_index, KL=kl.item(), indices=ids.tolist())
                break
            policy = -torch.minimum(ratio * normalized[ids], ratio.clamp(
                1 - config.clip_ratio, 1 + config.clip_ratio) * normalized[ids]).mean()
            value_loss = (value - target[ids]).square().mean() if value is not None else None
            fresh = mean + std * torch.randn(mean.shape, generator=generator)
            entropy = -squashed_log_probability(mean, std, fresh).mean()
            actor_loss = policy - config.entropy_coefficient * entropy
            weighted = config.value_coefficient * value_loss if value_loss is not None else None
            if detailed is None:
                detailed = dict(epoch=epoch, batch_index=batch_index, indices=ids.tolist(),
                                shared_actor_norm=None, shared_critic_norm=None, shared_cosine=None)
                if variant in ('A', 'B'):
                    shared = [p for name, p in network.named_parameters()
                              if name.startswith(('order_encoder.', 'arc_encoder.', 'vehicle_encoder.', 'body.'))]
                    ga, gc = vector(policy, shared), vector(weighted, shared)
                    product = ga.norm() * gc.norm()
                    detailed.update(shared_actor_norm=ga.norm().item(), shared_critic_norm=gc.norm().item(),
                                    shared_cosine=(torch.dot(ga, gc) / product).item() if product > 0 else None,
                                    shared_parameters=sum(p.numel() for p in shared))
            for optimizer in optim:
                optimizer.zero_grad(set_to_none=True)
            if isinstance(network, DecoupledNetwork):
                actor_loss.backward()
                weighted.backward()
                groups = (list(network.actor_model.parameters()), list(network.value_model.parameters()))
            else:
                (actor_loss if weighted is None else actor_loss + weighted).backward()
                groups = (list(network.parameters()),)
            norms = [torch.nn.utils.clip_grad_norm_(group, config.max_grad_norm,
                                                    error_if_nonfinite=True).item() for group in groups]
            for optimizer in optim:
                optimizer.step()
            if not all(torch.isfinite(p).all() for p in network.parameters()):
                raise FloatingPointError('nonfinite updated parameters')
            minibatches.append(dict(policy_loss=policy.item(), value_loss=None if value_loss is None else value_loss.item(),
                                    entropy=entropy.item(), approximate_kl=kl.item(), preclip_norms=norms,
                                    clip_coefficients=[min(1., config.max_grad_norm / (n + 1e-6)) for n in norms],
                                    clip_fraction=((ratio - 1).abs() > config.clip_ratio).float().mean().item()))
        if trigger is not None:
            break
    if not minibatches:
        raise FloatingPointError('no admissible minibatch')
    result = {key: sum(r[key] for r in minibatches) / len(minibatches)
              for key in ('policy_loss', 'entropy', 'approximate_kl', 'clip_fraction')}
    result.update(value_loss=None if variant == 'C' else sum(r['value_loss'] for r in minibatches) / len(minibatches),
                  gradient_norm=sum(r['preclip_norms'][0] for r in minibatches) / len(minibatches),
                  minibatches=len(minibatches), actor_steps=len(minibatches),
                  critic_steps=0 if variant == 'C' else len(minibatches), kl_early_stop=trigger is not None,
                  trigger=trigger, detailed_gradient=detailed, minibatch_records=minibatches,
                  standardized_advantage_mean=normalized.mean().item(),
                  standardized_advantage_std=normalized.std(unbiased=False).item(),
                  raw_advantage_mean=advantage.mean().item(), raw_advantage_std=advantage.std(unbiased=False).item(),
                  action_saturation_fraction=(torch.tanh(latent).abs() > .99).float().mean().item(),
                  all_gradients_finite=True)
    return result
