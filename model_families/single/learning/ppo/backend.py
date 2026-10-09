"""On-policy full-episode GAE and clipped PPO update; no business physics."""
import torch
from .networks import squashed_log_probability


def advantages(rewards, values, gae_lambda=.95, gamma=1.):
    if len(rewards) != len(values) or not rewards:
        raise ValueError("GAE requires a complete nonempty episode")
    result, carry, next_value = [], 0., 0.
    for reward, value in zip(reversed(rewards), reversed(values)):
        carry = reward+gamma*next_value-value+gamma*gae_lambda*carry
        result.append(carry)
        next_value = value
    result.reverse()
    return result, [a+v for a, v in zip(result, values)]


def update(network, optimizer, rows, config, generator):
    obs = [r["observation"] for r in rows]
    latent = torch.stack([r["latent"] for r in rows])
    old_logp = torch.tensor([r["logp"] for r in rows])
    adv = torch.tensor([r["advantage"] for r in rows])
    returns = torch.tensor([r["return"] for r in rows])
    if not all(torch.isfinite(x).all() for x in (latent, old_logp, adv, returns)):
        raise FloatingPointError("nonfinite PPO rollout")
    adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
    diagnostics = []
    stop = False
    for _ in range(config.epochs):
        for ids in torch.randperm(len(rows), generator=generator).split(config.minibatch_size):
            mean, std, value = network([obs[i] for i in ids.tolist()])
            logp = squashed_log_probability(mean, std, latent[ids])
            logratio = logp-old_logp[ids]
            if not all(torch.isfinite(x).all() for x in (mean, std, value, logp, logratio)):
                raise FloatingPointError("nonfinite PPO predictions/log probability")
            ratio = logratio.exp()
            kl = ((ratio-1)-logratio).mean()
            if not torch.isfinite(kl):
                raise FloatingPointError("nonfinite PPO KL")
            if kl.item() > config.target_kl:
                stop = True
                break
            policy_loss = -torch.minimum(ratio*adv[ids], ratio.clamp(
                1-config.clip_ratio, 1+config.clip_ratio)*adv[ids]).mean()
            value_loss = (value-returns[ids]).square().mean()
            # Fresh reparameterized samples estimate the transformed distribution
            # entropy. Latent Normal entropy would reward a different distribution.
            fresh = mean+std*torch.randn(mean.shape, generator=generator)
            entropy = -squashed_log_probability(mean, std, fresh).mean()
            loss = policy_loss+config.value_coefficient*value_loss-config.entropy_coefficient*entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm,
                                                  error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in network.parameters()):
                raise FloatingPointError("nonfinite updated PPO parameters")
            diagnostics.append(dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(),
                entropy=entropy.item(), approximate_kl=kl.item(), gradient_norm=norm.item(),
                clip_fraction=((ratio-1).abs()>config.clip_ratio).float().mean().item()))
        if stop:
            break
    if not diagnostics:
        raise FloatingPointError("PPO update produced no admissible minibatches")
    return {key: sum(d[key] for d in diagnostics)/len(diagnostics) for key in diagnostics[0]} | dict(
        minibatches=len(diagnostics), kl_early_stop=stop,
        action_saturation_fraction=(torch.tanh(latent).abs()>.99).float().mean().item())
