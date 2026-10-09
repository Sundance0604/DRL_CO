"""Complete-episode on-policy PPO and validation; checkpoint after each update."""
import random
import time
from pathlib import Path
import torch
from ...strategic.artifacts import write_json
from .backend import advantages, update
from .checkpoints import load_recovery, save_checkpoint
from .environment import PPOEnvironment
from .networks import PolicyNetwork, sample_action


def run_episode(episode, action):
    observation = episode.reset()
    while True:
        transition = episode.step(action(observation))
        if transition.terminated:
            return episode.final_ledger
        observation = transition.observation


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, (tuple, list)) else value


def train(factory, train_scenarios, validation_scenarios, schema, environment, baseline,
          config, output, resume=None):
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(target/"status.json", dict(status="RUNNING"))
    with torch.random.fork_rng():
        torch.manual_seed(config.training_seed)
        network = PolicyNetwork(len(schema.hubs), config.hidden_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    sampling = torch.Generator().manual_seed(config.training_seed+1001)
    minibatches = torch.Generator().manual_seed(config.training_seed+1002)
    data_rng = random.Random(config.training_seed+1003)
    first, completed, best_score, best_actor, best_update = 0, 0, None, None, None
    progress = dict(updates=0, episodes=0, best_validation_USD=None, training_seed=config.training_seed)
    if resume:
        try:
            progress, recovery = load_recovery(resume, network, optimizer, environment, config)
            first, completed, best_score = progress["updates"], progress["episodes"], progress["best_validation_USD"]
            sampling.set_state(recovery["sampling_rng"])
            minibatches.set_state(recovery["minibatch_rng"])
            data_rng.setstate(_tuples(recovery["data_rng"]))
            best_actor, best_update = recovery["best_actor"], progress.get("best_validation_update")
            if first >= config.updates:
                raise ValueError("resume requires a greater total updates budget")
        except BaseException as exc:
            write_json(target/"status.json", dict(status="FAILED", error=type(exc).__name__, message=str(exc)))
            raise
    try:
        for iteration in range(first, config.updates):
            rows, episodes = [], []
            network.train()
            for _ in range(config.episodes_per_update):
                scenario = train_scenarios[data_rng.randrange(len(train_scenarios))]
                episode = factory(scenario)
                env = PPOEnvironment(episode, config.reward_scale)
                observation, _ = env.reset()
                daily, rewards, values = [], [], []
                while True:
                    action, latent, logp, value = sample_action(network, observation, sampling)
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    if truncated:
                        raise RuntimeError("incomplete rollout cannot enter PPO")
                    daily.append(dict(observation=observation, latent=latent, logp=logp.item()))
                    rewards.append(reward)
                    values.append(value.item())
                    if terminated:
                        break
                    observation = next_observation
                adv, returns = advantages(rewards, values, config.gae_lambda, config.gamma)
                for row, a, ret in zip(daily, adv, returns):
                    row.update(advantage=a, return_=ret)
                    row["return"] = row.pop("return_")
                rows.extend(daily)
                completed += 1
                write_json(target/f"training-episode-{completed:06d}.json", dict(
                    scenario_id=scenario["id"], scenario_seed=scenario.get("seed"),
                    training_seed=config.training_seed, policy_version=iteration,
                    ledger=episode.final_ledger, metrics=episode.metrics, transitions=episode.records))
                episodes.append(episode.final_ledger["realized_profit"])
            diagnostic = update(network, optimizer, rows, config, minibatches)
            score = None
            network.eval()
            if (iteration+1) % config.validation_every == 0 or iteration+1 == config.updates:
                profits, validation = [], []
                for index, scenario in enumerate(validation_scenarios):
                    episode = factory(scenario)
                    ledger = run_episode(episode, lambda o: sample_action(network, o, sampling, True)[0].tolist())
                    profits.append(ledger["realized_profit"])
                    record = dict(scenario_id=scenario["id"], scenario_seed=scenario.get("seed"),
                                  training_seed=config.training_seed, update=iteration+1,
                                  ledger=ledger, metrics=episode.metrics, transitions=episode.records)
                    filename = f"validation-update-{iteration+1:04d}-scenario-{index:04d}.json"
                    write_json(target/filename, record)
                    validation.append(dict(scenario_id=scenario["id"], profit_USD=ledger["realized_profit"], file=filename))
                score = sum(profits)/len(profits)
                write_json(target/f"validation-{iteration+1:04d}.json", dict(update=iteration+1,
                    deterministic=True, scenarios=validation, mean_profit_USD=score))
            improved = score is not None and (best_score is None or score > best_score)
            if improved:
                best_score = score
                best_update = iteration+1
                best_actor = {k: v.detach().clone() for k, v in network.state_dict().items()
                              if not k.startswith("critic.")}
            progress = dict(updates=iteration+1, episodes=completed, best_validation_USD=best_score,
                            training_seed=config.training_seed, best_validation_update=best_update)
            recovery = dict(optimizer=optimizer.state_dict(), sampling_rng=sampling.get_state(),
                            minibatch_rng=minibatches.get_state(), data_rng=data_rng.getstate(), best_actor=best_actor)
            package = target/f"update-{iteration+1:04d}"
            save_checkpoint(package, network, environment, config, baseline, progress, recovery)
            if improved:
                write_json(target/"best.json", dict(checkpoint=package.name, validation_USD=score,
                                                   selection="maximum deterministic validation mean USD"))
            write_json(target/f"update-{iteration+1:04d}.json", dict(**diagnostic,
                training_mean_USD=sum(episodes)/len(episodes), validation_mean_USD=score,
                elapsed_seconds=time.monotonic()-started, transitions=len(rows), **progress))
        # Export a clean inference package even if resuming from an earlier run
        # whose best actor predates this attempt's updates.
        save_checkpoint(target/"last-inference", network, environment, config, baseline, progress)
        with torch.random.fork_rng():
            selected = PolicyNetwork(len(schema.hubs), config.hidden_size, False)
        selected.load_state_dict(best_actor, strict=True)
        save_checkpoint(target/"best-inference", selected, environment, config, baseline, progress)
        write_json(target/"best.json", dict(checkpoint="best-inference", validation_USD=best_score,
            update=best_update, selection="maximum deterministic validation mean USD; first tie retained"))
        write_json(target/"status.json", dict(status="COMPLETED", **progress))
        return progress
    except BaseException as exc:
        from ...strategic.contracts import EpisodeCancelled
        write_json(target/"status.json", dict(status="CANCELLED" if isinstance(exc, (EpisodeCancelled, KeyboardInterrupt))
                                             else "FAILED", error=type(exc).__name__, message=str(exc), **progress))
        raise
