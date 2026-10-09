"""Explicit research train/evaluate/preflight APIs; existing platform is unchanged."""
from dataclasses import asdict
from pathlib import Path
from ...data import validate_payload
from ...parameters import MatchingParameters
from ...strategic.artifacts import read_json, write_json
from ...strategic.baseline import FrozenBaseline
from ...strategic.contracts import ActionSchema, identity
from ...strategic.episode import StrategicEpisode
from ...strategic.observation import FeatureScales, dimensions
from .config import PPOConfig


def physical_signature(scenario):
    return dict(hubs=sorted(scenario["nodes"]), edges=sorted(
        [*sorted((a, b)), w] for a, b, w in scenario["edges"]),
        fleet_size=len(scenario["hubs"]), horizon=scenario["horizon"])


def scenario_content(scenario):
    # IDs, split labels and seeds may not disguise duplicate physical instances.
    return identity({k: v for k, v in scenario.items() if k not in ("id", "seed", "split")})


def validate_research_payload(dataset):
    """Larger explicit research pools, without changing platform upload limits."""
    scenarios = dataset.get("scenarios", [])
    if not scenarios:
        validate_payload(dataset)
    if len(scenarios) > 10000 or len({s["id"] for s in scenarios}) != len(scenarios):
        raise ValueError("research dataset size/duplicate scenario identity")
    for offset in range(0, len(scenarios), 100):
        validate_payload(dict(dataset, scenarios=scenarios[offset:offset+100]))


def prepare(configuration, dataset):
    validate_research_payload(dataset)
    allowed = {"parameters", "solver", "action", "feature_scales", "baseline", "ppo"}
    if set(configuration)-allowed:
        raise ValueError("unknown strategic run configuration fields")
    parameters = MatchingParameters.model_validate(configuration.get("parameters", {})).model_dump()
    from experiment_core.contracts import SolverParameters
    solver = SolverParameters.model_validate(configuration.get("solver", {})).model_dump()
    action = ActionSchema(**configuration["action"])
    scales = FeatureScales(**configuration.get("feature_scales", {}))
    model = configuration["baseline"]
    baseline = FrozenBaseline(read_json(model) if isinstance(model, str) else model)
    ppo = PPOConfig.model_validate(configuration.get("ppo", {}))
    scenarios = dataset["scenarios"]
    physics = physical_signature(scenarios[0])
    if any(physical_signature(s) != physics for s in scenarios):
        raise ValueError("first prototype requires one frozen network, horizon and fleet size per run")
    if set(action.hubs) != set(physics["hubs"]):
        raise ValueError("action/network hub mismatch")
    baseline.validate(action.hubs, physics["fleet_size"])
    by_split = {split: [s for s in scenarios if s["split"] == split] for split in ("train", "validation", "test")}
    split_identities = {}
    seen = {}
    for split, records in by_split.items():
        split_identities[split] = []
        for s in records:
            content = scenario_content(s)
            for key in (("content", content), ("seed", s.get("seed"))):
                if key[1] is not None and key in seen and seen[key] != split:
                    raise ValueError("train/validation/test overlap in content or scenario seed")
                seen[key] = split
            split_identities[split].append(dict(id=s["id"], seed=s.get("seed"), content_hash=content,
                                                full_hash=identity(s)))
    environment = dict(action=asdict(action), feature_scales=asdict(scales),
                       feature_dimensions=dimensions(len(action.hubs)), parameters=parameters,
                       solver=solver, physics=physics, data_splits=split_identities,
                       baseline_hash=baseline.hash)
    # Normalize tuples to JSON types before compatibility comparisons on resume.
    import json
    environment = json.loads(json.dumps(environment))
    factory = lambda s: StrategicEpisode(s, parameters, solver, action, scales, baseline,
                                          ppo.episode_timeout_seconds)
    return factory, by_split, action, environment, baseline, ppo


def train_run(configuration, dataset, output, resume=None):
    factory, splits, action, environment, baseline, config = prepare(configuration, dataset)
    if not splits["train"] or not splits["validation"]:
        raise ValueError("training requires independent train and validation scenarios")
    from .trainer import train
    return train(factory, splits["train"], splits["validation"], action, environment,
                 baseline, config, output, resume)


def evaluate_run(checkpoint, dataset, output, policy_seed=0, deterministic=True):
    validate_payload(dataset)
    from .policy import FrozenPolicy
    from .trainer import run_episode
    policy = FrozenPolicy(checkpoint, policy_seed, deterministic)
    environment = policy.environment
    schema, scales = ActionSchema(**environment["action"]), FeatureScales(**environment["feature_scales"])
    tests = [s for s in dataset["scenarios"] if s["split"] == "test"]
    if not tests:
        raise ValueError("formal evaluation requires test scenarios")
    used = [entry for split in ("train", "validation") for entry in environment["data_splits"][split]]
    for s in tests:
        if physical_signature(s) != environment["physics"]:
            raise ValueError("evaluation network/horizon/fleet compatibility mismatch")
        if any(s["id"] == e["id"] or scenario_content(s) == e["content_hash"] or
               (s.get("seed") is not None and s.get("seed") == e["seed"]) for e in used):
            raise ValueError("formal evaluation overlaps training/model selection")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    write_json(target/"status.json", dict(status="RUNNING"))
    results = []
    try:
        for index, scenario in enumerate(tests):
            episode = StrategicEpisode(scenario, environment["parameters"], environment["solver"],
                                        schema, scales, policy.baseline)
            ledger = run_episode(episode, policy.act)
            record = dict(scenario_id=scenario["id"], scenario_seed=scenario.get("seed"),
                policy_seed=policy_seed, deterministic=deterministic,
                training_seed=policy.manifest["progress"]["training_seed"],
                checkpoint=str(Path(checkpoint).resolve()), baseline_hash=policy.baseline.hash,
                ledger=ledger, metrics=episode.metrics, transitions=episode.records)
            write_json(target/f"evaluation-{index:04d}.json", record)
            results.append({k: v for k, v in record.items() if k != "transitions"})
        write_json(target/"evaluation.json", dict(results=results,
            mean_profit_USD=sum(r["ledger"]["realized_profit"] for r in results)/len(results)))
        write_json(target/"status.json", dict(status="COMPLETED", scenarios=len(results)))
        return results
    except BaseException as exc:
        write_json(target/"status.json", dict(status="CANCELLED" if isinstance(exc, KeyboardInterrupt)
                                             else "FAILED", error=type(exc).__name__, message=str(exc)))
        raise


def preflight_run(configuration, dataset, output):
    """Zero regression, reward settlement and fixed axis perturbation effects.

    Uses engineering training states, never validation/test for scale tuning.
    """
    factory, splits, schema, _, _, _ = prepare(configuration, dataset)
    if not splits["train"]:
        raise ValueError("preflight requires training engineering states")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    results = []
    zero = (0.,)*(2*len(schema.hubs))
    for index, scenario in enumerate(splits["train"]):
        episode = factory(scenario)
        obs = episode.reset()
        probes, rewards = [], []
        while True:
            reference = episode.probe(zero)
            hashes = []
            for axis in range(len(zero)):
                for sign in (-1., 1.):
                    signal = list(zero)
                    signal[axis] = sign
                    hashes.append(episode.probe(signal)["action_hash"])
            transition = episode.step(zero)
            if episode.records[-1]["action_hash"] != reference["action_hash"]:
                raise RuntimeError("zero signal probe/execution mismatch")
            rewards.append(transition.reward)
            probes.append(dict(period=obs.period, baseline_action_hash=reference["action_hash"],
                                perturbation_action_hashes=hashes,
                                changed=sum(h != reference["action_hash"] for h in hashes)))
            if transition.terminated:
                break
            obs = transition.observation
        result = dict(scenario_id=scenario["id"], ledger=episode.final_ledger,
                      reward_sum_USD=sum(rewards), probes=probes, transitions=episode.records)
        write_json(target/f"preflight-{index:04d}.json", result)
        results.append(result)
    write_json(target/"preflight.json", dict(scenarios=len(results),
        changed=sum(p["changed"] for r in results for p in r["probes"]),
        probes=sum(len(p["perturbation_action_hashes"]) for r in results for p in r["probes"]),
        note="No behavior changes is a budget warning; no profit advantage is established."))
    return results
