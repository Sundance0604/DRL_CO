"""Hashed inference packages and full-update-boundary recovery packages."""
import tempfile
from pathlib import Path
import torch
from ...strategic.artifacts import (ARTIFACT_VERSION, contained_file, dependency_identity,
                                    file_hash, protocols, read_json, source_identity, write_json)
from .networks import NETWORK_VERSION, PolicyNetwork


def save_checkpoint(directory, network, environment, config, baseline, progress, recovery=None):
    target = Path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("checkpoint destinations must be new")
    with tempfile.TemporaryDirectory(prefix=".ppo-checkpoint-", dir=target.parent) as scratch:
        stage = Path(scratch)/"package"
        stage.mkdir()
        actor_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()
                       if not k.startswith("critic.")}
        torch.save(actor_state, stage/"actor.pt")
        write_json(stage/"environment.json", environment)
        write_json(stage/"baseline.json", baseline.model)
        write_json(stage/"training-config.json", config.model_dump())
        if recovery is not None:
            torch.save(dict(network=network.state_dict(), **recovery), stage/"recovery.pt")
        files = {p.name: file_hash(p) for p in stage.iterdir() if p.is_file()}
        manifest = dict(schema_version=ARTIFACT_VERSION, protocols=protocols(), network=NETWORK_VERSION,
                        hidden_size=config.hidden_size, source_hash=source_identity(),
                        dependencies=dependency_identity(), baseline_hash=baseline.hash,
                        package="recovery" if recovery is not None else "inference",
                        progress=progress, artifacts=files)
        write_json(stage/"checkpoint.json", manifest)
        stage.rename(target)
    return manifest


def inspect_checkpoint(directory):
    root = Path(directory)
    manifest = read_json(root/"checkpoint.json")
    if (manifest.get("schema_version") != ARTIFACT_VERSION or manifest.get("protocols") != protocols()
            or manifest.get("network") != NETWORK_VERSION):
        raise ValueError("checkpoint family/protocol/network mismatch")
    if manifest["source_hash"] != source_identity() or manifest["dependencies"] != dependency_identity():
        raise ValueError("checkpoint source/dependency mismatch; use the original code snapshot")
    expected_files = {"actor.pt", "environment.json", "baseline.json", "training-config.json"}
    if manifest["package"] == "recovery":
        expected_files.add("recovery.pt")
    elif manifest["package"] != "inference":
        raise ValueError("unknown checkpoint package")
    if set(manifest["artifacts"]) != expected_files:
        raise ValueError("incomplete or unexpected artifact inventory")
    for relative, expected in manifest["artifacts"].items():
        if file_hash(contained_file(root, relative)) != expected:
            raise ValueError(f"checkpoint hash mismatch: {relative}")
    environment = read_json(contained_file(root, "environment.json"))
    from ...strategic.contracts import ActionSchema
    from ...strategic.observation import FeatureScales, dimensions
    action = ActionSchema(**environment["action"])
    FeatureScales(**environment["feature_scales"])
    if (environment["feature_dimensions"] != dimensions(len(action.hubs)) or
            set(environment["physics"]["hubs"]) != set(action.hubs)):
        raise ValueError("checkpoint observation dimensions/hub metadata mismatch")
    from ...strategic.baseline import FrozenBaseline
    baseline = FrozenBaseline(read_json(contained_file(root, "baseline.json")))
    if baseline.hash != manifest["baseline_hash"] or baseline.hash != environment["baseline_hash"]:
        raise ValueError("baseline identity mismatch")
    baseline.validate(action.hubs, environment["physics"]["fleet_size"])
    return manifest, environment, baseline


def load_actor(directory):
    manifest, environment, baseline = inspect_checkpoint(directory)
    # Initialization is private to this load, with no change to other runs' RNG.
    with torch.random.fork_rng():
        network = PolicyNetwork(len(environment["action"]["hubs"]), manifest["hidden_size"], False)
    state = torch.load(contained_file(directory, "actor.pt"), map_location="cpu", weights_only=True)
    network.load_state_dict(state, strict=True)
    network.eval()
    network.requires_grad_(False)
    return network, environment, baseline, manifest


def load_recovery(directory, network, optimizer, expected_environment, config):
    manifest, environment, baseline = inspect_checkpoint(directory)
    if environment != expected_environment or manifest["package"] != "recovery":
        raise ValueError("resume environment/data identity mismatch")
    from .config import PPOConfig
    old = PPOConfig.model_validate(read_json(contained_file(directory, "training-config.json")))
    if old.resume_identity() != config.resume_identity():
        raise ValueError("resume training configuration mismatch")
    recovery = torch.load(contained_file(directory, "recovery.pt"), map_location="cpu", weights_only=True)
    network.load_state_dict(recovery.pop("network"), strict=True)
    optimizer.load_state_dict(recovery.pop("optimizer"))
    return manifest["progress"], recovery
