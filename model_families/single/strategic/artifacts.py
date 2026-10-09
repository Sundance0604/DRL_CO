"""Controlled JSON artifacts, path containment, hashes and protocol identity."""
import hashlib
import json
import os
from pathlib import Path
import platform
import importlib.metadata
from .contracts import ACTION_VERSION, DECODER_VERSION, OBSERVATION_VERSION, REWARD_VERSION, identity

ARTIFACT_VERSION = "single-strategic-checkpoint/v1"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contained_file(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("checkpoint artifact path must be relative")
    target = (root/relative).resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("checkpoint artifact path escapes checkpoint")
    return target


def source_identity():
    repo = Path(__file__).resolve().parents[3]
    files = [*sorted((repo/"model_families"/"single").rglob("*.py")),
             *sorted((repo/"experiment_core").glob("*.py"))]
    for name in ("pyproject.toml", "uv.lock"):
        if (repo/name).is_file():
            files.append(repo/name)
    return identity({p.relative_to(repo).as_posix(): file_hash(p) for p in files})


def dependency_identity():
    return dict(python=platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ("torch", "numpy", "gurobipy", "pydantic")})


def protocols():
    return dict(family="single", observation=OBSERVATION_VERSION, action=ACTION_VERSION,
                decoder=DECODER_VERSION, reward=REWARD_VERSION,
                environment="single-complete-episode/v1", information="public-booked-only",
                terminal_policy="cooldown_realized", planning_horizon=1)
