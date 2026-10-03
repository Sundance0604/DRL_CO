from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from .contracts import PlatformError, strict_json

ROOT = Path(__file__).resolve().parents[1]


def workspace():
    path = Path(os.environ.get("DRL_WORKSPACE", str(ROOT / "workspace"))).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value
    ):
        raise PlatformError("INVALID_ID", "invalid resource identifier")
    return value


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, partial = tempfile.mkstemp(suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.unlink(partial)


def read_json(path):
    try:
        return strict_json(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise PlatformError("NOT_FOUND", "resource does not exist") from exc


def run_dir(run_id):
    return workspace() / "runs" / identifier(run_id)


def event(run_path, kind, **data):
    path = Path(run_path) / "events.jsonl"
    # Each run has one worker writer. Cancellation requests live in a separate file.
    sequence = sum(1 for _ in path.open(encoding="utf-8")) + 1 if path.exists() else 1
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            canonical({"sequence": sequence, "type": kind, **data}).decode() + "\n"
        )
        stream.flush()


def events(run_id, after=0):
    path = run_dir(run_id) / "events.jsonl"
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = strict_json(line)
            if item["sequence"] > after:
                result.append(item)
        except (PlatformError, KeyError):
            break  # a still-being-written final line is not an event yet
    return result
