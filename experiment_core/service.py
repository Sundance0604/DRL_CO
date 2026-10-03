from __future__ import annotations

import itertools
import json
import os
import platform
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from .contracts import BatchSpec, PlatformError, RunSpec
from .datasets import load_dataset
from .plugins import resolve
from .storage import (
    ROOT,
    atomic_json,
    digest,
    events,
    read_json,
    run_dir,
    workspace,
)

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "INTERRUPTED"}


def merge(base, patch):
    result = json.loads(json.dumps(base))
    for key, value in patch.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def plan(config):
    batch = (
        BatchSpec.model_validate(config)
        if "base_spec" in config
        else BatchSpec(base_spec=RunSpec.model_validate(config))
    )
    from .batching import expand_ranges
    sweeps = dict(batch.sweeps)
    for key, values in expand_ranges(batch.ranges).items():
        if key in sweeps:
            raise PlatformError("DUPLICATE_SWEEP", "range and explicit values overlap", key)
        sweeps[key] = values
    if any(not values for values in sweeps.values()):
        raise PlatformError("EMPTY_SWEEP", "each sweep must contain values")
    size = 1
    for values in sweeps.values():
        size *= len(values)
        if size > 1000:
            raise PlatformError("BATCH_SIZE", "parameter grid exceeds 1000 conditions")
    keys = list(sweeps)
    combinations = list(itertools.product(*sweeps.values())) if keys else [()]
    if (
        len(combinations)
        * len(batch.variants)
        * max(len(batch.base_spec.execution.policy_seeds), len(batch.base_spec.execution.training_seeds or []))
        > 1000
    ):
        raise PlatformError("BATCH_SIZE", "batch exceeds 1000 runs")
    specs = []
    for variant in batch.variants:
        for values in combinations:
            draft = merge(batch.base_spec.model_dump(), variant)
            if any(k in variant for k in ("model", "controller", "value_function")) and "framework_id" not in variant:
                draft["framework_id"] = None
            for key, value in zip(keys, values):
                from .batching import assign_parameter
                assign_parameter(draft, key, value)
            spec = resolve(draft)
            data, scenarios = load_dataset(spec.dataset)
            from .families import get_family
            f = get_family(spec.model.id)
            f.validate_selection(spec, data, scenarios)
            seeds = (spec.execution.training_seeds or spec.execution.policy_seeds) if f.is_training(spec) else spec.execution.policy_seeds
            if len(specs)+len(seeds)>1000:
                raise PlatformError("BATCH_SIZE","expanded variants and seeds exceed 1000 runs")
            condition = digest({"family":f.FAMILY_ID, "model":spec.model.model_dump(), "controller":spec.controller.model_dump(),
                                "value_function":spec.value_function.model_dump(), "evaluation":spec.evaluation.model_dump(),
                                "solver":spec.solver.model_dump()})
            for seed in seeds:
                specs.append(
                    {
                        "spec": spec.model_dump(),
                        "policy_seed": seed,
                        "scenario_ids": [s["id"] for s in scenarios],
                        "dataset_hash": spec.dataset.revision,
                        "condition_id": condition,
                        "scenario_seeds": {s["id"]:s.get("seed") for s in scenarios},
                        "training_seed": seed if f.is_training(spec) else None,
                        "statistical_unit": "independent-scenario",
                    }
                )
    return {
        "schema_version": "execution-plan/v1",
        "count": len(specs),
        "runs": specs,
        "resource_policy": "spawn isolated processes; no nested pools",
    }


def environment():
    import importlib.metadata as im

    versions = {}
    for name in (
        "numpy",
        "scipy",
        "torch",
        "fastapi",
        "pydantic",
        "gurobipy",
        "psutil",
    ):
        try:
            versions[name] = im.version(name)
        except im.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "os": platform.system(),
        "architecture": platform.machine(),
        "dependencies": versions,
        "device": "cpu",
    }


def code_files():
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]

    def get(*args):
        p = subprocess.run(git + list(args), cwd=ROOT, capture_output=True)
        return p.stdout if p.returncode == 0 else b""

    files = set(get("ls-files").decode().splitlines()) | set(
        get("ls-files", "--others", "--exclude-standard").decode().splitlines()
    )
    return get, [
        name
        for name in sorted(files)
        if (ROOT / name).is_file() and (ROOT / name).stat().st_size < 20 * 1024 * 1024
    ]


def current_code_hash():
    import hashlib

    _, files = code_files()
    return digest(
        {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    )


def code_record(target):
    get, files = code_files()
    sha = get("rev-parse", "HEAD").decode().strip()
    dirty = bool(get("status", "--porcelain"))
    record = {
        "git_sha": sha or None,
        "dirty": dirty,
        "code_available": bool(sha),
        "lock_hashes": {},
        "content_hash": current_code_hash(),
    }
    for name in ("uv.lock", "frontend/package-lock.json"):
        file = ROOT / name
        if file.exists():
            import hashlib

            record["lock_hashes"][name] = hashlib.sha256(file.read_bytes()).hexdigest()
    if files:
        # Includes untracked source files; ignored datasets, secrets and caches
        # never become source snapshots. A hash alone is not reproducibility.
        import zipfile

        with zipfile.ZipFile(
            target / "code.snapshot.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for relative in sorted(files):
                file = ROOT / relative
                if file.is_file() and file.stat().st_size < 20 * 1024 * 1024:
                    archive.write(file, relative)
        record["snapshot"] = "code.snapshot.zip"
    return record


class Coordinator:
    def __init__(self, max_parallel=1, cpu_budget=None):
        if not isinstance(max_parallel, int) or not 1 <= max_parallel <= 16:
            raise PlatformError(
                "RESOURCE_CONFIG", "max_parallel must be between 1 and 16"
            )
        self.root = workspace()
        self.guard = threading.RLock()
        self.max_parallel = max_parallel
        self.cpu_budget = cpu_budget or max(1, os.cpu_count() or 1)
        self.lock = (self.root / "coordinator.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.lock.seek(0)
                self.lock.write(b"0")
                self.lock.flush()
                self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock.close()
            raise PlatformError(
                "WORKSPACE_BUSY",
                "workspace already has a coordinator; connect to the running API",
                exit_code=3,
            ) from exc
        self.db = sqlite3.connect(self.root / "index.sqlite", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,batch TEXT,status TEXT,seed INTEGER,spec TEXT,submitted TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS requests(key TEXT PRIMARY KEY,hash TEXT,batch TEXT)"
        )
        self.db.commit()
        self.active = {}
        self.stopping = False
        # SQLite is an index, not the sole source of experimental facts.
        for manifest_path in (self.root / "runs").glob("*/manifest.json"):
            target = manifest_path.parent
            if self.db.execute(
                "SELECT 1 FROM runs WHERE id=?", (target.name,)
            ).fetchone():
                continue
            manifest = read_json(manifest_path)
            spec = read_json(target / "spec.resolved.json")
            submitted = read_json(target / "spec.submitted.json")
            state = read_json(target / "state.json")
            self.db.execute(
                "INSERT INTO runs VALUES(?,?,?,?,?,?)",
                (
                    target.name,
                    manifest["batch_id"],
                    state["status"],
                    manifest.get("training_seed") if manifest.get("training_seed") is not None else manifest["policy_seed"],
                    json.dumps(spec),
                    json.dumps(submitted),
                ),
            )
        self.db.commit()
        # A restarted coordinator does not silently resume or duplicate work.
        for rid, status in self.db.execute(
            "SELECT id,status FROM runs WHERE status='RUNNING'"
        ).fetchall():
            state_path = run_dir(rid) / "state.json"
            state = read_json(state_path) if state_path.exists() else {}
            if state.get("status") in TERMINAL:
                self._set_status(rid, state["status"])
            else:
                owner_path = run_dir(rid) / "owner.json"
                if owner_path.exists():
                    self._terminate_owned(read_json(owner_path))
                atomic_json(
                    state_path,
                    {
                        "status": "INTERRUPTED",
                        "reason": "coordinator restarted; no automatic resume",
                    },
                )
                self._set_status(rid, "INTERRUPTED")

    def _set_status(self, rid, status):
        self.db.execute("UPDATE runs SET status=? WHERE id=?", (status, rid))
        self.db.commit()

    @staticmethod
    def _terminate_owned(owner):
        import psutil

        try:
            process = psutil.Process(owner["pid"])
            if (
                abs(process.create_time() - owner["create_time"]) > 0.01
                or process.environ().get("DRL_RUN_TOKEN") != owner["token"]
            ):
                return False
            children = process.children(recursive=True)
            for child in reversed(children):
                child.terminate()
            process.terminate()
            _, alive = psutil.wait_procs(children + [process], timeout=2)
            for child in alive:
                child.kill()
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            return False

    def submit(self, config, idempotency_key=None):
        expanded = plan(config)
        if any(
            r["spec"]["solver"]["parameters"]["threads"] > self.cpu_budget
            for r in expanded["runs"]
        ):
            raise PlatformError(
                "CPU_BUDGET", "solver threads exceed coordinator CPU budget"
            )
        with self.guard:
            if idempotency_key:
                existing = self.db.execute(
                    "SELECT hash,batch FROM requests WHERE key=?", (idempotency_key,)
                ).fetchone()
                if existing:
                    if existing[0] != digest(config):
                        raise PlatformError(
                            "IDEMPOTENCY_CONFLICT",
                            "same key used for different request",
                        )
                    return self.batch(existing[1])
            bid = "b-" + uuid.uuid4().hex[:16]
            for item in expanded["runs"]:
                rid = "r-" + uuid.uuid4().hex[:20]
                target = run_dir(rid)
                target.mkdir(parents=True)
                resolved = item["spec"]
                atomic_json(target / "spec.submitted.json", config)
                atomic_json(target / "spec.resolved.json", resolved)
                atomic_json(target / "dataset.refs.json", resolved["dataset"])
                atomic_json(target / "environment.json", environment())
                from .plugins import descriptors

                atomic_json(target / "plugins.lock.json", descriptors(resolved["family"]))
                frozen, _ = load_dataset(RunSpec.model_validate(resolved).dataset)
                atomic_json(target / "dataset.snapshot.json", frozen)
                atomic_json(
                    target / "manifest.json",
                    {
                        "run_id": rid,
                        "batch_id": bid,
                        "policy_seed": item["policy_seed"] if item["training_seed"] is None else None,
                        "training_seed": item["training_seed"],
                        "scenario_seeds": item["scenario_seeds"],
                        "condition_id": item["condition_id"],
                        "family": resolved["family"],
                        "framework_id": resolved["framework_id"],
                        "statistical_unit": item["statistical_unit"],
                        "dataset_hash": item["dataset_hash"],
                        "scenario_ids": item["scenario_ids"],
                        "code": code_record(target),
                        "platform_version": "0.2.0",
                        "limitations": (["rollout conditional first collection batch is approximate"]
                                        if resolved["family"] == "single" and resolved["controller"]["id"] == "rollout" else [])
                                        + ["native macOS launch not tested"],
                    },
                )
                atomic_json(target / "state.json", {"status": "QUEUED"})
                self.db.execute(
                    "INSERT INTO runs VALUES(?,?,?,?,?,?)",
                    (
                        rid,
                        bid,
                        "QUEUED",
                        item["policy_seed"],
                        json.dumps(resolved),
                        json.dumps(config),
                    ),
                )
            if idempotency_key:
                self.db.execute(
                    "INSERT INTO requests VALUES(?,?,?)",
                    (idempotency_key, digest(config), bid),
                )
            self.db.commit()
            return self.batch(bid)

    def list_runs(self):
        with self.guard:
            return [
                self.run(r[0])
                for r in self.db.execute(
                    "SELECT id FROM runs ORDER BY rowid DESC"
                ).fetchall()
            ]

    def run(self, rid):
        from .storage import identifier

        identifier(rid)
        with self.guard:
            row = self.db.execute(
                "SELECT batch,status,seed,spec FROM runs WHERE id=?", (rid,)
            ).fetchone()
            if not row:
                raise PlatformError("NOT_FOUND", "run does not exist")
            result = {
                "run_id": rid,
                "batch_id": row[0],
                "status": row[1],
                "policy_seed": row[2],
                "spec": json.loads(row[3]),
                "events": events(rid),
            }
            metrics = run_dir(rid) / "metrics.json"
            manifest = read_json(run_dir(rid) / "manifest.json")
            result.update({key:manifest.get(key) for key in ("policy_seed","training_seed","scenario_seeds","condition_id","family","framework_id")})
            if metrics.exists():
                result["metrics"] = read_json(metrics)
            result["artifacts"] = [
                p.name
                for p in run_dir(rid).iterdir()
                if p.is_file()
                and p.name not in ("owner.json", "cancel.json")
                and not p.name.endswith(".partial")
            ]
            return result

    def batch(self, bid):
        with self.guard:
            ids = [
                r[0]
                for r in self.db.execute(
                    "SELECT id FROM runs WHERE batch=?", (bid,)
                ).fetchall()
            ]
            if not ids:
                raise PlatformError("NOT_FOUND", "batch does not exist")
            return {
                "batch_id": bid,
                "runs": [
                    {"run_id": rid, "status": self.run(rid)["status"]} for rid in ids
                ],
            }

    def cancel(self, rid):
        with self.guard:
            current = self.run(rid)
            if current["status"] == "QUEUED":
                atomic_json(run_dir(rid) / "state.json", {"status": "CANCELLED"})
                self._set_status(rid, "CANCELLED")
            elif current["status"] == "RUNNING":
                atomic_json(run_dir(rid) / "cancel.json", {"requested_at": time.time()})
            return self.run(rid)

    def rerun(self, rid, idempotency_key=None):
        run = self.run(rid)
        original = read_json(run_dir(rid) / "manifest.json")["code"]
        if original.get("content_hash") != current_code_hash():
            raise PlatformError(
                "CODE_VERSION_MISMATCH",
                "source changed since this run; use Copy configuration for a new-version experiment",
            )
        return self.submit(run["spec"], idempotency_key)

    def tick(self):
        with self.guard:
            used = 0
            for rid, job in list(self.active.items()):
                process = job["process"]
                target = run_dir(rid)
                cancel = target / "cancel.json"
                timed_out = time.monotonic() - job["start"] > job["timeout"]
                if cancel.exists() or timed_out:
                    if not job.get("cancel_started"):
                        job["cancel_started"] = time.monotonic()
                        if timed_out:
                            atomic_json(
                                cancel, {"requested_at": time.time(), "timeout": True}
                            )
                    elif time.monotonic() - job["cancel_started"] > 3:
                        self._terminate_owned(job["owner"])
                        atomic_json(
                            target / "state.json",
                            {
                                "status": "TIMEOUT" if timed_out else "CANCELLED",
                                "forced": True,
                            },
                        )
                if process.poll() is not None:
                    state = read_json(target / "state.json")
                    status = (
                        state["status"] if state.get("status") in TERMINAL else "FAILED"
                    )
                    self._set_status(rid, status)
                    job["stdout"].close()
                    job["stderr"].close()
                    del self.active[rid]
                else:
                    used += job["threads"]
            if self.stopping:
                return
            for rid, seed, raw in self.db.execute(
                "SELECT id,seed,spec FROM runs WHERE status='QUEUED' ORDER BY rowid"
            ).fetchall():
                if len(self.active) >= self.max_parallel:
                    break
                spec = json.loads(raw)
                threads = spec["solver"]["parameters"]["threads"]
                if used + threads > self.cpu_budget:
                    continue
                target = run_dir(rid)
                # Execute the source captured at submission; queued tasks never
                # silently use a subsequently edited working tree.
                import zipfile

                code = target / "code"
                code.mkdir(exist_ok=True)
                with zipfile.ZipFile(target / "code.snapshot.zip") as archive:
                    for name in archive.namelist():
                        if not (code / name).resolve().is_relative_to(code.resolve()):
                            raise PlatformError(
                                "CODE_SNAPSHOT_PATH", "unsafe source snapshot"
                            )
                    archive.extractall(code)
                token = uuid.uuid4().hex
                stdout = (target / "stdout.log").open("wb")
                stderr = (target / "stderr.log").open("wb")
                env = os.environ.copy()
                env.update(
                    DRL_WORKSPACE=str(self.root),
                    DRL_RUN_TOKEN=token,
                    PYTHONDONTWRITEBYTECODE="1",
                    OMP_NUM_THREADS="1",
                )
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "experiment_core.worker",
                        rid,
                        str(seed),
                    ],
                    cwd=code,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                )
                import psutil

                owner = {
                    "pid": process.pid,
                    "create_time": psutil.Process(process.pid).create_time(),
                    "token": token,
                }
                atomic_json(target / "owner.json", owner)
                self.active[rid] = {
                    "process": process,
                    "stdout": stdout,
                    "stderr": stderr,
                    "owner": owner,
                    "start": time.monotonic(),
                    "threads": threads,
                    "timeout": spec["execution"]["timeout_seconds"],
                }
                self._set_status(rid, "RUNNING")
                used += threads

    def close(self):
        if self.lock.closed:
            return
        with self.guard:
            self.stopping = True
            for rid in self.active:
                self.cancel(rid)
        limit = time.monotonic() + 5
        while self.active and time.monotonic() < limit:
            self.tick()
            time.sleep(0.05)
        for job in self.active.values():
            self._terminate_owned(job["owner"])
            job["stdout"].close()
            job["stderr"].close()
        self.db.close()
        if os.name == "nt":
            import msvcrt

            self.lock.seek(0)
            msvcrt.locking(self.lock.fileno(), msvcrt.LK_UNLCK, 1)
        self.lock.close()
