from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import PurePosixPath
import numpy as np
from scipy.stats import t as student_t
from .contracts import PlatformError, RunSpec
from .datasets import load_dataset, save_dataset
from .storage import ROOT, canonical, digest, read_json, run_dir


def comparison(run_ids, metric="operating_profit"):
    if len(run_ids) != 2:
        raise PlatformError("COMPARISON_ARITY", "choose exactly two runs")
    specs = [
        RunSpec.model_validate(read_json(run_dir(r) / "spec.resolved.json"))
        for r in run_ids
    ]
    a, b = specs
    from .metrics import DEFINITIONS

    definition = next((d for d in DEFINITIONS if d["key"] == metric and a.model.id in d["compatible_model_families"]), None)
    if (
        definition is None
        or a.model.id not in definition["compatible_model_families"]
        or definition["sense"] not in {"minimize", "maximize"}
    ):
        raise PlatformError(
            "METRIC_NOT_COMPARABLE",
            "choose a measured operational metric compatible with this model family",
        )
    source_hashes = [
        read_json(run_dir(r) / "manifest.json")["code"]["content_hash"] for r in run_ids
    ]
    if source_hashes[0] != source_hashes[1]:
        raise PlatformError(
            "COMPARISON_CODE_MISMATCH", "compare runs from the same source snapshot"
        )
    from .families import get_family
    if not get_family(a.model.id).supports_samples(a):
        raise PlatformError(
            "STATISTICS_NOT_APPLICABLE",
            "analytical/calibration outputs are parameter studies, not independent policy scenario trials",
        )
    keys = [
        "accounting_version",
        "information_set",
        "terminal_policy",
        "warmup_periods",
    ]
    if (
        a.dataset.revision != b.dataset.revision
        or a.model.id != b.model.id
        or a.model.parameters != b.model.parameters
        or any(getattr(a.evaluation, k) != getattr(b.evaluation, k) for k in keys)
    ):
        raise PlatformError(
            "INCOMPATIBLE_COMPARISON",
            "pairing requires identical data, physical model, parameters, information and accounting",
        )
    groups = []
    for rid in run_ids:
        state = read_json(run_dir(rid) / "state.json")
        if state["status"] != "COMPLETED":
            raise PlatformError(
                "FAILED_RUN_COMPARISON",
                f"run {rid} is {state['status']}; failures are not omitted",
            )
        rows = read_json(run_dir(rid) / "metrics.json")["rows"]
        grouped = {}
        for row in rows:
            value = row["metrics"].get(metric)
            if not isinstance(value, (int, float)):
                raise PlatformError("METRIC_UNAVAILABLE", "selected metric unavailable")
            grouped.setdefault(row["scenario_id"], []).append(value)
        groups.append({sid: float(np.mean(values)) for sid, values in grouped.items()})
    if set(groups[0]) != set(groups[1]):
        raise PlatformError("UNPAIRED_SCENARIOS", "scenario sets must match exactly")
    pairs = [
        {
            "scenario_id": sid,
            "a": groups[0][sid],
            "b": groups[1][sid],
            "difference": groups[0][sid] - groups[1][sid],
        }
        for sid in sorted(groups[0])
    ]
    diffs = np.array([p["difference"] for p in pairs])
    n = len(diffs)
    mean = float(diffs.mean())
    half = (
        float(student_t.ppf(0.975, n - 1) * diffs.std(ddof=1) / np.sqrt(n))
        if n >= 2
        else None
    )
    baseline = float(np.mean(list(groups[1].values())))
    improvement = mean if definition["sense"] == "maximize" else -mean
    return {
        "metric": metric,
        "direction": "A-B",
        "metric_sense": definition["sense"],
        "pairs": pairs,
        "n": n,
        "mean_difference": mean,
        "confidence_interval_95": [mean - half, mean + half]
        if half is not None
        else None,
        "ci_reason": None
        if half is not None
        else "at least two independent scenarios required",
        "percent_improvement": improvement / baseline * 100 if baseline > 0 else None,
        "percent_reason": None if baseline > 0 else "baseline is zero or negative",
        "failed_runs": 0,
        "method": "scenario paired Student-t; policy seeds averaged within scenario",
    }


def export_bundle(rid):
    target = run_dir(rid)
    spec = RunSpec.model_validate(read_json(target / "spec.resolved.json"))
    data, _ = load_dataset(spec.dataset)
    files = {"dataset.json": canonical(data), "spec.json": canonical(spec.model_dump())}
    for name in (
        "manifest.json",
        "environment.json",
        "plugins.lock.json",
        "code.snapshot.zip",
        "checkpoint.json",
        "weights.json",
        "weights.pt",
        "metrics.json", "trace.json", "metric-definitions.json", "charts.json", "events.jsonl",
    ):
        path = target / name
        if path.exists():
            files["run/" + name] = path.read_bytes()
    for name in ("uv.lock", "frontend/package-lock.json"):
        if (ROOT / name).exists():
            files["locks/" + name] = (ROOT / name).read_bytes()
    from .plugins import schemas

    files["schemas.json"] = canonical(schemas())
    from .families import get_family
    ref = get_family(spec.model.id).checkpoint_reference(spec)
    if ref:
        source = run_dir(ref.parameters["checkpoint_run"])
        for name in ("checkpoint.json", "weights.json", "weights.pt"):
            if (source / name).exists():
                files["checkpoint/" + name] = (source / name).read_bytes()
    files["checksums.json"] = canonical(
        {
            "schema_version": "reproduction/v1",
            "data_included": True,
            "dataset_hash": spec.dataset.revision,
            "files": {
                name: {"sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
                for name, value in files.items()
            },
        }
    )
    out = target / "reproduction.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return {"run_id": rid, "artifact_id": "reproduction.zip", "data_included": True}


def validate_bundle(content):
    if len(content) > 100 * 1024 * 1024:
        raise PlatformError("BUNDLE_SIZE", "bundle too large")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise PlatformError("BUNDLE_DUPLICATE", "duplicate archive paths")
            if (
                sum(i.file_size for i in archive.infolist()) > 200 * 1024 * 1024
                or len(names) > 10000
            ):
                raise PlatformError("BUNDLE_SIZE", "expanded bundle too large")
            for name in names:
                p = PurePosixPath(name)
                if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
                    raise PlatformError("BUNDLE_PATH", "unsafe archive path")
            from .contracts import strict_json

            manifest = strict_json(archive.read("checksums.json").decode())
            if manifest["schema_version"] != "reproduction/v1":
                raise PlatformError("BUNDLE_SCHEMA", "unknown bundle version")
            if set(names) != (set(manifest["files"]) | {"checksums.json"}):
                raise PlatformError("BUNDLE_FILES", "unexpected or missing files")
            files = {name: archive.read(name) for name in manifest["files"]}
            for name, info in manifest["files"].items():
                if (
                    len(files[name]) != info["size"]
                    or hashlib.sha256(files[name]).hexdigest() != info["sha256"]
                ):
                    raise PlatformError("BUNDLE_HASH", "checksum mismatch")
            spec = RunSpec.model_validate(strict_json(files["spec.json"].decode()))
            data = strict_json(files["dataset.json"].decode())
            from .datasets import validate_payload

            validate_payload(data)
            if (
                digest(data) != spec.dataset.revision
                or digest(data) != manifest["dataset_hash"]
            ):
                raise PlatformError("BUNDLE_DATA_HASH", "frozen data hash mismatch")
            source_manifest = strict_json(files["run/manifest.json"].decode())
            return {
                "valid": True,
                "data_included": True,
                "files": len(files),
                "spec": spec.model_dump(),
                "dataset": data,
                "source_code": source_manifest["code"],
                "code_execution": "not executed; inspect code and set up matching environment before rerun",
            }
    except (zipfile.BadZipFile, KeyError, UnicodeError) as exc:
        raise PlatformError(
            "INVALID_BUNDLE", "bundle is incomplete or not a valid ZIP"
        ) from exc


def reproduce(content):
    checked = validate_bundle(content)
    spec = checked["spec"]
    from .service import current_code_hash

    if checked["source_code"].get("content_hash") != current_code_hash():
        raise PlatformError(
            "CODE_VERSION_MISMATCH",
            "restore matching source before reproducing; bundled code is never auto-executed",
        )
    save_dataset(spec["dataset"]["dataset_id"], checked["dataset"])
    # Never unpickle checkpoint/code from an imported archive. Existing registered
    # checkpoint must already be available, otherwise validation refuses the run.
    from .service import plan

    return {
        "validated": True,
        "plan": plan(spec),
        "exact_code_verified": True,
        "reason": "current source matches recorded snapshot; no untrusted code is loaded",
    }
