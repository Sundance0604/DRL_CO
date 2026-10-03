from __future__ import annotations

import io
import json
import time
import zipfile
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from experiment_core.contracts import (
    DatasetRef,
    PlatformError,
    RunSpec,
    BHHParameters,
    strict_json,
)
from experiment_core.datasets import generate, load_dataset, save_dataset, import_orders
from experiment_core.plugins import resolve
from experiment_core.service import Coordinator, plan
from experiment_core.storage import digest, read_json, run_dir, atomic_json
from experiment_core.runner import matching
from experiment_core.reproduction import comparison, export_bundle, validate_bundle
from experiment_core.bhh import (
    inverse_f,
    capacity,
    direct_capacity,
    steady,
    finite,
    rolling,
)
from platform_api.app import create_app


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DRL_WORKSPACE", str(tmp_path))
    return tmp_path


def config(isolated, split="test"):
    d = generate(
        {
            "dataset_id": "tiny",
            "seeds": [101, 102],
            "splits": ["train", "test"],
            "horizon": 3,
            "orders_per_step": 1,
            "first_mile": "none",
        }
    )
    return {
        "name": "tiny",
        "dataset": {"dataset_id": "tiny", "revision": d["revision"], "split": split},
        "model": {"id": "single_level_matching"},
    }


def completed(co, batch):
    until = time.monotonic() + 30
    while time.monotonic() < until:
        co.tick()
        status = co.batch(batch["batch_id"])
        if all(
            r["status"] in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
            for r in status["runs"]
        ):
            return status
        time.sleep(0.03)
    pytest.fail("worker did not complete")


def test_strict_unknown_fields_and_nonfinite():
    with pytest.raises(PlatformError):
        strict_json('{"v":NaN}')
    with pytest.raises(ValidationError):
        RunSpec.model_validate({"name": "bad", "unknown": 4})
    with pytest.raises(PlatformError):
        resolve(
            {
                "name": "bad",
                "dataset": {"dataset_id": "x", "revision": "a"},
                "model": {"id": "single_level_matching"},
                "controller": {"id": "rolling_horizon"},
            }
        )


def test_frozen_revision_and_mapping(isolated):
    cfg = config(isolated)
    ref = DatasetRef.model_validate(cfg["dataset"])
    data, _ = load_dataset(ref)
    assert generate(data["provenance"]["config"])["revision"] == ref.revision
    duplicate = json.loads(json.dumps(data))
    duplicate["scenarios"][0]["orders"] *= 2
    with pytest.raises(PlatformError, match="unique"):
        save_dataset("bad", duplicate)
    template = dict(data["scenarios"][0])
    template.pop("orders")
    o = data["scenarios"][0]["orders"][0]
    manifest = import_orders(
        "imported", json.dumps([o]), "json", {k: k for k in o}, template
    )
    assert manifest["family"] == "single_level_matching"
    path = isolated / "datasets" / "tiny" / ref.revision / "normalized.json"
    path.write_text(json.dumps(data | {"provenance": {}}))
    with pytest.raises(PlatformError, match="hash"):
        load_dataset(ref)


def test_plan_sweeps_and_pure_step(isolated):
    cfg = config(isolated)
    assert (
        plan(
            {"base_spec": cfg, "sweeps": {"model.parameters.cross_hub": [False, True]}}
        )["count"]
        == 2
    )
    spec = resolve(cfg)
    _, scenarios = load_dataset(spec.dataset)
    step = matching(spec, scenarios[0], 0, lambda: None, stop_period=0)
    assert not step["committed"] and step["before"]["period"] == 0
    assert step["plan"]["state_hash"] == digest(step["before"])
    metrics, trace = matching(spec, scenarios[0], 0, lambda: None)
    assert trace[0]["after"]["period"] == 1 and metrics["arrivals"] == 3


def test_baseline_adapter_equivalence(isolated):
    cfg = config(isolated)
    spec = resolve(cfg)
    _, scenarios = load_dataset(spec.dataset)
    from experiment_core.datasets import matching_state
    from model.mt_prototype import run

    sim, orders = matching_state(
        scenarios[0], spec.model.parameters, spec.solver.parameters.model_dump()
    )
    original = run(
        sim.net, scenarios[0]["hubs"], orders, True, 10, horizon=scenarios[0]["horizon"]
    )
    metrics, _ = matching(spec, scenarios[0], 0, lambda: None)
    assert metrics["operating_profit"] == pytest.approx(original["J"])
    assert (
        metrics["assigned"] == original["assigned"]
        and metrics["delivered"] == original["delivered"]
    )
    sim.begin_period([])
    decision = sim.solve_plan(True, 10)
    sim.commit_plan(decision)
    profit = sim.J
    with pytest.raises(RuntimeError, match="stale"):
        sim.commit_plan(decision)
    assert sim.J == profit


def test_queue_idempotence_lock_cancel_reproduction(isolated):
    cfg = config(isolated)
    co = Coordinator()
    try:
        with pytest.raises(PlatformError, match="coordinator"):
            Coordinator()
        batch = co.submit(cfg, "request-1")
        assert co.submit(cfg, "request-1") == batch
        status = completed(co, batch)
        rid = status["runs"][0]["run_id"]
        assert status["runs"][0]["status"] == "COMPLETED", read_json(
            run_dir(rid) / "state.json"
        )
        assert [e["sequence"] for e in co.run(rid)["events"]] == list(
            range(1, len(co.run(rid)["events"]) + 1)
        )
        second = co.submit(cfg)
        rid2 = completed(co, second)["runs"][0]["run_id"]
        assert comparison([rid, rid2])["mean_difference"] == 0
        assert comparison([rid, rid2])["confidence_interval_95"] is None
        export_bundle(rid)
        assert validate_bundle((run_dir(rid) / "reproduction.zip").read_bytes())[
            "valid"
        ]
        third = co.submit(cfg)
        co.cancel(third["runs"][0]["run_id"])
        assert co.batch(third["batch_id"])["runs"][0]["status"] == "CANCELLED"
    finally:
        co.close()


def test_archive_traversal():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("../bad", "bad")
    with pytest.raises(PlatformError, match="unsafe"):
        validate_bundle(out.getvalue())


def test_diagnostic_redaction_preserves_json(isolated):
    from experiment_core.diagnostics import diagnose
    from experiment_core.storage import ROOT, event

    target = run_dir("diagnostic-test")
    target.mkdir(parents=True)
    atomic_json(
        target / "state.json",
        {"status": "FAILED", "workspace_token": "credential-value"},
    )
    event(target, "error", message=f"Failed at {ROOT}; token=private-credential")
    diagnose("diagnostic-test")
    with zipfile.ZipFile(target / "diagnostic.zip") as archive:
        report = json.loads(archive.read("diagnostic.json"))
    assert report["state"]["workspace_token"] == "<redacted>"
    assert "private-credential" not in report["errors"][0]["message"]
    assert str(ROOT) not in report["errors"][0]["message"]


def test_comparison_metric_direction_and_source_contract(isolated):
    # Controlled metric fixtures exercise comparison semantics, not model quality.
    spec = resolve(config(isolated)).model_dump()
    for rid, pending in (("compare-a", 8), ("compare-b", 10)):
        target = run_dir(rid)
        target.mkdir(parents=True)
        atomic_json(target / "spec.resolved.json", spec)
        atomic_json(target / "state.json", {"status": "COMPLETED"})
        atomic_json(target / "manifest.json", {"code": {"content_hash": "same"}})
        atomic_json(
            target / "metrics.json",
            {"rows": [{"scenario_id": "one", "metrics": {"pending": pending}}]},
        )
    result = comparison(["compare-a", "compare-b"], "pending")
    assert result["mean_difference"] == -2
    assert result["percent_improvement"] == 20
    assert result["metric_sense"] == "minimize"
    with pytest.raises(PlatformError, match="measured operational"):
        comparison(["compare-a", "compare-b"], "upper_bound")
    atomic_json(
        run_dir("compare-b") / "manifest.json", {"code": {"content_hash": "changed"}}
    )
    with pytest.raises(PlatformError, match="same source"):
        comparison(["compare-a", "compare-b"], "pending")


def test_restart_index_rebuild_and_exact_code_guard(isolated, monkeypatch):
    cfg = config(isolated)
    co = Coordinator()
    rid = completed(co, co.submit(cfg))["runs"][0]["run_id"]
    assert (run_dir(rid) / "code" / "experiment_core" / "worker.py").exists()
    co.close()
    # Disposable fixture-owned database only; JSON facts must survive its loss.
    (isolated / "index.sqlite").unlink()
    co = Coordinator()
    try:
        assert co.run(rid)["status"] == "COMPLETED"
        monkeypatch.setattr(
            "experiment_core.service.current_code_hash", lambda: "different"
        )
        with pytest.raises(PlatformError, match="source changed"):
            co.rerun(rid)
    finally:
        co.close()


def test_running_cancel_and_timeout(isolated):
    cfg = config(isolated) | {"execution": {"timeout_seconds": 0.01}}
    co = Coordinator()
    try:
        rid = completed(co, co.submit(cfg))["runs"][0]["run_id"]
        assert co.run(rid)["status"] == "TIMEOUT"
        cfg["execution"] = {"timeout_seconds": 300}
        batch = co.submit(cfg)
        rid = batch["runs"][0]["run_id"]
        co.tick()
        co.cancel(rid)
        assert completed(co, batch)["runs"][0]["status"] == "CANCELLED"
    finally:
        co.close()


def test_api_security_validation_sse(isolated):
    cfg = config(isolated)
    with TestClient(create_app(token="token")) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.post("/api/v1/batches", json=cfg).status_code == 403
        assert (
            client.get(
                "/api/v1/session", headers={"Origin": "https://evil.test"}
            ).status_code
            == 403
        )
        headers = {"X-Workspace-Token": "token"}
        assert client.post(
            "/api/v1/experiments/validate", json=cfg, headers=headers
        ).json()["valid"]
        bad = cfg | {"junk": True}
        r = client.post("/api/v1/experiments/validate", json=bad, headers=headers)
        assert (
            r.status_code == 422 and r.json()["error"]["details"][0]["path"] == "/junk"
        )
        r = client.get("/api/v1/runs/not-found/artifacts/secrets.json")
        assert r.status_code == 404
        batch = client.post("/api/v1/batches", json=cfg, headers=headers).json()
        rid = batch["runs"][0]["run_id"]
        until = time.monotonic() + 30
        while client.get(f"/api/v1/runs/{rid}").json()["status"] not in {
            "COMPLETED",
            "FAILED",
        }:
            assert time.monotonic() < until
            time.sleep(0.1)
        response = client.get(f"/api/v1/runs/{rid}/events?stream=true")
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "data:" in response.text
        ids = [
            int(line[4:])
            for line in response.text.splitlines()
            if line.startswith("id: ")
        ]
        assert ids == sorted(set(ids))
        response = client.get(
            f"/api/v1/runs/{rid}/events?stream=true",
            headers={"Last-Event-ID": str(ids[-1])},
        )
        assert "data:" not in response.text


def test_sac_training_and_held_out_evaluation(isolated):
    d = generate(
        {
            "dataset_id": "sac-tiny",
            "family": "legacy_dispatch",
            "seeds": [91, 92],
            "splits": ["train", "test"],
            "horizon": 4,
            "orders_per_step": 2,
        }
    )
    cfg = {
        "name": "SAC training",
        "dataset": {
            "dataset_id": "sac-tiny",
            "revision": d["revision"],
            "split": "train",
        },
        "model": {"id": "legacy_dispatch"},
        "controller": {
            "id": "train_sac",
            "parameters": {"epochs": 2, "learning_rate": 0.0002},
        },
    }
    co = Coordinator()
    try:
        rid = completed(co, co.submit(cfg))["runs"][0]["run_id"]
        assert co.run(rid)["status"] == "COMPLETED", read_json(
            run_dir(rid) / "state.json"
        )
        metadata = read_json(run_dir(rid) / "checkpoint.json")
        assert metadata["feature_schema"] == "candidate-sac/v1"
        evaluated = cfg | {
            "dataset": cfg["dataset"] | {"split": "test"},
            "controller": {
                "id": "candidate_sac",
                "parameters": {"checkpoint_run": rid},
            },
        }
        assert completed(co, co.submit(evaluated))["runs"][0]["status"] == "COMPLETED"
        with pytest.raises(PlatformError, match="held-out"):
            plan(evaluated | {"dataset": cfg["dataset"]})
    finally:
        co.close()


def test_training_eval_bound_and_bhh_worker(isolated):
    cfg = config(isolated, "train")
    co = Coordinator()
    try:
        train = cfg | {"controller": {"id": "train_value", "parameters": {"epochs": 3}}}
        rid = completed(co, co.submit(train))["runs"][0]["run_id"]
        assert co.run(rid)["status"] == "COMPLETED", read_json(
            run_dir(rid) / "state.json"
        )
        evaluated = cfg | {
            "dataset": cfg["dataset"] | {"split": "test"},
            "value_function": {
                "id": "learned_hub_time",
                "parameters": {"checkpoint_run": rid},
            },
        }
        assert completed(co, co.submit(evaluated))["runs"][0]["status"] == "COMPLETED"
        with pytest.raises(PlatformError, match="held-out"):
            plan(evaluated | {"dataset": cfg["dataset"]})
        boundcfg = cfg | {
            "dataset": cfg["dataset"] | {"split": "test"},
            "controller": {"id": "oracle_lp"},
            "evaluation": {"information_set": "oracle"},
        }
        bid = completed(co, co.submit(boundcfg))["runs"][0]["run_id"]
        assert co.run(bid)["status"] == "COMPLETED"
        assert (
            co.run(bid)["metrics"]["rows"][0]["metrics"]["bound_direction"]
            == "maximize-upper"
        )
        bhh = generate(
            {
                "dataset_id": "bhh",
                "family": "bhh",
                "seeds": [1],
                "splits": ["test"],
                "horizon": 2,
            }
        )
        study = {
            "name": "bhh",
            "dataset": {"dataset_id": "bhh", "revision": bhh["revision"]},
            "model": {"id": "bhh_steady"},
            "solver": {"backend": "cpu"},
            "evaluation": {
                "information_set": "oracle",
                "accounting_version": "bhh-cost-v1",
            },
        }
        assert completed(co, co.submit(study))["runs"][0]["status"] == "COMPLETED"
    finally:
        co.close()


def test_bhh_capacity_and_steady_identity():
    p = BHHParameters().model_dump()
    assert capacity(0, 4, p) == 0 and capacity(2, 0.1, p) == 0
    for a, b in ((0.05, 0.6), (0, 0.6), (0.05, 0)):
        q = inverse_f(2, a, b)
        assert a * q + b * q**0.5 == pytest.approx(2)
    assert direct_capacity(1, 1, p) == 0
    result = steady(p)
    assert abs(result["decomposition"]["identity_residual"]) < 1e-6
    assert result["integer"]["loss"] >= -1e-6
    assert result["hub"]["wave"] == pytest.approx(62, abs=3)


def test_bhh_shared_fleet_and_commitments():
    p = BHHParameters(hv_fleet=[1, 1], av_fleet=[1, 1]).model_dump()
    s = {
        "horizon": 2,
        "orders": [
            dict(
                id="o",
                departure="0",
                destination="1",
                passenger=2,
                book_time=0,
                start_time=0,
                end_time=10,
                penalty=1000,
            )
        ],
    }
    solver = {"time_limit_seconds": 10, "mip_gap": 0, "threads": 1, "seed": 0}
    x, report = finite(s, p, solver)
    assert report["delivered_load"] == pytest.approx(2)
    _, zero = finite(s, p | {"hv_fleet": [0, 0]}, solver)
    assert zero["delivered_load"] == 0
    fixed, metrics, windows = rolling(
        s,
        p,
        solver,
        {"planning_horizon": 8, "completion_extension": 2, "commit_periods": 2},
        lambda: None,
    )
    assert metrics["delivered_load"] == pytest.approx(2)
    assert metrics["global_gap"] is None
    assert metrics["business_cost"] == pytest.approx(report["business_cost"])
    for key, value in fixed.items():
        if value > 1e-8 and key.startswith("g:"):
            assert int(key.split(":")[-1]) <= 10
