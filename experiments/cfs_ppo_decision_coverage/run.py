"""Preregistered all-period axis controls, independent of frozen earlier attempts."""
import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from experiments.cfs_ppo_signal_action_diagnostic.run import (
    Budget, BudgetLimited, probe_at, verify_hashes, PARENT, PREPARED)
from experiments.cfs_ppo_strategic_control.study_common import ROOT, sha, now
from model_families.single.strategic.artifacts import read_json, write_json, source_identity
from model_families.single.strategic.contracts import identity

PACKAGE = Path(__file__).resolve().parent
PREVIOUS = ROOT/"experiments/cfs_ppo_signal_action_diagnostic/runs/signal_action_v1"
CONDITION = "western_multicluster-n80-k3"
SCENES = (23004, 23005)
OLD_TIMES = (4, 8, 12, 16, 20, 23)


def decision_identity(plan):
    """Ignore top-level enumeration order; keep every route's internal order."""
    fields = {k: sorted(copy.deepcopy(plan[k]), key=lambda x:json.dumps(x, sort_keys=True))
              for k in ("assignments", "routes", "repositions", "expired")}
    return identity(dict(period=plan["period"], **fields))


def axis_controls(dim):
    if dim != 10:
        raise ValueError("this registered study requires five hubs / ten axes")
    return [dict(axis=i, sign=sign, signal=[sign if j == i else 0. for j in range(dim)])
            for i in range(dim) for sign in (-1., 1.)]


def sources():
    paths = [*sorted(PACKAGE.glob("*.py")), PACKAGE/"README.md",
        ROOT/"tests/test_ppo_decision_coverage.py",
        ROOT/"experiments/cfs_ppo_signal_action_diagnostic/run.py",
        ROOT/"experiments/cfs_ppo_signal_action_diagnostic/analyse.py",
        ROOT/"experiments/cfs_ppo_strategic_control/study_execution.py"]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def history_hashes():
    hashes = {}
    for parent in (PARENT, PREVIOUS):
        final = read_json(parent/"final_manifest.json")
        if final["status"] != "COMPLETED":
            raise RuntimeError("history is not frozen")
        verify_hashes(parent, final["final_artifact_hashes"])
        hashes.update({str(parent/r):h for r,h in final["final_artifact_hashes"].items()})
        for name in ("manifest.json", "final_manifest.json"):
            hashes[str(parent/name)] = sha(parent/name)
    return hashes


def ensure_disjoint(new, old):
    from model_families.single.learning.ppo.runner import scenario_content
    if ({s["seed"] for s in new} & {s["seed"] for s in old} or
        {scenario_content(s) for s in new} & {scenario_content(s) for s in old}):
        raise RuntimeError("new scenes overlap previously used seeds/content")


def prepare(attempt):
    from model_families.single.data import validate_payload
    from model_families.single.learning.ppo.runner import physical_signature
    target = Path(attempt).resolve()
    if target.exists():
        raise FileExistsError("new attempt required")
    history = history_hashes()
    original = read_json(PREPARED/"manifest.json")
    previous = read_json(PREVIOUS/"final_manifest.json")
    if source_identity() != original["source_identity"] or source_identity() != previous["source_identity"]:
        raise RuntimeError("frozen physics/learning source changed")
    source_path = PREPARED/"data"/CONDITION/"diagnostic.json"
    provenance_path = source_path.with_name("diagnostic-provenance.json")
    for p in (source_path, provenance_path):
        if sha(p) != original["input_hashes"][p.relative_to(PREPARED).as_posix()]:
            raise RuntimeError("prepared scene/provenance changed")
    dataset = read_json(source_path)
    scenes = [s for s in dataset["scenarios"] if s["seed"] in SCENES]
    if len(scenes) != 2 or {s["seed"] for s in scenes} != set(SCENES):
        raise RuntimeError("registered scenes missing")
    validate_payload({**dataset, "scenarios":scenes})
    old = read_json(PREVIOUS/"data"/(CONDITION+".json"))["scenarios"]
    for role in ("train", "validation", "diagnostic"):
        old += read_json(PARENT/"data"/CONDITION/(role+".json"))["scenarios"]
    old += read_json(PARENT/"sealed"/CONDITION/"test.json")["scenarios"]
    for role in ("pilot_train", "pilot_validation"):
        old += read_json(PREPARED/"data"/CONDITION/(role+".json"))["scenarios"]
    ensure_disjoint(scenes, old)
    item = read_json(PREVIOUS/"data"/(CONDITION+".json"))
    cfg = item["config"]
    known = physical_signature(item["scenarios"][0])
    if any(physical_signature(s) != known or s["horizon"] != 24 for s in scenes):
        raise RuntimeError("new scenario physics/horizon differs")
    axis_controls(2*len(cfg["action"]["hubs"]))
    source_inputs = {str(p):sha(p) for p in (source_path, provenance_path, PREPARED/"manifest.json",
        PREVIOUS/"data"/(CONDITION+".json"), Path(cfg["baseline"]),
        ROOT/"reports/笔记本数值实验规范_假设驱动与逐级验证.md")}
    target.mkdir(parents=True)
    for folder in ("protocol", "data", "raw", "analysis", "figures", "qa"):
        (target/folder).mkdir()
    protocol = dict(version="ppo-decision-coverage/v1", question="Does fixed six-period sampling miss axis-responsive states?",
        role="mechanism exploration, not profit comparison", created_at=now(), condition=CONDITION,
        scenario_seeds=list(SCENES), periods=list(range(24)), previous_periods=list(OLD_TIMES),
        state_source="natural MC zero-residual trajectory", controls="zero and all ten +/-1 axes",
        primary_action_identity="canonical top-level lists; internal route order preserved",
        active_subset="pending>=1 and idle>=1; not proof of alternative feasibility",
        maximum_solves=1200, expected_solves=1056, wall_seconds=1800, disk_limit_bytes=2*2**30,
        workers=1, solver_threads=1, no_training=True, no_profit_selection=True,
        decision="omitted responsive states: sampling omission local evidence; otherwise examine support/physical constraints before training",
        independent_sample_unit="scenario (n=2); 48 states are nested observations",
        data_boundary="Previously prepared CFS-derived simulated scenes, first numerical use; not new real observation dates")
    write_json(target/"protocol/preregistration.json", protocol)
    write_json(target/"data"/(CONDITION+".json"), dict(config=cfg, scenarios=scenes))
    local = {p.relative_to(target).as_posix():sha(p) for f in ("protocol", "data")
             for p in sorted((target/f).glob("*.json"))}
    write_json(target/"manifest.json", dict(status="PREPARED", created_at=now(),
        source_identity=source_identity(), source_files=sources(), history_hashes=history,
        input_source_hashes=source_inputs, input_hashes=local))
    write_json(target/"qa/preparation.json", dict(status="PASSED", scenarios=2, overlap=0,
        physics_unchanged=True, all_prior_deliverable_hashes_checked=len(history)))
    print("PREPARED 2 new W80 scenes / all 24 periods / no training", flush=True)


def load(attempt, statuses):
    target = Path(attempt).resolve()
    manifest = read_json(target/"manifest.json")
    if manifest["status"] not in statuses:
        raise RuntimeError("stage must not repeat: "+manifest["status"])
    if manifest["source_identity"] != source_identity() or manifest["source_files"] != sources():
        raise RuntimeError("frozen numerical/analysis/source changed")
    verify_hashes(target, manifest["input_hashes"])
    verify_hashes(Path("/"), manifest["input_source_hashes"])
    return target, manifest


def run(attempt):
    import psutil
    from experiments.cfs_ppo_strategic_control.study_execution import episode, replay
    from model_families.single.runner import state_snapshot
    from model_families.single.strategic.observation import free_release
    from model_families.single.rolling.controller import observed_state
    target, manifest = load(attempt, {"PREPARED"})
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    budget = Budget(seconds=1800, solves=1200)
    manifest.update(status="RUNNING", numerical_started_at=now())
    write_json(target/"manifest.json", manifest)
    item = read_json(target/"data"/(CONDITION+".json"))
    total = 0
    try:
        for scenario in item["scenarios"]:
            ep = episode(scenario, item["config"])
            ep.timeout_seconds = 1800
            ep._cancelled = budget.check
            obs = ep.reset()
            states = []
            while True:
                budget.check()
                zero = (0.,)*10
                cache = {}
                refkey, reference, _ = probe_at(ep, obs, zero, cache, budget)
                reference_canonical = decision_identity(reference["plan"])
                logical = [dict(method="MC", axis=None, sign=None, signal=list(zero),
                    probe_key=refkey, cached=False, changed=False, raw_hash_changed=False)]
                for control in axis_controls(10):
                    key, probe, hit = probe_at(ep, obs, control["signal"], cache, budget)
                    logical.append(dict(method="axis", **control, probe_key=key, cached=hit,
                        changed=decision_identity(probe["plan"]) != reference_canonical,
                        raw_hash_changed=probe["action_hash"] != reference["action_hash"]))
                before = state_snapshot(ep._sim)
                snapshot = observed_state(ep._sim)
                vehicles = []
                for v in snapshot.vehs:
                    hub, when = free_release(snapshot, v)
                    vehicles.append(dict(vehicle_id=str(v.id), phase=v.phase, hub=v.loc,
                        committed_free_hub=hub, committed_free_time=when, terminal_release=when>=snapshot.T))
                states.append(dict(condition=CONDITION, scenario_seed=scenario["seed"], scenario_id=scenario["id"],
                    period=obs.period, horizon=snapshot.T, before_state=before, observation=asdict(obs),
                    observation_hash=obs.hash, pending_order_count=len(ep._sim.pool),
                    idle_vehicle_count=sum(v.phase=="idle" for v in ep._sim.vehs),
                    vehicles=vehicles, reference_key=refkey, reference_canonical_hash=reference_canonical,
                    logical=logical, probes=cache))
                budget.claim()
                tr = ep.step(zero)
                if ep.records[-1]["action_hash"] != reference["action_hash"]:
                    raise RuntimeError("MC committed action differs from same-state zero probe")
                if tr.terminated:
                    break
                obs = tr.observation
            if [s["period"] for s in states] != list(range(24)):
                raise RuntimeError("missing/duplicate registered period")
            trajectory = dict(transitions=ep.records, ledger=ep.final_ledger, metrics=ep.metrics)
            replay_result = replay(scenario, item["config"], trajectory)
            write_json(target/"raw"/f"s{scenario['seed']}.json", dict(condition=CONDITION,
                scenario_seed=scenario["seed"], states=states, trajectory=trajectory,
                replay=replay_result, completed_at=now()))
            total += len(states)
            if sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) > 2*2**30:
                raise BudgetLimited("registered disk limit reached")
            print("scene", scenario["seed"], "states", total, "solver calls", budget.solves, flush=True)
        if total != 48 or budget.solves != 1056:
            raise RuntimeError("registered coverage/call accounting mismatch")
        raw_hashes = {p.relative_to(target).as_posix():sha(p) for p in sorted((target/"raw").glob("*.json"))}
        write_json(target/"qa/numerical.json", dict(status="PASSED", states=48, probes=1008,
            actual_solver_calls=budget.solves, wall_seconds=time.monotonic()-budget.started,
            MC_trajectory_periods_replayed=48, all_optimal=True, independent_probe_compile_cash_checked=True,
            all_probe_live_states_unchanged=True, raw_hashes=raw_hashes,
            peak_working_set_bytes=getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)))
        manifest.update(status="NUMERICAL_COMPLETE_QA_PENDING", numerical_completed_at=now(), raw_hashes=raw_hashes)
        write_json(target/"manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="BUDGET_LIMITED" if isinstance(exc, BudgetLimited) else "INVALID", stopped_at=now())
        write_json(target/"manifest.json", manifest)
        write_json(target/"qa/stop.json", dict(status=manifest["status"], error=type(exc).__name__,
            message=str(exc), actual_solver_calls=budget.solves))
        if "states" in locals():
            write_json(target/"raw/partial.json", dict(completed=False, states=states,
                partial_transitions=ep.records, scenario_seed=scenario["seed"]))
        raise


def freeze(attempt):
    target, manifest = load(attempt, {"NUMERICAL_COMPLETE_QA_PENDING"})
    for name in ("numerical", "analysis", "tests", "visual-review"):
        if read_json(target/"qa"/(name+".json"))["status"] != "PASSED":
            raise RuntimeError("required QA missing: "+name)
    tests = read_json(target/"qa/tests.json")
    if sha(target/tests["xml_file"]) != tests["xml_sha256"]:
        raise RuntimeError("test XML changed")
    verify_hashes(target, manifest["raw_hashes"])
    verify_hashes(Path("/"), manifest["history_hashes"])
    verify_hashes(target, read_json(target/"qa/analysis.json")["output_hashes"])
    verify_hashes(target, read_json(target/"qa/visual-review.json")["reviewed_hashes"])
    hashes = {p.relative_to(target).as_posix():sha(p) for p in sorted(target.rglob("*"))
              if p.is_file() and p.name not in ("manifest.json", "final_manifest.json")}
    manifest.update(status="COMPLETED", completed_at=now(), final_artifact_hashes=hashes,
        scope="mechanism exploration; not learned policy or profitability validation")
    write_json(target/"manifest.json", manifest)
    write_json(target/"final_manifest.json", manifest)
    verify_hashes(target, hashes)
    print("COMPLETED;", len(hashes), "deliverable hashes verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "freeze"))
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    for k,v in dict(PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                    OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1").items():
        if os.environ.get(k) != v:
            raise RuntimeError("required environment: "+k)
    globals()[args.stage](args.attempt)
