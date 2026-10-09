"""Fixed, budgeted probes on public MC states; never trains or changes policies."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

from model_families.single.strategic.artifacts import read_json, write_json, source_identity
from model_families.single.strategic.contracts import identity
from experiments.cfs_ppo_strategic_control.study_common import ROOT, sha, now

PACKAGE = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/cfs_ppo_strategic_control/runs/cfs_laptop_ppo_v1"
PREPARED = ROOT / "experiments/cfs_ppo_strategic_control/runs/cfs_first_ppo_v1"
CONDITIONS = ("california_corridor-n80-k3", "western_multicluster-n80-k3")
SEEDS = (30001, 30002, 30003)
SCENES = (23002, 23003)
TIMES = (4, 8, 12, 16, 20, 23)
MULTIPLIERS = (0., .5, 1., 2., 4.)


class BudgetLimited(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds=1800, solves=1000):
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.limit = solves
        self.solves = 0

    def check(self):
        if time.monotonic() >= self.deadline:
            raise BudgetLimited("registered 30-minute numerical budget reached")
        return False

    def claim(self):
        self.check()
        if self.solves >= self.limit:
            raise BudgetLimited("registered solver-call limit reached")
        self.solves += 1


def scaled_signal(signal, multiplier):
    if not math.isfinite(multiplier) or multiplier < 0 or any(
            not math.isfinite(x) or abs(x) > 1 for x in signal):
        raise ValueError("invalid actor signal/multiplier")
    raw = [multiplier*x for x in signal]
    return tuple(max(-1., min(1., x)) for x in raw), sum(abs(x) > 1 for x in raw)


def source_files():
    paths = [*sorted(PACKAGE.glob("*.py")), PACKAGE/"README.md",
             ROOT/"tests/test_ppo_signal_action_diagnostic.py",
             ROOT/"tests/test_single_strategic_ppo.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def verify_hashes(root, hashes):
    for relative, expected in hashes.items():
        if sha(root / relative) != expected:
            raise RuntimeError("frozen source/input changed: " + relative)


def hash_history():
    """Verify the complete frozen laptop deliverables, not only their summary."""
    final = read_json(PARENT / "final_manifest.json")
    if final["status"] != "COMPLETED":
        raise RuntimeError("parent laptop attempt is not frozen")
    verify_hashes(PARENT, final["final_artifact_hashes"])
    return {str(PARENT / relative): digest for relative, digest in final["final_artifact_hashes"].items()} | {
        str(PARENT / "final_manifest.json"): sha(PARENT / "final_manifest.json")}


def prepare(attempt):
    from model_families.single.learning.ppo.policy import FrozenPolicy
    from model_families.single.learning.ppo.runner import scenario_content, physical_signature
    from model_families.single.data import validate_payload
    target = Path(attempt).resolve()
    if target.exists():
        raise FileExistsError("attempt output must be new")
    print("Verifying frozen parent hashes (read only)", flush=True)
    history = hash_history()
    original = read_json(PREPARED / "manifest.json")
    if original["source_identity"] != source_identity():
        raise RuntimeError("original physics/learning source identity changed")
    selected = read_json(PARENT / "protocol/selected-models.json")["selected"]
    inputs = {}
    prepared_scenes = {}
    for name in CONDITIONS:
        cfg_path = PARENT / "protocol" / (name+".json")
        cfg = read_json(cfg_path)
        cfg["baseline"] = str(PARENT / cfg["baseline"])
        source_path = PREPARED / "data" / name / "diagnostic.json"
        relative = source_path.relative_to(PREPARED).as_posix()
        if sha(source_path) != original["input_hashes"][relative]:
            raise RuntimeError("prepared diagnostic dataset changed")
        dataset = read_json(source_path)
        scenes = [s for s in dataset["scenarios"] if s["seed"] in SCENES]
        if len(scenes) != 2 or {s["seed"] for s in scenes} != set(SCENES):
            raise RuntimeError("missing preregistered diagnostic scenes")
        validate_payload({**dataset, "scenarios": scenes})
        old = []
        for role in ("train", "validation", "diagnostic"):
            old += read_json(PARENT / "data" / name / (role+".json"))["scenarios"]
        old += read_json(PARENT / "sealed" / name / "test.json")["scenarios"]
        old += read_json(PREPARED / "data" / name / "pilot_train.json")["scenarios"]
        old += read_json(PREPARED / "data" / name / "pilot_validation.json")["scenarios"]
        if {scenario_content(s) for s in scenes} & {scenario_content(s) for s in old}:
            raise RuntimeError("new diagnostic content overlaps previously used scenes")
        if set(SCENES) & {s["seed"] for s in old}:
            raise RuntimeError("new diagnostic seeds overlap previously used scenes")
        policies = []
        for seed in SEEDS:
            row = next(r for r in selected if r["condition"] == name and r.get("seed") == seed)
            cp = PARENT / row["checkpoint"]
            policy = FrozenPolicy(cp)
            if any(physical_signature(s) != policy.environment["physics"] for s in scenes):
                raise RuntimeError("diagnostic physics differs from frozen policy")
            policies.append(dict(seed=seed, checkpoint=str(cp), update=row["update"]))
        cal = next(r for r in selected if r["condition"] == name and r.get("method") == "CAL")
        prepared_scenes[name] = dict(config=cfg, scenarios=scenes, policies=policies,
            cal_signal=[cal["near"], cal["far"]]*len(cfg["action"]["hubs"]))
        for path in (cfg_path, Path(cfg["baseline"]), source_path):
            inputs[str(path)] = sha(path)
    # Include original provenance and source checks without editing the source.
    for name in CONDITIONS:
        p = PREPARED / "data" / name / "diagnostic-provenance.json"
        rel = p.relative_to(PREPARED).as_posix()
        if sha(p) != original["input_hashes"][rel]:
            raise RuntimeError("source scenario provenance changed")
        inputs[str(p)] = sha(p)
    inputs[str(PREPARED / "manifest.json")] = sha(PREPARED / "manifest.json")
    norm = ROOT/"reports/笔记本数值实验规范_假设驱动与逐级验证.md"
    inputs[str(norm)] = sha(norm)
    target.mkdir(parents=True)
    for folder in ("protocol", "data", "raw", "analysis", "figures", "qa"):
        (target/folder).mkdir()
    protocol = dict(version="ppo-signal-action-diagnostic/v1", role="mechanism-exploration",
        created_at=now(), conditions=list(CONDITIONS), scenario_seeds=list(SCENES),
        training_seeds=list(SEEDS), periods=list(TIMES), multipliers=list(MULTIPLIERS),
        state_source="MC zero-residual trajectory; full public snapshots",
        axis_controls="all 10 axes, -1 and +1; no profit-based selection",
        active_subset="pending_order_count>=1 and idle_vehicle_count>=1; not proof of alternate feasibility",
        source_scenarios="prepared previously; first numerical use; CFS-generated temporal fields",
        wall_seconds=1800, maximum_solves=1000, expected_maximum_solves=984,
        disk_limit_bytes=2*2**30, workers=1, solver_threads=1,
        interpretation="signal and integer action mechanism only, not profit superiority or all-space search",
        next_step="decide after diagnostics; no automatic training or parameter search",
        learning_source_identity=source_identity(), diagnostic_sources=source_files())
    write_json(target/"protocol/preregistration.json", protocol)
    for name, item in prepared_scenes.items():
        write_json(target/"data"/(name+".json"), item)
    local_hashes = {p.relative_to(target).as_posix(): sha(p) for f in ("protocol", "data")
                    for p in sorted((target/f).rglob("*.json"))}
    write_json(target/"manifest.json", dict(status="PREPARED", created_at=now(),
        source_identity=source_identity(), source_files=source_files(), input_hashes=local_hashes,
        history_hashes=history, input_source_hashes=inputs))
    write_json(target/"qa/preparation.json", dict(status="PASSED", all_parent_deliverables=len(history)-1,
        models=6, scenes=4, trained_validation_test_pilot_overlap=0, physics_and_baseline_checked=True))
    print("Prepared 4 new diagnostic scenes and 6 frozen actors", flush=True)


def load(attempt, expected):
    target = Path(attempt).resolve()
    manifest = read_json(target/"manifest.json")
    if manifest["status"] not in expected:
        raise RuntimeError("attempt stage cannot be restarted: " + manifest["status"])
    if manifest["source_identity"] != source_identity() or manifest["source_files"] != source_files():
        raise RuntimeError("frozen numerical/diagnostic sources changed")
    verify_hashes(target, manifest["input_hashes"])
    verify_hashes(Path("/"), manifest["input_source_hashes"])
    return target, manifest


def audit_probe(ep, obs, signal, record):
    """Independently compile returned integer routes and recalculate coefficients."""
    from model_families.single.planning import compile_action
    from model_families.single.rolling.controller import observed_state
    from model_families.single.runner import serialize_plan
    from model_families.single.strategic.decoder import ReleaseSignalProvider
    from model_families.single.strategic.contracts import StrategicSignal
    from experiments.cfs_ppo_strategic_control.study_execution import action_from_plan, physical_hash
    info = {}
    compiled = compile_action(observed_state(ep._sim), action_from_plan(record["plan"]), info)
    serialized = serialize_plan(compiled, obs.period, obs.hash, info)
    if (physical_hash(serialized) != record["action_hash"] or
            info["accounting"] != record["plan"]["accounting"] or
            not math.isclose(compiled[0], record["plan"]["operating_profit"], abs_tol=1e-6, rel_tol=1e-10)):
        raise RuntimeError("probe integer execution cash/action mismatch")
    if record["plan"]["solver"]["status"] != "OPTIMAL":
        raise RuntimeError("probe solver not optimal")
    provider = ReleaseSignalProvider(ep._baseline, ep.schema,
        StrategicSignal(tuple(signal), obs.period, obs.revision, obs.hash), ep._sim.T)
    for row in record["coefficients"]:
        value = provider.marginal_release_value(row["time"], row["hub"], row["supply_slot"], {})
        if not math.isclose(value, row["value"], abs_tol=1e-8, rel_tol=1e-12):
            raise RuntimeError("probe release coefficient mismatch")


def probe_at(ep, obs, signal, cache, budget):
    from model_families.single.runner import state_snapshot
    signal = tuple(signal)
    key = identity(dict(observation=obs.hash, revision=obs.revision, signal=signal))
    if key in cache:
        return key, cache[key], True
    before = identity(state_snapshot(ep._sim))
    budget.claim()
    record = ep.probe(signal)
    audit_probe(ep, obs, signal, record)
    if identity(state_snapshot(ep._sim)) != before:
        raise RuntimeError("probe mutated live physics")
    cache[key] = record
    return key, record, False


def run(attempt):
    from dataclasses import asdict
    from model_families.single.learning.ppo.policy import FrozenPolicy
    from model_families.single.strategic.episode import StrategicEpisode
    from model_families.single.strategic.baseline import FrozenBaseline
    from model_families.single.strategic.contracts import ActionSchema
    from model_families.single.strategic.observation import FeatureScales
    from model_families.single.parameters import MatchingParameters
    from experiment_core.contracts import SolverParameters
    from model_families.single.runner import state_snapshot
    from experiments.cfs_ppo_strategic_control.study_execution import replay
    import torch
    import psutil
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    target, manifest = load(attempt, {"PREPARED"})
    budget = Budget()
    manifest.update(status="RUNNING", numerical_started_at=now())
    write_json(target/"manifest.json", manifest)
    total_states = total_logical = total_replays = 0
    try:
        for name in CONDITIONS:
            item = read_json(target/"data"/(name+".json"))
            cfg = item["config"]
            policies = [(p["seed"], FrozenPolicy(p["checkpoint"])) for p in item["policies"]]
            for scenario in item["scenarios"]:
                ep = StrategicEpisode(scenario,
                    MatchingParameters.model_validate(cfg["parameters"]).model_dump(),
                    SolverParameters.model_validate(cfg["solver"]).model_dump(),
                    ActionSchema(**cfg["action"]), FeatureScales(**cfg["feature_scales"]),
                    FrozenBaseline(read_json(cfg["baseline"])), timeout_seconds=1800,
                    cancelled=budget.check)
                obs = ep.reset()
                state_rows = []
                while True:
                    budget.check()
                    zero = (0.,)*(2*len(ep.schema.hubs))
                    if obs.period in TIMES:
                        cache, logical, actors = {}, [], []
                        refkey, ref, _ = probe_at(ep, obs, zero, cache, budget)
                        def take(method, signal, seed=None, multiplier=None, axis=None, sign=None):
                            nonlocal total_logical
                            k, r, hit = probe_at(ep, obs, signal, cache, budget)
                            logical.append(dict(method=method, training_seed=seed, multiplier=multiplier,
                                axis=axis, sign=sign, signal=list(signal), probe_key=k, cached=hit,
                                changed=r["action_hash"] != ref["action_hash"]))
                            total_logical += 1
                        logical.append(dict(method="MC", training_seed=None, multiplier=None,
                            axis=None, sign=None, signal=list(zero), probe_key=refkey, cached=False, changed=False))
                        total_logical += 1
                        take("CAL", item["cal_signal"])
                        for seed, policy in policies:
                            a = policy.act(obs)
                            if a != policy.act(obs):
                                raise RuntimeError("deterministic actor failed repeatability")
                            actors.append(dict(training_seed=seed, signal=list(a)))
                            for c in MULTIPLIERS:
                                signal, clipped = scaled_signal(a, c)
                                take("PPO_scaled", signal, seed=seed, multiplier=c)
                                logical[-1]["clipped_components"] = clipped
                        for axis in range(len(zero)):
                            for sign in (-1., 1.):
                                a = list(zero)
                                a[axis] = sign
                                take("axis", a, axis=axis, sign=sign)
                        state_rows.append(dict(condition=name, scenario_seed=scenario["seed"],
                            scenario_id=scenario["id"], period=obs.period,
                            before_state=state_snapshot(ep._sim), observation=asdict(obs), observation_hash=obs.hash,
                            pending_order_count=len(ep._sim.pool),
                            idle_vehicle_count=sum(v.phase == "idle" for v in ep._sim.vehs),
                            reference_key=refkey, actors=actors, logical=logical, probes=cache))
                        total_states += 1
                    budget.claim()
                    transition = ep.step(zero)
                    if obs.period in TIMES and ep.records[-1]["action_hash"] != ref["action_hash"]:
                        raise RuntimeError("zero-signal probe differs from committed MC action")
                    if transition.terminated:
                        break
                    obs = transition.observation
                trajectory = dict(transitions=ep.records, ledger=ep.final_ledger, metrics=ep.metrics)
                replay(scenario, cfg, trajectory)
                total_replays += len(ep.records)
                record = dict(condition=name, scenario_seed=scenario["seed"], states=state_rows,
                    trajectory=trajectory, completed_at=now())
                write_json(target/"raw"/(name+f"-s{scenario['seed']}.json"), record)
                size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
                if size > 2*2**30:
                    raise BudgetLimited("registered 2 GiB disk budget reached")
                write_json(target/"qa/progress.json", dict(states=total_states, logical_probes=total_logical,
                    actual_solver_calls=budget.solves, wall_seconds=time.monotonic()-budget.started))
                print(name, scenario["seed"], "states", total_states, "solves", budget.solves, flush=True)
        if total_states != 24 or total_logical != 888 or total_replays != 96:
            raise RuntimeError("incomplete preregistered scene/state/probe coverage")
        raw_hashes = {p.relative_to(target).as_posix(): sha(p) for p in sorted((target/"raw").glob("*.json"))}
        write_json(target/"qa/numerical.json", dict(status="PASSED", states=total_states,
            logical_probes=total_logical, actual_solver_calls=budget.solves,
            trajectory_periods_replayed=total_replays, all_optimal=True,
            every_unique_probe_independently_compiled=True, all_coefficient_queries_recomputed=True,
            deterministic_actors_repeated=True, all_probe_live_states_unchanged=True,
            wall_seconds=time.monotonic()-budget.started, raw_hashes=raw_hashes,
            peak_working_set_bytes=getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)))
        manifest.update(status="NUMERICAL_COMPLETE_QA_PENDING", numerical_completed_at=now(), raw_hashes=raw_hashes)
        write_json(target/"manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="BUDGET_LIMITED" if isinstance(exc, BudgetLimited) else "INVALID", stopped_at=now())
        write_json(target/"manifest.json", manifest)
        # Preserve the current scene as partial evidence, never a completed sample.
        if "state_rows" in locals():
            write_json(target/"raw/partial.json", dict(condition=name, scenario_seed=scenario["seed"],
                states=state_rows, partial_transitions=ep.records, completed=False))
        write_json(target/"qa/stop.json", dict(status=manifest["status"], error=type(exc).__name__,
            message=str(exc), solver_calls=budget.solves, wall_seconds=time.monotonic()-budget.started))
        raise


def freeze(attempt):
    target, manifest = load(attempt, {"NUMERICAL_COMPLETE_QA_PENDING"})
    for filename in ("numerical.json", "analysis.json", "tests.json", "visual-review.json"):
        if read_json(target/"qa"/filename)["status"] != "PASSED":
            raise RuntimeError("required QA missing: " + filename)
    tests = read_json(target/"qa/tests.json")
    if sha(target/tests["xml_file"]) != tests["xml_sha256"]:
        raise RuntimeError("actual test XML changed")
    verify_hashes(target, manifest["raw_hashes"])
    verify_hashes(Path("/"), manifest["history_hashes"])
    analysis = read_json(target/"qa/analysis.json")
    verify_hashes(target, analysis["output_hashes"])
    visual = read_json(target/"qa/visual-review.json")
    verify_hashes(target, visual["reviewed_hashes"])
    all_files = {p.relative_to(target).as_posix(): sha(p) for p in sorted(target.rglob("*"))
                 if p.is_file() and p.name not in {"manifest.json", "final_manifest.json"}}
    manifest.update(status="COMPLETED", completed_at=now(), final_artifact_hashes=all_files,
        scientific_scope="mechanism exploration; no profitability claims")
    write_json(target/"manifest.json", manifest)
    write_json(target/"final_manifest.json", manifest)
    verify_hashes(target, all_files)
    print("COMPLETED and all", len(all_files), "deliverable hashes verified", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "freeze"))
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    for k, v in dict(PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                     OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1").items():
        if os.environ.get(k) != v:
            raise RuntimeError("required numerical environment: " + k)
    globals()[args.stage](args.attempt)


if __name__ == "__main__":
    main()
