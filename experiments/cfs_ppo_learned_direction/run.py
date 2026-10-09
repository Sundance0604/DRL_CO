"""Frozen actor probes at exactly reconstructed public MC boundaries."""
import argparse
import copy
from dataclasses import asdict
import os
from pathlib import Path
import time

from experiments.cfs_ppo_signal_action_diagnostic.run import (
    Budget as OldBudget, BudgetLimited, probe_at, audit_probe, scaled_signal,
    verify_hashes, PARENT)
from experiments.cfs_ppo_decision_coverage.run import (
    decision_identity, PREVIOUS, CONDITION, SCENES)
from experiments.cfs_ppo_strategic_control.study_common import ROOT, sha, now
from model_families.single.strategic.artifacts import read_json, write_json, source_identity
from model_families.single.strategic.contracts import identity

PACKAGE = Path(__file__).resolve().parent
COVERAGE = ROOT/"experiments/cfs_ppo_decision_coverage/runs/full_period_v1"
SEEDS = (30001, 30002, 30003)
MULTIPLIERS = (1., 4.)


class Budget(OldBudget):
    def __init__(self, seconds=600, solves=288):
        super().__init__(seconds, solves)

    def check(self):
        if time.monotonic() >= self.deadline:
            raise BudgetLimited("registered ten-minute numerical budget reached")
        return False


def advance_frozen_mc(ep, transition):
    """Replay the committed integer action, never invoke a MILP or policy step."""
    from experiments.cfs_ppo_strategic_control.study_execution import action_from_plan, close, physical_hash
    from model_families.single.runner import state_snapshot, serialize_plan
    from model_families.single.planning import compile_action
    from model_families.single.accounting import completion_bound, cooldown_tick
    obs = ep._observation
    if (obs.hash != transition["observation_hash"] or
            identity(state_snapshot(ep._sim)) != transition["before_state_hash"]):
        raise RuntimeError("frozen MC before/observation mismatch")
    old = transition["plan"]
    info = dict(old["solver"])
    plan = compile_action(ep._sim, action_from_plan(old), info)
    serial = serialize_plan(plan, obs.period, obs.hash, info)
    if (physical_hash(serial) != transition["action_hash"] or
            info["accounting"] != old["accounting"] or not close(plan[0], old["operating_profit"])):
        raise RuntimeError("frozen MC integer action/cash mismatch")
    ep._sim.commit_plan(plan)
    terminal = ep._sim.t == ep._sim.T
    if terminal:
        bound = completion_bound(ep._sim)
        ep._sim.pool = {}
        while any(v.phase != "idle" for v in ep._sim.vehs):
            if ep._sim.t >= bound:
                raise RuntimeError("frozen MC cooldown incomplete")
            cooldown_tick(ep._sim)
    if identity(state_snapshot(ep._sim)) != transition["after_state_hash"]:
        raise RuntimeError("frozen MC after mismatch")
    return None if terminal else ep._publish()


def sources():
    paths = [*sorted(PACKAGE.glob("*.py")), PACKAGE/"README.md", ROOT/"tests/test_ppo_learned_direction.py",
        ROOT/"experiments/cfs_ppo_signal_action_diagnostic/run.py",
        ROOT/"experiments/cfs_ppo_signal_action_diagnostic/analyse.py",
        ROOT/"experiments/cfs_ppo_decision_coverage/run.py",
        ROOT/"experiments/cfs_ppo_decision_coverage/analyse.py",
        ROOT/"experiments/cfs_ppo_strategic_control/study_execution.py"]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def history_hashes():
    hashes = {}
    for parent in (PARENT, PREVIOUS, COVERAGE):
        final = read_json(parent/"final_manifest.json")
        if final["status"] != "COMPLETED":
            raise RuntimeError("history not COMPLETED")
        verify_hashes(parent, final["final_artifact_hashes"])
        hashes.update({str(parent/r):h for r,h in final["final_artifact_hashes"].items()})
        for name in ("manifest.json", "final_manifest.json"):
            hashes[str(parent/name)] = sha(parent/name)
    return hashes


def prepare(attempt):
    from model_families.single.learning.ppo.policy import FrozenPolicy
    from model_families.single.learning.ppo.runner import physical_signature
    from model_families.single.strategic.baseline import FrozenBaseline
    from model_families.single.strategic.contracts import ActionSchema
    target = Path(attempt).resolve()
    if target.exists():
        raise FileExistsError("new attempt required")
    history = history_hashes()
    item = read_json(COVERAGE/"data"/(CONDITION+".json"))
    prior = read_json(PREVIOUS/"data"/(CONDITION+".json"))
    if source_identity() != read_json(COVERAGE/"manifest.json")["source_identity"]:
        raise RuntimeError("physics/learning source changed")
    if item["config"] != prior["config"] or {s["seed"] for s in item["scenarios"]} != set(SCENES):
        raise RuntimeError("frozen scenes/config differ")
    policies = prior["policies"]
    if {p["seed"] for p in policies} != set(SEEDS) or any(p["update"] != 4 for p in policies):
        raise RuntimeError("registered selected actors differ")
    for p in policies:
        actor = FrozenPolicy(p["checkpoint"])
        if any(physical_signature(s) != actor.environment["physics"] for s in item["scenarios"]):
            raise RuntimeError("actor physics mismatch")
        if (actor.baseline.hash != FrozenBaseline(read_json(item["config"]["baseline"])).hash or
                ActionSchema(**actor.environment["action"]) != ActionSchema(**item["config"]["action"]) or
                actor.environment["feature_scales"] != item["config"]["feature_scales"]):
            raise RuntimeError("actor baseline/action/features mismatch")
    target.mkdir(parents=True)
    for f in ("protocol", "data", "raw", "analysis", "figures", "qa"):
        (target/f).mkdir()
    protocol = dict(version="ppo-learned-direction/v1", created_at=now(), condition=CONDITION,
        scenario_seeds=list(SCENES), periods=list(range(24)), training_seeds=list(SEEDS),
        selected_updates=[4,4,4], multipliers=list(MULTIPLIERS),
        coverage_source=str(COVERAGE), state_source="same frozen MC development states; no new independent test",
        subgroup="prior axis-responsive, fixed before this batch; preserve all states",
        primary="canonical action changes relative to frozen zero probe; not profit advantage",
        maximum_solves=288, wall_seconds=600, disk_limit_bytes=2*2**30,
        workers=1, solver_threads=1, no_training=True, no_profit_selection=True,
        sample_unit="two development scenarios; states/models/multipliers nested, descriptive only",
        next_step="one hypothesis recommendation; no automatic further experiment")
    write_json(target/"protocol/preregistration.json", protocol)
    write_json(target/"data"/(CONDITION+".json"), dict(**item, policies=policies))
    local = {p.relative_to(target).as_posix():sha(p) for f in ("protocol","data") for p in (target/f).glob("*.json")}
    write_json(target/"manifest.json", dict(status="PREPARED", created_at=now(), source_identity=source_identity(),
        source_files=sources(), history_hashes=history, input_hashes=local))
    write_json(target/"qa/preparation.json", dict(status="PASSED", history_files=len(history),
        models=3, reused_development_scenarios=2, states=48, no_new_independent_test=True))
    print("PREPARED: 48 reused public states; three update-4 actors; no training", flush=True)


def load(attempt, statuses):
    target = Path(attempt).resolve()
    manifest = read_json(target/"manifest.json")
    if manifest["status"] not in statuses:
        raise RuntimeError("stage cannot repeat: "+manifest["status"])
    if manifest["source_identity"] != source_identity() or manifest["source_files"] != sources():
        raise RuntimeError("frozen source changed")
    verify_hashes(target, manifest["input_hashes"])
    return target, manifest


def run(attempt):
    import torch
    import psutil
    from model_families.single.learning.ppo.policy import FrozenPolicy
    from experiments.cfs_ppo_strategic_control.study_execution import episode, replay
    from model_families.single.runner import state_snapshot
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    target, manifest = load(attempt, {"PREPARED"})
    verify_hashes(Path("/"), manifest["history_hashes"])
    item = read_json(target/"data"/(CONDITION+".json"))
    budget = Budget()
    manifest.update(status="RUNNING", numerical_started_at=now())
    write_json(target/"manifest.json", manifest)
    total = logical_count = 0
    try:
        actors = [(p["seed"],FrozenPolicy(p["checkpoint"])) for p in item["policies"]]
        for scenario in item["scenarios"]:
            frozen = read_json(COVERAGE/"raw"/f"s{scenario['seed']}.json")
            replay_result = replay(scenario, item["config"], frozen["trajectory"])
            ep = episode(scenario, item["config"])
            ep.timeout_seconds = 600
            ep._cancelled = budget.check
            obs = ep.reset()
            rows = []
            for state, tr in zip(frozen["states"], frozen["trajectory"]["transitions"], strict=True):
                budget.check()
                if (asdict(obs) != state["observation"] or state_snapshot(ep._sim) != state["before_state"]):
                    raise RuntimeError("public state reconstruction mismatch")
                ref = state["probes"][state["reference_key"]]
                audit_probe(ep, obs, (0.,)*10, ref)
                cache = {state["reference_key"]:copy.deepcopy(ref)}
                logical, outputs = [], []
                for seed, policy in actors:
                    a = policy.act(obs)
                    if a != policy.act(obs):
                        raise RuntimeError("deterministic actor differs")
                    outputs.append(dict(training_seed=seed, signal=list(a)))
                    for multiplier in MULTIPLIERS:
                        signal, clipped = scaled_signal(a, multiplier)
                        key, p, cached = probe_at(ep, obs, signal, cache, budget)
                        logical.append(dict(training_seed=seed, multiplier=multiplier, signal=list(signal),
                            clipped_components=clipped, probe_key=key, cached=cached,
                            changed=decision_identity(p["plan"]) != state["reference_canonical_hash"]))
                rows.append(dict(scenario_seed=scenario["seed"], period=obs.period,
                    observation_hash=obs.hash, reference_key=state["reference_key"],
                    prior_axis_responsive=any(r["changed"] for r in state["logical"] if r["method"]=="axis"),
                    actors=outputs, logical=logical, probes=cache))
                total += 1; logical_count += len(logical)
                obs = advance_frozen_mc(ep, tr)
            if obs is not None or [r["period"] for r in rows] != list(range(24)):
                raise RuntimeError("incomplete frozen trajectory")
            write_json(target/"raw"/f"s{scenario['seed']}.json", dict(states=rows, replay=replay_result, completed_at=now()))
            if sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) > 2*2**30:
                raise BudgetLimited("disk limit reached")
            print("scene",scenario["seed"],"states",total,"solver calls",budget.solves,flush=True)
        budget.check()
        if total != 48 or logical_count != 288 or budget.solves > 288:
            raise RuntimeError("registered coverage/cost mismatch")
        hashes = {p.relative_to(target).as_posix():sha(p) for p in (target/"raw").glob("*.json")}
        write_json(target/"qa/numerical.json", dict(status="PASSED", states=48, logical_probes=288,
            actual_solver_calls=budget.solves, baseline_solver_calls=0, wall_seconds=time.monotonic()-budget.started,
            peak_working_set_bytes=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss),
            public_boundaries_reconstructed=48, full_MC_cooldown_replay=True,
            actor_repeatability=True, independent_compile_cash_coefficients=True, live_probe_unchanged=True,
            all_optimal=True, raw_hashes=hashes))
        manifest.update(status="NUMERICAL_COMPLETE_QA_PENDING", numerical_completed_at=now(), raw_hashes=hashes)
        write_json(target/"manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="BUDGET_LIMITED" if isinstance(exc,BudgetLimited) else "INVALID",stopped_at=now())
        write_json(target/"manifest.json",manifest)
        write_json(target/"qa/stop.json",dict(status=manifest["status"],error=type(exc).__name__,message=str(exc),solves=budget.solves))
        if "rows" in locals():
            write_json(target/"raw/partial.json",dict(completed=False,states=rows))
        raise


def freeze(attempt):
    target, manifest = load(attempt,{"NUMERICAL_COMPLETE_QA_PENDING"})
    for name in ("preparation","numerical","analysis","tests","visual-review"):
        if read_json(target/"qa"/(name+".json"))["status"] != "PASSED":
            raise RuntimeError("required QA missing: "+name)
    tests = read_json(target/"qa/tests.json")
    if sha(target/tests["xml_file"]) != tests["xml_sha256"]:
        raise RuntimeError("XML changed")
    verify_hashes(Path("/"),manifest["history_hashes"])
    verify_hashes(target,manifest["raw_hashes"])
    for name,field in (("analysis","output_hashes"),("visual-review","reviewed_hashes")):
        verify_hashes(target,read_json(target/"qa"/(name+".json"))[field])
    hashes = {p.relative_to(target).as_posix():sha(p) for p in target.rglob("*") if p.is_file()
              and p.name not in ("manifest.json","final_manifest.json")}
    manifest.update(status="COMPLETED",completed_at=now(),final_artifact_hashes=hashes,
        scope="reused development states, learned-direction mechanism only, not profitability evidence")
    write_json(target/"manifest.json",manifest)
    write_json(target/"final_manifest.json",manifest)
    verify_hashes(target,hashes)
    print("COMPLETED",len(hashes),"deliverable hashes verified",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","run","freeze"))
    parser.add_argument("--attempt",required=True)
    args=parser.parse_args()
    for k,v in dict(PYTHONHASHSEED="0",PYTHONIOENCODING="utf-8",PYTHONDONTWRITEBYTECODE="1",
                    OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1").items():
        if os.environ.get(k)!=v:
            raise RuntimeError("required environment: "+k)
    globals()[args.stage](args.attempt)
