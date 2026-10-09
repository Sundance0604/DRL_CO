"""Stage-gated first CFS PPO experiment. No hidden budget approval or result selection."""
import argparse
import copy
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
from model_families.single.strategic.artifacts import read_json, write_json
from .study_common import CONDITIONS, key, prepare, load, run_config, prepared_train_data, data, now, sha
from .study_execution import episode, run_record, run_rh4, replay, physical_hash


def ensure_environment():
    for name, expected in dict(PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1").items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"worker environment requires {name}={expected}")


def journal(target, stage, status, **extra):
    path = target/"qa/run-journal.json"
    rows = read_json(path) if path.exists() else []
    rows.append(dict(at=now(), stage=stage, status=status, **extra))
    write_json(path, rows)


def summary_record(record, condition, method, file):
    metric = record["metrics"]
    if "transitions" in record:
        solver_times = [r["plan"]["solver"].get("runtime", 0) for r in record["transitions"]]
    else:
        solver_times = [r["plan"]["solver"].get("runtime", 0) for r in record["original_trace"] if not r.get("drain")]
    return dict(condition=condition, method=method, scenario_id=record["scenario_id"],
                scenario_seed=record["scenario_seed"], profit_USD=record["ledger"]["realized_profit"],
                delivered=metric.get("delivered", metric.get("completed_orders", 0)),
                cancelled=metric.get("cancelled", metric.get("cancelled_orders", 0)),
                unfinished=metric["unfinished_committed"],
                episode_seconds=metric.get("episode_seconds", record.get("episode_seconds")),
                solver_times=solver_times, policy_seconds=record.get("policy_seconds", 0),
                output_bytes=Path(file).stat().st_size, file=str(file))


def pilot(attempt):
    from model_families.single.learning.ppo.runner import train_run
    from model_families.single.learning.ppo.checkpoints import load_actor
    import torch
    import psutil
    target, manifest = load(attempt)
    if read_json(target/"qa/engineering.json")["status"] != "PASSED":
        raise RuntimeError("engineering gate required")
    summaries, training = [], []
    started_all = time.perf_counter()
    peak_rss = psutil.Process().memory_info().rss
    for condition in CONDITIONS:
        name, cfg = key(condition), run_config(target, condition, 29999, True)
        output = target/"preflight"/"pilot-training"/name
        started = time.perf_counter()
        train_run(cfg, prepared_train_data(target, condition, True), output)
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        duration = time.perf_counter()-started
        files = list(output.glob("training-episode-*.json"))+list(output.glob("validation-update-*-scenario-*.json"))
        rows = []
        for file in files:
            record = read_json(file)
            stage = "training" if file.name.startswith("training") else "validation"
            rows.append(summary_record(record, name, stage, file))
        training.append(dict(condition=name, wall_seconds=duration, train_episodes=16,
            validation_episodes=32, rows=rows, checkpoint_bytes=sum(p.stat().st_size for p in output.glob("update-*/*")),
            total_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())))
        if condition == CONDITIONS[0]:
            resumed = target/"preflight/pilot-resumed"/name
            train_run(cfg, prepared_train_data(target, condition, True), resumed, output/"update-0002")
            full, *_ = load_actor(output/"last-inference")
            continued, *_ = load_actor(resumed/"last-inference")
            if not all(torch.equal(v, continued.state_dict()[k]) for k, v in full.state_dict().items()):
                raise RuntimeError("CFS update-boundary resume differs from continuous training")
            write_json(target/"qa/CFS-resume.json", dict(status="PASSED", condition=name,
                uninterrupted=str(output), resumed=str(resumed), boundary_update=2, exact_tensor_equality=True))
        output = target/"preflight"/"pilot-baselines"/name
        output.mkdir(parents=True, exist_ok=False)
        scenes = data(target, condition, "pilot_validation")["scenarios"]
        for method in ("MC", "M0", "RH4"):
            local = copy.deepcopy(cfg)
            if method == "M0":
                local["baseline"] = dict(method="zero")
            for scenario in scenes:
                record = run_rh4(scenario, local) if method == "RH4" else run_record(scenario, local, lambda o: (0.,)*10, method)
                file = output/f"{method}-{scenario['seed']}.json"
                write_json(file, record)
                if method != "RH4":
                    replay(scenario, local, record)
                summaries.append(summary_record(record, name, method, file))
        for near in (-.5, 0., .5):
            for far in (-.5, 0., .5):
                fixed = (near, far)*5
                for scenario in scenes[:2]:
                    record = run_record(scenario, cfg, lambda o: fixed, "CAL")
                    file = output/f"CAL-{near}-{far}-{scenario['seed']}.json"
                    write_json(file, record)
                    replay(scenario, cfg, record)
                    summaries.append(summary_record(record, name, "CAL", file))
        print("Pilot passed", name, f"training wall {duration:.1f}s", flush=True)
    estimate = {}
    for condition in CONDITIONS:
        name = key(condition)
        sample = next(r for r in training if r["condition"] == name)
        # Exact formal train/update/validation ratio is 8/4 and 1 checkpoint
        # validation every 16 updates instead of every pilot update.
        train_rows = [r for r in sample["rows"] if r["method"] == "training"]
        val_rows = [r for r in sample["rows"] if r["method"] == "validation"]
        train_time = sum(r["episode_seconds"] for r in train_rows)
        validation_time = sum(r["episode_seconds"] for r in val_rows)
        overhead = max(0., sample["wall_seconds"]-train_time-validation_time)
        mc = [r for r in summaries if r["condition"] == name and r["method"] == "MC"]
        m0 = [r for r in summaries if r["condition"] == name and r["method"] == "M0"]
        rh4 = [r for r in summaries if r["condition"] == name and r["method"] == "RH4"]
        calibrated = [r for r in summaries if r["condition"] == name and r["method"] == "CAL"]
        per_training = train_time/16
        per_validation = validation_time/32
        per_update = overhead/4
        formal_training = 5*(1024*per_training+256*per_validation+128*per_update)
        calibration = 9*32*np.mean([r["episode_seconds"] for r in calibrated])
        evaluation = 80*(5*per_validation+2*np.mean([r["episode_seconds"] for r in mc])+
                          np.mean([r["episode_seconds"] for r in m0])+np.mean([r["episode_seconds"] for r in rh4]))
        byte_per_episode = sum(r["output_bytes"] for r in sample["rows"])/48
        checkpoint_bytes = sample["checkpoint_bytes"]/4
        additional_bytes = byte_per_episode*(5120+1280+720+288+160)+checkpoint_bytes*640
        estimate[name] = dict(formal_training_seconds=formal_training, calibration_seconds=float(calibration),
            formal_evaluation_seconds=float(evaluation), projected_total_seconds=float(formal_training+calibration+evaluation),
            per_training_episode_seconds=per_training, per_validation_episode_seconds=per_validation,
            episode_p95_seconds=float(np.quantile([r["episode_seconds"] for r in sample["rows"]], .95)),
            episode_max_seconds=max(r["episode_seconds"] for r in sample["rows"]),
            projected_output_bytes=float(additional_bytes), checkpoint_bytes_per_update=checkpoint_bytes)
    seconds = sum(r["projected_total_seconds"] for r in estimate.values())
    estimated_bytes = sum(r["projected_output_bytes"] for r in estimate.values())
    budget = dict(status="AWAITING_HUMAN_CONFIRMATION", measured_at=now(), conditions=estimate,
        projected_serial_hours=seconds/3600, conservative_serial_hours=seconds*2/3600,
        projected_output_GiB=estimated_bytes/2**30, conservative_output_GiB=estimated_bytes*2/2**30,
        free_storage_GiB=shutil.disk_usage(target).free/2**30,
        peak_process_RSS_GiB=peak_rss/2**30, logical_CPUs=os.cpu_count(),
        pilot_elapsed_seconds=time.perf_counter()-started_all, formal_episodes=22224,
        extra_costs="replay, independent validation recomputation and diagnostics add time; 2x reserve included",
        inference_mode="deterministic", workers=1)
    write_json(target/"preflight/pilot-summary.json", dict(training=training, baselines=summaries))
    write_json(target/"protocol/budget-proposal.json", budget)
    manifest.update(status="PILOT_PASSED_BUDGET_PENDING", pilot_at=now())
    write_json(target/"manifest.json", manifest)
    print("Budget proposal", budget["conservative_serial_hours"], "hours;", budget["conservative_output_GiB"], "GiB", flush=True)
    return budget


def budget_gate(target):
    path = target/"protocol/budget-confirmation.json"
    if not path.exists():
        raise RuntimeError("formal execution requires measured budget confirmation from the user")
    approval = read_json(path)
    if approval.get("approved") is not True or approval.get("proposal_sha256") != sha(target/"protocol/budget-proposal.json"):
        raise RuntimeError("budget approval missing or proposal changed")
    return approval


def formal_training(attempt):
    from model_families.single.learning.ppo.runner import train_run
    target, manifest = load(attempt)
    budget_gate(target)
    for condition in CONDITIONS:
        name = key(condition)
        for seed in range(30001, 30006):
            output = target/"training"/name/str(seed)
            if output.exists():
                state = read_json(output/"status.json")
                if state["status"] != "COMPLETED" or state["episodes"] != 1024:
                    raise RuntimeError("partial training retained; explicit boundary recovery/new attempt required")
                continue
            cfg = run_config(target, condition, seed)
            journal(target, "training", "STARTED", condition=name, seed=seed)
            train_run(cfg, prepared_train_data(target, condition), output)
            journal(target, "training", "COMPLETED", condition=name, seed=seed)
            print("Formal training complete", name, seed, flush=True)
    manifest.update(status="FORMAL_TRAINING_COMPLETE", formal_training_at=now())
    write_json(target/"manifest.json", manifest)


def select(attempt):
    from model_families.single.learning.ppo.policy import FrozenPolicy
    target, manifest = load(attempt)
    budget_gate(target)
    selected, validation_audit = [], []
    for condition in CONDITIONS:
        name, cfg = key(condition), run_config(target, condition, 30001)
        output = target/"validation"/name
        output.mkdir(exist_ok=False)
        scenes = data(target, condition, "validation")["scenarios"]
        grid = []
        for near in (-.5, 0., .5):
            for far in (-.5, 0., .5):
                signal = (near, far)*5
                profits = []
                for scenario in scenes:
                    record = run_record(scenario, cfg, lambda o: signal, "CAL")
                    file = output/f"CAL-{near}-{far}-{scenario['seed']}.json"
                    write_json(file, record)
                    replay(scenario, cfg, record)
                    profits.append(record["ledger"]["realized_profit"])
                grid.append(dict(near=near, far=far, mean_profit_USD=sum(profits)/32, profits=profits))
                print("CAL candidate complete", name, near, far, flush=True)
        best = sorted(grid, key=lambda row: (-row["mean_profit_USD"], (row["near"], row["far"]) != (0., 0.),
                       row["near"]**2+row["far"]**2, row["near"], row["far"]))[0]
        mc = next(r for r in grid if r["near"] == r["far"] == 0.)
        write_json(output/"CAL-selection.json", dict(selected=best, candidates=grid,
            tie="exact mean tie: prefer zero, then squared norm, then lexicographic near/far"))
        write_json(output/"MC-summary.json", mc)
        for seed in range(30001, 30006):
            training = target/"training"/name/str(seed)
            state = read_json(training/"status.json")
            if state["status"] != "COMPLETED" or state["episodes"] != 1024:
                raise RuntimeError("training incomplete")
            scores = [read_json(p) for p in sorted(training.glob("validation-[0-9][0-9][0-9][0-9].json"))]
            if len(scores) != 8:
                raise RuntimeError("missing validation checkpoints")
            best_validation = max(scores, key=lambda row: row["mean_profit_USD"])
            decision = read_json(training/"best.json")
            if decision["update"] != best_validation["update"]:
                raise RuntimeError("validation selection mismatch")
            checkpoint = training/"best-inference"
            policy = FrozenPolicy(checkpoint)
            recomputed = []
            for scenario in scenes:
                record = run_record(scenario, cfg, policy.act, f"PPO-{seed}-validation-recompute")
                file = output/f"PPO-{seed}-{scenario['seed']}.json"
                write_json(file, record)
                replay(scenario, cfg, record)
                recomputed.append(record["ledger"]["realized_profit"])
            original = [r["profit_USD"] for r in best_validation["scenarios"]]
            if not np.allclose(recomputed, original, atol=1e-6, rtol=1e-10):
                raise RuntimeError("selected checkpoint per-scenario validation differs from independent recomputation")
            validation_audit.append(dict(condition=name, seed=seed, update=decision["update"], status="PASSED"))
            selected.append(dict(condition=name, seed=seed, checkpoint=str(checkpoint.relative_to(target)),
                files={p.name: sha(p) for p in checkpoint.iterdir() if p.is_file()}, selected_update=decision["update"],
                validation_mean_USD=best_validation["mean_profit_USD"]))
        selected.append(dict(condition=name, method="CAL", near=best["near"], far=best["far"]))
    w80_mc = read_json(target/"validation/western_multicluster-n80-k3/MC-summary.json")["mean_profit_USD"]
    write_json(target/"protocol/selected-models.json", dict(selected_at=now(), formal_results_seen=False,
        selected=selected, W80_noninferiority_delta_USD=.01*max(abs(w80_mc), 1000.),
        MC_W80_validation_mean_USD=w80_mc))
    write_json(target/"qa/validation-selection.json", dict(status="PASSED", rows=validation_audit))
    manifest.update(status="MODELS_SELECTED_FROZEN", selected_at=now())
    write_json(target/"manifest.json", manifest)


def evaluation(attempt):
    from model_families.single.learning.ppo.runner import evaluate_run
    target, manifest = load(attempt)
    budget_gate(target)
    selected = read_json(target/"protocol/selected-models.json")
    for condition in CONDITIONS:
        name, cfg = key(condition), run_config(target, condition, 30001)
        output = target/"evaluation"/name
        output.mkdir(exist_ok=False)
        sealed = read_json(target/"sealed"/name/"test.json")
        rows = []
        for seed in range(30001, 30006):
            choice = next(r for r in selected["selected"] if r.get("condition") == name and r.get("seed") == seed)
            checkpoint = target/choice["checkpoint"]
            if any(sha(checkpoint/file) != expected for file, expected in choice["files"].items()):
                raise RuntimeError("frozen selected checkpoint changed")
            run_output = output/f"PPO-{seed}"
            evaluate_run(checkpoint, sealed, run_output)
            for file, scenario in zip(sorted(run_output.glob("evaluation-*.json")), sealed["scenarios"]):
                record = read_json(file)
                replay(scenario, cfg, record)
                rows.append(summary_record(record, name, f"PPO-{seed}", file))
            print("Formal evaluation complete", name, seed, flush=True)
        cal = next(r for r in selected["selected"] if r.get("condition") == name and r.get("method") == "CAL")
        signal = (cal["near"], cal["far"])*5
        for method in ("MC", "CAL", "M0", "RH4"):
            folder = output/method
            folder.mkdir()
            local = copy.deepcopy(cfg)
            if method == "M0":
                local["baseline"] = dict(method="zero")
            for scenario in sealed["scenarios"]:
                policy = (lambda o: signal) if method == "CAL" else (lambda o: (0.,)*10)
                record = run_rh4(scenario, local) if method == "RH4" else run_record(scenario, local, policy, method)
                file = folder/f"scenario-{scenario['seed']}.json"
                write_json(file, record)
                if method != "RH4":
                    replay(scenario, local, record)
                rows.append(summary_record(record, name, method, file))
            print("Formal baseline complete", name, method, flush=True)
        write_json(output/"summary.json", dict(rows=rows, scenarios=80, methods=9, status="COMPLETED"))
    manifest.update(status="FORMAL_EVALUATION_COMPLETE", formal_evaluation_at=now())
    write_json(target/"manifest.json", manifest)


def diagnostics(attempt):
    from model_families.single.learning.ppo.policy import FrozenPolicy
    target, manifest = load(attempt)
    selected = read_json(target/"protocol/selected-models.json")["selected"]
    summaries = []
    for condition in CONDITIONS:
        name, cfg = key(condition), run_config(target, condition, 30001)
        output = target/"diagnostics"/name
        output.mkdir(exist_ok=False)
        choice = next(r for r in selected if r.get("condition") == name and r.get("seed") == 30001)
        policy = FrozenPolicy(target/choice["checkpoint"])
        cal = next(r for r in selected if r.get("condition") == name and r.get("method") == "CAL")
        cal_signal = (cal["near"], cal["far"])*5
        for scenario in data(target, condition, "diagnostic")["scenarios"]:
            ep = episode(scenario, cfg)
            obs = ep.reset()
            same_state = []
            while True:
                ppo_signal = policy.act(obs)
                mc, ppo, calibrated = ep.probe((0.,)*10), ep.probe(ppo_signal), ep.probe(cal_signal)
                same_state.append(dict(period=obs.period, observation_hash=obs.hash, state_source="MC zero residual trajectory",
                    PPO_signal=ppo_signal, CAL_signal=cal_signal,
                    PPO_coefficients=ppo["coefficients"], MC_coefficients=mc["coefficients"],
                    CAL_coefficients=calibrated["coefficients"], PPO_action_hash=ppo["action_hash"],
                    MC_action_hash=mc["action_hash"], CAL_action_hash=calibrated["action_hash"],
                    PPO_MC_disagreement=ppo["action_hash"] != mc["action_hash"],
                    PPO_CAL_disagreement=ppo["action_hash"] != calibrated["action_hash"]))
                transition = ep.step((0.,)*10)
                if transition.terminated:
                    break
                obs = transition.observation
            record = dict(condition=name, scenario_seed=scenario["seed"], training_seed=30001,
                          same_state=same_state, trajectory=dict(ledger=ep.final_ledger,
                                                               metrics=ep.metrics, transitions=ep.records))
            file = output/f"scenario-{scenario['seed']}.json"
            write_json(file, record)
            replay(scenario, cfg, record["trajectory"])
            values = [a for r in same_state for a in r["PPO_signal"]]
            lookups = {}
            for row in same_state:
                for coefficient in row["PPO_coefficients"]:
                    lookups[coefficient["lookup"]] = lookups.get(coefficient["lookup"], 0)+1
            summaries.append(dict(condition=name, scenario_seed=scenario["seed"], periods=24,
                PPO_MC_disagreement=sum(r["PPO_MC_disagreement"] for r in same_state)/24,
                PPO_CAL_disagreement=sum(r["PPO_CAL_disagreement"] for r in same_state)/24,
                nonzero_fraction=sum(abs(v)>1e-9 for v in values)/len(values),
                saturation_fraction=sum(abs(v)>.99 for v in values)/len(values), lookup_counts=lookups))
    write_json(target/"diagnostics/summary.json", dict(rows=summaries, training_seed=30001,
        selection="predefined all 16 diagnostic seeds, plots seed 23000", state_source="MC", support_count="unknown"))
    manifest.update(status="DIAGNOSTICS_COMPLETE", diagnostics_at=now())
    write_json(target/"manifest.json", manifest)


def audit(attempt):
    target, _ = load(attempt)
    audited, training, evaluation = [], [], []
    for condition in CONDITIONS:
        name, cfg = key(condition), run_config(target, condition, 30001)
        scenarios = {s["id"]: s for split in ("train", "validation") for s in data(target, condition, split)["scenarios"]}
        for seed in range(30001, 30006):
            folder = target/"training"/name/str(seed)
            for file in sorted(folder.glob("training-episode-*.json"))+sorted(folder.glob("validation-update-*-scenario-*.json")):
                record = read_json(file)
                audited.append(dict(file=str(file.relative_to(target)), **replay(scenarios[record["scenario_id"]], cfg, record)))
            updates = [read_json(p) for p in sorted(folder.glob("update-*.json"))]
            training.append(dict(condition=name, seed=seed, wall_seconds=updates[-1]["elapsed_seconds"],
                KL_early_stop_count=sum(u["kl_early_stop"] for u in updates),
                mean_saturation=float(np.mean([u["action_saturation_fraction"] for u in updates])), updates=128,
                bytes=sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())))
        evaluation.extend(read_json(target/"evaluation"/name/"summary.json")["rows"])
        print("Full trajectory replay passed", name, flush=True)
    write_json(target/"qa/full-trajectory-replay.json", dict(status="PASSED", rows=audited, episodes=len(audited)))
    write_json(target/"analysis/computation.json", dict(training=training, evaluation=evaluation,
        failed_runs=0, formal_episode_count=22224, completed_at=now()))
    manifest = read_json(target/"manifest.json")
    for name, expected in {**manifest["history_hashes"], **manifest["historical_config_hashes"]}.items():
        if sha(Path(__file__).resolve().parents[2]/name) != expected:
            raise RuntimeError("historical input changed during study: "+name)
    if sha(manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise RuntimeError("CFS source changed during study")
    write_json(target/"qa/historical-readonly.json", dict(status="PASSED", at=now(), files=len(manifest["history_hashes"]),
        CFS_source_sha256=manifest["source"]["sha256"]))


def finish(attempt):
    target, manifest = load(attempt)
    for name in ("engineering", "data-isolation", "CFS-resume", "validation-selection", "full-trajectory-replay",
                 "historical-readonly", "statistical-recalculation", "final-tests", "visual-review"):
        if read_json(target/"qa"/(name+".json"))["status"] != "PASSED":
            raise RuntimeError("delivery QA not passed: "+name)
    if not (target/"ppo_first_cfs_report.md").exists():
        raise RuntimeError("report missing")
    manifest.update(status="COMPLETED", completed_at=now(),
        final_artifact_hashes={p.relative_to(target).as_posix(): sha(p) for p in sorted(target.rglob("*"))
                               if p.is_file() and p.name not in ("manifest.json", "final_manifest.json")})
    write_json(target/"manifest.json", manifest)
    write_json(target/"final_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description="Stage-gated round-eight scientific study")
    parser.add_argument("stage", choices=("prepare", "engineering", "pilot", "train", "select", "evaluate", "diagnose", "audit", "analyse", "deliver"))
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    ensure_environment()
    if args.stage == "prepare":
        prepare(args.attempt)
        return
    target, _ = load(args.attempt)
    journal(target, args.stage, "STARTED")
    try:
        if args.stage == "engineering":
            from .study_execution import engineering
            engineering(target)
        elif args.stage == "pilot":
            pilot(target)
        elif args.stage == "train":
            formal_training(target)
        elif args.stage == "select":
            select(target)
        elif args.stage == "evaluate":
            evaluation(target)
        elif args.stage == "diagnose":
            diagnostics(target)
        elif args.stage == "audit":
            audit(target)
        elif args.stage == "analyse":
            from .study_analysis import figures, report
            figures(target)
            report(target)
        elif args.stage == "deliver":
            finish(target)
        if args.stage != "deliver":
            journal(target, args.stage, "COMPLETED")
    except BaseException as exc:
        write_json(target/"qa/stop_condition_failure.json", dict(status="STOP_CONDITION_TRIGGERED",
            stage=args.stage, at=now(), error=type(exc).__name__, message=str(exc)))
        journal(target, args.stage, "FAILED", error=str(exc))
        manifest = read_json(target/"manifest.json")
        manifest.update(status="STOP_CONDITION_TRIGGERED", failed_at=now(), failed_stage=args.stage)
        write_json(target/"manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
