"""Laptop-scale mechanism experiment using existing models and fresh held-outs."""
import argparse
import csv
import logging
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from model_families.single.strategic.artifacts import read_json, write_json, source_identity
from model_families.single.strategic.contracts import EncodedObservation, VehicleObservation
from ..study_common import ROOT, PACKAGE, CONDITIONS, key, now, sha, payload, study_sources
from ..study_execution import episode, run_record, replay
from ..study_run import summary_record, journal
from ..laptop.run import SEEDS, configure_process, resource_guard, check_tests, independent_bootstrap, sources as laptop_sources

TEST_SEEDS = tuple(range(22012, 22020))
TOL = 1e-6


def sources():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def restore_observation(blob, expected_hash):
    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, (list, tuple)) else value
    fields = dict(blob)
    fields["vehicles"] = tuple(VehicleObservation(**{k:tuples(v) for k,v in row.items()}) for row in blob["vehicles"])
    for name in ("hubs", "global_features", "hub_features", "orders"):
        fields[name] = tuples(fields[name])
    result = EncodedObservation(**fields)
    if result.hash != expected_hash:
        raise RuntimeError("restored public observation hash differs from frozen trace")
    return result


def constant_signal(signals, dimensions):
    matrix = np.asarray(signals, dtype=float)
    if matrix.ndim != 2 or not len(matrix) or matrix.shape[1] != dimensions:
        raise ValueError("fixed signal needs a nonempty full-dimensional validation matrix")
    if not np.isfinite(matrix).all() or np.any(np.abs(matrix) > 1):
        raise ValueError("invalid validation signal")
    return tuple(float(x) for x in matrix.mean(axis=0))


def fresh_guard(scenarios, used):
    from model_families.single.learning.ppo.runner import scenario_content
    if len({s["id"] for s in scenarios}) != len(scenarios) or len({s["seed"] for s in scenarios}) != len(scenarios):
        raise RuntimeError("duplicate new test identity")
    contents = [scenario_content(s) for s in scenarios]
    if len(set(contents)) != len(contents):
        raise RuntimeError("duplicate new physical instance")
    for s,content in zip(scenarios,contents):
        if s["split"] != "test" or any(s["id"] == e["id"] or s["seed"] == e["seed"] or content == e["content_hash"] for e in used):
            raise RuntimeError("new test overlaps earlier used data")


def paired_statistics(difference, family_size=3, replications=20000, seed=32000):
    x = np.asarray(difference,dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or min(x.shape) < 1:
        raise ValueError("finite crossed difference matrix required")
    if not isinstance(family_size,int) or family_size < 1:
        raise ValueError("positive comparison family size required")
    rng = np.random.default_rng(seed)
    direct=[]
    for start in range(0,replications,1000):
        n = min(1000,replications-start)
        a = rng.integers(0,x.shape[0],(n,x.shape[0]))
        b = rng.integers(0,x.shape[1],(n,x.shape[1]))
        direct.extend(x[a[:,:,None],b[:,None,:]].mean(axis=(1,2)))
    direct = np.asarray(direct)
    independent = independent_bootstrap(x,replications,seed)
    alpha=.05/family_size
    qs=[.025,.975,alpha/2,1-alpha/2]
    if not np.allclose(np.quantile(direct,qs),np.quantile(independent,qs),atol=1e-8,rtol=1e-12):
        raise RuntimeError("independent primary bootstrap recalculation failed")
    return dict(mean_USD=float(x.mean()),seed_mean_USD=x.mean(axis=1).tolist(),
        scene_mean_USD=x.mean(axis=0).tolist(),marginal_95=np.quantile(direct,[.025,.975]).tolist(),
        simultaneous_interval=np.quantile(direct,[alpha/2,1-alpha/2]).tolist(),
        interval_level=1-alpha,family_size=family_size,replications=replications,bootstrap_seed=seed,
        seed_positive_count=int(sum(v>TOL for v in x.mean(axis=1))),
        scene_seed_positive_fraction=float(np.mean(x>TOL)),equivalent_fraction=float(np.mean(np.abs(x)<=TOL)),
        same_profit_tolerance_USD=TOL)


def load(target):
    target=Path(target).resolve()
    m=read_json(target/"manifest.json")
    if m["status"] in ("COMPLETED","STOP_CONDITION_TRIGGERED"):
        raise RuntimeError("completed or failed attempts are immutable")
    if m["source_identity"]!=source_identity() or m["study_sources"]!=study_sources() or m["mechanism_sources"]!=sources() or m["laptop_sources"]!=laptop_sources():
        raise RuntimeError("frozen numerical sources changed")
    for relative,digest in m["input_hashes"].items():
        if sha(target/relative)!=digest:
            raise RuntimeError("frozen mechanism input changed: "+relative)
    for relative,digest in m["parent_checks"].items():
        if sha(ROOT/relative)!=digest:
            raise RuntimeError("parent evidence changed: "+relative)
    return target,m


def config(target,name):
    cfg=read_json(target/"protocol"/(name+".json"))
    cfg["baseline"]=str(target/cfg["baseline"])
    return cfg


def prepare(target,parent):
    from model_families.single.learning.ppo.policy import FrozenPolicy
    from model_families.single.learning.ppo.runner import scenario_content, validate_research_payload
    target,parent=Path(target).resolve(),Path(parent).resolve()
    old=read_json(parent/"final_manifest.json")
    if old["status"]!="COMPLETED" or old["source_identity"]!=source_identity() or old["study_sources"]!=study_sources() or old["laptop_sources"]!=laptop_sources():
        raise RuntimeError("completed parent model experiment is not compatible")
    # Verify all frozen parent files, not merely the selected actor weights.
    for relative,digest in old["final_artifact_hashes"].items():
        if sha(parent/relative)!=digest:
            raise RuntimeError("completed parent artifact changed: "+relative)
    heavy=PACKAGE/"runs/cfs_first_ppo_v1"
    heavy_manifest=read_json(heavy/"manifest.json")
    if heavy_manifest["source_identity"]!=source_identity() or heavy_manifest["study_sources"]!=study_sources():
        raise RuntimeError("original frozen CFS pool source mismatch")
    target.mkdir(parents=True,exist_ok=False)
    for directory in ("protocol","models","baselines","data","calibration","evaluation","diagnostics","analysis","figures","qa"):
        (target/directory).mkdir()
    protocol=dict(schema_version="cfs-dynamic-vs-hub-constant/v1",created_at=now(),
        authorization="用户：好，按照你的指示进行；承接动态信号与每枢纽固定信号机制验证建议",
        conditions=CONDITIONS,training_seeds=list(SEEDS),new_training_episodes=0,
        validation_scenario_seeds=list(range(21000,21008)),test_scenario_seeds=list(TEST_SEEDS),
        constant_rule="Per checkpoint, equal-weight arithmetic mean of all 8x24 deterministic outputs on frozen MC validation observations; one near/far pair per hub",
        primary="DYNAMIC minus HUB_CONSTANT per paired checkpoint and scene; remove all time/state dependence from actor output",
        comparison_family=3,bootstrap_replications=20000,bootstrap_seed=32000,
        secondary="MC and previously validation-selected CAL means, descriptive only",
        inference="deterministic",worker_count=1,torch_threads=1,solver_threads=1,
        test_scene_count_per_condition=8,evaluation_episodes=192,diagnostic_MC_episodes=3,
        same_state_probe_solves=432,diagnostic_scenario_seed=22012,
        signal_observation_probe="All 8 new MC test trajectories; no policy tuning",
        same_state_action_probe="Seed 22012 only, predefined, all 3 models, dynamic and frozen constant on identical MC states",
        money_zero_tolerance_USD=TOL,scientific_scope="Exploratory mechanism test for existing short-trained models; no new learning and no claim about optimum static policies or longer-trained RL",
        frozen_before_test=True)
    write_json(target/"protocol/preregistration.json",protocol)
    parent_checks={str((parent/"final_manifest.json").relative_to(ROOT)):sha(parent/"final_manifest.json"),
                   str((heavy/"manifest.json").relative_to(ROOT)):sha(heavy/"manifest.json")}
    used=read_json(parent/"data/identities.json")
    choices=read_json(parent/"protocol/selected-models.json")["selected"]
    constants,identity_rows=[],[]
    wall_budget,byte_budget=0.,0.
    for condition in CONDITIONS:
        name=key(condition)
        cfg=read_json(parent/"protocol"/(name+".json"))
        if cfg["solver"]["threads"]!=1:
            raise RuntimeError("single solver thread required")
        shutil.copyfile(parent/"baselines"/(name+".json"),target/"baselines"/(name+".json"))
        write_json(target/"protocol"/(name+".json"),cfg)
        # Candidate scenarios were generated earlier but have never been evaluated.
        source=heavy/"sealed"/name/"test.json"
        expected=heavy_manifest["input_hashes"][source.relative_to(heavy).as_posix()]
        if sha(source)!=expected:
            raise RuntimeError("original sealed test pool changed")
        scenes=[s for s in read_json(source)["scenarios"] if s["seed"] in TEST_SEEDS]
        if [s["seed"] for s in scenes]!=list(TEST_SEEDS):
            raise RuntimeError("new test seed subset mismatch")
        fresh_guard(scenes,[r for r in used if r["condition"]==name])
        validate_research_payload(payload(scenes))
        write_json(target/"data"/(name+".json"),payload(scenes))
        source_provenance=heavy/"sealed"/name/"test-provenance.json"
        if sha(source_provenance)!=heavy_manifest["input_hashes"][source_provenance.relative_to(heavy).as_posix()]:
            raise RuntimeError("sealed provenance changed")
        positions=[i for i,s in enumerate(read_json(source)["scenarios"]) if s["seed"] in TEST_SEEDS]
        provenance=read_json(source_provenance)
        write_json(target/"data"/(name+"-provenance.json"),[provenance[i] for i in positions])
        identity_rows.extend(dict(condition=name,id=s["id"],seed=s["seed"],content_hash=scenario_content(s)) for s in scenes)
        calibration_dir=target/"calibration"/name
        calibration_dir.mkdir()
        observations=[]
        for validation_seed in range(21000,21008):
            record=read_json(parent/"validation"/name/f"CAL-0.0-0.0-{validation_seed}.json")
            if record["scenario_seed"]!=validation_seed or len(record["transitions"])!=24:
                raise RuntimeError("MC validation reference mismatch")
            for tr in record["transitions"]:
                observations.append((validation_seed,restore_observation(tr["observation"],tr["observation_hash"])))
        for seed in SEEDS:
            choice=next(r for r in choices if r["condition"]==name and r.get("seed")==seed)
            cp=parent/choice["checkpoint"]
            destination=target/"models"/name/str(seed)
            shutil.copytree(cp,destination)
            if any(sha(destination/p)!=h for p,h in choice["files"].items()):
                raise RuntimeError("copied frozen actor mismatch")
            policy=FrozenPolicy(destination)
            check_tests(policy,scenes)
            signals=[policy.act(obs) for _,obs in observations]
            fixed=constant_signal(signals,2*len(cfg["action"]["hubs"]))
            trace=[dict(scenario_seed=s,period=obs.period,observation_hash=obs.hash,signal=list(a))
                   for (s,obs),a in zip(observations,signals)]
            write_json(calibration_dir/f"{seed}.json",dict(rows=trace,fixed_signal=fixed,observations=len(trace),
                component_std=np.asarray(signals).std(axis=0).tolist(),rule=protocol["constant_rule"]))
            constants.append(dict(condition=name,training_seed=seed,fixed_signal=fixed,
                checkpoint=destination.relative_to(target).as_posix(),parent_checkpoint=choice["checkpoint"],
                actor_SHA256=choice["files"]["actor.pt"],validation_observations=len(trace)))
        cal=next(r for r in choices if r["condition"]==name and r.get("method")=="CAL")
        constants.append(cal)
        previous=read_json(parent/"evaluation"/name/"summary.json")["rows"]
        ppo=[r for r in previous if r["method"].startswith("PPO-")]
        mc=[r for r in previous if r["method"]=="MC"]
        calibr=[r for r in previous if r["method"]=="CAL"]
        wall_budget+=8*(3*np.mean([r["episode_seconds"] for r in ppo])+4*np.mean([r["episode_seconds"] for r in mc])+np.mean([r["episode_seconds"] for r in calibr]))
        byte_budget+=65*np.mean([r["output_bytes"] for r in mc+ppo])
        print("Validation-only per-hub constants frozen",name,flush=True)
    write_json(target/"protocol/frozen-policies.json",dict(at=now(),test_outcomes_seen=False,policies=constants))
    write_json(target/"data/identities.json",identity_rows)
    write_json(target/"qa/data-isolation.json",dict(status="PASSED",new_scene_count=24,
        reused_prior_test_scenes=0,train_validation_prior_test_diagnostic_overlap=False))
    write_json(target/"protocol/budget.json",dict(projected_minutes=float(wall_budget*1.5/60),
        conservative_minutes=float(wall_budget*3/60),projected_output_GiB=float(byte_budget/2**30),
        conservative_output_GiB=float(byte_budget*2/2**30),new_training=0,workers=1))
    with zipfile.ZipFile(target/"protocol/source.snapshot.zip","w",zipfile.ZIP_DEFLATED) as archive:
        paths=[*sorted((ROOT/"model_families/single").rglob("*.py")),*sorted((ROOT/"experiment_core").glob("*.py")),
               *sorted(PACKAGE.glob("*.py")),*sorted((PACKAGE/"laptop").glob("*.py")),*sorted(Path(__file__).parent.glob("*.py")),ROOT/"pyproject.toml",ROOT/"uv.lock"]
        for p in paths:
            archive.write(p,p.relative_to(ROOT))
    manifest=dict(status="PREPARED",schema_version="cfs-mechanism/v1",created_at=now(),
        source_identity=source_identity(),study_sources=study_sources(),laptop_sources=laptop_sources(),
        mechanism_sources=sources(),parent_checks=parent_checks,source=old["source"],
        parent_directory=str(parent.relative_to(ROOT)),parent_artifacts=old["final_artifact_hashes"],
        input_hashes={p.relative_to(target).as_posix():sha(p) for folder in ("protocol","models","baselines","data","calibration")
                      for p in sorted((target/folder).rglob("*")) if p.is_file()})
    write_json(target/"manifest.json",manifest)
    print("Mechanism protocol ready",read_json(target/"protocol/budget.json"),flush=True)


def evaluate(target):
    from model_families.single.learning.ppo.policy import FrozenPolicy
    frozen=read_json(target/"protocol/frozen-policies.json")["policies"]
    all_rows=[]
    for condition in CONDITIONS:
        load(target)
        name=key(condition)
        cfg=config(target,name)
        scenes=read_json(target/"data"/(name+".json"))["scenarios"]
        output=target/"evaluation"/name
        output.mkdir()
        methods=[(f"DYNAMIC-{seed}",seed) for seed in SEEDS]+[(f"HUB_CONSTANT-{seed}",seed) for seed in SEEDS]+[("MC",None),("CAL",None)]
        for method,seed in methods:
            directory=output/method
            directory.mkdir()
            if seed:
                choice=next(r for r in frozen if r["condition"]==name and r.get("training_seed")==seed)
                if method.startswith("DYNAMIC"):
                    policy=FrozenPolicy(target/choice["checkpoint"])
                    check_tests(policy,scenes)
                    act=policy.act
                else:
                    signal=tuple(choice["fixed_signal"])
                    act=lambda obs:signal
            else:
                cal=next(r for r in frozen if r["condition"]==name and r.get("method")=="CAL")
                signal=(cal["near"],cal["far"])*len(cfg["action"]["hubs"]) if method=="CAL" else (0.,)*10
                act=lambda obs:signal
            for scene in scenes:
                r=run_record(scene,cfg,act,method)
                r.update(training_seed=seed,mechanism="dynamic" if method.startswith("DYNAMIC") else "fixed")
                p=directory/f"scenario-{scene['seed']}.json"
                write_json(p,r)
                replay(scene,cfg,r)
                all_rows.append(summary_record(r,name,method,p))
            print("New-scene evaluation + replay passed",name,method,flush=True)
        resource_guard(target)
    write_json(target/"evaluation/summary.json",dict(rows=all_rows,episode_count=len(all_rows),status="COMPLETED"))
    if len(all_rows)!=192:
        raise RuntimeError("missing evaluation episodes")
    write_json(target/"qa/evaluation-replay.json",dict(status="PASSED",episodes=192,unfinished_committed=0))


def diagnose(target):
    from model_families.single.learning.ppo.policy import FrozenPolicy
    frozen=read_json(target/"protocol/frozen-policies.json")["policies"]
    summaries=[]
    for condition in CONDITIONS:
        name=key(condition)
        cfg=config(target,name)
        directory=target/"diagnostics"/name
        directory.mkdir()
        choices=[next(r for r in frozen if r["condition"]==name and r.get("training_seed")==seed) for seed in SEEDS]
        policies=[FrozenPolicy(target/r["checkpoint"]) for r in choices]
        signals=[[] for _ in SEEDS]
        for scene_seed in TEST_SEEDS:
            record=read_json(target/"evaluation"/name/"MC"/f"scenario-{scene_seed}.json")
            for tr in record["transitions"]:
                obs=restore_observation(tr["observation"],tr["observation_hash"])
                for i,policy in enumerate(policies):
                    signals[i].append(policy.act(obs))
        scenes=read_json(target/"data"/(name+".json"))["scenarios"]
        scene=next(s for s in scenes if s["seed"]==22012)
        ep=episode(scene,cfg)
        obs=ep.reset()
        rows=[]
        while True:
            probes=[]
            for seed,choice,policy in zip(SEEDS,choices,policies):
                signal=policy.act(obs)
                dynamic=ep.probe(signal)
                static=ep.probe(choice["fixed_signal"])
                probes.append(dict(training_seed=seed,dynamic_signal=signal,fixed_signal=choice["fixed_signal"],
                    dynamic_action_hash=dynamic["action_hash"],fixed_action_hash=static["action_hash"],
                    action_differs=dynamic["action_hash"]!=static["action_hash"],
                    dynamic_coefficients=dynamic["coefficients"],fixed_coefficients=static["coefficients"]))
            rows.append(dict(period=obs.period,observation_hash=obs.hash,probes=probes))
            tr=ep.step((0.,)*10)
            if tr.terminated:
                break
            obs=tr.observation
        trajectory=dict(ledger=ep.final_ledger,metrics=ep.metrics,transitions=ep.records)
        replay(scene,cfg,trajectory)
        write_json(directory/"same-state-22012.json",dict(rows=rows,trajectory=trajectory,state_source="MC",scenario_seed=22012))
        for i,(seed,choice) in enumerate(zip(SEEDS,choices)):
            x=np.asarray(signals[i]);fixed=np.asarray(choice["fixed_signal"])
            summary=dict(condition=name,training_seed=seed,observations=len(x),
                fixed_signal=choice["fixed_signal"],component_std=x.std(axis=0).tolist(),
                mean_abs_signal_deviation=float(np.abs(x-fixed).mean()),max_abs_signal_deviation=float(np.abs(x-fixed).max()),
                same_state_action_difference_count=sum(r["probes"][i]["action_differs"] for r in rows),
                same_state_action_probe_periods=len(rows))
            summaries.append(summary)
            write_json(directory/f"MC-state-signals-{seed}.json",dict(rows=x.tolist(),**summary))
        print("All models same-state diagnosis passed",name,flush=True)
    write_json(target/"diagnostics/summary.json",dict(rows=summaries,signal_reference_states=192,
        integer_action_reference_scene=22012,additional_episodes=3,probe_solves=432))
    write_json(target/"qa/diagnostic-replay.json",dict(status="PASSED",episodes=3,same_state_probes=432))


def analyse(target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"svg.fonttype":"none"})
    rows=read_json(target/"evaluation/summary.json")["rows"]
    means,comparisons=[],[]
    for condition in CONDITIONS:
        name=key(condition)
        values={m:np.asarray([r["profit_USD"] for r in sorted(rows,key=lambda r:r["scenario_seed"]) if r["condition"]==name and r["method"]==m])
                for m in [*(f"DYNAMIC-{s}" for s in SEEDS),*(f"HUB_CONSTANT-{s}" for s in SEEDS),"MC","CAL"]}
        if any(v.shape!=(8,) for v in values.values()):
            raise RuntimeError("incomplete paired crossed evaluation")
        dynamic=np.stack([values[f"DYNAMIC-{s}"] for s in SEEDS])
        static=np.stack([values[f"HUB_CONSTANT-{s}"] for s in SEEDS])
        comparisons.append(dict(condition=name,**paired_statistics(dynamic-static)))
        means.append(dict(condition=name,DYNAMIC=float(dynamic.mean()),HUB_CONSTANT=float(static.mean()),
            MC=float(values["MC"].mean()),CAL=float(values["CAL"].mean()),
            dynamic_seed_means=dynamic.mean(axis=1).tolist(),constant_seed_means=static.mean(axis=1).tolist()))
    write_json(target/"analysis/results.json",dict(means=means,primary_comparisons=comparisons,scope="exploratory mechanism ablation"))
    fields=("condition","method","scenario_id","scenario_seed","profit_USD","delivered","cancelled","unfinished","episode_seconds","policy_seconds")
    with (target/"analysis/paired-test-results.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    write_json(target/"qa/statistical-recalculation.json",dict(status="PASSED",comparisons=3,replications=20000,
        implementations="direct crossed resampling and independent count-weighted einsum",family_size=3))
    files=[]
    def save(fig,stem):
        for ext in ("png","pdf","svg"):
            p=target/"figures"/(stem+"."+ext)
            fig.savefig(p,dpi=600,bbox_inches="tight");files.append(p.relative_to(target).as_posix())
        plt.close(fig)
    titles=("CA / 80 orders / 3 vehicles","West / 40 orders / 8 vehicles","West / 80 orders / 3 vehicles")
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    for ax,r,title in zip(axes,comparisons,titles):
        lo,hi=r["simultaneous_interval"]
        ax.plot([lo,hi],[0,0],lw=3)
        ax.plot(r["mean_USD"],0,"o")
        ax.axvline(0,color="grey",lw=.8)
        ax.set(title=title,xlabel="Dynamic minus hub constant (USD)",yticks=[],ylim=(-.5,.5))
        if max(abs(lo),abs(hi))<1e-5:
            ax.set_xlim(-1,1)
            ax.text(.03,.88,"All paired profits equal within $1e-6",transform=ax.transAxes,fontsize=8)
        else:
            ax.text(.03,.88,"Family-adjusted 98.333% interval",transform=ax.transAxes,fontsize=8)
    save(fig,"01-dynamic-increment")
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    for ax,r,title in zip(axes,means,titles):
        methods=("DYNAMIC","HUB_CONSTANT","MC","CAL")
        ax.bar(range(4),[r[m] for m in methods],color=["C0","C1",".6","C2"])
        ax.set(title=title,xticks=range(4),xticklabels=["Dynamic","Hub fixed","MC","CAL"],ylabel="Mean realized profit (USD)")
    save(fig,"02-profit-means")
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    diag=read_json(target/"diagnostics/summary.json")["rows"]
    for ax,c,title in zip(axes,CONDITIONS,titles):
        group=[r for r in diag if r["condition"]==key(c)]
        for i,r in enumerate(group):
            ax.plot(range(10),r["component_std"],"o-",label=str(r["training_seed"]))
        ax.set(title=title,xlabel="Hub near/far index",ylabel="Signal SD across 192 MC states",xticks=range(0,10,2))
        ax.grid(alpha=.2)
    axes[-1].legend(title="Frozen model seed",fontsize=7)
    save(fig,"03-state-sensitivity")
    write_json(target/"analysis/figures.json",dict(files=files,PNG_DPI=600))
    print("Three primary comparisons independently recalculated; figures written",flush=True)


def audit(target):
    _,manifest=load(target)
    parent=ROOT/manifest["parent_directory"]
    for relative,digest in manifest["parent_artifacts"].items():
        if sha(parent/relative)!=digest:
            raise RuntimeError("parent experiment was modified: "+relative)
    if sha(manifest["source"]["path"])!=manifest["source"]["sha256"]:
        raise RuntimeError("CFS SQLite source changed")
    write_json(target/"qa/parent-readonly.json",dict(status="PASSED",parent_files=len(manifest["parent_artifacts"]),
        source_SHA256=manifest["source"]["sha256"],at=now()))


def report(target):
    result=read_json(target/"analysis/results.json")
    resource=read_json(target/"analysis/execution.json")
    diag=read_json(target/"diagnostics/summary.json")["rows"]
    lines=["# PPO 动态信号与每枢纽固定信号：轻量机制实验","",
        "本次复用上一轮九个短训练模型，检验其输出随时间／公开状态变化的增量价值。原已完成实验和模型均保留。没有重新训练、调整网络或选择新的 checkpoint。","",
        f"新增数值运行 {resource['wall_seconds']/60:.2f} 分钟，进程峰值工作集 {resource['peak_working_set_bytes']/2**30:.3f} GiB，报告生成前新增产物 {resource['output_bytes']/2**30:.3f} GiB。CPU 单进程、Torch/BLAS/MILP 各 1 线程，Windows 低于正常优先级。","",
        "## 冻结对照与数据","",
        "每个模型在 8 个独立验证场景的 MC 轨迹上输出 192 个确定性信号；按每个 hub 的 near/far 分量分别求等权平均，得到 10 维固定信号。固定的是残差信号；原 MC 底座的时间／供给价值及 H1 解码器仍按原规则运行。均值规则预先登记，没有按验证收益优化这十个常数。","",
        "三个工况分别采用此前未评价的测试种子 22012–22019，各 8 个场景。每个场景比较三个动态 PPO、与之配对的三个每枢纽固定信号，以及 MC、上一轮验证选择的 CAL。模型和固定信号在任何新测试利润产生前冻结。测试不参与均值计算或选择。合计 192 次测试日，另有 3 次诊断 MC 日和 432 次同状态探针。","",
        "主比较是每工况的动态减固定，按训练 seed 和场景交叉重抽 20,000 次，三比较使用 Bonferroni 98.333% 区间；由独立计数加权实现复算。普通 95% 区间和每 seed 差值同时保存。MC、CAL 只报告描述性均值。","",
        "## 实现利润（USD／场景）","","| 工况 | 动态 PPO | 每枢纽固定信号 | MC | CAL |","|---|---:|---:|---:|---:|"]
    for r in result["means"]:
        lines.append(f"| {r['condition']} | {r['DYNAMIC']:.2f} | {r['HUB_CONSTANT']:.2f} | {r['MC']:.2f} | {r['CAL']:.2f} |")
    lines += ["","## 动态适应的增量","","| 工况 | 动态−固定 USD | 普通 95% 区间 | 三比较调整区间 | 正增益 seed | 利润相同配对比例 |","|---|---:|---|---|---:|---:|"]
    for r in result["primary_comparisons"]:
        lo,hi=r["marginal_95"];a,b=r["simultaneous_interval"]
        lines.append(f"| {r['condition']} | {r['mean_USD']:.2f} | [{lo:.2f}, {hi:.2f}] | [{a:.2f}, {b:.2f}] | {r['seed_positive_count']}/3 | {r['equivalent_fraction']:.2%} |")
    positive=[r["condition"] for r in result["primary_comparisons"] if r["simultaneous_interval"][0]>TOL]
    equal=[r["condition"] for r in result["primary_comparisons"] if r["equivalent_fraction"]==1.]
    lines += ["",f"调整区间下界严格高于零的工况：{', '.join(positive) if positive else '无'}。全部配对利润在 1e−6 USD 内相同的工况：{', '.join(equal) if equal else '无'}。","",
        "利润相同表示这批场景没有观察到动态输出的额外现金价值；不保证所有路线或动作完全相同，也不证明所有动态策略都无效。若区间包含零，只能判定当前小样本不能给出稳定正增量证据，不能宣称统计等效。","",
        "## 状态变化与整数动作","",
        "信号变化在全部 8 个新测试场景的 MC 状态上测量。整数动作比较使用预先指定的种子 22012：对完全同一 MC 状态、同一模型分别求解动态与固定信号，不用不同策略各自轨迹的后续差异充当同状态证据。","",
        "| 工况 | 模型 seed | 平均绝对信号偏离固定值 | 最大偏离 | 同状态整数动作不同 |","|---|---:|---:|---:|---:|"]
    for r in diag:
        lines.append(f"| {r['condition']} | {r['training_seed']} | {r['mean_abs_signal_deviation']:.6f} | {r['max_abs_signal_deviation']:.6f} | {r['same_state_action_difference_count']}/{r['same_state_action_probe_periods']} |")
    lines += ["","## 解释边界与可核验文件","",
        "固定均值消除了 actor 的全部时间和状态依赖，但不是经过充分优化的最优静态策略。采用 MC 验证状态作为统一参考，也可能与动态策略自身访问的状态分布不同。本次模型仅接受过 32 个训练日，有 3 个 seed、每工况 8 个新测试场景。结论只涉及这些冻结模型和场景。","",
        "全部 195 次日轨迹完整冷却结算并独立回放，检查奖励望远镜求和、原始美元账本、整数动作编译、容量、就绪时间、截止时间及未完成承诺为零。非 OPTIMAL、非有限值或回放异常立即停止并保留失败记录。旧实验全文件 SHA256 和原 CFS SQLite SHA256 在运行后复核。","",
        "- `protocol/preregistration.json`、`frozen-policies.json`：预登记规则、九个模型及固定向量。","- `calibration/*/*.json`：逐验证状态的确定性输出及均值。","- `analysis/results.json`、`paired-test-results.csv`：配对差值、区间及逐场景结果。","- `diagnostics/*/same-state-22012.json`：同状态两种信号、系数与完整整数动作 hash。","- `figures/01`–`03`：动态增量、利润均值、状态敏感性；PDF、SVG、600 dpi PNG。","- `qa/`、`final_manifest.json`：检查证据和最终文件 hash。",""]
    (target/"mechanism_experiment_report.md").write_text("\n".join(lines),encoding="utf-8")


def run_all(target):
    started=time.perf_counter()
    for stage,fn in (("evaluate",evaluate),("diagnose",diagnose),("analyse",analyse),("audit",audit)):
        load(target);journal(target,stage,"STARTED")
        fn(target);journal(target,stage,"COMPLETED")
    resources=resource_guard(target)
    write_json(target/"analysis/execution.json",dict(wall_seconds=time.perf_counter()-started,
        peak_working_set_bytes=resources["peak_working_set_bytes"],output_bytes=sum(p.stat().st_size for p in target.rglob("*") if p.is_file()),
        evaluation_episodes=192,diagnostic_episodes=3,probe_solves=432))
    report(target)
    m=read_json(target/"manifest.json");m.update(status="NUMERICAL_COMPLETE_QA_PENDING",at=now())
    write_json(target/"manifest.json",m)
    print("Mechanism numerical run complete; tests and visual acceptance pending",flush=True)


def recover_analysis(target, failed):
    """Retain a failed export attempt; use its verified outcomes without rerunning."""
    import ast
    from datetime import datetime
    target,failed=Path(target).resolve(),Path(failed).resolve()
    old=read_json(failed/"manifest.json")
    failure=read_json(failed/"qa/stop_condition_failure.json")
    if old["status"]!="STOP_CONDITION_TRIGGERED" or failure["error"]!="TypeError" or failure["message"]!="Object of type int64 is not JSON serializable":
        raise RuntimeError("analysis recovery only accepts the recorded scalar export error")
    for name in ("evaluation-replay","diagnostic-replay"):
        if read_json(failed/"qa"/(name+".json"))["status"]!="PASSED":
            raise RuntimeError("recoverable numerical execution did not finish")
    for relative,digest in old["input_hashes"].items():
        if sha(failed/relative)!=digest:
            raise RuntimeError("failed-attempt frozen input changed: "+relative)
    if old["source_identity"]!=source_identity() or old["study_sources"]!=study_sources() or old["laptop_sources"]!=laptop_sources():
        raise RuntimeError("simulation/model sources changed since recorded execution")
    relative=Path(__file__).relative_to(ROOT).as_posix()
    with zipfile.ZipFile(failed/"protocol/source.snapshot.zip") as z:
        before=z.read(relative).decode("utf-8")
    after=Path(__file__).read_text(encoding="utf-8")
    def blocks(source):
        return {node.name:ast.get_source_segment(source,node) for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
    before_functions,after_functions=blocks(before),blocks(after)
    unchanged=("restore_observation","constant_signal","fresh_guard","config","prepare","evaluate","diagnose")
    if any(before_functions[n]!=after_functions[n] for n in unchanged):
        raise RuntimeError("recovery cannot change calibration, testing or diagnostics")
    expected_stats=before_functions["paired_statistics"].replace("seed_positive_count=sum(v>TOL for v in x.mean(axis=1)),", "seed_positive_count=int(sum(v>TOL for v in x.mean(axis=1))),")
    if after_functions["paired_statistics"]!=expected_stats:
        raise RuntimeError("recovery changed statistical computation beyond scalar serialization")
    target.mkdir(parents=True,exist_ok=False)
    for folder in ("protocol","models","baselines","data","calibration","diagnostics","qa"):
        shutil.copytree(failed/folder,target/folder,ignore=shutil.ignore_patterns("stop_condition_failure.json"))
    shutil.copyfile(failed/"protocol/source.snapshot.zip",target/"protocol/source.v1.snapshot.zip")
    (target/"evaluation").mkdir();(target/"analysis").mkdir();(target/"figures").mkdir()
    shutil.copyfile(failed/"evaluation/summary.json",target/"evaluation/summary.json")
    original_artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(failed.rglob("*")) if p.is_file()}
    write_json(target/"protocol/recovery.json",dict(at=now(),failed_attempt=failed.relative_to(ROOT).as_posix(),
        reason=failure,operation="analysis only; reuse immutable original raw episodes by reference",
        new_training_episodes=0,new_evaluation_episodes=0,new_probe_solves=0,
        unchanged_numerical_functions=list(unchanged),statistics_change="int cast of positive seed count only",
        raw_evaluation_directory=(failed/"evaluation").relative_to(ROOT).as_posix()))
    with zipfile.ZipFile(target/"protocol/source.snapshot.zip","w",zipfile.ZIP_DEFLATED) as archive:
        paths=[*sorted((ROOT/"model_families/single").rglob("*.py")),*sorted((ROOT/"experiment_core").glob("*.py")),
               *sorted(PACKAGE.glob("*.py")),*sorted((PACKAGE/"laptop").glob("*.py")),*sorted(Path(__file__).parent.glob("*.py")),ROOT/"pyproject.toml",ROOT/"uv.lock"]
        for p in paths:
            archive.write(p,p.relative_to(ROOT))
    manifest=dict(old,status="PREPARED",created_at=now(),mechanism_sources=sources(),
        recovered_from=failed.relative_to(ROOT).as_posix(),recovery_artifacts=original_artifacts)
    manifest["parent_checks"]={**old["parent_checks"],**original_artifacts}
    manifest["input_hashes"]={p.relative_to(target).as_posix():sha(p) for folder in ("protocol","models","baselines","data","calibration","diagnostics","evaluation")
                              for p in sorted((target/folder).rglob("*")) if p.is_file()}
    write_json(target/"manifest.json",manifest)
    started=time.perf_counter()
    analyse(target);audit(target)
    resources=resource_guard(target)
    journal_rows=read_json(failed/"qa/run-journal.json")
    original_seconds=(datetime.fromisoformat(failure["at"])-datetime.fromisoformat(journal_rows[0]["at"])).total_seconds()
    original_memory=read_json(failed/"qa/resources-last.json")["peak_working_set_bytes"]
    write_json(target/"analysis/execution.json",dict(wall_seconds=original_seconds+time.perf_counter()-started,
        original_seconds=original_seconds,recovery_seconds=time.perf_counter()-started,
        peak_working_set_bytes=max(original_memory,resources["peak_working_set_bytes"]),
        memory_scope="maximum of original recorded working set and recovery process peak",
        output_bytes=sum(p.stat().st_size for p in target.rglob("*") if p.is_file())+sum(p.stat().st_size for p in failed.rglob("*") if p.is_file()),
        evaluation_episodes=192,diagnostic_episodes=3,probe_solves=432,additional_solver_runs_during_recovery=0))
    report(target)
    report_path=target/"mechanism_experiment_report.md"
    note="\n## 分析恢复记录\n\n原 `../cfs_dynamic_vs_constant_v1` 的 195 次日轨迹和全部探针均已完成并通过回放，随后在保存统计结果时因 NumPy int64 无法直接写入 JSON 而停止。原目录与失败记录保持完整。当前目录仅修正正增益种子计数的整数导出，AST 源码比较确认校准、测试和诊断函数未变化；利润、重抽样、区间计算和模型选择规则未改变。没有重复训练、评价或探针求解。原始测试日记录仍在 v1 的 `evaluation/`，当前逐场景表引用那些已冻结文件。当前报告资源用量合并原执行和恢复分析阶段，磁盘包括保留的两个目录。\n"
    report_path.write_text(report_path.read_text(encoding="utf-8")+note,encoding="utf-8")
    manifest=read_json(target/"manifest.json");manifest.update(status="NUMERICAL_COMPLETE_QA_PENDING",at=now())
    write_json(target/"manifest.json",manifest)
    print("Analysis-only recovery complete; zero new solver runs",flush=True)


def deliver(target):
    _,m=load(target)
    for name in ("data-isolation","evaluation-replay","diagnostic-replay","statistical-recalculation","parent-readonly","final-tests","visual-review"):
        if read_json(target/"qa"/(name+".json"))["status"]!="PASSED":
            raise RuntimeError("final QA missing: "+name)
    m.update(status="COMPLETED",completed_at=now(),
        final_artifact_hashes={p.relative_to(target).as_posix():sha(p) for p in sorted(target.rglob("*"))
                              if p.is_file() and p.name not in ("manifest.json","final_manifest.json")})
    write_json(target/"manifest.json",m);write_json(target/"final_manifest.json",m)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","all","deliver","recover-analysis"))
    parser.add_argument("--attempt",type=Path,required=True)
    parser.add_argument("--parent",type=Path,default=PACKAGE/"runs/cfs_laptop_ppo_v1")
    parser.add_argument("--failed",type=Path)
    args=parser.parse_args();configure_process()
    if args.stage=="prepare":
        prepare(args.attempt,args.parent);return
    if args.stage=="recover-analysis":
        if args.failed is None:
            parser.error("recover-analysis requires --failed")
        recover_analysis(args.attempt,args.failed);return
    target,_=load(args.attempt)
    try:
        (run_all if args.stage=="all" else deliver)(target)
    except BaseException as exc:
        write_json(target/"qa/stop_condition_failure.json",dict(status="STOP_CONDITION_TRIGGERED",at=now(),error=type(exc).__name__,message=str(exc)))
        m=read_json(target/"manifest.json");m.update(status="STOP_CONDITION_TRIGGERED",at=now())
        write_json(target/"manifest.json",m);raise


if __name__=="__main__":
    main()
