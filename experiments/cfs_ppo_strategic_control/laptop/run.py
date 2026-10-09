"""Preregistered small experiment; complete traces and strict selection isolation."""
import argparse
import copy
import csv
import os
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from model_families.single.strategic.artifacts import read_json, write_json, source_identity
from ..study_common import ROOT, PACKAGE, CONDITIONS, key, now, sha, payload, study_sources, load as shared_load, data, run_config, prepared_train_data
from ..study_execution import run_record, run_rh4, replay, episode
from ..study_run import summary_record, ensure_environment, journal
from ..study_analysis import bootstrap_crossed

SEEDS = (30001, 30002, 30003)
COUNTS = dict(train=32, validation=8, test=12, diagnostic=2)
UPDATES, BATCH, VALIDATION_EVERY = 8, 4, 4


def sources():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def load(attempt):
    target, manifest = shared_load(attempt)
    if manifest["laptop_sources"] != sources():
        raise RuntimeError("frozen laptop sources changed; retain this attempt and create a new one")
    for path, digest in manifest["inherited_checks"].items():
        if sha(ROOT/path) != digest:
            raise RuntimeError("inherited engineering evidence changed: "+path)
    return target, manifest


def resource_guard(target):
    import psutil
    process = psutil.Process()
    memory = process.memory_info()
    state = dict(at=now(), RSS_bytes=memory.rss,
                 peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss),
                 free_disk_bytes=shutil.disk_usage(target).free)
    write_json(target/"qa/resources-last.json", state)
    if state["free_disk_bytes"] < 5*2**30:
        raise RuntimeError("less than 5 GiB free disk; preserving partial attempt")
    return state


def configure_process():
    ensure_environment()
    import torch
    import psutil
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Change only this experiment process, never machine-wide settings.
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def prepare(attempt, inherited):
    previous, parent = shared_load(inherited)
    for gate in ("engineering", "data-isolation", "CFS-resume"):
        if read_json(previous/"qa"/(gate+".json"))["status"] != "PASSED":
            raise RuntimeError("engineering prerequisite failed: "+gate)
    target = Path(attempt).resolve()
    target.mkdir(parents=True, exist_ok=False)
    for folder in ("protocol", "data", "sealed", "baselines", "training", "validation", "evaluation", "diagnostics", "analysis", "figures", "qa"):
        (target/folder).mkdir()
    inherited_checks = {str((previous/"manifest.json").relative_to(ROOT)): sha(previous/"manifest.json")}
    for gate in ("engineering", "data-isolation", "CFS-resume", "preparation-tests", "statistical-precheck"):
        p = previous/"qa"/(gate+".json")
        if p.exists():
            inherited_checks[str(p.relative_to(ROOT))] = sha(p)
    protocol = dict(schema_version="cfs-laptop-exploration/v1", created_at=now(),
        amendment_authorization="用户：不行，我的电脑是笔记本，改成一个轻量化的实验",
        inference="deterministic", conditions=CONDITIONS, training_seeds=list(SEEDS),
        counts=COUNTS, updates=UPDATES, episodes_per_update=BATCH, validation_every=VALIDATION_EVERY,
        total_training_episodes=288, total_episode_executions=978,
        episode_breakdown=dict(training=288, training_validation=144, CAL_selection=216,
                               selected_validation_recomputation=72, held_out_test=252, diagnostic_MC_trajectories=6),
        subset_rule="First seeds in each previously frozen unused formal split; never select by outcomes",
        splits=dict(train=list(range(20000,20032)), validation=list(range(21000,21008)),
                    test=list(range(22000,22012)), diagnostic=list(range(23000,23002))),
        workers=1, torch_threads=1, solver_threads=1, process_priority="below normal on Windows",
        CAL_grid=[-.5,0.,.5], CAL_tie="zero, smaller squared norm, lexicographic near/far",
        bootstrap_seed=31000, bootstrap_replications=20000, simultaneous_family_size=6,
        scientific_scope="Exploratory only: 3 training seeds and 12 synthetic CFS scenarios per condition; no confirmatory superiority claim",
        training_change="Budget only; same architecture, coefficients, scales, physics, optimizer, epochs and rewards",
        model_selection="deterministic validation means at updates 4 and 8, first tie retained; all choices frozen before tests",
        diagnostic_model_seed=30001, diagnostic_state_source="MC zero-residual trajectory",
        accounting="full cooldown realized USD ledger, gamma=1; M0 uses same H1 decoder with zero value",
        noninferiority="W80 descriptive one-sided bound versus -0.01*max(abs(MC validation mean),1000); exploratory",
        hard_wall_minutes=45, inherited_attempt=str(previous.relative_to(ROOT)))
    write_json(target/"protocol/preregistration.json", protocol)
    identities = []
    from model_families.single.learning.ppo.runner import scenario_content, prepare as validate
    for c in CONDITIONS:
        name = key(c)
        for folder in ("data", "sealed"):
            (target/folder/name).mkdir()
        for split, count in COUNTS.items():
            folder = "sealed" if split == "test" else "data"
            source = previous/folder/name/(split+".json")
            scenarios = read_json(source)["scenarios"][:count]
            if [s["seed"] for s in scenarios] != protocol["splits"][split]:
                raise RuntimeError("frozen subset seed mismatch")
            write_json(target/folder/name/(split+".json"), payload(scenarios))
            source_provenance = previous/folder/name/(split+"-provenance.json")
            write_json(target/folder/name/(split+"-provenance.json"), read_json(source_provenance)[:count])
            for s in scenarios:
                identities.append(dict(condition=name, split=split, id=s["id"], seed=s["seed"], content_hash=scenario_content(s)))
        for suffix in (".json", "-scales.json"):
            shutil.copyfile(previous/"baselines"/(name+suffix), target/"baselines"/(name+suffix))
        cfg = read_json(previous/"protocol"/(name+".json"))
        cfg["ppo"].update(updates=UPDATES, episodes_per_update=BATCH, validation_every=VALIDATION_EVERY)
        if cfg["solver"]["threads"] != 1:
            raise RuntimeError("solver must use one thread")
        write_json(target/"protocol"/(name+".json"), cfg)
        validate(run_config(target,c,SEEDS[0]), prepared_train_data(target,c))
        subset = [r for r in identities if r["condition"] == name]
        if len({r["content_hash"] for r in subset}) != len(subset):
            raise RuntimeError("subset contains duplicate physical scenarios")
    write_json(target/"data/identities.json", identities)
    write_json(target/"qa/data-isolation.json", dict(status="PASSED", selected_scenarios=len(identities),
        inherited_no_history_collision=True, disjoint_ids_seeds_content=True, selected_by_outcomes=False))
    shutil.copyfile(previous/"protocol/task.md", target/"protocol/original-task.md")
    pilot = read_json(previous/"preflight/pilot-summary.json")
    estimated_seconds, estimated_bytes = 0., 0.
    for sample in pilot["training"]:
        estimated_seconds += sample["wall_seconds"]*3
        mean_bytes = sample["total_bytes"]/48
        estimated_bytes += mean_bytes*326 + sample["checkpoint_bytes"]*6
        baseline_times = [r["episode_seconds"] for r in pilot["baselines"] if r["condition"] == sample["condition"]]
        estimated_seconds += np.mean(baseline_times)*182
    budget = dict(status="USER_REQUESTED_LIGHTWEIGHT_REVISION", projected_minutes=float(estimated_seconds/60),
        conservative_minutes=float(estimated_seconds*2/60), projected_output_GiB=float(estimated_bytes/2**30),
        conservative_output_GiB=float(estimated_bytes*2/2**30), full_protocol_training_fraction=288/15360,
        workers=1, solver_threads=1, torch_threads=1, requires_GPU=False,
        excludes_original_preparation_disk=True, inherited_pilot_SHA256=sha(previous/"preflight/pilot-summary.json"))
    write_json(target/"protocol/budget.json", budget)
    manifest = dict(schema_version="cfs-laptop-study/v1", status="PREPARED", created_at=now(),
        source=parent["source"], source_identity=source_identity(), study_sources=study_sources(),
        laptop_sources=sources(), inherited_checks=inherited_checks,
        history_hashes=parent["history_hashes"], historical_config_hashes=parent["historical_config_hashes"],
        input_hashes={})
    with zipfile.ZipFile(target/"protocol/source.snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        paths = [*sorted((ROOT/"model_families/single").rglob("*.py")), *sorted((ROOT/"experiment_core").glob("*.py")),
                 *sorted(PACKAGE.glob("*.py")), *sorted(Path(__file__).parent.glob("*.py")), ROOT/"pyproject.toml", ROOT/"uv.lock"]
        for p in paths:
            archive.write(p, p.relative_to(ROOT))
    manifest["input_hashes"] = {p.relative_to(target).as_posix():sha(p) for folder in ("protocol","data","sealed","baselines")
                                for p in sorted((target/folder).rglob("*")) if p.is_file()}
    write_json(target/"manifest.json", manifest)
    print("Laptop protocol frozen:", budget, flush=True)
    return target


def train(target):
    from model_families.single.learning.ppo.runner import train_run
    records = []
    for c in CONDITIONS:
        name = key(c)
        scenarios = {s["id"]:s for split in ("train","validation") for s in data(target,c,split)["scenarios"]}
        for seed in SEEDS:
            load(target)
            cfg = run_config(target,c,seed)
            output = target/"training"/name/str(seed)
            journal(target,"train","STARTED",condition=name,seed=seed)
            started = time.perf_counter()
            progress = train_run(cfg, prepared_train_data(target,c), output)
            if progress["episodes"] != UPDATES*BATCH or progress["updates"] != UPDATES:
                raise RuntimeError("training budget mismatch")
            for p in sorted(output.glob("training-episode-*.json"))+sorted(output.glob("validation-update-*-scenario-*.json")):
                r = read_json(p)
                replay(scenarios[r["scenario_id"]], cfg, r)
            records.append(dict(condition=name,seed=seed,wall_seconds=time.perf_counter()-started,
                                episodes=progress["episodes"], replay_episodes=48, **resource_guard(target)))
            journal(target,"train","COMPLETED",condition=name,seed=seed)
            print("Training + full replay passed", name,seed,progress,flush=True)
    write_json(target/"qa/training-replay.json", dict(status="PASSED", episodes=432, rows=records))


def select(target):
    from model_families.single.learning.ppo.policy import FrozenPolicy
    choices, audits = [], []
    for c in CONDITIONS:
        load(target)
        name,cfg = key(c),run_config(target,c,SEEDS[0])
        output = target/"validation"/name
        output.mkdir(parents=True,exist_ok=False)
        scenes = data(target,c,"validation")["scenarios"]
        grid=[]
        for near in (-.5,0.,.5):
            for far in (-.5,0.,.5):
                values=[]
                signal=(near,far)*len(cfg["action"]["hubs"])
                for s in scenes:
                    r=run_record(s,cfg,lambda o:signal,"CAL")
                    write_json(output/f"CAL-{near}-{far}-{s['seed']}.json",r)
                    replay(s,cfg,r)
                    values.append(r["ledger"]["realized_profit"])
                grid.append(dict(near=near,far=far,mean_profit_USD=float(np.mean(values)),profits=values))
        best=sorted(grid,key=lambda r:(-r["mean_profit_USD"],(r["near"],r["far"])!=(0.,0.),r["near"]**2+r["far"]**2,r["near"],r["far"]))[0]
        mc=next(r for r in grid if r["near"]==r["far"]==0.)
        write_json(output/"CAL-selection.json",dict(selected=best,candidates=grid))
        write_json(output/"MC-summary.json",mc)
        choices.append(dict(condition=name,method="CAL",near=best["near"],far=best["far"]))
        for seed in SEEDS:
            folder=target/"training"/name/str(seed)
            scores=[read_json(p) for p in sorted(folder.glob("validation-[0-9][0-9][0-9][0-9].json"))]
            if len(scores)!=2 or [r["update"] for r in scores]!=[4,8]:
                raise RuntimeError("validation checkpoint schedule mismatch")
            best_score=max(scores,key=lambda r:r["mean_profit_USD"])
            decision=read_json(folder/"best.json")
            if decision["update"]!=best_score["update"]:
                raise RuntimeError("selected checkpoint mismatch")
            checkpoint=folder/"best-inference"
            policy=FrozenPolicy(checkpoint)
            values=[]
            for s in scenes:
                r=run_record(s,cfg,policy.act,"PPO-validation")
                write_json(output/f"PPO-{seed}-{s['seed']}.json",r)
                replay(s,cfg,r)
                values.append(r["ledger"]["realized_profit"])
            if not np.allclose(values,[r["profit_USD"] for r in best_score["scenarios"]],atol=1e-6,rtol=1e-10):
                raise RuntimeError("independent validation profit mismatch")
            choices.append(dict(condition=name,seed=seed,checkpoint=str(checkpoint.relative_to(target)),
                files={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()},update=decision["update"],
                validation_mean_USD=float(np.mean(values))))
            audits.append(dict(condition=name,seed=seed,status="PASSED"))
        print("All model choices validated",name,"CAL",best["near"],best["far"],flush=True)
        resource_guard(target)
    delta=.01*max(abs(read_json(target/"validation/western_multicluster-n80-k3/MC-summary.json")["mean_profit_USD"]),1000.)
    write_json(target/"protocol/selected-models.json",dict(selected_at=now(),test_results_seen=False,selected=choices,W80_delta_USD=delta))
    manifest=read_json(target/"manifest.json")
    manifest["input_hashes"]["protocol/selected-models.json"]=sha(target/"protocol/selected-models.json")
    write_json(target/"manifest.json",manifest)
    write_json(target/"qa/validation-selection.json",dict(status="PASSED",rows=audits,calibration_episodes=216,recomputed_episodes=72))


def check_tests(policy,scenes):
    from model_families.single.learning.ppo.runner import physical_signature, scenario_content
    used=[e for role in ("train","validation") for e in policy.environment["data_splits"][role]]
    for s in scenes:
        if s["split"]!="test" or physical_signature(s)!=policy.environment["physics"]:
            raise RuntimeError("test physics/split incompatibility")
        if any(s["id"]==e["id"] or s["seed"]==e["seed"] or scenario_content(s)==e["content_hash"] for e in used):
            raise RuntimeError("held-out test overlaps training or selection")


def evaluate(target):
    from model_families.single.learning.ppo.policy import FrozenPolicy
    choices=read_json(target/"protocol/selected-models.json")["selected"]
    for c in CONDITIONS:
        load(target)
        name,cfg=key(c),run_config(target,c,SEEDS[0])
        scenes=read_json(target/"sealed"/name/"test.json")["scenarios"]
        output=target/"evaluation"/name
        output.mkdir(parents=True,exist_ok=False)
        rows=[]
        cal=next(r for r in choices if r["condition"]==name and r.get("method")=="CAL")
        cal_signal=(cal["near"],cal["far"])*len(cfg["action"]["hubs"])
        for method in [*(f"PPO-{s}" for s in SEEDS),"MC","CAL","M0","RH4"]:
            folder=output/method
            folder.mkdir()
            local=copy.deepcopy(cfg)
            policy=lambda o:(0.,)*10
            if method.startswith("PPO-"):
                seed=int(method[4:])
                choice=next(r for r in choices if r["condition"]==name and r.get("seed")==seed)
                cp=target/choice["checkpoint"]
                if any(sha(cp/p)!=h for p,h in choice["files"].items()):
                    raise RuntimeError("frozen checkpoint changed")
                frozen=FrozenPolicy(cp)
                check_tests(frozen,scenes)
                policy=frozen.act
            elif method=="CAL":
                policy=lambda o:cal_signal
            elif method=="M0":
                local["baseline"]=dict(method="zero")
            for s in scenes:
                r=run_rh4(s,local) if method=="RH4" else run_record(s,local,policy,method)
                p=folder/f"scenario-{s['seed']}.json"
                write_json(p,r)
                if method!="RH4":
                    replay(s,local,r)
                rows.append(summary_record(r,name,method,p))
            print("Test + replay passed",name,method,len(scenes),flush=True)
        write_json(output/"summary.json",dict(status="COMPLETED",scenarios=len(scenes),methods=7,rows=rows))
        resource_guard(target)
    write_json(target/"qa/test-replay.json",dict(status="PASSED",episodes=252,all_optimal=True,unfinished_committed=0))


def diagnose(target):
    from model_families.single.learning.ppo.policy import FrozenPolicy
    choices=read_json(target/"protocol/selected-models.json")["selected"]
    summaries=[]
    for c in CONDITIONS:
        name,cfg=key(c),run_config(target,c,SEEDS[0])
        out=target/"diagnostics"/name
        out.mkdir(parents=True,exist_ok=False)
        choice=next(r for r in choices if r["condition"]==name and r.get("seed")==30001)
        policy=FrozenPolicy(target/choice["checkpoint"])
        cal=next(r for r in choices if r["condition"]==name and r.get("method")=="CAL")
        cal_signal=(cal["near"],cal["far"])*len(cfg["action"]["hubs"])
        for s in data(target,c,"diagnostic")["scenarios"]:
            ep=episode(s,cfg)
            obs=ep.reset()
            rows=[]
            while True:
                signal=policy.act(obs)
                mc,p,ca=ep.probe((0.,)*10),ep.probe(signal),ep.probe(cal_signal)
                rows.append(dict(period=obs.period,signal=list(signal),observation_hash=obs.hash,
                    PPO_coefficients=p["coefficients"],MC_coefficients=mc["coefficients"],
                    PPO_MC_disagreement=p["action_hash"]!=mc["action_hash"],
                    PPO_CAL_disagreement=p["action_hash"]!=ca["action_hash"]))
                tr=ep.step((0.,)*10)
                if tr.terminated:
                    break
                obs=tr.observation
            record=dict(same_state=rows,trajectory=dict(ledger=ep.final_ledger,metrics=ep.metrics,transitions=ep.records))
            replay(s,cfg,record["trajectory"])
            write_json(out/f"scenario-{s['seed']}.json",record)
            values=np.asarray([r["signal"] for r in rows])
            lookup={}
            for r in rows:
                for coef in r["PPO_coefficients"]:
                    lookup[coef["lookup"]]=lookup.get(coef["lookup"],0)+1
            summaries.append(dict(condition=name,scenario_seed=s["seed"],training_seed=30001,
                PPO_MC_disagreement=float(np.mean([r["PPO_MC_disagreement"] for r in rows])),
                PPO_CAL_disagreement=float(np.mean([r["PPO_CAL_disagreement"] for r in rows])),
                nonzero_fraction=float(np.mean(np.abs(values)>1e-9)),saturation_fraction=float(np.mean(np.abs(values)>.99)),
                lookup_counts=lookup))
        print("Same-state diagnosis complete",name,flush=True)
    write_json(target/"diagnostics/summary.json",dict(rows=summaries,state_source="MC",selection="seed 30001 predefined; diagnostic scenes 23000 and 23001",support_count="unknown"))


def independent_bootstrap(x,replications=20000,seed=31000):
    x=np.asarray(x,dtype=float)
    rng=np.random.default_rng(seed)
    samples=[]
    for start in range(0,replications,1000):
        count=min(1000,replications-start)
        a=rng.integers(0,len(x),(count,len(x)))
        b=rng.integers(0,x.shape[1],(count,x.shape[1]))
        wa=np.stack([np.bincount(r,minlength=len(x)) for r in a])
        wb=np.stack([np.bincount(r,minlength=x.shape[1]) for r in b])
        samples.extend(np.einsum("bi,ij,bj->b",wa,x,wb)/(len(x)*x.shape[1]))
    return np.asarray(samples)


def analyse(target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"svg.fonttype":"none","pdf.fonttype":42})
    stats,means,csv_rows=[],[],[]
    figure_paths=[]
    all_rows=[]
    arrays={}
    for c in CONDITIONS:
        name=key(c)
        rows=read_json(target/"evaluation"/name/"summary.json")["rows"]
        all_rows.extend(rows)
        for r in rows:
            csv_rows.append({k:r[k] for k in ("condition","method","scenario_id","scenario_seed","profit_USD","delivered","cancelled","unfinished","episode_seconds","policy_seconds")})
        values={m:np.asarray([r["profit_USD"] for r in sorted(rows,key=lambda r:r["scenario_seed"]) if r["method"]==m]) for m in [*(f"PPO-{s}" for s in SEEDS),"MC","CAL","M0","RH4"]}
        if any(v.shape!=(12,) for v in values.values()):
            raise RuntimeError("missing crossed test observations")
        ppo=np.stack([values[f"PPO-{s}"] for s in SEEDS])
        arrays[name]=(values,ppo)
        means.append(dict(condition=name,**{m:float(v.mean()) for m,v in values.items()},PPO=float(ppo.mean())))
        for baseline in ("MC","CAL"):
            diff=ppo-values[baseline]
            result=bootstrap_crossed(diff)
            other=independent_bootstrap(diff)
            alpha=.05/6
            for field,q in (("marginal_95",[.025,.975]),("bonferroni_99_1667",[alpha/2,1-alpha/2]),("one_sided_95_lower",.05)):
                if not np.allclose(result[field],np.quantile(other,q),atol=1e-8,rtol=1e-12):
                    raise RuntimeError("independent bootstrap mismatch")
            stats.append(dict(condition=name,comparison="PPO-"+baseline,**result))
    with (target/"analysis/test_results.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    write_json(target/"analysis/profit-comparison.json",dict(means=means,comparisons=stats,scope="exploratory",training_seeds=3,test_scenarios=12))
    write_json(target/"qa/statistical-recalculation.json",dict(status="PASSED",implementation="independent count-weighted einsum versus direct crossed indexing",comparisons=6,replications=20000))

    def save(fig,name):
        for ext in ("pdf","svg","png"):
            p=target/"figures"/(name+"."+ext)
            fig.savefig(p,dpi=600,bbox_inches="tight")
            figure_paths.append(str(p.relative_to(target)))
        plt.close(fig)

    titles=("CA / 80 orders / 3 vehicles","West / 40 orders / 8 vehicles","West / 80 orders / 3 vehicles")
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    for ax,c,title in zip(axes,CONDITIONS,titles):
        for seed in SEEDS:
            folder=target/"training"/key(c)/str(seed)
            updates=[read_json(folder/f"update-{u:04d}.json") for u in range(1,9)]
            ax.plot(range(1,9),[u["training_mean_USD"] for u in updates],alpha=.5,label=f"train {seed}")
            ax.plot([4,8],[updates[u-1]["validation_mean_USD"] for u in (4,8)],"o--",label=f"valid {seed}")
        ax.set(title=title,xlabel="PPO update",ylabel="Realized profit (USD)")
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=6,ncol=2)
    save(fig,"01-learning-curves")
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    for ax,c,title in zip(axes,CONDITIONS,titles):
        comparison=[r for r in stats if r["condition"]==key(c)]
        for i,r in enumerate(comparison):
            lo,hi=r["bonferroni_99_1667"]
            ax.plot([lo,hi],[i,i],color=f"C{i}")
            ax.plot(r["mean_USD"],i,"o",color=f"C{i}")
        ax.axvline(0,color="grey",lw=.8)
        ax.set(title=title,xlabel="Paired gain (USD), adjusted interval",yticks=[0,1],yticklabels=["PPO - MC","PPO - CAL"],ylim=(-.5,1.5))
    save(fig,"02-paired-effects")
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    for ax,c,title in zip(axes,CONDITIONS,titles):
        values,ppo=arrays[key(c)]
        for method,x in [("PPO mean across seeds",ppo.mean(axis=0)),*((m,values[m]) for m in ("MC","CAL","M0","RH4"))]:
            ax.step(np.sort(x),np.arange(1,13)/12,where="post",label=method)
        ax.set(title=title,xlabel="Realized profit (USD)",ylabel="Empirical CDF")
    axes[-1].legend(fontsize=7)
    save(fig,"03-profit-distributions")
    fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for col,(c,title) in enumerate(zip(CONDITIONS,titles)):
        record=read_json(target/"diagnostics"/key(c)/"scenario-23000.json")
        matrix=np.asarray([r["signal"] for r in record["same_state"]]).T
        image=axes[0,col].imshow(matrix,aspect="auto",vmin=-1,vmax=1,cmap="coolwarm",extent=(-.5,23.5,9.5,-.5))
        axes[0,col].set(title=title,xlabel="Decision period",ylabel="Hub near/far signal index")
        rows=record["same_state"]
        for field,label in (("PPO_MC_disagreement","PPO vs MC"),("PPO_CAL_disagreement","PPO vs CAL")):
            axes[1,col].step([r["period"] for r in rows],[int(r[field]) for r in rows],where="mid",label=label)
        axes[1,col].set(xlabel="MC trajectory period",ylabel="Integer action differs",yticks=[0,1],ylim=(-.1,1.1))
    axes[1,-1].legend(fontsize=7)
    fig.colorbar(image,ax=list(axes[0]),label="Frozen actor signal")
    save(fig,"04-signal-and-action-diagnostics")
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    methods=("PPO","MC","CAL","M0","RH4")
    for ax,c,title in zip(axes,CONDITIONS,titles):
        rows=[r for r in all_rows if r["condition"]==key(c)]
        for i,m in enumerate(methods):
            sample=[r for r in rows if (r["method"].startswith("PPO-") if m=="PPO" else r["method"]==m)]
            ax.bar(i-.18,np.mean([r["delivered"] for r in sample]),width=.35,color="C0",label="Delivered" if i==0 else None)
            ax.bar(i+.18,np.mean([r["cancelled"] for r in sample]),width=.35,color="C1",label="Cancelled" if i==0 else None)
        ax.set(title=title,xticks=range(5),xticklabels=methods,ylabel="Mean orders per scenario")
    axes[-1].legend(fontsize=7)
    save(fig,"05-service-outcomes")
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    for ax,c,title in zip(axes,CONDITIONS,titles):
        rows=[r for r in all_rows if r["condition"]==key(c)]
        total,solver,policy=[],[],[]
        for m in methods:
            sample=[r for r in rows if (r["method"].startswith("PPO-") if m=="PPO" else r["method"]==m)]
            total.append(np.mean([r["episode_seconds"] for r in sample]))
            solver.append(np.mean([sum(r["solver_times"]) for r in sample]))
            policy.append(np.mean([r["policy_seconds"] for r in sample]) if m=="PPO" else 0.)
        ax.bar(range(5),total,color=".8",label="Total episode")
        ax.bar(range(5),solver,color="C0",label="MILP reported runtime")
        ax.bar(range(5),policy,bottom=solver,color="C2",label="Actor inference (PPO)")
        ax.set(title=title,xticks=range(5),xticklabels=methods,ylabel="Mean wall time (s)")
    axes[-1].legend(fontsize=6)
    save(fig,"06-computation-cost")
    write_json(target/"analysis/figures.json",dict(files=figure_paths,PNG_DPI=600))
    print("Statistics independently verified; six figure families written",flush=True)


def audit(target):
    _,manifest=load(target)
    for path,digest in {**manifest["history_hashes"],**manifest["historical_config_hashes"]}.items():
        if sha(ROOT/path)!=digest:
            raise RuntimeError("protected historical file changed: "+path)
    if sha(manifest["source"]["path"])!=manifest["source"]["sha256"]:
        raise RuntimeError("CFS SQLite changed")
    write_json(target/"qa/historical-readonly.json",dict(status="PASSED",source_SHA256=manifest["source"]["sha256"],at=now()))


def report(target):
    result=read_json(target/"analysis/profit-comparison.json")
    execution=read_json(target/"analysis/execution.json")
    diagnostic=read_json(target/"diagnostics/summary.json")["rows"]
    choices=read_json(target/"protocol/selected-models.json")
    lines=["# 第八轮 PPO 笔记本轻量数值实验", "", "本次按用户要求改为轻量探索。原完整任务书预算未执行；原准备、工程预检及 pilot 保留于 `../cfs_first_ppo_v1`。本目录是单独冻结的新实验，结果不作为原任务书大样本验证完成的证据。", "",
        f"新增实验执行 {execution['wall_seconds']/60:.2f} 分钟；执行进程记录的最高工作集 {execution['peak_working_set_bytes']/2**30:.3f} GiB；报告生成前新增产物 {execution['output_bytes']/2**30:.3f} GiB。单进程、CPU、Torch/BLAS/MILP 各 1 线程，Windows 低于正常优先级。未计入此前工程预检与 pilot 成本。", "",
        "## 冻结设置", "", "三个原工况，各 3 个训练种子（30001–30003）；每个模型 8 次更新 × 4 个完整日 = 32 个训练 episode，共 288 个（原 15,360 个的 1.875%）。训练池 32、验证 8、测试 12、诊断 2 个场景／工况，分别使用此前未参与 pilot 的固定种子前缀。", "",
        "验证仅在第 4、8 次更新进行，取确定性验证平均利润最高的 checkpoint，并列保留第一次。CAL 枚举共享 near/far 的九组参数，仅按验证利润选择。所有工况的模型和 CAL 参数在打开测试结果前一次冻结。测试对同 12 个场景比较三个 PPO seed 与 MC、CAL、M0、RH4；不选最好测试 seed。", "",
        "网络 hidden=64、PPO 4 epochs、固定尺度、学习率、gamma=1、reward scale=1000、原物理规则、非 bucket bank1 MC 底座全部保持。M0 是同 H=1 价值 MILP 的零底座；与原旧单期入口存在同值决策差异，不能宣称两者所有动作完全相同。利润来自完整承诺冷却后的权威实现账本，战略系数与 tie-break 不作现金奖励。", "",
        "## 测试实现利润（USD／场景）", "", "| 工况 | PPO（三种子均值） | MC | CAL | M0 | RH4 |", "|---|---:|---:|---:|---:|---:|"]
    for r in result["means"]:
        lines.append(f"| {r['condition']} | {r['PPO']:.2f} | {r['MC']:.2f} | {r['CAL']:.2f} | {r['M0']:.2f} | {r['RH4']:.2f} |")
    lines += ["", "## 配对差值与不确定性", "", "20,000 次交叉 bootstrap 同时重抽训练 seed 和场景；六个 PPO−MC/PPO−CAL 比较使用 Bonferroni 99.1667% 区间，另保存普通 95% 区间。独立计数加权实现重算并核对所有区间。只有 3 个 seed、12 个场景，尾部分位数和泛化判断均不稳；这些区间是探索性描述，不能据此宣布强化学习优越或方向失败。", "", "| 工况 | 比较 | 平均差 USD | 普通 95% 区间 | 六比较调整区间 | 正增益 seed |", "|---|---|---:|---|---|---:|"]
    for r in result["comparisons"]:
        lo,hi=r["marginal_95"]; blo,bhi=r["bonferroni_99_1667"]
        positives=sum(v>0 for v in r["seed_mean_USD"])
        lines.append(f"| {r['condition']} | {r['comparison']} | {r['mean_USD']:.2f} | [{lo:.2f}, {hi:.2f}] | [{blo:.2f}, {bhi:.2f}] | {positives}/3 |")
    w80=next(r for r in result["comparisons"] if r["condition"]=="western_multicluster-n80-k3" and r["comparison"]=="PPO-MC")
    lines += ["", f"W80 描述性非劣检视：验证预定容忍额 δ={choices['W80_delta_USD']:.2f} USD，单侧 95% 下界 {w80['one_sided_95_lower']:.2f} USD；是否超过 −δ：{w80['one_sided_95_lower']>-choices['W80_delta_USD']}。这也只是轻量探索。", "", "## 控制诊断", "", "固定使用训练 seed 30001，在两个独立诊断场景的 MC 轨迹上对同一观测分别求解 PPO、MC、CAL，检查系数变化和整数动作变化，不用各自轨迹间的差异替代同状态证据。", "", "| 工况 | PPO−MC 整数动作不同率 | 非零信号比例 | 信号饱和比例 |", "|---|---:|---:|---:|"]
    for c in CONDITIONS:
        rows=[r for r in diagnostic if r["condition"]==key(c)]
        lines.append(f"| {key(c)} | {np.mean([r['PPO_MC_disagreement'] for r in rows]):.2%} | {np.mean([r['nonzero_fraction'] for r in rows]):.2%} | {np.mean([r['saturation_fraction'] for r in rows]):.2%} |")
    lines += ["", "底座查询类型保存在 `diagnostics/summary.json`。历史 MC 表不含逐格支持计数，支持量标记 unknown，不能用网格、插值或钳制次数代替支持量。", "", "## 审计与文件", "", "训练及其验证 432 日、CAL 选择 216 日、选定模型验证重算 72 日、测试 252 日、诊断 MC 轨迹 6 日，共 978 次 episode 执行均保留原始记录并检查回放。诊断另外求解同状态探针；回放本身无需再次求解。各日完整结算，任何非 OPTIMAL、非有限数、账本或可行性异常立即停止并保留失败记录。", "",
        "工程零信号、隐藏未来信息边界、原 CFS 恢复一致性检查复用同一源码／物理／尺度的已通过证据，其来源和 SHA256 列于 manifest；本次数据隔离、验证独立重算、全轨迹回放、统计独立复算另行执行。最终测试和图形验收状态见 `qa/final-tests.json`、`qa/visual-review.json`。历史源码、参数、MC 表与 4.9 GB 原 CFS SQLite 的 hash 复核通过。", "",
        "- `protocol/preregistration.json`：减量协议与用户指示；`protocol/budget.json`：基于 pilot 的预算。", "- `protocol/selected-models.json`：预先冻结的九个 PPO 包与三个 CAL 参数。", "- `analysis/test_results.csv`、`analysis/profit-comparison.json`：逐场景数据及完整差值区间。", "- `training/*/*/best-inference`：仅推理包；同目录 `update-*` 是可恢复包。", "- `figures/01`–`06`：训练曲线、配对差值、利润分布、同状态信号诊断、服务量与计算成本，均提供 PDF、SVG、600 dpi PNG。", "",
        "本实验训练很短，且合成场景并非真实日期重放。若收益小或不稳定，首先只能判定当前轻量配置没有给出稳定改进证据；不能推断更充分训练、更多场景或其他观测建模必然无效。后续是否增加预算应由本次现象决定。", ""]
    (target/"laptop_experiment_report.md").write_text("\n".join(lines),encoding="utf-8")


def run_all(target):
    started=time.perf_counter()
    for stage,fn in (("train",train),("select",select),("evaluate",evaluate),("diagnose",diagnose),("audit",audit),("analyse",analyse)):
        load(target)
        if time.perf_counter()-started>45*60:
            raise TimeoutError("laptop experiment exceeded 45 minute wall budget")
        journal(target,stage,"STARTED")
        fn(target)
        journal(target,stage,"COMPLETED")
    resources=resource_guard(target)
    write_json(target/"analysis/execution.json",dict(wall_seconds=time.perf_counter()-started,
        peak_working_set_bytes=resources["peak_working_set_bytes"],
        output_bytes=sum(p.stat().st_size for p in target.rglob("*") if p.is_file()),episodes=978))
    report(target)
    manifest=read_json(target/"manifest.json")
    manifest.update(status="NUMERICAL_COMPLETE_QA_PENDING",numerical_completed_at=now())
    write_json(target/"manifest.json",manifest)
    print("Laptop numerical run complete; final tests and visual review pending",flush=True)


def finish(target):
    _,manifest=load(target)
    for name in ("data-isolation","training-replay","validation-selection","test-replay","statistical-recalculation","historical-readonly","final-tests","visual-review"):
        if read_json(target/"qa"/(name+".json"))["status"]!="PASSED":
            raise RuntimeError("delivery gate incomplete: "+name)
    manifest.update(status="COMPLETED",completed_at=now(),
        final_artifact_hashes={p.relative_to(target).as_posix():sha(p) for p in sorted(target.rglob("*"))
                              if p.is_file() and p.name not in ("manifest.json","final_manifest.json")})
    write_json(target/"manifest.json",manifest)
    write_json(target/"final_manifest.json",manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","all","deliver"))
    parser.add_argument("--attempt",type=Path,required=True)
    parser.add_argument("--inherited",type=Path,default=PACKAGE/"runs/cfs_first_ppo_v1")
    args=parser.parse_args()
    configure_process()
    if args.stage=="prepare":
        prepare(args.attempt,args.inherited)
        return
    target,_=load(args.attempt)
    try:
        (run_all if args.stage=="all" else finish)(target)
    except BaseException as exc:
        write_json(target/"qa/stop_condition_failure.json",dict(status="STOP_CONDITION_TRIGGERED",at=now(),error=type(exc).__name__,message=str(exc)))
        manifest=read_json(target/"manifest.json")
        manifest.update(status="STOP_CONDITION_TRIGGERED",failed_at=now())
        write_json(target/"manifest.json",manifest)
        raise


if __name__=="__main__":
    main()
