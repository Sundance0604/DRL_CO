"""Full-cooldown executed policy ladder; each run retains failure and detailed t."""
import time
from pathlib import Path
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from model_families.single.value.providers import load_provider
from .common import OUTPUT,key,parameters,rho_levels,map_jobs,write_csv,file_hash
from .simulation import simulate

LADDER=[
    ("P0-Myopic","myopic",None),("P1-Myopic-Fluid","myopic+V","fluid"),
    ("P2-Myopic-Table","myopic+V","table"),("P3-Myopic-Linear","myopic+V","linear"),
    ("P4-Myopic-Chebyshev","myopic+V","chebyshev"),("P5-Myopic-Spline","myopic+V","spline"),
    ("P6-RH4","RH4",None),("P7-RH4-Fluid","RH4+V","fluid"),
    ("P8-RH4-Chebyshev","RH4+V","chebyshev"),("P9-RH4-Spline","RH4+V","spline"),
    ("P10-RH8","RH8",None),("D-OracleH8","OracleH8",None)]


def policy_job(job):
    target=OUTPUT/"raw/runs"/job["id"]
    identity={k:job[k] for k in ["data_hash","model_hash","code_hash"]}
    if (target/"summary.json").exists():
        prior=read_json(target/"summary.json")
        assert all(prior[k]==v for k,v in identity.items()),"run identity mismatch"
        if prior["status"]=="COMPLETED":return prior
    target.mkdir(parents=True,exist_ok=True)
    row=dict(id=job["id"],stage=job["stage"],condition=job["condition"],seed=job["scenario"]["seed"],
        policy=job["label"],information="oracle diagnostic only" if job["policy"]=="OracleH8" else "online",
        scenario_hash=digest(job["scenario"]),**identity,policy_seed=0,training_seed=None,
        accounting_version="single-realized-v1")
    start=time.perf_counter()
    try:
        sim,orders=matching_state(job["scenario"],job["parameters"],job["solver"])
        provider=load_provider(Path(job["model"])) if job["model"] else None
        metrics,steps,_=simulate(sim,orders,job["policy"],provider,job["prior"],trace=True,timeout=job["timeout"])
        row.update({k:v for k,v in metrics.items() if isinstance(v,(int,float))})
        row.update(status="COMPLETED",service_rate=metrics["delivered"]/job["orders"])
        atomic_json(target/"metrics.json",metrics)
        atomic_json(target/"trace.json",steps)
        atomic_json(target/"decision-ledger.json",[dict(period=s["period"],accounting=s["plan"]["accounting"]) for s in steps if not s.get("drain")])
        atomic_json(target/"realized-ledger.json",dict(final=metrics,
            by_t=[dict(period=s["period"],ledger=s["realized_ledger"]) for s in steps]))
        row["trace_hash"]=file_hash(target/"trace.json")
    except Exception as exc:
        row.update(status="FAILED",error=str(exc),daily_wall_seconds=time.perf_counter()-start)
    atomic_json(target/"spec.json",job)
    atomic_json(target/"summary.json",row)
    return row


def run_ladder(cfg,baseline,data,priors,manifest,seeds,stage):
    jobs=[]
    model_hashes=read_json(OUTPUT/"manifests/model_freeze.json")["models"]
    for c in cfg["conditions"]:
        cid=key(c)
        for seed in seeds:
            scenario=data[c["region"],c["orders"],c["fleet"],seed]["scenarios"][0]
            for label,policy,method in LADDER:
                model=OUTPUT/"models"/f"{cid}-{method}.json" if method else None
                if model and file_hash(model)!=model_hashes[model.name]:raise RuntimeError("frozen model changed")
                jobs.append(dict(id=f"{stage}-{cid}-s{seed}-{label}",stage=stage,condition=cid,scenario=scenario,
                    label=label,policy=policy,model=str(model) if model else None,orders=c["orders"],
                    prior=priors[cid],parameters=parameters(baseline,rho_levels(scenario,baseline)["rho_max"]),
                    solver=baseline["solver"],data_hash=digest(scenario),model_hash=file_hash(model) if model else None,
                    code_hash=manifest["code_hash"],timeout=cfg["run_timeout_seconds"]))
    rows,wall=map_jobs(policy_job,jobs,stage,cfg["max_parallel"])
    rows.sort(key=lambda r:r["id"]);write_csv(OUTPUT/"processed"/(stage+"_policies.csv"),rows)
    atomic_json(OUTPUT/"processed"/(stage+"_policies.json"),rows)
    stage_info=dict(count=len(rows),failures=sum(r["status"]!="COMPLETED" for r in rows),wall_seconds=wall,seeds=seeds)
    if stage_info["failures"]:raise RuntimeError("failed samples retained; inference blocked")
    return stage_info
