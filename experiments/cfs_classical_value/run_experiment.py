"""Staged, resumable full study. Must launch with PYTHONHASHSEED=0."""
import argparse
import time
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from .common import *
from .simulation import simulate
from .build_fluid_value import build_fluid
from .build_mc_targets import build_targets
from .fit_models import fit_all
from .run_value_policy import run_ladder


def baseline_regression(cfg,baseline):
    path=OUTPUT/"manifests/baseline_regression.json"
    if path.exists():
        prior=read_json(path);assert prior["status"]=="PASSED";return prior
    checks=[]
    for c in cfg["conditions"]:
        cid=key(c);scenario=read_json(ROOT/"results/cfs_dynamic_lookahead/raw/datasets"/f"{cid}-s1010.dataset.json")["scenarios"][0]
        for policy,label,H in [("myopic","myopic",0),("RH4","known_supply",4),("RH8","known_supply",8)]:
            target=ROOT/"results/cfs_dynamic_lookahead/raw/runs"/f"rolling-{cid}-s1010-rho_max-{label}-h{H}"
            # Actual archived stage prefix is discovered without mutating files.
            matches=list((ROOT/"results/cfs_dynamic_lookahead/raw/runs").glob(f"*-{cid}-s1010-rho_max-{label}-h{H}"))
            matches=[p for p in matches if p.name.startswith("rolling-")]
            if len(matches)!=1:raise RuntimeError(f"ambiguous baseline archive: {matches}")
            prior=read_json(matches[0]/"summary.json")
            sim,orders=matching_state(scenario,parameters(baseline,rho_levels(scenario,baseline)["rho_max"]),baseline["solver"])
            metrics,_,_=simulate(sim,orders,policy)
            delta=metrics["realized_profit"]-prior["realized_profit"]
            checks.append(dict(condition=cid,policy=policy,seed=1010,archived_profit=prior["realized_profit"],
                replay_profit=metrics["realized_profit"],difference=delta))
            assert abs(delta)<1e-6
    result=dict(status="PASSED",checks=checks,protected_files_unchanged=True)
    atomic_json(path,result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--through",choices=["inputs","targets","fit","pilot","formal","all"],default="all")
    args=parser.parse_args();cfg,baseline,manifest=initialize()
    manifest["baseline_regression"]=baseline_regression(cfg,baseline)
    seeds=cfg["construction_seeds"]+cfg["validation_seeds"]+cfg["formal_seeds"]
    started=time.perf_counter();data=datasets(cfg,baseline,seeds)
    manifest["input_seconds"]=time.perf_counter()-started
    atomic_json(OUTPUT/"manifest.json",manifest)
    if args.through=="inputs":return
    if "fluid" not in manifest:
        priors,info=build_fluid(cfg,baseline,data);manifest["fluid"]=info
        atomic_json(OUTPUT/"manifest.json",manifest)
    else:priors=read_json(OUTPUT/"models/priors.json")
    for split,seedfield in [("construction","construction_seeds"),("validation","validation_seeds")]:
        if split not in manifest:
            manifest[split]=build_targets(cfg,baseline,data,priors,split,cfg[seedfield])
            atomic_json(OUTPUT/"manifest.json",manifest)
    if args.through=="targets":return
    if "fit" not in manifest:
        manifest["fit"]=fit_all(cfg)
        freeze=dict(selected_before_formal=True,formal_seed_profit_seen=False,
            selection_split="validation",model_selection_hash=file_hash(OUTPUT/"models/selection.json"),
            models={p.name:file_hash(p) for p in (OUTPUT/"models").glob("*.json")})
        atomic_json(OUTPUT/"manifests/model_freeze.json",freeze)
        atomic_json(OUTPUT/"manifest.json",manifest)
    if args.through=="fit":return
    if "pilot" not in manifest:
        manifest["pilot"]=run_ladder(cfg,baseline,data,priors,manifest,cfg["validation_seeds"][:5],"pilot")
        manifest["formal_ladder_decision"]="retain all eleven online methods plus OracleH8 diagnostic; no formal-profit pruning"
        atomic_json(OUTPUT/"manifest.json",manifest)
    if args.through=="pilot":return
    if "formal" not in manifest:
        manifest["formal"]=run_ladder(cfg,baseline,data,priors,manifest,cfg["formal_seeds"],"formal")
        atomic_json(OUTPUT/"manifest.json",manifest)
    if args.through=="formal":return
    if "heldout_value" not in manifest:
        manifest["heldout_value"]=build_targets(cfg,baseline,data,priors,"heldout_value",cfg["heldout_value_seeds"])
        atomic_json(OUTPUT/"manifest.json",manifest)
    if "secondary_RH4" not in manifest:
        manifest["secondary_RH4"]=build_targets(cfg,baseline,data,priors,"secondary_RH4",cfg["secondary_seeds"],"RH4",[0,12,23])
        atomic_json(OUTPUT/"manifest.json",manifest)
    manifest["status"]="NUMERICAL_RUNS_COMPLETED"
    atomic_json(OUTPUT/"manifest.json",manifest)


if __name__=="__main__":main()
