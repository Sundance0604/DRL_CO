"""Identity, source preservation, frozen inputs and study-only execution."""
import copy
import json
import os
import platform
import time
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from experiment_core.storage import atomic_json, read_json, digest
from experiments.cfs_rho_endofday.run_experiment import ROOT, file_hash, code_hashes, write_csv
from experiments.cfs_dynamic_lookahead.common import freeze_data, rho_levels

OUTPUT = ROOT/"results/cfs_classical_value"
CONFIG = ROOT/"experiments/cfs_classical_value/configs/main.json"
PROTECTED = ["model_families/single/network.py","model_families/single/cfs_adapter.py",
             "model_families/single/accounting.py","model_families/single/engine/mt_prototype.py",
             "model_families/single/rolling/route_milp.py","model_families/single/rolling/controller.py",
             "model_families/single/runner.py"]


def key(c):
    return f'{c["region"]}-n{c["orders"]}-k{c["fleet"]}'


def nodes(T, n=7):
    raw = (1+np.cos((2*np.arange(1,n+1)-1)*np.pi/(2*n)))/2*(T-1)
    return sorted(set(np.rint(raw).astype(int).tolist()))


def parameters(baseline, rho):
    return {k:baseline[k] for k in ["speed","capacity","loaded_cost","empty_cost",
                "waiting_cost","expiry_cost","max_delay"]} | dict(
                    cross_hub=True,reposition=False,pickup_radius=rho,pickup_rule="network_radius",tie_break=1.)


def source_hashes():
    h=code_hashes()
    for directory in ["experiments/cfs_classical_value","model_families/single/value"]:
        for p in (ROOT/directory).rglob("*.py"):
            if p.name=="analysis.py":
                continue
            h[p.relative_to(ROOT).as_posix()]=file_hash(p)
    for name in ["experiments/cfs_dynamic_lookahead/common.py","tests/test_classical_value.py"]:
        h[name]=file_hash(ROOT/name)
    return h


def initialize():
    cfg=read_json(CONFIG); baseline=read_json(ROOT/cfg["baseline_config"])
    splits=[set(cfg[k]) for k in ["construction_seeds","validation_seeds","formal_seeds"]]
    assert not any(splits[i]&splits[j] for i in range(3) for j in range(i))
    assert set(cfg["heldout_value_seeds"])<=splits[2]
    assert os.environ.get("PYTHONHASHSEED")=="0","launch with PYTHONHASHSEED=0"
    OUTPUT.mkdir(parents=True,exist_ok=True)
    old=read_json(ROOT/"results/cfs_dynamic_lookahead/manifest.json")
    for name in PROTECTED:
        assert file_hash(ROOT/name)==old["source_code_sha256"][name],f"frozen baseline changed: {name}"
    hashes=source_hashes()
    identity=dict(config_hash=digest(cfg),code_hash=digest(hashes))
    path=OUTPUT/"manifest.json"
    if path.exists():
        manifest=read_json(path)
        assert all(manifest[k]==v for k,v in identity.items()),"source/config changed: new output required"
        return cfg,baseline,manifest
    source=Path(baseline["data_source"])
    print("Verifying read-only CFS SHA256...",flush=True)
    assert file_hash(source)==old["source"]["sha256"]
    for d in ["targets","models","raw","processed","manifests"]:
        (OUTPUT/d).mkdir(exist_ok=True)
    snapshot=OUTPUT/"raw/source.snapshot.zip"
    with zipfile.ZipFile(snapshot,"w",zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(ROOT/name,name)
        z.write(CONFIG,"classical-config.json")
    manifest=dict(**identity,config=cfg,baseline_config=baseline,status="INITIALIZED",
        source=old["source"],source_code_sha256=hashes,python=platform.python_version(),
        hash_seed=0,policy_seed=0,training_seed=None,solver=baseline["solver"],
        value_provider_version="single-marginal-release/v1",source_snapshot_sha256=file_hash(snapshot),
        baseline_reports={name:file_hash(ROOT/name) for name in [
            "reports/cfs_rho_endofday/simulation_report.md",
            "reports/cfs_dynamic_lookahead/dynamic_lookahead_report.md"]},
        baseline_code_digest=old["source_code_digest"],
        baseline_manifest_sha256=file_hash(ROOT/"results/cfs_dynamic_lookahead/manifest.json"),
        protected_hashes={name:file_hash(ROOT/name) for name in PROTECTED},
        actual_projected_chebyshev_nodes=nodes(baseline["horizon"],cfg["chebyshev_node_count"]),
        resource_target="Remove currently FREE idle vehicles at queried hub only; retain all committed vehicles; J(q free vehicles)-J(q-1) on identical future path. Reindex vehicle IDs; accounting constants cancel.",
        features=dict(remaining="(T-release_time)/T",other_idle="idle vehicles outside queried hub",
            incoming="known committed final idle releases to hub strictly after current observation and by release_time+4",
            backlog="currently observed hub-origin total load / capacity",
            expected="construction-only mean hub bookings per period * remaining periods",
            q="ordered free resource slot; integer vehicles"),
        boundary="At T no further new control permitted; added FREE vehicle changes no commitment, so marginal=0. Committed execution costs/fare remain in full-cooldown ledger and cancel between paired worlds.",
        scope="No neural/RL, tensor bases or oracle-training. Wide rho only. MC empirical future resampling within each split, not survey population conditional inference.")
    atomic_json(path,manifest)
    return cfg,baseline,manifest


def datasets(cfg,baseline,seeds):
    existing={}
    for c in cfg["conditions"]:
        for seed in seeds:
            path=OUTPUT/"raw/datasets"/f'{key(c)}-s{seed}.dataset.json'
            if path.exists():
                existing[c["region"],c["orders"],c["fleet"],seed]=read_json(path)
    if len(existing)==len(cfg["conditions"])*len(seeds):
        return existing
    return freeze_data(cfg,baseline,OUTPUT,cfg["conditions"],seeds)


def map_jobs(fn,jobs,label,workers=3):
    start=time.perf_counter(); rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fn,j) for j in jobs]
        for f in as_completed(futures):
            rows.append(f.result())
            if len(rows)%10==0 or len(rows)==len(jobs):
                print(f"{label}: {len(rows)}/{len(jobs)}, elapsed {time.perf_counter()-start:.1f}s",flush=True)
    return rows,time.perf_counter()-start
