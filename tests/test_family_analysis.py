from __future__ import annotations
import ast
import copy
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from experiment_core.analysis import summary, build_panels, extract_runs, create_analysis, export_results
from experiment_core.analysis_contracts import AnalysisSpec
from experiment_core.batching import expand_ranges
from experiment_core.contracts import PlatformError, RangeSweep, DatasetRef, Execution
from experiment_core.datasets import generate, load_dataset
from experiment_core.families import all_families
from experiment_core.plugins import resolve
from experiment_core.runner import execute
from experiment_core.service import plan
from experiment_core.storage import atomic_json, read_json, run_dir, ROOT, digest
from platform_api.app import create_app

@pytest.fixture
def isolated(tmp_path,monkeypatch):
    monkeypatch.setenv("DRL_WORKSPACE",str(tmp_path))
    return tmp_path

def make_runs():
    dataset=generate({"dataset_id":"analysis-tiny","seeds":[331,332,333],"splits":["test"],
                      "horizon":4,"num_cities":4,"num_vehicles":3,"orders_per_step":1,"first_mile":"none"})
    data,scenarios=load_dataset(DatasetRef(dataset_id=dataset["dataset_id"],revision=dataset["revision"]))
    ids=[]
    for i,(controller,value,cost) in enumerate([("myopic","zero",1),("myopic","fluid_dual",1),
                                               ("myopic","zero",2),("myopic","zero",1)]):
        rid="r-test-"+str(i); ids.append(rid); target=run_dir(rid);target.mkdir(parents=True)
        spec=resolve({"name":"analysis "+str(i),"dataset":{"dataset_id":dataset["dataset_id"],"revision":dataset["revision"]},
                      "model":{"id":"single_level_matching","parameters":{"empty_cost":cost,"loaded_cost":1}},
                      "controller":{"id":controller},"value_function":{"id":value}})
        cond=digest({k:spec.model_dump()[k] for k in ("model","controller","value_function","evaluation","solver")})
        atomic_json(target/"spec.resolved.json",spec.model_dump())
        atomic_json(target/"spec.submitted.json",spec.model_dump())
        atomic_json(target/"manifest.json",{"run_id":rid,"batch_id":"b-test","policy_seed":i,"code":{"content_hash":"test-same-source"},"condition_id":cond})
        atomic_json(target/"dataset.snapshot.json",data)
        execute(spec,data,scenarios,target,i,lambda *_a,**_k:None,lambda:False)
        atomic_json(target/"state.json",{"status":"COMPLETED"})
    return ids

def test_owned_import_boundaries():
    for family in ("single","legacy","bhh"):
        for path in (ROOT/"model_families"/family).rglob("*.py"):
            tree=ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ""] if isinstance(node,ast.ImportFrom) else []
                for name in names:
                    assert not name.startswith(("drl_co.","model.","experiment_core.bhh","experiment_core.spatial")),str(path)+":"+name
                    assert not name.startswith(tuple("model_families."+other+"." for other in ("single","legacy","bhh") if other!=family)),str(path)+":"+name

def test_family_catalogue_and_research_context():
    families={f.FAMILY_ID:f.manifest() for f in all_families()}
    assert len({f["accent"] for f in families.values()})==3
    assert all(f["generation_schema"] and f["math"]["equations"] and f["templates"] and f["trace"] for f in families.values())
    single=families["single"]
    assert all(not f["training_required"] for f in single["frameworks"] if f["id"] not in {"learned"})
    with pytest.raises(PlatformError,match="research family"):
        resolve({"name":"cross","dataset":{"dataset_id":"x","revision":"x"},"model":{"id":"single_level_matching"},"controller":{"id":"candidate_sac"}})
    with pytest.raises(PlatformError,match="framework"):
        resolve({"name":"wrong","framework_id":"rollout","dataset":{"dataset_id":"x","revision":"x"},"model":{"id":"single_level_matching"}})

@pytest.mark.parametrize("family,model,path,start,stop,accounting,backend",[
 ("single_level_matching","single_level_matching","model.parameters.empty_cost",.1,.3,"legacy-assignment-v1","gurobi"),
 ("legacy_dispatch","legacy_dispatch","model.parameters.capacity",1,3,"legacy-assignment-v1","gurobi"),
 ("bhh","bhh_steady","model.parameters.tau",1,3,"bhh-cost-v1","cpu"),
])
def test_all_family_numeric_batch(isolated,family,model,path,start,stop,accounting,backend):
    cfg={"dataset_id":"grid","family":family,"seeds":[401,402],"splits":["test"],"horizon":3,"orders_per_step":1}
    d=generate(cfg)
    base={"name":"grid","dataset":{"dataset_id":"grid","revision":d["revision"]},"model":{"id":model},
          "solver":{"backend":backend},"evaluation":{"accounting_version":accounting},"execution":{"policy_seeds":[1,2]}}
    result=plan({"base_spec":base,"ranges":{path:{"start":start,"stop":stop,"step":(stop-start)/2}}})
    assert result["count"]==6
    assert len({r["condition_id"] for r in result["runs"]})==3
    assert all(r["scenario_seeds"]=={"s-401":401,"s-402":402} for r in result["runs"])

def test_ranges_joint_constraints_and_seed_contract(isolated):
    values=expand_ranges({"x":RangeSweep(start=.1,stop=.3,step=.1),"y":RangeSweep(start=1,stop=8,step=2,scale="log")})
    assert values=={"x":[.1,.2,.3],"y":[1,2,4,8]}
    for seeds in ([1,1],[],[-1]):
        with pytest.raises(ValidationError):Execution(training_seeds=seeds)
    d=generate({"dataset_id":"b","family":"bhh","seeds":[3],"splits":["test"],"horizon":2,"orders_per_step":1})
    base={"name":"b","dataset":{"dataset_id":"b","revision":d["revision"]},"model":{"id":"bhh_finite"},"controller":{"id":"rolling_horizon"},"evaluation":{"accounting_version":"bhh-cost-v1"}}
    with pytest.raises(ValidationError):
        plan({"base_spec":base,"sweeps":{"controller.parameters.commit_periods":[30]}})
    with pytest.raises(PlatformError,match="declared"):
        plan({"base_spec":base,"sweeps":{"model.parameters.single_capacity":[3]}})

def test_condition_samples_and_full_trace(isolated):
    ids=make_runs()
    data=extract_runs(ids)
    panels,stats,_=build_panels(data,AnalysisSpec(run_ids=ids,kind="sensitivity",x_parameter="model.parameters.empty_cost"))
    assert len(panels)==1
    assert all(s["n"]==3 for s in stats) # duplicate policy runs never increase n
    for rid in ids:
        trace=read_json(run_dir(rid)/"trace.json")
        metrics=read_json(run_dir(rid)/"metrics.json")["rows"]
        for s,m in zip(trace,metrics):
            assert len(s["steps"])==4
            assert all(all(t["invariants"].values()) for t in s["steps"])
            assert sum(t["measurements"]["period_profit"] for t in s["steps"])==pytest.approx(m["metrics"]["operating_profit"])
            assert sum(t["measurements"]["period_business_cost"] for t in s["steps"])==pytest.approx(m["metrics"]["business_cost"])
            assert sum(t["measurements"]["assigned_delta"] for t in s["steps"])==m["metrics"]["assigned"]
            assert sum(t["measurements"]["delivered_delta"] for t in s["steps"])==m["metrics"]["delivered"]
            assert {"observation","before","after","arrivals","plan"}<=set(s["steps"][0])

def test_statistical_replication_and_ci():
    s=summary([1,2,3])
    assert s["mean"]==2 and s["sd"]==1 and s["n"]==3
    assert s["ci_low"]==pytest.approx(-.4841377,abs=1e-6)
    assert summary([4])["ci_low"] is None
    with pytest.raises(PlatformError):summary([float("nan")])

def test_paired_guard_and_missing_measurements(isolated):
    ids=make_runs()
    data=extract_runs(ids[:2])
    _,stats,pairs=build_panels(data,AnalysisSpec(run_ids=ids[:2],kind="paired"))
    assert len(pairs)==3 and stats[0]["n"]==3
    missing=copy.deepcopy(data);missing["observations"]=missing["observations"][:-1]
    with pytest.raises(PlatformError,match="exact same"):build_panels(missing,AnalysisSpec(run_ids=ids[:2],kind="paired"))
    unavailable=copy.deepcopy(data);unavailable["observations"][0]["metrics"]["operating_profit"]=None
    with pytest.raises(PlatformError,match="unavailable"):build_panels(unavailable,AnalysisSpec(run_ids=ids[:2]))
    atomic_json(run_dir(ids[0])/"state.json",{"status":"FAILED"})
    with pytest.raises(PlatformError,match="silently"):extract_runs(ids[:2])

@pytest.mark.parametrize("kind",["comparison","sensitivity","heatmap","timeseries","paired"])
def test_paper_renderer_and_portable_bundle(isolated,kind):
    ids=make_runs()
    selected=ids[:2] if kind in {"comparison","paired"} else ids
    cfg={"run_ids":selected,"kind":kind,"columns":2,"dpi":300}
    if kind in {"sensitivity","heatmap"}:cfg["x_parameter"]="model.parameters.empty_cost"
    if kind=="heatmap":cfg["y_parameter"]="model.parameters.loaded_cost"
    if kind=="timeseries":cfg["metrics"]=["cumulative_profit","pool_after"]
    result=create_analysis(cfg);target=Path(result["local_path"])
    assert result["plots"]==["figure.pdf","figure.svg","figure.png"]
    assert (target/"figure.pdf").read_bytes().startswith(b"%PDF")
    assert "<svg" in (target/"figure.svg").read_text(encoding="utf-8")
    from PIL import Image
    assert Image.open(target/"figure.png").size[0]==2100
    with zipfile.ZipFile(target/"analysis.zip") as archive:
        assert {"observations.csv","periods.csv","states.jsonl","statistics.csv","plotting.py","plotting-config.json","plot-data.json"}<=set(archive.namelist())
    if kind=="comparison":
        p=subprocess.run([sys.executable,str(target/"plotting.py")],cwd=target,capture_output=True,text=True,timeout=30)
        assert p.returncode==0,p.stderr

def test_raw_export_and_family_scoped_api(isolated):
    ids=make_runs();result=export_results(ids)
    with (Path(result["local_path"])/"periods.csv").open(encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    assert rows and {"period","scenario_seed","policy_seed","condition_id","instance_key","metric","value"}<=set(rows[0])
    from experiment_core.service import Coordinator
    co=Coordinator()
    try:
        app=create_app(coordinator=co,token="test-token")
        with TestClient(app) as client:
            families=client.get("/api/v1/families").json()
            assert len(families)==3
            plugins=client.get("/api/v1/plugins?family=single").json()
            assert all(p["family"]=="single" for p in plugins)
            assert client.get("/api/v1/datasets?family=legacy").json()==[]
            assert client.post("/api/v1/analyses",json={"run_ids":ids[:2]}).status_code==403
            response=client.post("/api/v1/analyses",headers={"x-workspace-token":"test-token"},json={"run_ids":ids[:2],"dpi":300})
            assert response.status_code==200,response.text
            aid=response.json()["analysis_id"]
            assert client.get("/api/v1/analyses/"+aid+"/artifacts/figure.png").status_code==200
            assert client.get("/api/v1/analyses/"+aid+"/artifacts/owner.json").status_code==404
    finally:co.close()
