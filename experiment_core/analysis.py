"""Common statistical pipeline. Research extraction is delegated to an owned adapter."""
from __future__ import annotations
import csv
import json
import math
import threading
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from .analysis_contracts import AnalysisSpec
from .contracts import PlatformError
from .families import get_family
from .storage import atomic_json, digest, identifier, read_json, run_dir, workspace

_RENDER_GUARD = threading.Lock()
# Renderer interface is replaceable without changing experiment or observation storage.
def _matplotlib(config, panels, target):
    from .paper_renderer import render
    with _RENDER_GUARD:
        return render(config, panels, target)
RENDERERS = {"matplotlib":_matplotlib}

def summary(values, confidence=.95):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.all(np.isfinite(a)):
        raise PlatformError("NONFINITE_ANALYSIS","all observations must be finite")
    n=len(a); mean=float(a.mean()); sd=float(a.std(ddof=1)) if n>1 else None
    half=float(student_t.ppf((1+confidence)/2,n-1)*sd/math.sqrt(n)) if n>1 else None
    return {"n":n,"mean":mean,"sd":sd,"minimum":float(a.min()),"maximum":float(a.max()),
            "ci_low":mean-half if half is not None else None,"ci_high":mean+half if half is not None else None,
            "ci_reason":None if n>1 else "at least two independent scenarios required",
            "method":"scenario-level Student-t; policy seeds averaged within frozen instance"}

def extract_runs(run_ids):
    parts=[]
    for rid in run_ids:
        target=run_dir(rid)
        state=read_json(target/"state.json")
        if state["status"]!="COMPLETED":
            raise PlatformError("FAILED_RUN_ANALYSIS",f"{rid} is {state['status']}; incomplete runs cannot be silently omitted")
        spec=read_json(target/"spec.resolved.json")
        f=get_family(spec["model"]["id"])
        if not hasattr(f,"analysis_adapter"):
            raise PlatformError("ANALYSIS_NOT_AVAILABLE",f"{f.FAMILY_ID} has not opted into the analysis protocol yet")
        parts.append((f.FAMILY_ID, f.analysis_adapter().extract(target,spec,read_json(target/"manifest.json"))))
    if len({p[0] for p in parts}) != 1:
        raise PlatformError("CROSS_FAMILY_ANALYSIS","research families cannot be pooled")
    if sum(len(p[1]["periods"]) for p in parts)>200000:
        raise PlatformError("ANALYSIS_SIZE","analysis exceeds 200000 period records")
    return {"family":parts[0][0],"observations":[r for _,p in parts for r in p["observations"]],
            "periods":[r for _,p in parts for r in p["periods"]],"states":[r for _,p in parts for r in p["states"]],
            "definitions":parts[0][1]["definitions"],"trace_schema":parts[0][1]["trace_schema"],
            "runs":[r for _,p in parts for r in p["runs"]],
            "datasets":{key:value for _,p in parts for key,value in p["datasets"].items()}}

def scalar(row, metric):
    value=row["metrics"].get(metric)
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
        raise PlatformError("METRIC_UNAVAILABLE",f"{metric} unavailable for {row['run_id']}; choose compatible frameworks or trace metrics")
    return float(value)

def reduced_context(row, config):
    physical=json.loads(json.dumps(row["physical"]))
    for axis in (config.x_parameter,config.y_parameter):
        if not axis: continue
        if not axis.startswith("model.parameters."):
            # Policy axes are allowed, but must not be removed from the physical model.
            continue
        keys=axis.split(".")[2:]; ptr=physical["model"]["parameters"]
        for key in keys[:-1]: ptr=ptr[int(key)] if isinstance(ptr,list) else ptr[key]
        if isinstance(ptr,list): ptr[int(keys[-1])]="<sweep>"
        else: ptr.pop(keys[-1],None)
    return digest({"dataset":row["dataset_hash"],"physical":physical})

def build_panels(data, config):
    observations=data["periods"] if config.kind=="timeseries" else data["observations"]
    if not observations:
        raise PlatformError("TRACE_UNAVAILABLE","this selection has no per-period trace")
    if config.kind!="timeseries":
        registered={d["key"]:d for d in data["definitions"]}
        for metric in config.metrics:
            if metric not in registered:
                raise PlatformError("METRIC_UNAVAILABLE",f"unregistered family metric: {metric}")
            if config.kind=="paired" and registered[metric]["sense"] not in {"maximize","minimize"}:
                raise PlatformError("METRIC_NOT_COMPARABLE","paired comparison requires an operational metric")
    for axis in (config.x_parameter,config.y_parameter):
        if axis:
            for row in observations:
                value=row["parameters"].get(axis)
                if not isinstance(value,(int,float)) or isinstance(value,bool):
                    raise PlatformError("NONNUMERIC_AXIS",f"{axis} must be numeric in every selected condition")
    strata=defaultdict(list)
    for row in observations:
        strata[reduced_context(row,config)].append(row)
    panels=[]; stats=[]; pairs=[]
    for context,rows in sorted(strata.items()):
        dataset=rows[0]["dataset_id"]
        for metric in config.metrics:
            # Never average parameter configurations. Multiple policy seeds within a
            # physical instance estimate that instance's expected policy result.
            groups=defaultdict(lambda:defaultdict(list))
            representatives={}
            for row in rows:
                x=row["period"] if config.kind=="timeseries" else row["parameters"].get(config.x_parameter,0)
                y=row["parameters"].get(config.y_parameter,0)
                key=(row["condition_id"],x,y)
                groups[key][row["instance_key"]].append(scalar(row,metric))
                representatives[key]=row
            reduced={key:{instance:float(np.mean(v)) for instance,v in values.items()} for key,values in groups.items()}
            label=lambda row: config.labels.get(row["framework"],row["framework"])
            if config.kind=="paired":
                if len(reduced)!=2:
                    raise PlatformError("PAIRED_CONDITIONS","select exactly two conditions within each physical/data stratum")
                a,b=sorted(reduced,key=lambda k:(label(representatives[k]),k))
                if set(reduced[a])!=set(reduced[b]):
                    raise PlatformError("UNPAIRED_SCENARIOS","paired conditions must contain the exact same frozen instances")
                aa,bb=representatives[a],representatives[b]
                values=[reduced[a][s]-reduced[b][s] for s in sorted(reduced[a])]
                s=summary(values,config.confidence)
                paired_label=label(aa)+" − "+label(bb)
                for instance in sorted(reduced[a]):
                    pairs.append({"context":context,"metric":metric,"instance_key":instance,
                                  "condition_a":a[0],"condition_b":b[0],"a":reduced[a][instance],
                                  "b":reduced[b][instance],"difference":reduced[a][instance]-reduced[b][instance]})
                series=[{"label":paired_label,"points":[{"x":0,**s}],"samples":values}]
                stats.append({"context":context,"metric":metric,"series":paired_label,**s})
            else:
                series_by_key={}
                # Exclude axis coordinates from series identity; keep all other
                # policy and physical parameters, including checkpoints.
                for key,values in sorted(reduced.items()):
                    row=representatives[key]; remaining=dict(row["parameters"])
                    for axis in (config.x_parameter,config.y_parameter): remaining.pop(axis,None)
                    identity=digest({"framework":row["framework"],"remaining":remaining})
                    title=label(row)
                    if identity not in series_by_key:
                        series_by_key[identity]={"label":title,"points":[],"samples":[],"identity":identity}
                    s=summary(list(values.values()),config.confidence)
                    series_by_key[identity]["points"].append({"x":key[1],"y":key[2],**s})
                    series_by_key[identity]["samples"].extend(values.values())
                    stats.append({"context":context,"metric":metric,"condition_id":key[0],"series":title,"x":key[1],"y":key[2],**s})
                series=list(series_by_key.values())
                # Distinct controller configurations receive explicit suffixes.
                names=[s["label"] for s in series]
                for s in series:
                    if names.count(s["label"])>1: s["label"]+=" ["+s["identity"][:6]+"]"
                if config.kind=="comparison" and any(len(s["points"])!=1 for s in series):
                    raise PlatformError("CONDITION_POOLING","use sensitivity plots for multiple parameter conditions")
            definition=next((d for d in data["definitions"] if d["key"]==metric),{})
            panel={"title":dataset+" · "+context[:6],"metric":metric,"context":context,
                   "xlabel":"Period t" if config.kind=="timeseries" else config.x_parameter or "",
                   "y_parameter":config.y_parameter,"ylabel":definition.get("label",metric)+" ("+definition.get("unit","trace units")+")",
                   "series":series}
            if config.kind=="heatmap":
                # Each algorithm gets its own panel; unobserved grid cells stay blank.
                for s in series: panels.append({**panel,"title":panel["title"]+" · "+s["label"],"series":[s]})
            else: panels.append(panel)
    if len(panels)>24:
        raise PlatformError("PANEL_LIMIT","more than 24 panels; filter experiments or metrics")
    return panels,stats,pairs

def write_csv(path, rows):
    keys=sorted({key for row in rows for key in row})
    with Path(path).open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def flat_rows(records):
    out=[]
    for row in records:
        meta={k:v for k,v in row.items() if k not in {"metrics","physical","parameters"}}
        for metric,value in row["metrics"].items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                out.append({**meta,**row["parameters"],"metric":metric,"value":value})
    return out

def save_data(target,data):
    atomic_json(target/"data-contract.json",{"schema_version":"analysis-data/v1","trace_schema":data["trace_schema"],
        "sample_unit":"frozen physical scenario; parameter conditions are not replications",
        "numeric_table_layout":"long-form metric/value; native JSON records preserve nulls and metric reasons"})
    atomic_json(target/"runs.json",data["runs"])
    atomic_json(target/"datasets.json",data["datasets"])
    atomic_json(target/"metric-definitions.json",data["definitions"])
    write_csv(target/"observations.csv",flat_rows(data["observations"]))
    write_csv(target/"periods.csv",flat_rows(data["periods"]))
    atomic_json(target/"observations.json",data["observations"])
    atomic_json(target/"periods.json",data["periods"])
    # Nested order/resource states, decisions, full accounting and diagnostics are
    # lossless JSONL, separate from tidy numeric tables.
    with (target/"states.jsonl").open("w",encoding="utf-8") as stream:
        for state in data["states"]: stream.write(json.dumps(state,ensure_ascii=False,allow_nan=False)+"\n")

def bundle(target):
    with zipfile.ZipFile(target/"analysis.zip","w",zipfile.ZIP_DEFLATED) as z:
        for file in sorted(target.iterdir()):
            if file.is_file() and file.name!="analysis.zip": z.write(file,file.name)

def create_analysis(config):
    config=AnalysisSpec.model_validate(config)
    data=extract_runs(config.run_ids)
    panels,statistics,pairs=build_panels(data,config)
    aid="a-"+uuid.uuid4().hex[:20]
    target=workspace()/"analyses"/aid; target.mkdir(parents=True)
    save_data(target,data)
    atomic_json(target/"plotting-config.json",config.model_dump())
    atomic_json(target/"plot-data.json",panels)
    write_csv(target/"statistics.csv",statistics); write_csv(target/"paired-differences.csv",pairs)
    from . import paper_renderer
    (target/"plotting.py").write_bytes(Path(paper_renderer.__file__).read_bytes())
    from importlib.metadata import version
    (target/"requirements.txt").write_text("matplotlib=="+version("matplotlib")+"\nnumpy=="+version("numpy")+"\n",encoding="utf-8")
    try:
        plots=RENDERERS[config.renderer](config.model_dump(),panels,target)
    except Exception as exc:
        atomic_json(target/"state.json",{"status":"FAILED","message":str(exc)})
        raise PlatformError("RENDER_FAILED",str(exc)) from exc
    manifest={"schema_version":"analysis-artifact/v1","analysis_id":aid,"family":data["family"],"status":"COMPLETED",
              "config":config.model_dump(),"local_path":str(target),"plots":plots,"panels":len(panels),
              "sample_unit":"frozen physical scenario; policy seeds averaged within instance",
              "inference":"pointwise Student-t CIs, not simultaneous bands; independence assumed across generated scenarios",
              "warnings":["No CI when n < 2. Different data/physical conditions are faceted, not pooled.",
                          "Augmented objective depends on the value function; it is diagnostic, not policy profit.",
                          "No imputation or smoothing of missing heatmap cells or time periods."],
              "source_hashes":sorted({r["source_hash"] for r in data["observations"]}),
              "statistics":statistics}
    atomic_json(target/"manifest.json",manifest); bundle(target)
    return get_analysis(aid)

def get_analysis(aid):
    target=workspace()/"analyses"/identifier(aid)
    result=read_json(target/"manifest.json")
    result["artifacts"]=[p.name for p in target.iterdir() if p.is_file() and not p.name.endswith(".partial")]
    return result

def list_analyses(family=None):
    root=workspace()/"analyses"
    result=[]
    for path in sorted(root.glob("a-*/manifest.json"),key=lambda p:p.stat().st_mtime,reverse=True):
        item=read_json(path)
        if not family or item["family"]==family: result.append(item)
    return result

def export_results(run_ids):
    # Raw export requires no chart choice and preserves all recorded measurements.
    data=extract_runs(run_ids)
    eid="a-"+uuid.uuid4().hex[:20]
    target=workspace()/"analyses"/eid; target.mkdir(parents=True)
    save_data(target,data)
    atomic_json(target/"manifest.json",{"schema_version":"analysis-artifact/v1","analysis_id":eid,
                "family":data["family"],"status":"COMPLETED","local_path":str(target),"plots":[],
                "config":{"run_ids":run_ids,"kind":"raw-export"},"sample_unit":"frozen scenario"})
    bundle(target)
    return get_analysis(eid)
