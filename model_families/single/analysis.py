"""Single-owned conversion from research records to the platform analysis protocol."""
from experiment_core.contracts import PlatformError
from experiment_core.storage import digest, read_json
from .metrics import DEFINITIONS

def extract(target, spec, manifest):
    if spec["controller"]["id"] == "train_value":
        raise PlatformError("TRAINING_NOT_EVALUATION", "fit runs are not held-out policy observations")
    if (target / "dataset.snapshot.json").exists():
        data = read_json(target / "dataset.snapshot.json")
    else:
        from experiment_core.datasets import load_dataset
        from experiment_core.contracts import DatasetRef
        data, _ = load_dataset(DatasetRef.model_validate(spec["dataset"]))
    scenarios = {s["id"]: s for s in data["scenarios"]}
    traces = {s["scenario_id"]: s for s in read_json(target / "trace.json")}
    result, periods, states = [], [], []
    params = {}
    for kind in ("model", "controller", "value_function"):
        def flatten(value, prefix):
            if isinstance(value, dict):
                for k, v in value.items(): flatten(v, prefix+"."+k)
            elif isinstance(value, list):
                for k, v in enumerate(value): flatten(v, prefix+"."+str(k))
            else: params[prefix] = value
        flatten(spec[kind]["parameters"], kind+".parameters")
    physical = {"model":spec["model"], "evaluation":spec["evaluation"],
                "solver":spec["solver"], "source":manifest["code"]["content_hash"]}
    framework = spec.get("framework_id") or spec["controller"]["id"]+"."+spec["value_function"]["id"]
    for record in read_json(target / "metrics.json")["rows"]:
        sid = record["scenario_id"]
        scenario = scenarios[sid]
        # Same frozen physical events must not inflate n even under renamed IDs.
        payload = {k:v for k,v in scenario.items() if k not in {"id","split","seed"}}
        common = {"run_id":manifest["run_id"], "family":"single", "framework":framework,
                  "condition_id":manifest.get("condition_id") or digest({"physical":physical,"controller":spec["controller"],"value":spec["value_function"]}),
                  "dataset_id":spec["dataset"]["dataset_id"], "dataset_hash":spec["dataset"]["revision"],
                  "scenario_id":sid, "instance_key":digest(payload), "scenario_seed":scenario.get("seed"),
                  "policy_seed":record.get("policy_seed"), "training_seed":record.get("training_seed"),
                  "physical_key":digest(physical), "source_hash":physical["source"], "parameters":params,
                  "physical":physical}
        result.append({**common, "metrics":record["metrics"]})
        for step in traces.get(sid, {}).get("steps", []):
            periods.append({**common,"period":step["period"],"metrics":step.get("measurements",{})})
            states.append({"run_id":manifest["run_id"],"scenario_id":sid,"instance_key":common["instance_key"],
                           "period":step["period"],"trace":step})
    return {"observations":result,"periods":periods,"states":states,
            "runs":[{"run_id":manifest["run_id"],"spec":spec,"manifest":manifest}],
            "datasets":{spec["dataset"]["revision"]:data},
            "definitions":DEFINITIONS, "trace_schema":"single-trace-step/v2"}
