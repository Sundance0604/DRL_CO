"""Small, real, reproducible acceptance experiment; all writes in local workspace."""
import json
import time
from .storage import atomic_json, workspace, read_json, run_dir
from .datasets import generate
from .service import Coordinator, TERMINAL
from .analysis import create_analysis, export_results

def main():
    co=Coordinator()
    try:
        d=generate({"dataset_id":"phase1-acceptance","seeds":[21001,21002,21003,21004],"splits":["test"],
                    "horizon":6,"num_cities":4,"num_vehicles":3,"orders_per_step":2,"first_mile":"none"})
        config={"base_spec":{"name":"Phase 1 computational experiment","dataset":{"dataset_id":d["dataset_id"],"revision":d["revision"]},
                            "model":{"id":"single_level_matching","parameters":{"loaded_cost":1,"empty_cost":1}}},
                "variants":[{"controller":{"id":"myopic"},"value_function":{"id":"zero"}},
                            {"controller":{"id":"myopic"},"value_function":{"id":"fluid_dual"}},
                            {"controller":{"id":"rollout","parameters":{"samples":2}},"value_function":{"id":"zero"}}],
                "ranges":{"model.parameters.empty_cost":{"start":1,"stop":2,"step":1}}}
        batch=co.submit(config)
        until=time.monotonic()+240
        while time.monotonic()<until:
            co.tick(); batch=co.batch(batch["batch_id"])
            if all(r["status"] in TERMINAL for r in batch["runs"]):break
            time.sleep(.1)
        if any(r["status"]!="COMPLETED" for r in batch["runs"]):raise RuntimeError(json.dumps(batch))
        ids=[r["run_id"] for r in batch["runs"]]
        baseline=[r for r in ids if co.run(r)["spec"]["model"]["parameters"]["empty_cost"]==1]
        figures={}
        for kind,chosen,extra in [
            ("comparison",baseline,{"metrics":["operating_profit","delivered_order_rate"]}),
            ("sensitivity",ids,{"x_parameter":"model.parameters.empty_cost"}),
            ("timeseries",baseline,{"metrics":["cumulative_profit","pool_after","idle_vehicles","assigned_delta"]}),
            ("paired",baseline[:2],{}),
            ("heatmap",ids,{"x_parameter":"model.parameters.empty_cost","y_parameter":"model.parameters.loaded_cost"}),
        ]:
            a=create_analysis({"run_ids":chosen,"kind":kind,**extra});figures[kind]=a["analysis_id"]
        raw=export_results(ids)
        for family,model,params,backend,accounting in [
            ("legacy_dispatch","legacy_dispatch",{"capacity":1},"gurobi","legacy-assignment-v1"),
            ("bhh","bhh_steady",{"tau":1,"wave_max":10},"cpu","bhh-cost-v1")]:
            data=generate({"dataset_id":"phase1-"+family,"family":family,"seeds":[22001,22002],"splits":["test"],"horizon":3,"orders_per_step":1})
            b=co.submit({"base_spec":{"name":"Phase 1 "+family+" grid","dataset":{"dataset_id":data["dataset_id"],"revision":data["revision"]},
                                     "model":{"id":model,"parameters":params},"solver":{"backend":backend},"evaluation":{"accounting_version":accounting}},
                         "ranges":{"model.parameters."+("tau" if family=="bhh" else "capacity"):{"start":1,"stop":2,"step":1}}})
            limit=time.monotonic()+90
            while time.monotonic()<limit:
                co.tick(); b=co.batch(b["batch_id"])
                if all(r["status"] in TERMINAL for r in b["runs"]):break
                time.sleep(.1)
            assert all(r["status"]=="COMPLETED" for r in b["runs"]),b
        report={"batch":batch,"figures":figures,"raw_export":raw["analysis_id"],
                "conditions":6,"independent_scenarios_per_condition":4,"policy_seeds_per_condition":1,
                "status":"passed"}
        atomic_json(workspace()/"phase1-validation.json",report)
        print(json.dumps(report))
    finally:co.close()

if __name__=="__main__":main()
