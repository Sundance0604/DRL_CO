from __future__ import annotations
import copy
import hashlib
import math
import random
import time
import numpy as np
from experiment_core.contracts import PlatformError
from experiment_core.storage import atomic_json, digest, read_json, run_dir
def execute(spec, data, scenarios, target, policy_seed, emit, cancelled):
    start = time.monotonic()
    random.seed(policy_seed)
    np.random.seed(policy_seed)

    def check():
        if cancelled():
            raise PlatformError("CANCELLED", "cancellation requested", exit_code=6)
        if time.monotonic() - start > spec.execution.timeout_seconds:
            raise PlatformError("TIMEOUT", "run timeout exceeded", exit_code=5)

    rows, traces, training = [], [], []
    agent = None
    for i, scenario in enumerate(scenarios):
        check()
        emit(
            "scenario_started", scenario_id=scenario["id"], progress=i / len(scenarios)
        )
        if spec.model.id == "bhh_steady":
            from .optimization import steady

            metrics = steady(spec.model.parameters)
            trace = []
        elif spec.model.id == "bhh_spatial":
            from .spatial import calibrate

            metrics = calibrate(spec.model.parameters)
            trace = []
        else:
            from .bhh import finite, rolling

            if spec.controller.id == "rolling_horizon":
                solution, metrics, trace = rolling(
                    scenario,
                    spec.model.parameters,
                    spec.solver.parameters.model_dump(),
                    spec.controller.parameters,
                    check,
                )
            else:
                solution, metrics = finite(
                    scenario, spec.model.parameters, spec.solver.parameters.model_dump()
                )
                trace = [
                    {
                        "period": 0,
                        "plan": {k: v for k, v in solution.items() if v > 1e-8},
                        "committed": False,
                        "oracle": True,
                    }
                ]
            metrics["order_rate_reason"] = (
                "BHH records represent divisible commodities, not indivisible orders"
            )
        rows.append(
            {
                "scenario_id": scenario["id"],
                "policy_seed": None if spec.controller.id.startswith("train_") else policy_seed,
                "training_seed": policy_seed if spec.controller.id.startswith("train_") else None,
                "scenario_seed": scenario.get("seed"),
                "family": spec.family,
                "framework_id": spec.framework_id,
                "metrics": metrics,
            }
        )
        if spec.execution.save_trace:
            traces.append(
                {
                    "scenario_id": scenario["id"],
                    "network": {"nodes": scenario["nodes"], "edges": scenario["edges"]},
                    "steps": trace,
                }
            )
        emit(
            "scenario_completed",
            scenario_id=scenario["id"],
            progress=(i + 1) / len(scenarios),
            metrics=metrics,
        )
    atomic_json(target / "trace.json", traces)
    if spec.model.id == "bhh_finite":
        from .bhh import capacity, direct_capacity

        p = spec.model.parameters
        table = {}
        for city in (0, 1):
            table[f"local_city_{city}"] = [
                {"vehicles": v, "duration": d, "capacity": capacity(v, d, p, city)}
                for v in range(sum(p["hv_fleet"]) + 1)
                for d in range(1, 13)
            ]
            table[f"direct_direction_{city}"] = [
                {
                    "vehicles": v,
                    "duration": d,
                    "capacity": 0 if d == p["tau"] else direct_capacity(v, d, p, city),
                }
                for v in range(sum(p["hv_fleet"]) + 1)
                for d in range(1, 13)
            ]
        atomic_json(target / "capacity-table.json", table)
    from .metrics import DEFINITIONS

    atomic_json(target / "metric-definitions.json", DEFINITIONS)
    atomic_json(
        target / "charts.json",
        {
            "schema_version": "chart-artifact/v1",
            "type": "scenario-bars",
            "data_ref": "metrics.json",
            "encodings": {
                "x": "scenario_id",
                "y": "business_cost"
                if spec.model.id.startswith("bhh")
                else "operating_profit",
            },
            "units": {"y": "currency"},
            "metric_defs": "metric-definitions.json",
            "javascript_allowed": False,
        },
    )
    atomic_json(
        target / "metrics.json",
        {
            "schema_version": "run-metrics/v1",
            "rows": rows,
            "elapsed_seconds": time.monotonic() - start,
        },
    )
    return rows
