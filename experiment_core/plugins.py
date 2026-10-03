from __future__ import annotations

from . import contracts as c
from .storage import read_json, run_dir

REGISTRY = {
    "model": {
        "single_level_matching": c.MatchingParameters,
        "legacy_dispatch": c.LegacyParameters,
        "bhh_steady": c.BHHSteadyParameters,
        "bhh_finite": c.BHHParameters,
        "bhh_spatial": c.SpatialParameters,
    },
    "controller": {
        "myopic": c.MyopicParameters,
        "rollout": c.RolloutParameters,
        "rolling_horizon": c.RollingParameters,
        "candidate_sac": c.SacParameters,
        "train_value": c.TrainParameters,
        "train_sac": c.SacTrainParameters,
        "oracle_lp": c.MyopicParameters,
        "oracle_mip": c.MyopicParameters,
    },
    "value_function": {
        "zero": c.MyopicParameters,
        "fluid_dual": c.FluidParameters,
        "learned_hub_time": c.LearnedParameters,
    },
    "solver_backend": {"gurobi": c.SolverParameters, "cpu": c.SolverParameters},
}


def descriptors():
    profiles = {
        "single_level_matching": "indivisible-orders/abstract-hub-network/v1",
        "legacy_dispatch": "legacy-candidate-city/v1",
        "bhh_steady": "balanced-stationary-unconstrained-fleet/v1",
        "bhh_finite": "two-city-divisible-stops/shared-fleet/v1",
        "bhh_spatial": "fixed-disc-points/exact-small-tsp/v1",
    }
    return [
        {
            "kind": kind,
            "id": name,
            "version": "1",
            "parameters_schema": cls.model_json_schema(),
            "ui_schema": {"section": kind},
            "capabilities": {
                "resume": False,
                "trace": name
                in {"single_level_matching", "legacy_dispatch", "bhh_finite"},
                "window_plan": name == "bhh_finite",
                "dataset_profile": profiles.get(name),
                "device": "cpu",
                "training_method": {
                    "train_sac": "reinforcement-learning",
                    "train_value": "supervised-fluid-dual-regression",
                }.get(name),
                "arbitrary_imports": False,
            },
        }
        for kind, entries in REGISTRY.items()
        for name, cls in entries.items()
    ]


def resolve(spec):
    spec = c.RunSpec.model_validate(spec)
    for kind in ("model", "controller", "value_function"):
        ref = getattr(spec, kind)
        cls = REGISTRY[kind].get(ref.id)
        if cls is None:
            raise c.PlatformError(
                "PLUGIN_NOT_FOUND", f"unknown {kind} plugin", f"/{kind}/id"
            )
        ref.parameters = cls.model_validate(ref.parameters).model_dump()
    m, controller, value = spec.model.id, spec.controller.id, spec.value_function.id
    allowed = {
        "single_level_matching": {
            "myopic",
            "rollout",
            "train_value",
            "oracle_lp",
            "oracle_mip",
        },
        "legacy_dispatch": {"myopic", "candidate_sac", "train_sac"},
        "bhh_steady": {"myopic"},
        "bhh_finite": {"myopic", "rolling_horizon"},
        "bhh_spatial": {"myopic"},
    }
    if controller not in allowed[m] or (
        m != "single_level_matching" and value != "zero"
    ):
        raise c.PlatformError(
            "UNSUPPORTED_COMBINATION",
            "model/controller/value combination not implemented",
        )
    if controller in ("rollout", "train_value") and value != "zero":
        raise c.PlatformError(
            "UNSUPPORTED_COMBINATION", "rollout/training requires zero value reference"
        )
    if spec.solver.backend != (
        "cpu" if m in ("bhh_steady", "bhh_spatial") else "gurobi"
    ):
        raise c.PlatformError(
            "SOLVER_MISMATCH",
            "solver backend incompatible with model",
            "/solver/backend",
        )
    if spec.evaluation.warmup_periods != 0:
        raise c.PlatformError(
            "UNSUPPORTED_ACCOUNTING", "warm-up measurement is not implemented"
        )
    if spec.evaluation.accounting_version != (
        "bhh-cost-v1" if m.startswith("bhh") else "legacy-assignment-v1"
    ):
        raise c.PlatformError(
            "ACCOUNTING_MISMATCH", "accounting version does not match physical model"
        )
    if controller in ("oracle_lp", "oracle_mip") and (
        value != "zero"
        or spec.evaluation.information_set != "oracle"
        or spec.evaluation.terminal_policy != "report_pending"
    ):
        raise c.PlatformError(
            "BOUND_PROFILE",
            "bounds require oracle, zero value and report_pending accounting",
        )
    if (
        spec.evaluation.information_set == "oracle"
        and m in ("single_level_matching", "legacy_dispatch")
        and controller not in ("oracle_lp", "oracle_mip")
    ):
        raise c.PlatformError(
            "UNSUPPORTED_INFORMATION",
            "these adapters implement online policies, not oracle bounds",
        )
    if (
        m == "bhh_finite"
        and controller == "myopic"
        and spec.evaluation.information_set != "oracle"
    ):
        raise c.PlatformError(
            "ORACLE_REQUIRED", "full-horizon BHH solve uses all arrivals; mark oracle"
        )
    if controller in ("train_value", "train_sac") and spec.dataset.split != "train":
        raise c.PlatformError(
            "TRAIN_SPLIT_REQUIRED", "value training requires train-only selection"
        )
    if value == "learned_hub_time" or controller == "candidate_sac":
        ref = spec.value_function if value == "learned_hub_time" else spec.controller
        metadata = read_json(
            run_dir(ref.parameters["checkpoint_run"]) / "checkpoint.json"
        )
        expected = (
            "hub-time-value/v1" if value == "learned_hub_time" else "candidate-sac/v1"
        )
        if metadata.get("feature_schema") != expected:
            raise c.PlatformError(
                "CHECKPOINT_FAMILY", "checkpoint is not compatible with selected plugin"
            )
    return spec


def schemas():
    return {
        "run-spec": c.RunSpec.model_json_schema(),
        "batch-spec": c.BatchSpec.model_json_schema(),
        "dataset-generation": c.Generation.model_json_schema(),
        "application-config": c.ApplicationConfig.model_json_schema(),
        **{f"{d['kind']}.{d['id']}": d["parameters_schema"] for d in descriptors()},
    }
