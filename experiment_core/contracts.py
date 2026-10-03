from __future__ import annotations

import json
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_assignment=True
    )


class ApplicationConfig(Contract):
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: Annotated[int, Field(ge=1024, le=65535)] = 8765
    max_parallel: Annotated[int, Field(ge=1, le=16)] = 1
    cpu_budget: Annotated[int, Field(ge=1, le=256)] | None = None


class PluginRef(Contract):
    id: Identifier
    version: Literal["1"] = "1"
    parameters: dict = Field(default_factory=dict)


class DatasetRef(Contract):
    dataset_id: Identifier
    revision: Identifier
    scenario_ids: list[Identifier] = Field(default_factory=list)
    split: Literal["train", "validation", "test", "all"] = "test"


class SolverParameters(Contract):
    time_limit_seconds: Positive = 30
    mip_gap: Annotated[float, Field(ge=0, le=1)] = 0
    threads: Annotated[int, Field(ge=1, le=16)] = 1
    seed: Annotated[int, Field(ge=0, le=2_000_000_000)] = 0


class SolverRef(Contract):
    backend: Literal["gurobi", "cpu"] = "gurobi"
    parameters: SolverParameters = Field(default_factory=SolverParameters)


class Evaluation(Contract):
    information_set: Literal["online", "oracle"] = "online"
    accounting_version: str = (
        "legacy-assignment-v1"
    )
    terminal_policy: Literal["report_pending", "drain_committed"] = "report_pending"
    warmup_periods: Annotated[int, Field(ge=0)] = 0
    metrics: list[str] = Field(
        default_factory=lambda: ["operating_profit", "assigned", "delivered", "pending"]
    )


class Execution(Contract):
    training_seeds: Annotated[list[Annotated[int, Field(ge=0, le=2_000_000_000)]], Field(min_length=1, max_length=100)] | None = None
    policy_seeds: list[Annotated[int, Field(ge=0, le=2_000_000_000)]] = Field(
        default_factory=lambda: [0], min_length=1, max_length=100
    )
    save_trace: bool = True
    timeout_seconds: Positive = 300

    @model_validator(mode="after")
    def unique_seeds(self):
        for values in (self.policy_seeds, self.training_seeds):
            if values is not None and len(values) != len(set(values)):
                raise ValueError("replication seeds must be unique")
        return self


class RunSpec(Contract):
    schema_version: Literal["experiment-spec/v1"] = "experiment-spec/v1"
    family: Identifier | None = None
    framework_id: Identifier | None = None
    name: str = Field(min_length=1, max_length=200)
    dataset: DatasetRef
    model: PluginRef
    controller: PluginRef = Field(default_factory=lambda: PluginRef(id="myopic"))
    value_function: PluginRef = Field(default_factory=lambda: PluginRef(id="zero"))
    solver: SolverRef = Field(default_factory=SolverRef)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    execution: Execution = Field(default_factory=Execution)


class RangeSweep(Contract):
    start: float
    stop: float
    step: Positive
    scale: Literal["linear", "log"] = "linear"


class BatchSpec(Contract):
    schema_version: Literal["batch-spec/v1"] = "batch-spec/v1"
    base_spec: RunSpec
    variants: list[dict] = Field(
        default_factory=lambda: [{}], min_length=1, max_length=100
    )
    sweeps: dict[str, list] = Field(default_factory=dict)
    ranges: dict[str, "RangeSweep"] = Field(default_factory=dict)


class PlatformError(Exception):
    def __init__(self, code, message, path="", exit_code=2):
        self.code, self.message, self.path, self.exit_code = (
            code,
            message,
            path,
            exit_code,
        )
        super().__init__(message)

    def as_dict(self):
        return {
            "error": {"code": self.code, "message": self.message, "path": self.path}
        }


def strict_json(text):
    def reject(value):
        raise PlatformError("NONFINITE_JSON", f"nonfinite JSON number: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PlatformError("DUPLICATE_JSON_KEY", f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(text, parse_constant=reject, object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise PlatformError("INVALID_JSON", str(exc)) from exc


# Backward-compatible public names. New platform code uses family-owned schemas.
def __getattr__(name):
    from importlib import import_module
    aliases = {
        "MatchingParameters": "single", "RolloutParameters": "single",
        "FluidParameters": "single", "LearnedParameters": "single", "TrainParameters": "single",
        "LegacyParameters": "legacy", "SacParameters": "legacy", "SacTrainParameters": "legacy",
        "BHHParameters": "bhh", "BHHSteadyParameters": "bhh", "SpatialParameters": "bhh", "RollingParameters":"bhh",
    }
    if name == "Generation":
        return import_module("model_families.single.generation").Generation
    if name in aliases:
        return getattr(import_module(f"model_families.{aliases[name]}.parameters"), name)
    raise AttributeError(name)
