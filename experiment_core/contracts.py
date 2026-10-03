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
    accounting_version: Literal["legacy-assignment-v1", "bhh-cost-v1"] = (
        "legacy-assignment-v1"
    )
    terminal_policy: Literal["report_pending", "drain_committed"] = "report_pending"
    warmup_periods: Annotated[int, Field(ge=0)] = 0
    metrics: list[str] = Field(
        default_factory=lambda: ["operating_profit", "assigned", "delivered", "pending"]
    )


class Execution(Contract):
    policy_seeds: list[Annotated[int, Field(ge=0, le=2_000_000_000)]] = Field(
        default_factory=lambda: [0], min_length=1, max_length=100
    )
    save_trace: bool = True
    timeout_seconds: Positive = 300


class RunSpec(Contract):
    schema_version: Literal["experiment-spec/v1"] = "experiment-spec/v1"
    name: str = Field(min_length=1, max_length=200)
    dataset: DatasetRef
    model: PluginRef
    controller: PluginRef = Field(default_factory=lambda: PluginRef(id="myopic"))
    value_function: PluginRef = Field(default_factory=lambda: PluginRef(id="zero"))
    solver: SolverRef = Field(default_factory=SolverRef)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    execution: Execution = Field(default_factory=Execution)


class BatchSpec(Contract):
    schema_version: Literal["batch-spec/v1"] = "batch-spec/v1"
    base_spec: RunSpec
    variants: list[dict] = Field(
        default_factory=lambda: [{}], min_length=1, max_length=100
    )
    sweeps: dict[str, list] = Field(default_factory=dict)


class MatchingParameters(Contract):
    speed: Positive = 20
    capacity: Annotated[int, Field(ge=1, le=100)] = 7
    loaded_cost: Nonnegative = 10
    empty_cost: Nonnegative = 10
    waiting_cost: Nonnegative = 0
    expiry_cost: Nonnegative = 300
    max_delay: Annotated[int, Field(ge=0, le=12)] = 3
    tie_break: Nonnegative = 1
    cross_hub: bool = True
    reposition: bool = False


class MyopicParameters(Contract):
    pass


class RolloutParameters(Contract):
    samples: Annotated[int, Field(ge=1, le=16)] = 2
    candidate_cross_hub: list[bool] = Field(
        default_factory=lambda: [False, True], min_length=1
    )
    expected_orders_per_step: Annotated[int, Field(ge=0, le=100)] = 2


class RollingParameters(Contract):
    planning_horizon: Annotated[int, Field(ge=2, le=100)] = 12
    commit_periods: Annotated[int, Field(ge=1, le=100)] = 2
    completion_extension: Annotated[int, Field(ge=0, le=100)] = 6

    @model_validator(mode="after")
    def horizons(self):
        if self.commit_periods > self.planning_horizon:
            raise ValueError("commit_periods must not exceed planning_horizon")
        return self


class FluidParameters(Contract):
    recompute: bool = False
    multiplier: Nonnegative = 1
    expected_orders_per_step: Annotated[int, Field(ge=0, le=100)] = 2


class LearnedParameters(Contract):
    checkpoint_run: Identifier
    multiplier: Nonnegative = 1


class TrainParameters(Contract):
    epochs: Annotated[int, Field(ge=1, le=1000)] = 30
    learning_rate: Positive = 0.01


class SacTrainParameters(TrainParameters):
    learning_rate: Positive = 0.0003


class LegacyParameters(Contract):
    capacity: Annotated[int, Field(ge=1, le=100)] = 7


class SacParameters(Contract):
    checkpoint_run: Identifier
    stochastic: bool = False


class BHHParameters(Contract):
    a: Nonnegative = 0.05
    b: Nonnegative = 0.6
    rho: Nonnegative = 0.2
    a_by_city: list[Nonnegative] | None = Field(
        default=None, min_length=2, max_length=2
    )
    b_by_city: list[Nonnegative] | None = Field(
        default=None, min_length=2, max_length=2
    )
    rho_by_city: list[Nonnegative] | None = Field(
        default=None, min_length=2, max_length=2
    )
    period_duration: Positive = 1
    tau: Annotated[int, Field(ge=1, le=40)] = 2
    demand_rate: Positive = 30
    hv_capacity: Annotated[int, Field(ge=1, le=100)] = 12
    av_capacity: Annotated[int, Field(ge=1, le=1000)] = 120
    hv_cost: Nonnegative = 40
    av_cost: Nonnegative = 15
    waiting_cost: Positive = 5
    finite_waiting_cost: Nonnegative = 0
    resort_cost: Nonnegative = 0
    hv_fleet: list[Annotated[int, Field(ge=0, le=20)]] = Field(
        default_factory=lambda: [4, 4], min_length=2, max_length=2
    )
    av_fleet: list[Annotated[int, Field(ge=0, le=20)]] = Field(
        default_factory=lambda: [2, 2], min_length=2, max_length=2
    )
    wave_max: Annotated[float, Field(ge=1, le=10000)] = 1000
    mode: Literal["hybrid", "direct", "hub"] = "hybrid"


class SpatialParameters(Contract):
    radius: Positive = 1
    speed: Positive = 1
    samples: Annotated[int, Field(ge=1, le=32)] = 4
    seed: Annotated[int, Field(ge=0)] = 11
    stop_counts: list[Annotated[int, Field(ge=1, le=8)]] = Field(
        default_factory=lambda: [1, 2, 4, 6], min_length=1, max_length=8
    )


class BHHSteadyParameters(BHHParameters):
    tau: Positive = 2
    period_duration: Literal[1] = 1


class Generation(Contract):
    schema_version: Literal["dataset-generation/v1"] = "dataset-generation/v1"
    dataset_id: Identifier
    family: Literal["single_level_matching", "legacy_dispatch", "bhh"] = (
        "single_level_matching"
    )
    seeds: list[Annotated[int, Field(ge=0, le=2_000_000_000)]] = Field(
        default_factory=lambda: [10001, 10002, 10003], min_length=1, max_length=100
    )
    splits: list[Literal["train", "validation", "test"]] = Field(
        default_factory=lambda: ["train", "validation", "test"], min_length=1
    )
    horizon: Annotated[int, Field(ge=1, le=200)] = 8
    num_vehicles: Annotated[int, Field(ge=0, le=100)] = 5
    num_cities: Annotated[int, Field(ge=2, le=20)] = 8
    orders_per_step: Annotated[int, Field(ge=0, le=100)] = 2
    first_mile: Literal["batch", "direct", "none"] = "batch"

    @model_validator(mode="after")
    def unique(self):
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("scenario seeds must be unique")
        if len(self.splits) not in (1, len(self.seeds)):
            raise ValueError("provide one split or one per seed")
        return self


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
