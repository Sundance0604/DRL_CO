from typing import Annotated, Literal
from pydantic import Field, model_validator
from experiment_core.contracts import Contract, Positive, Nonnegative, Identifier

class RollingParameters(Contract):
    planning_horizon: Annotated[int, Field(ge=2, le=100)] = 12
    commit_periods: Annotated[int, Field(ge=1, le=100)] = 2
    completion_extension: Annotated[int, Field(ge=0, le=100)] = 6

    @model_validator(mode="after")
    def horizons(self):
        if self.commit_periods > self.planning_horizon:
            raise ValueError("commit_periods must not exceed planning_horizon")
        return self


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
