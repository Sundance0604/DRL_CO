from typing import Annotated, Literal
from pydantic import Field, model_validator
from experiment_core.contracts import Contract, Positive, Nonnegative, Identifier

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
    pickup_radius: Nonnegative = 0
    pickup_rule: Literal["network_radius", "legacy_adjacency"] = "network_radius"
    reposition: bool = False


class MyopicParameters(Contract):
    pass


class SingleRollingParameters(Contract):
    planning_horizon: Annotated[int, Field(ge=1, le=100, description="H: jointly plan [t, min(t+H,T)); only currently booked orders are visible.")] = 6
    commit_periods: Annotated[int, Field(ge=1, le=100, description="k: execute the first k periods before replanning; must not exceed H. New bookings wait for the next replan.")] = 1
    completion_extension: Annotated[int, Field(ge=0, le=100, description="Diagnostic completion padding beyond the decision window. Automatically extended for physical commitments; not a hard truncation cap or an extra dispatch horizon.")] = 6
    planner_id: Annotated[Identifier, Field(description="Registered window algorithm; replacing it does not change control, state updates or accounting.")] = "route_milp"

    @model_validator(mode="after")
    def valid_prefix(self):
        if self.commit_periods > self.planning_horizon:
            raise ValueError("commit_periods must not exceed planning_horizon")
        return self


class RolloutParameters(Contract):
    samples: Annotated[int, Field(ge=1, le=16)] = 2
    candidate_cross_hub: list[bool] = Field(
        default_factory=lambda: [False, True], min_length=1
    )
    expected_orders_per_step: Annotated[int, Field(ge=0, le=100)] = 2


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
