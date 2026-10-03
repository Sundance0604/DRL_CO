from typing import Annotated, Literal
from pydantic import Field, model_validator
from experiment_core.contracts import Contract, Positive, Nonnegative, Identifier

class Generation(Contract):
    schema_version: Literal["dataset-generation/v1"] = "dataset-generation/v1"
    dataset_id: Identifier
    family: Literal["single_level_matching"] = "single_level_matching"
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
