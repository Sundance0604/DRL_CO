from typing import Annotated, Literal
from pydantic import Field, model_validator
from experiment_core.contracts import Contract, Positive, Nonnegative, Identifier

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
