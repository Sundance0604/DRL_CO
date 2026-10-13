"""Explicit direct-price identities, incompatible with residual checkpoints."""
from dataclasses import dataclass
import math
from model_families.single.strategic.contracts import ActionSchema, StrategicSignal

OBSERVATION_VERSION = 'single-direct-public-no-mc/v1'
ACTION_VERSION = 'single-direct-near-far-price/v1'


@dataclass(frozen=True)
class DirectSchema(ActionSchema):
    version: str = ACTION_VERSION

    def __post_init__(self):
        object.__setattr__(self,'hubs',tuple(self.hubs))
        object.__setattr__(self,'scales',tuple(float(x) for x in self.scales))
        if (self.version!=ACTION_VERSION or not self.hubs or len(set(self.hubs))!=len(self.hubs)
                or len(self.hubs)!=len(self.scales) or any(not math.isfinite(x) or x<=0 for x in self.scales)):
            raise ValueError('invalid direct action schema')


@dataclass(frozen=True)
class DirectSignal(StrategicSignal):
    schema_version: str = ACTION_VERSION

    def __post_init__(self):
        object.__setattr__(self,'values',tuple(float(x) for x in self.values))
        if self.schema_version!=ACTION_VERSION or any(not math.isfinite(x) or abs(x)>1 for x in self.values):
            raise ValueError('invalid direct signal')
