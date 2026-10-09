"""Immutable, versioned public observations and strategic actions."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

OBSERVATION_VERSION = "single-strategic-observation/v1"
ACTION_VERSION = "single-near-far-residual/v1"
DECODER_VERSION = "single-strategic-h1/v1"
REWARD_VERSION = "single-full-cooldown-increment/v1"


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ActionSchema:
    hubs: tuple[str, ...]
    scales: tuple[float, ...]
    version: str = ACTION_VERSION

    def __post_init__(self):
        object.__setattr__(self, "hubs", tuple(self.hubs))
        object.__setattr__(self, "scales", tuple(float(s) for s in self.scales))
        if (self.version != ACTION_VERSION or not self.hubs or
                len(set(self.hubs)) != len(self.hubs) or len(self.hubs) != len(self.scales) or
                any(not math.isfinite(s) or s <= 0 for s in self.scales)):
            raise ValueError("invalid action schema, hub order or frozen USD scales")


@dataclass(frozen=True)
class StrategicSignal:
    values: tuple[float, ...]  # interleaved near, far for each schema hub
    period: int
    revision: int
    observation_hash: str
    schema_version: str = ACTION_VERSION

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))
        if self.schema_version != ACTION_VERSION or any(
                not math.isfinite(v) or abs(v) > 1 for v in self.values):
            raise ValueError("signal must contain finite bounded actions")


@dataclass(frozen=True)
class VehicleObservation:
    features: tuple[float, ...]
    arcs: tuple[tuple[float, ...], ...]
    commitments: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class EncodedObservation:
    period: int
    revision: int
    hubs: tuple[str, ...]
    global_features: tuple[float, ...]
    hub_features: tuple[tuple[float, ...], ...]
    orders: tuple[tuple[float, ...], ...]
    vehicles: tuple[VehicleObservation, ...]
    version: str = OBSERVATION_VERSION

    @property
    def hash(self):
        return identity(asdict(self))


@dataclass(frozen=True)
class Transition:
    observation: EncodedObservation | None
    reward: float
    terminated: bool
    transition_id: str


class EpisodeCancelled(RuntimeError):
    pass
