"""Scenario persistence, including compatibility with the original pickle."""

from __future__ import annotations

import pickle
from pathlib import Path

from model_families.legacy.engine.domain.city_graph import CityGraph
from model_families.legacy.engine.domain.order import Order
from model_families.legacy.engine.domain.vehicle import Vehicle


class _LegacyScenarioUnpickler(pickle.Unpickler):
    """Map class paths stored by the original flat repository layout."""

    _CLASS_MAP = {
        ("CITY_GRAPH", "CityGraph"): CityGraph,
        ("ORDER", "Order"): Order,
        ("VEHICLE", "Vehicle"): Vehicle,
    }

    def find_class(self, module: str, name: str):
        mapped = self._CLASS_MAP.get((module, name))
        return mapped if mapped is not None else super().find_class(module, name)


def load_scenario(path: Path):
    """Load vehicles, orders and graph from a current or legacy scenario."""
    with Path(path).open("rb") as handle:
        data = _LegacyScenarioUnpickler(handle).load()
    return data["Vehicles"], data["Total_order"], data["G"]
