"""Algorithm boundary: immutable scheduling metadata, isolated state, domain actions."""
from dataclasses import dataclass
from typing import Protocol, Callable
from ..planning import PeriodAction


@dataclass(frozen=True)
class WindowRequest:
    state: object
    start: int
    decision_end: int  # exclusive, clipped to the real decision horizon T
    commit_end: int    # exclusive, clipped to T, never beyond decision_end
    completion_extension: int
    checkpoint: Callable[[], None]
    preview_orders: tuple = ()  # ONLY oracle-preview requests; never live state


@dataclass(frozen=True)
class WindowSolution:
    actions: dict[int, PeriodAction]
    diagnostics: dict


class WindowPlanner(Protocol):
    def plan(self, request: WindowRequest) -> WindowSolution: ...
