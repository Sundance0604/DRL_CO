"""Explicit planner registration, no user-supplied dynamic imports."""
from experiment_core.contracts import PlatformError


def create_planner(name):
    if name == "route_milp":
        from .route_milp import RouteMILPPlanner
        return RouteMILPPlanner()
    raise PlatformError("WINDOW_PLANNER", "unregistered single window planner: "+name)
