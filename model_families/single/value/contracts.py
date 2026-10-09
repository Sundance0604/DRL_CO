"""Pre-evaluated marginal resource values; no nonlinear solver expressions."""
from typing import Protocol

VERSION = "single-marginal-release/v1"


class ValueProvider(Protocol):
    def marginal_release_value(self, time, hub, supply_slot, online_state_summary) -> float:
        """q-th FREE vehicle at (time, hub), conditional on the observed summary."""


class ZeroValueProvider:
    def marginal_release_value(self, time, hub, supply_slot, online_state_summary):
        return 0.0


def add_release_slots(model, releases, provider, summary, horizon, max_slots):
    """Ordered slots prevent reusing q=1 and cherry-picking a later high-value slot.

    releases[(time,hub)] are binary path-flow indicators. Every vehicle ends in
    one sink exactly once. Prefix constraints are valid even for nonmonotone or
    NEGATIVE marginal estimates; no concavity assumption is made.
    """
    import math
    import gurobipy as gp
    from gurobipy import GRB
    terms, coefficients = [], []
    for (when, hub), indicators in sorted(releases.items()):
        if when >= horizon:
            continue  # new free supply cannot initiate controls after T
        slots = [model.addVar(vtype=GRB.BINARY) for _ in range(max_slots)]
        model.addConstr(gp.quicksum(slots) == gp.quicksum(indicators))
        for q, z in enumerate(slots, 1):
            if q > 1:
                model.addConstr(z <= slots[q-2])
            value = float(provider.marginal_release_value(when, hub, q, summary))
            if not math.isfinite(value):
                raise ValueError("nonfinite marginal release coefficient")
            terms.append(value*z)
            coefficients.append(dict(time=when,hub=hub,supply_slot=q,value=value))
    return terms, coefficients
