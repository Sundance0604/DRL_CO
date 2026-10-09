"""A frozen residual signal -> existing H=1 value MILP -> compiled action."""
import math
from dataclasses import dataclass
from ..planning import PeriodAction, compile_action
from ..rolling.contracts import WindowRequest
from .contracts import identity


class ReleaseSignalProvider:
    def __init__(self, baseline, schema, signal, horizon):
        if len(signal.values) != 2*len(schema.hubs) or not 0 <= signal.period < horizon:
            raise ValueError("action dimension or decision boundary mismatch")
        self.baseline, self.schema, self.signal, self.horizon = baseline, schema, signal, horizon
        self._index = {h: i for i, h in enumerate(schema.hubs)}
        self.queries = {}

    def marginal_release_value(self, time, hub, supply_slot, online_state_summary):
        if hub not in self._index or supply_slot < 1 or int(supply_slot) != supply_slot:
            raise ValueError("invalid release query")
        if not math.isfinite(time) or time < self.signal.period:
            raise ValueError("release precedes decision")
        key = (time, hub, supply_slot)
        if key not in self.queries:
            base, mode = self.baseline.query(time, hub, supply_slot, self.horizon)
            i = self._index[hub]
            ell = (time-self.signal.period)/(self.horizon-self.signal.period)
            correction = 0. if time >= self.horizon else self.schema.scales[i] * (
                (1-ell)*self.signal.values[2*i] + ell*self.signal.values[2*i+1])
            value = base+correction
            if not math.isfinite(value):
                raise ValueError("nonfinite strategic coefficient")
            self.queries[key] = dict(time=time, hub=hub, supply_slot=supply_slot,
                                     base=base, residual=correction, value=value, lookup=mode,
                                     support_count=None)  # legacy table has no per-cell count
        return self.queries[key]["value"]


@dataclass(frozen=True)
class DecodeResult:
    action: object
    plan: object
    serialized: dict
    action_hash: str
    coefficients: tuple[dict, ...]


class StrategicDecoder:
    def __init__(self, schema, baseline, planner_factory=None):
        self.schema, self.baseline = schema, baseline
        self.planner_factory = planner_factory

    def decode(self, snapshot, observation, signal, checkpoint):
        if ((signal.period, signal.revision, signal.observation_hash) !=
                (observation.period, observation.revision, observation.hash) or
                (snapshot.t, snapshot.revision) != (signal.period, signal.revision) or
                observation.hubs != self.schema.hubs):
            raise RuntimeError("stale or incompatible strategic signal")
        provider = ReleaseSignalProvider(self.baseline, self.schema, signal, snapshot.T)
        factory = self.planner_factory
        if factory is None:
            from ..value.route_milp import ValueRouteMILPPlanner
            factory = ValueRouteMILPPlanner
        summary = dict(T=snapshot.T, period=snapshot.t)
        from ..rolling.controller import observed_state
        result = factory(provider, summary).plan(WindowRequest(
            observed_state(snapshot), snapshot.t, snapshot.t+1, snapshot.t+1, 0, checkpoint))
        if result.diagnostics.get("status") != "OPTIMAL":
            raise RuntimeError("strategic decoder requires OPTIMAL; incomplete solve is not termination")
        action = result.actions.get(snapshot.t, PeriodAction(snapshot.t))
        info = dict(result.diagnostics)
        plan = compile_action(snapshot, action, info)
        from ..runner import serialize_plan
        serialized = serialize_plan(plan, snapshot.t, observation.hash, info)
        physical = {k: serialized[k] for k in ("period", "assignments", "routes", "repositions", "expired")}
        return DecodeResult(action, plan, serialized, identity(physical), tuple(provider.queries.values()))
