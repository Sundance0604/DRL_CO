"""Read-only non-bucket MC table, with explicit interpolation audit."""
import copy
import math
import numpy as np
from bisect import bisect_left
from .contracts import identity


class FrozenBaseline:
    def __init__(self, model):
        self._model = copy.deepcopy(model)
        if model.get("method") not in ("table", "zero"):
            raise ValueError("only a frozen non-bucket MC table or explicit zero base is supported")
        self._groups = {}
        if model["method"] == "table":
            for hub, slots in model["values"].items():
                for q, group in slots.items():
                    ts, vs = tuple(group["times"]), tuple(group["values"])
                    if (int(q) < 1 or not ts or len(ts) != len(vs) or
                            any(not math.isfinite(x) for x in (*ts, *vs)) or
                            any(b <= a for a, b in zip(ts, ts[1:]))):
                        raise ValueError("invalid MC table grid")
                    self._groups[hub, int(q)] = (ts, vs)
        self.hash = identity(self._model)

    @property
    def model(self):
        return copy.deepcopy(self._model)

    def validate(self, hubs, fleet):
        if self._model["method"] == "table":
            if set(self._model["values"]) != set(hubs):
                raise ValueError("MC table hub set mismatch")
            for h in hubs:
                for q in range(1, fleet + 1):
                    if (h, q) not in self._groups:
                        raise ValueError(f"missing MC support for hub={h}, q={q}; no implicit fallback")

    def query(self, time, hub, q, horizon):
        if time >= horizon:
            return 0., "terminal"
        if self._model["method"] == "zero":
            return 0., "explicit_zero"
        ts, vs = self._groups[hub, q]
        i = bisect_left(ts, time)
        if i < len(ts) and ts[i] == time:
            return float(vs[i]), "grid"
        if i == 0:
            return float(vs[0]), "left_clamp"
        if i == len(ts):
            return float(vs[-1]), "right_clamp"
        # Preserve the parent provider's exact interpolation arithmetic for zero
        # signal regression, including ties in the integer decoder.
        return float(np.interp(time, ts, vs)), "interpolation"

    def marginal_release_value(self, time, hub, supply_slot, online_state_summary):
        return self.query(time, hub, supply_slot, online_state_summary["T"])[0]
