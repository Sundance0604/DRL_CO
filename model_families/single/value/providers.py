"""JSON-serializable additive approximations with no learning-based policy."""
import json
import numpy as np
from scipy.interpolate import BSpline
from .contracts import ZeroValueProvider

FEATURES=["remaining","other_idle","incoming","backlog","expected","q"]


def feature_row(when,hub,q,online):
    return dict(remaining=max(0,(online["T"]-when)/online["T"]),q=float(q),
        hub=hub,other_idle=float(sum(n for h,n in online["idle"].items() if h!=hub)),
        incoming=float(sum(x["hub"]==hub and online["period"]<x["time"]<=when+4 for x in online["incoming"])),
        backlog=float(online["backlog"].get(hub,0)),
        expected=float(online["prior"]["rate"].get(hub,0)*max(0,online["T"]-when)))


raw_features=feature_row


def design(rows,meta):
    n=len(rows);parts=[np.ones((n,1))]
    parts.append(np.asarray([[float(r["hub"]==h) for h in meta["hubs"][1:]] for r in rows]))
    for name in FEATURES:
        x=np.asarray([float(r[name]) for r in rows])
        b=meta["bases"][name]
        if b["kind"]=="linear":
            a=((x-b["center"])/b["scale"])[:,None]
        elif b["kind"]=="chebyshev":
            z=np.clip(2*(x-b["lower"])/b["scale"]-1,-1,1)
            a=np.polynomial.chebyshev.chebvander(z,b["degree"])[:,1:]
        else:
            # Clip to training domain. No uncontrolled polynomial extrapolation.
            x=np.clip(x,b["lower"],b["upper"])
            knots=np.asarray(b["knots"]);c=np.eye(len(knots)-4)
            a=BSpline(knots,c,3)(x)@np.asarray(b["transform"])
            a=a-np.asarray(b["center"])
        parts.append(a)
    return np.column_stack(parts)


class RegressionValueProvider:
    def __init__(self,model):
        self.model=model
    def marginal_release_value(self,time,hub,supply_slot,online_state_summary):
        if time>=online_state_summary["T"]:return 0.
        row=raw_features(time,hub,supply_slot,online_state_summary)
        return float((design([row],self.model)@np.asarray(self.model["coefficients"])).item())


class LinearRidgeValueProvider(RegressionValueProvider):pass
class ChebyshevValueProvider(RegressionValueProvider):pass
class SplineGAMValueProvider(RegressionValueProvider):pass


class MCTableValueProvider:
    def __init__(self,model):self.model=model
    def marginal_release_value(self,time,hub,supply_slot,online_state_summary):
        if time>=online_state_summary["T"]:return 0.
        group=self.model["values"][hub][str(supply_slot)]
        return float(np.interp(time,group["times"],group["values"]))


class FluidDualValueProvider(MCTableValueProvider):
    """Offline fluid LP shadow-price surface; state-independent baseline."""


def load_provider(path):
    model=json.loads(path.read_text(encoding="utf-8"))
    classes=dict(table=MCTableValueProvider,fluid=FluidDualValueProvider,
        linear=LinearRidgeValueProvider,chebyshev=ChebyshevValueProvider,spline=SplineGAMValueProvider)
    return classes[model["method"]](model)
