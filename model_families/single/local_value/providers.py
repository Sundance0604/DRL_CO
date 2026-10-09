"""Fixed pre-solve coefficients, batch evaluation and q-marginalized correction."""
import copy
import numpy as np
from scipy.interpolate import BSpline
from experiment_core.storage import read_json, digest
from model_families.single.value.providers import load_provider
from model_families.single.planning import committed_arc_load

FEATURES=['remaining','duration','deadhead','loaded','compatible_load','expiring_load',
          'incoming','release_idle','origin_idle','expected','wait','same_hub']
CONTEXT_FIELDS={'candidate_id','decision_time','departure_time','release_time','release_hub',
    'vehicle_origin_hub','pickup_hub','wait_time','deadhead_distance','loaded_distance',
    'route_duration','compatible_order_count','compatible_backlog_load','compatible_expiring_count',
    'compatible_expiring_load','residual_capacity','origin_idle_delta','known_release_delta','kind'}

def context(sim,vehicle,route,when,release,hub,candidate_id):
    origin=vehicle.loc; cap=sim.parameters['capacity']
    compatible=[] if route is None else [o for o in route['serve'] if o.book_time<=sim.t]
    assert route is None or len(compatible)==len(route['serve']), 'unbooked order in deployable context'
    exp=[o for o in compatible if max(when+1,o.start_time)+sim.net.tau[o.departure,o.destination]>o.end_time]
    p=origin if route is None else route['p']
    r=dict(candidate_id=str(candidate_id),decision_time=sim.t,departure_time=when,
        release_time=release,release_hub=hub,vehicle_origin_hub=origin,pickup_hub=p,
        wait_time=release-when if route is None else route['delay'],
        deadhead_distance=0. if route is None else sim.net.d[origin,p],
        loaded_distance=0. if route is None else sim.net.d[p,route['e']],route_duration=release-when,
        compatible_order_count=len(compatible),compatible_backlog_load=sum(o.passenger for o in compatible)/cap,
        compatible_expiring_count=len(exp),compatible_expiring_load=sum(o.passenger for o in exp)/cap,
        residual_capacity=cap-max(committed_arc_load(sim.net,vehicle).values(),default=0),
        origin_idle_delta=-int(route is not None and vehicle.phase=='idle'),known_release_delta=1,
        kind='wait' if route is None else 'route')
    assert set(r)==CONTEXT_FIELDS
    return r

def features(ctx,online):
    assert set(ctx)<=CONTEXT_FIELDS and set(ctx)==CONTEXT_FIELDS
    t=ctx['release_time']; hub=ctx['release_hub']; origin=ctx['vehicle_origin_hub']
    return dict(hub=hub,time=t,remaining=max(0,(online['T']-t)/online['T']),
        duration=ctx['route_duration'],deadhead=ctx['deadhead_distance'],loaded=ctx['loaded_distance'],
        compatible_load=ctx['compatible_backlog_load'],expiring_load=ctx['compatible_expiring_load'],
        incoming=sum(x['hub']==hub and online['period']<x['time']<=t+4 for x in online['incoming'])+1,
        release_idle=online['idle'][hub],origin_idle=online['idle'][origin]+ctx['origin_idle_delta'],
        expected=online['prior']['rate'].get(hub,0)*max(0,online['T']-t),
        wait=ctx['wait_time'],same_hub=float(origin==hub))

def design(rows,meta,deployed=False):
    n=len(rows); hubs=meta['hubs']; qs=meta['qs']
    parts=[np.ones((n,1)),np.array([[float(r['hub']==h) for h in hubs] for r in rows])]
    if deployed: parts.append(np.tile(meta['q_weights'],(n,1)))
    else: parts.append(np.array([[float(int(r['q'])==q) for q in qs] for r in rows]))
    for f in FEATURES:
        x=np.array([r[f] for r in rows],float); b=meta['bases'][f]
        if b['kind']=='linear': a=((x-b['center'])/b['scale'])[:,None]
        elif b['kind']=='chebyshev':
            a=np.polynomial.chebyshev.chebvander(np.clip(2*(x-b['lower'])/b['scale']-1,-1,1),b['degree'])[:,1:]
        else:
            x=np.clip(x,b['lower'],b['upper']);kn=np.array(b['knots'])
            a=BSpline(kn,np.eye(len(kn)-4),3)(x)@np.array(b['transform'])-np.array(b['center'])
        parts.append(a)
    return np.column_stack(parts)

class LegacyAdapter:
    """None/context both preserve an old provider exactly; no local correction."""
    def __init__(self,legacy): self.legacy=legacy
    def marginal_release_value(self,time,hub,supply_slot,online_state_summary,action_context=None):
        return self.legacy.marginal_release_value(time,hub,supply_slot,online_state_summary)
    def corrections(self,contexts,online): return np.zeros(len(contexts))

class LocalValueProvider:
    def __init__(self,model,slot):
        self.model=copy.deepcopy(model); self.slot=slot
        self.beta=np.array(model['coefficients'])
    def corrections(self,contexts,online):
        if not contexts: return np.zeros(0)
        rows=[features(c,online) for c in contexts]
        out=design(rows,self.model,True)@self.beta
        out=np.where([c['release_time']>=online['T'] for c in contexts],0.,out)
        assert np.isfinite(out).all()
        return out
    def marginal_release_value(self,time,hub,supply_slot,online_state_summary,action_context=None):
        base=self.slot.marginal_release_value(time,hub,supply_slot,online_state_summary)
        return float(base if action_context is None else base+self.corrections([action_context],online_state_summary)[0])

def load_local(path):
    model=read_json(path)
    if model['method'] in ('table','fluid','legacy'): return LegacyAdapter(load_provider(path))
    from model_families.single.value.providers import MCTableValueProvider
    return LocalValueProvider(model,MCTableValueProvider(model['slot_model']))
