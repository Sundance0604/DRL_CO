"""Price encoding only; original MILP, observation, execution and cash are reused."""
import copy
import math
from dataclasses import dataclass
import numpy as np
from experiments.cfs_direct_ppo_v3.control import DirectEpisode, DirectProvider
from experiments.cfs_direct_ppo_v3.contracts import DirectSchema, DirectSignal, OBSERVATION_VERSION
from experiments.cfs_direct_ppo_v3.observation import FeatureScales
from model_families.single.strategic.contracts import StrategicSignal, identity
from model_families.single.strategic.episode import StrategicEpisode
from model_families.single.strategic.decoder import DecodeResult
from model_families.single.value.route_milp import ValueRouteMILPPlanner
from model_families.single.rolling.contracts import WindowRequest
from model_families.single.rolling.controller import observed_state
from model_families.single.planning import compile_action, PeriodAction
from model_families.single.runner import serialize_plan

VERSION = 'single-structured-q-affine-price/v1'


@dataclass(frozen=True)
class QSignal(StrategicSignal):
    schema_version: str = VERSION

    def __post_init__(self):
        object.__setattr__(self, 'values', tuple(float(v) for v in self.values))
        if self.schema_version != VERSION or any(not math.isfinite(v) or abs(v)>1 for v in self.values):
            raise ValueError('invalid structured signal')


class QProvider:
    def __init__(self, schema, signal, horizon, fleet):
        if len(signal.values)!=4*len(schema.hubs) or not 0<=signal.period<horizon or fleet<1:
            raise ValueError('structured dimension/boundary')
        self.schema, self.signal, self.horizon, self.fleet = schema, signal, horizon, fleet
        self.queries = {}

    def marginal_release_value(self, time, hub, supply_slot, online_state_summary):
        if hub not in self.schema.hubs or int(supply_slot)!=supply_slot or not 1<=supply_slot<=self.fleet:
            raise ValueError('invalid q')
        if not math.isfinite(time) or time<self.signal.period:
            raise ValueError('release precedes control')
        key=(time,hub,supply_slot)
        if key not in self.queries:
            i=self.schema.hubs.index(hub);n=2*len(self.schema.hubs)
            ell=(time-self.signal.period)/(self.horizon-self.signal.period)
            psi=(supply_slot-1)/(self.fleet-1) if self.fleet>1 else 0.
            a=self.signal.values
            alpha=(1-ell)*a[2*i]+ell*a[2*i+1]
            beta=(1-ell)*a[n+2*i]+ell*a[n+2*i+1]
            v=0. if time>=self.horizon else self.schema.scales[i]*(alpha+beta*psi)
            self.queries[key]=dict(time=time,hub=hub,supply_slot=supply_slot,value=v,
                direct_price=v,lookup='structured-q',alpha=alpha,beta=beta,psi=psi,
                support_count=None)
        return self.queries[key]['value']


def basis(schema, period, horizon, fleet, coefficients):
    """Actual slot objective columns, alpha10 then beta10; normalized by USD scale."""
    n=2*len(schema.hubs);out=np.zeros((len(coefficients),2*n))
    for r,c in enumerate(coefficients):
        i=schema.hubs.index(c['hub']);ell=(c['time']-period)/(horizon-period)
        psi=(c['supply_slot']-1)/(fleet-1) if fleet>1 else 0.
        out[r,2*i:2*i+2]=(1-ell,ell)
        out[r,n+2*i:n+2*i+2]=out[r,2*i:2*i+2]*psi
    return out


class AuditPlanner(ValueRouteMILPPlanner):
    """Read the finished original model before disposal. Never change its constraints."""
    def _plan(self, request, model):
        result=super()._plan(request,model)
        coeff=result.diagnostics['release_coefficients'];vs=model.getVars()
        zs=vs[-len(coeff):] if coeff else []
        if any(abs(v.Obj-c['value'])>1e-7 for v,c in zip(zs,coeff)):
            raise RuntimeError('slot objective extraction failed')
        result.diagnostics['slot_occupancy']=[dict(time=c['time'],hub=c['hub'],q=c['supply_slot'],
            occupied=int(v.X>.5)) for v,c in zip(zs,coeff)]
        # Equality-row quotient removes flow-balance constants and fixed variables.
        # This is an affine relaxation, not a proof of the exact integer feasible hull.
        if getattr(self.provider,'rank_audit',False) and coeff:
            b=basis(self.provider.schema,request.start,request.state.T,len(request.state.vehs),coeff)
            cmat=np.zeros((len(vs),b.shape[1]));cmat[-len(coeff):]=b
            constraints=model.getConstrs();a=model.getA().toarray()
            eq=a[[j for j,c in enumerate(constraints) if c.Sense=='=']]
            fixed=[j for j,v in enumerate(vs) if v.LB==v.UB]
            if fixed:eq=np.vstack((eq,np.eye(len(vs))[fixed]))
            residual=cmat-eq.T@np.linalg.lstsq(eq.T,cmat,rcond=1e-10)[0]
            n=cmat.shape[1]//2
            result.diagnostics['coefficient_rank']=dict(alpha=int(np.linalg.matrix_rank(residual[:,:n],tol=1e-8)),
                expanded=int(np.linalg.matrix_rank(residual,tol=1e-8)),tolerance=1e-8,
                interpretation='equality-quotient affine relaxation; NOT exact integer-hull rank')
        return result


class QDecoder:
    def __init__(self,schema):self.schema=schema;self.rank_audit=False

    def decode(self,snapshot,observation,signal,checkpoint):
        if (not isinstance(signal,QSignal) or observation.version!=OBSERVATION_VERSION or
            (snapshot.t,snapshot.revision)!=(signal.period,signal.revision) or
            (signal.period,signal.revision,signal.observation_hash)!=(observation.period,observation.revision,observation.hash)
            or observation.hubs!=self.schema.hubs):raise RuntimeError('stale structured signal')
        provider=QProvider(self.schema,signal,snapshot.T,len(snapshot.vehs));provider.rank_audit=self.rank_audit
        result=AuditPlanner(provider,dict(T=snapshot.T,period=snapshot.t)).plan(
            WindowRequest(observed_state(snapshot),snapshot.t,snapshot.t+1,snapshot.t+1,0,checkpoint))
        if result.diagnostics['status']!='OPTIMAL':raise RuntimeError('structured solver not OPTIMAL')
        action=result.actions.get(snapshot.t,PeriodAction(snapshot.t));info=dict(result.diagnostics)
        plan=compile_action(snapshot,action,info);serial=serialize_plan(plan,snapshot.t,observation.hash,info)
        return DecodeResult(action,plan,serial,identity({k:serial[k] for k in
            ('period','assignments','routes','repositions','expired')}),tuple(provider.queries.values()))


class QEpisode(DirectEpisode):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._decoder=QDecoder(self.schema)

    def step(self,values):
        if hasattr(self._cancelled,'claim'):self._cancelled.claim()
        o=self._observation
        signal=values if isinstance(values,QSignal) else QSignal(tuple(values),o.period,o.revision,o.hash)
        return StrategicEpisode.step(self,signal)

    def probe(self,values):
        if self.status!='DECISION_READY':raise RuntimeError('not ready')
        if hasattr(self._cancelled,'claim'):self._cancelled.claim()
        o=self._observation
        result=self._decoder.decode(observed_state(self._sim),o,QSignal(tuple(values),o.period,o.revision,o.hash),self._check)
        return dict(action_hash=result.action_hash,plan=copy.deepcopy(result.serialized),coefficients=list(result.coefficients))


def episode(scene,cfg,budget=None):
    if 'baseline' in cfg:raise ValueError('no MC in q episode')
    from model_families.single.parameters import MatchingParameters
    from experiment_core.contracts import SolverParameters
    return QEpisode(scene,MatchingParameters.model_validate(cfg['parameters']).model_dump(),
                    SolverParameters.model_validate(cfg['solver']).model_dump(),DirectSchema(**cfg['action']),
                    FeatureScales(**cfg['feature_scales']),cancelled=budget)


def upgrade(ep):
    """A replayed historical prefix is retained byte-for-byte, without a new solve."""
    ep.__class__=QEpisode;ep._decoder=QDecoder(ep.schema)
    return ep


def replay(scene,cfg,record):
    """Independently check q prices, then unchanged historical physical replay.

    Only the old replay's *price view* is adapted; actions, cash, plans and hashes
    are NOT changed. It independently recompiles all integer actions/cooldown.
    """
    from experiments.cfs_direct_ppo_v3.qa import replay as physical
    schema=DirectSchema(**cfg['action']);adapted=copy.deepcopy(record)
    count=0;horizon=scene['horizon']
    for tr in adapted['transitions']:
        signal=tr['signal'];values=signal['values'];n=2*len(schema.hubs)
        if signal['schema_version']==VERSION:
            q=QSignal(**signal);fleet=len(tr['observation']['vehicles'])
            for c in tr['coefficients']:
                i=schema.hubs.index(c['hub']);ell=(c['time']-q.period)/(horizon-q.period)
                psi=(c['supply_slot']-1)/(fleet-1) if fleet>1 else 0.
                expected=schema.scales[i]*((1-ell)*(values[2*i]+values[n+2*i]*psi)+
                                          ell*(values[2*i+1]+values[n+2*i+1]*psi))
                if c['time']>=horizon or abs(expected-c['value'])>1e-7:raise RuntimeError('q price independent mismatch')
                count+=1
            signal['values']=values[:n];signal['schema_version']=DirectSignal.__dataclass_fields__['schema_version'].default
            old=DirectProvider(schema,DirectSignal(**signal),horizon)
            for c in tr['coefficients']:
                c['value']=c['direct_price']=old.marginal_release_value(c['time'],c['hub'],c['supply_slot'],{})
    if count and record['transitions'][-1]['period']!=horizon-1:
        raise RuntimeError('q independent replay expects full day')
    result=physical(scene,cfg,adapted);result['q_price_coefficients']=count
    return result
