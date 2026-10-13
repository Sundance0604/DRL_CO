"""Direct prices -> unchanged H1 MILP -> unchanged execution and cash."""
import copy
import math
from model_families.single.strategic.episode import StrategicEpisode
from model_families.single.strategic.decoder import DecodeResult
from model_families.single.strategic.contracts import identity
from model_families.single.planning import PeriodAction, compile_action
from model_families.single.rolling.contracts import WindowRequest
from model_families.single.rolling.controller import observed_state
from model_families.single.runner import serialize_plan
from .contracts import DirectSchema, DirectSignal, OBSERVATION_VERSION
from .observation import DirectObserver, FeatureScales


class DirectProvider:
    def __init__(self,schema,signal,horizon):
        if len(signal.values)!=2*len(schema.hubs) or not 0<=signal.period<horizon:
            raise ValueError('direct price dimension/boundary mismatch')
        self.schema,self.signal,self.horizon=schema,signal,horizon
        self.queries={}

    def marginal_release_value(self,time,hub,supply_slot,online_state_summary):
        if hub not in self.schema.hubs or supply_slot<1 or int(supply_slot)!=supply_slot:
            raise ValueError('invalid release query')
        if not math.isfinite(time) or time<self.signal.period:
            raise ValueError('release precedes decision')
        key=(time,hub,supply_slot)
        if key not in self.queries:
            i=self.schema.hubs.index(hub)
            ell=(time-self.signal.period)/(self.horizon-self.signal.period)
            value=0. if time>=self.horizon else self.schema.scales[i]*(
                (1-ell)*self.signal.values[2*i]+ell*self.signal.values[2*i+1])
            if not math.isfinite(value):raise ValueError('nonfinite direct price')
            self.queries[key]=dict(time=time,hub=hub,supply_slot=supply_slot,value=value,
                direct_price=value,lookup='direct-terminal-zero' if time>=self.horizon else 'actor-direct',
                support_count=None)
        return self.queries[key]['value']


class DirectDecoder:
    def __init__(self,schema,planner_factory=None):
        self.schema,self.planner_factory=schema,planner_factory

    def decode(self,snapshot,observation,signal,checkpoint):
        if (observation.version!=OBSERVATION_VERSION or not isinstance(signal,DirectSignal) or
            (signal.period,signal.revision,signal.observation_hash)!=(observation.period,observation.revision,observation.hash)
            or (snapshot.t,snapshot.revision)!=(signal.period,signal.revision) or observation.hubs!=self.schema.hubs):
            raise RuntimeError('stale/incompatible direct signal')
        from model_families.single.value.route_milp import ValueRouteMILPPlanner
        provider=DirectProvider(self.schema,signal,snapshot.T)
        result=(self.planner_factory or ValueRouteMILPPlanner)(provider,dict(T=snapshot.T,period=snapshot.t)).plan(
            WindowRequest(observed_state(snapshot),snapshot.t,snapshot.t+1,snapshot.t+1,0,checkpoint))
        if result.diagnostics.get('status')!='OPTIMAL':raise RuntimeError('direct solver not OPTIMAL')
        action=result.actions.get(snapshot.t,PeriodAction(snapshot.t))
        info=dict(result.diagnostics);plan=compile_action(snapshot,action,info)
        serial=serialize_plan(plan,snapshot.t,observation.hash,info)
        return DecodeResult(action,plan,serial,identity({k:serial[k] for k in
            ('period','assignments','routes','repositions','expired')}),tuple(provider.queries.values()))


class DirectEpisode(StrategicEpisode):
    def __init__(self,scenario,parameters,solver,schema,scales,timeout_seconds=600.,cancelled=None):
        super().__init__(scenario,parameters,solver,schema,scales,None,timeout_seconds,cancelled,
                         DirectDecoder(schema))
        self._observer=DirectObserver(schema,scales)

    def reset(self):
        import time
        from model_families.single.data import matching_state
        from model_families.single.strategic.reward import LedgerReward
        self._started=time.monotonic();self._check()
        self._sim,self._orders=matching_state(copy.deepcopy(self._scenario),self._parameters,self._solver)
        if self._sim.T<1 or not self._sim.vehs:raise ValueError('invalid direct episode')
        self._schedule={}
        for o in self._orders.values():self._schedule.setdefault(o.book_time,[]).append(o)
        self._history,self._rows,self._records={ },[],[]
        self._reward=LedgerReward();self._final_ledger=None;self.status='DECISION_READY'
        return self._publish()

    def step(self,values):
        if hasattr(self._cancelled,'claim'):self._cancelled.claim()
        obs=self._observation
        signal=values if isinstance(values,DirectSignal) else DirectSignal(tuple(values),obs.period,obs.revision,obs.hash)
        return super().step(signal)

    def probe(self,values):
        if self.status!='DECISION_READY':raise RuntimeError('direct episode not ready')
        self._check()
        if hasattr(self._cancelled,'claim'):self._cancelled.claim()
        obs=self._observation
        result=self._decoder.decode(observed_state(self._sim),obs,
            DirectSignal(tuple(values),obs.period,obs.revision,obs.hash),self._check)
        return dict(action_hash=result.action_hash,plan=copy.deepcopy(result.serialized),coefficients=copy.deepcopy(result.coefficients))


def episode(scenario,cfg,cancelled=None):
    from model_families.single.parameters import MatchingParameters
    from experiment_core.contracts import SolverParameters
    if 'baseline' in cfg:raise ValueError('Direct configuration must contain no baseline')
    return DirectEpisode(scenario,MatchingParameters.model_validate(cfg['parameters']).model_dump(),
        SolverParameters.model_validate(cfg['solver']).model_dump(),DirectSchema(**cfg['action']),
        FeatureScales(**cfg['feature_scales']),cancelled=cancelled)
