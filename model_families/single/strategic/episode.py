"""The sole owner of live physics, hidden schedule and execution cash."""
import copy
import time
from dataclasses import asdict
from ..accounting import completion_bound, cooldown_tick
from ..data import matching_state
from ..rolling.controller import observed_state
from .contracts import EpisodeCancelled, StrategicSignal, Transition, identity
from .decoder import StrategicDecoder
from .observation import PublicObserver
from .reward import LedgerReward


class StrategicEpisode:
    def __init__(self, scenario, parameters, solver, schema, scales, baseline,
                 timeout_seconds=600., cancelled=None, decoder=None):
        self._scenario = copy.deepcopy(scenario)
        self._parameters, self._solver = copy.deepcopy(parameters), copy.deepcopy(solver)
        self._baseline = baseline
        self.schema = schema
        self._observer = PublicObserver(schema, scales)
        self._decoder = decoder or StrategicDecoder(schema, baseline)
        self.timeout_seconds = timeout_seconds
        self._cancelled = cancelled or (lambda: False)
        self.status = "NEW"
        self._records = []

    def _check(self):
        if self._cancelled():
            raise EpisodeCancelled("strategic episode cancelled")
        if time.monotonic()-self._started > self.timeout_seconds:
            raise TimeoutError("strategic episode wall time exhausted")

    def _publish(self):
        arrivals = self._schedule.get(self._sim.t, [])
        self._sim.begin_period(arrivals)
        for o in arrivals:
            key = o.departure, o.destination
            self._history[key] = self._history.get(key, 0)+1
        snapshot = observed_state(self._sim)
        self._observation = self._observer.build(snapshot, self._history, self._baseline)
        return self._observation

    def reset(self):
        self._started = time.monotonic()
        self._check()
        self._sim, self._orders = matching_state(copy.deepcopy(self._scenario),
                                                self._parameters, self._solver)
        if self._sim.T < 1 or not self._sim.vehs:
            raise ValueError("episode needs a positive horizon and nonempty fleet")
        self._baseline.validate(self.schema.hubs, len(self._sim.vehs))
        self._schedule = {}
        for o in self._orders.values():
            self._schedule.setdefault(o.book_time, []).append(o)
        self._history, self._rows, self._records = {}, [], []
        self._reward = LedgerReward()
        self._final_ledger = None
        self.status = "DECISION_READY"
        return self._publish()

    def step(self, values):
        if self.status != "DECISION_READY":
            raise RuntimeError("episode is not ready for a decision")
        try:
            self._check()
            observation = self._observation
            from ..runner import state_snapshot
            before_hash = identity(state_snapshot(self._sim))
            signal = values if isinstance(values, StrategicSignal) else StrategicSignal(
                tuple(values), observation.period, observation.revision, observation.hash)
            snapshot = observed_state(self._sim)
            self.status = "EXECUTING"
            decoded = self._decoder.decode(snapshot, observation, signal, self._check)
            # A planner may mutate its isolated snapshot; recompile on live state
            # validates both physics and the original period/revision guard.
            from ..planning import compile_action
            if (decoded.plan.period, decoded.plan.revision) != (self._sim.t, self._sim.revision):
                raise RuntimeError("decoder changed snapshot period/revision")
            live_info = {}
            live_plan = compile_action(self._sim, decoded.action, live_info)
            if live_info["accounting"] != decoded.serialized["accounting"]:
                raise RuntimeError("snapshot/live compilation accounting mismatch")
            self._sim.commit_plan(live_plan)
            self._rows.append(dict(period=signal.period, plan=decoded.serialized))
            terminal = self._sim.t == self._sim.T
            if terminal:
                self.status = "COOLDOWN"
                cap = completion_bound(self._sim)
                self._sim.pool = {}
                while any(v.phase != "idle" for v in self._sim.vehs):
                    self._check()
                    if self._sim.t >= cap:
                        raise RuntimeError("cooldown bound reached with unfinished commitments")
                    cooldown_tick(self._sim)
            reward, ledger = self._reward.advance(self._sim, self._orders, self._rows, terminal)
            transition_id = identity(dict(index=len(self._rows)-1, observation=observation.hash,
                                          action=decoded.action_hash))
            self._records.append(dict(transition_id=transition_id, period=signal.period,
                observation_hash=observation.hash, signal=asdict(signal), action_hash=decoded.action_hash,
                observation=asdict(observation), before_state_hash=before_hash,
                after_state_hash=identity(state_snapshot(self._sim)),
                coefficients=list(decoded.coefficients), plan=decoded.serialized, reward_USD=reward,
                cumulative_profit_USD=ledger["realized_profit"], ledger=ledger))
            if terminal:
                self._final_ledger = ledger
                self.status = "TERMINATED"
                next_observation = None
            else:
                next_observation = self._publish()
                self.status = "DECISION_READY"
            return Transition(next_observation, reward, terminal, transition_id)
        except EpisodeCancelled:
            self.status = "CANCELLED"
            raise
        except Exception:
            self.status = "FAILED"
            raise

    def probe(self, values):
        """Read-only engineering diagnostics on the current public boundary."""
        if self.status != "DECISION_READY":
            raise RuntimeError("episode is not ready for a probe")
        self._check()
        obs = self._observation
        signal = StrategicSignal(tuple(values), obs.period, obs.revision, obs.hash)
        result = self._decoder.decode(observed_state(self._sim), obs, signal, self._check)
        return dict(action_hash=result.action_hash, plan=copy.deepcopy(result.serialized),
                    coefficients=copy.deepcopy(result.coefficients))

    @property
    def records(self):
        return copy.deepcopy(self._records)

    @property
    def final_ledger(self):
        if self.status != "TERMINATED":
            raise RuntimeError("profit is not settled")
        return copy.deepcopy(self._final_ledger)

    @property
    def metrics(self):
        ledger = self.final_ledger
        return dict(**ledger, **self._sim.S, cooldown_periods=self._sim.t-self._sim.T,
                    episode_seconds=time.monotonic()-self._started,
                    solver_calls=len(self._rows), solver_seconds=sum(
                        row["plan"]["solver"].get("runtime", 0) for row in self._rows),
                    execution_events=[dict(vehicle_id=v.id, segments=copy.deepcopy(v.segments),
                                           pickups=dict(v.pickups), deliveries=dict(v.deliveries))
                                      for v in self._sim.vehs])
