"""H/k receding-horizon control, independent of the planning algorithm.

Inspired by BHH's window/frozen-prefix mechanism, not its business model.
No future scenario records are passed to planners. Existing commitments live in
the simulator; discarded window suffixes never become real vehicle state.
"""
import copy
from types import SimpleNamespace
from ..engine.demand import ready_time
from ..planning import PeriodAction, compile_action
from .contracts import WindowRequest, WindowSolution


def observed_order(o, t):
    fields=("id","departure","destination","passenger","book_time","end_time","revenue","penalty")
    return SimpleNamespace(**{k:copy.deepcopy(getattr(o,k)) for k in fields},start_time=ready_time(o,t))


def observed_state(sim):
    state=sim.clone()
    # Do not share mutable order objects with the live simulator or expose batch
    # outcomes that are only revealed at a future cutoff.
    def observed(o):
        return observed_order(o,sim.t)
    state.pool={oid:observed(o) for oid,o in sim.pool.items()}
    for v in state.vehs:
        v.orders={oid:observed(o) for oid,o in v.orders.items()}
    # A planner may change its input snapshot; it may not change the live graph.
    state.net=copy.deepcopy(sim.net)
    return state


class RollingController:
    information_mode = "known_supply"

    def build_request(self, sim, end, commit):
        return WindowRequest(observed_state(sim),sim.t,end,commit,
                             self.parameters["completion_extension"],self.checkpoint)

    def __init__(self, parameters, checkpoint, planner=None):
        from .registry import create_planner
        self.parameters=parameters
        self.checkpoint=checkpoint
        self.planner=planner if planner is not None else create_planner(parameters["planner_id"])
        self.commit_end=-1
        self.solution=None
        self.windows=[]

    def decide(self, sim, info):
        self.checkpoint()
        if not 0<=sim.t<sim.T:
            raise RuntimeError("rolling decisions must stop at T")
        replan=self.solution is None or sim.t>=self.commit_end
        if replan:
            end=min(sim.t+self.parameters["planning_horizon"],sim.T)
            commit=min(sim.t+self.parameters["commit_periods"],end)
            request=self.build_request(sim,end,commit)
            solution=self.planner.plan(request)
            if not isinstance(solution,WindowSolution):
                raise RuntimeError("planner must return WindowSolution")
            for period,action in solution.actions.items():
                if not request.start<=period<request.decision_end or action.period!=period:
                    raise RuntimeError("planner action outside decision window")
            self.solution,self.commit_end=solution,commit
            metadata=dict(solution.diagnostics,window_start=sim.t,decision_end=end,commit_end=commit,
                          planner_id=self.parameters["planner_id"],visible_orders=sorted(sim.pool),
                          information_mode=self.information_mode,
                          preview_order_ids=sorted(o.id for o in request.preview_orders),
                          information=("ORACLE frozen future bookings; NOT deployable" if self.information_mode=='oracle_preview' else
                                       "currently booked orders only; unrevealed ready time uses current upper bound"))
            self.windows.append(metadata)
        info.update(controller="oracle_preview" if self.information_mode=='oracle_preview' else "rolling_horizon",replanned=replan,
                    status=self.windows[-1].get("status","WINDOW_PLAN") if replan else "FROZEN_PREFIX",
                    runtime=self.windows[-1].get("runtime",0) if replan else 0,
                    candidate_route_count=self.windows[-1].get("candidate_route_count",0) if replan else 0,
                    window=self.windows[-1])
        action=self.solution.actions.get(sim.t,PeriodAction(sim.t))
        return compile_action(sim,action,info)


class OraclePreviewController(RollingController):
    """Information-relaxation diagnostic, explicitly NOT a deployable policy."""
    information_mode = "oracle_preview"

    def __init__(self,parameters,checkpoint,preview_source,planner=None):
        if parameters["commit_periods"] != 1:
            raise ValueError("oracle_preview must replan every period")
        super().__init__(parameters,checkpoint,planner)
        self.preview_source=preview_source

    def build_request(self,sim,end,commit):
        # H is the existing platform's number of decision periods, [t,t+H).
        # Reveal through t+H inclusive as requested. The boundary booking has no
        # dispatch slot in this window and cannot be served early.
        preview=tuple(observed_order(o,sim.t) for o in
                      self.preview_source(sim.t,sim.t+self.parameters["planning_horizon"]))
        if any(not sim.t<o.book_time<=sim.t+self.parameters["planning_horizon"] for o in preview):
            raise RuntimeError("oracle preview outside declared booking interval")
        return WindowRequest(observed_state(sim),sim.t,end,commit,
                             self.parameters["completion_extension"],self.checkpoint,preview)
