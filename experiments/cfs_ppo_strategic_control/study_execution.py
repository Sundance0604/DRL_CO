"""Read-only replay, same-state diagnostics and auxiliary baseline execution."""
import copy
import math
import time
from pathlib import Path
from model_families.single.data import matching_state
from model_families.single.parameters import MatchingParameters
from experiment_core.contracts import SolverParameters
from model_families.single.planning import PeriodAction, VehicleAction, compile_action
from model_families.single.runner import serialize_plan, state_snapshot
from model_families.single.accounting import realized_ledger, cooldown_tick, completion_bound
from model_families.single.rolling.controller import observed_state
from model_families.single.strategic.baseline import FrozenBaseline
from model_families.single.strategic.contracts import ActionSchema, StrategicSignal, identity
from model_families.single.strategic.decoder import ReleaseSignalProvider
from model_families.single.strategic.episode import StrategicEpisode
from model_families.single.strategic.observation import FeatureScales, PublicObserver
from model_families.single.strategic.artifacts import read_json, write_json
from .study_common import CONDITIONS, key, load, now, run_config, data, sha


def action_from_plan(plan):
    orders = {}
    for assignment in plan["assignments"]:
        orders.setdefault(int(assignment["vehicle_id"]), []).append(assignment["order_id"])
    vehicles = []
    for route in plan["routes"]:
        vehicle = int(route["vehicle_id"])
        vehicles.append(VehicleAction(vehicle, {k: v for k, v in route.items() if k != "vehicle_id"},
                                      tuple(orders.get(vehicle, []))))
    for row in plan["repositions"]:
        vehicles.append(VehicleAction(int(row["vehicle_id"]), reposition_to=row["hub"]))
    return PeriodAction(plan["period"], tuple(vehicles))


def physical_hash(plan):
    return identity({k: plan[k] for k in ("period", "assignments", "routes", "repositions", "expired")})


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-10)


def replay(scenario, cfg, record):
    parameters = MatchingParameters.model_validate(cfg["parameters"]).model_dump()
    solver = SolverParameters.model_validate(cfg["solver"]).model_dump()
    sim, orders = matching_state(copy.deepcopy(scenario), parameters, solver)
    base_cfg = cfg["baseline"]
    base = FrozenBaseline(read_json(base_cfg) if isinstance(base_cfg, str) else base_cfg)
    schema, scales = ActionSchema(**cfg["action"]), FeatureScales(**cfg["feature_scales"])
    observer, history, rows, total = PublicObserver(schema, scales), {}, [], 0.
    for index, transition in enumerate(record["transitions"]):
        t = sim.t
        arrivals = [o for o in orders.values() if o.book_time == t]
        sim.begin_period(arrivals)
        for o in arrivals:
            history[o.departure, o.destination] = history.get((o.departure, o.destination), 0)+1
        obs = observer.build(observed_state(sim), history, base)
        if obs.hash != transition["observation_hash"] or identity(state_snapshot(sim)) != transition["before_state_hash"]:
            raise RuntimeError("replay public observation or before-state mismatch")
        if identity(transition["observation"]) != obs.hash:
            raise RuntimeError("recorded observation/hash mismatch")
        signal = StrategicSignal(**transition["signal"])
        provider = ReleaseSignalProvider(base, schema, signal, sim.T)
        for coefficient in transition["coefficients"]:
            value = provider.marginal_release_value(coefficient["time"], coefficient["hub"], coefficient["supply_slot"], {})
            if not close(value, coefficient["value"]):
                raise RuntimeError("replay strategic coefficient mismatch")
        plan_data = transition["plan"]
        info = dict(plan_data["solver"])
        plan = compile_action(sim, action_from_plan(plan_data), info)
        serialized = serialize_plan(plan, t, obs.hash, info)
        if (physical_hash(serialized) != transition["action_hash"] or
                info["accounting"] != plan_data["accounting"] or not close(plan[0], plan_data["operating_profit"])):
            raise RuntimeError("integer action/cash replay mismatch")
        if info["status"] != "OPTIMAL":
            raise RuntimeError("incomplete solve recorded as a completed episode")
        sim.commit_plan(plan)
        rows.append(dict(period=t, plan=serialized))
        terminal = index == len(record["transitions"])-1
        if terminal:
            if sim.t != sim.T:
                raise RuntimeError("incomplete decision sequence")
            bound = completion_bound(sim)
            sim.pool = {}
            while any(v.phase != "idle" for v in sim.vehs):
                if sim.t >= bound:
                    raise RuntimeError("replay cooldown failed")
                cooldown_tick(sim)
        if identity(state_snapshot(sim)) != transition["after_state_hash"]:
            raise RuntimeError("after-state replay mismatch")
        ledger = realized_ledger(sim, orders, rows)
        if any(not close(ledger[k], value) for k, value in transition["ledger"].items()
               if isinstance(value, (int, float))):
            raise RuntimeError("incremental ledger replay mismatch")
        previous = total
        total = ledger["realized_profit"]
        if not close(transition["reward_USD"], total-previous):
            raise RuntimeError("reward replay mismatch")
    if not close(sum(r["reward_USD"] for r in record["transitions"]), record["ledger"]["realized_profit"]):
        raise RuntimeError("reward telescoping failed")
    if record["ledger"] != ledger or ledger["unfinished_committed"]:
        raise RuntimeError("final realized ledger replay mismatch")
    for vehicle in sim.vehs:
        for segment in vehicle.segments:
            if not 0 <= segment["load"] <= parameters["capacity"]:
                raise RuntimeError("replay link capacity failed")
        for oid, pickup in vehicle.pickups.items():
            if pickup < orders[oid].start_time:
                raise RuntimeError("replay pickup readiness failed")
        for oid, delivered in vehicle.deliveries.items():
            if delivered > orders[oid].end_time:
                raise RuntimeError("replay delivery deadline failed")
    return dict(status="PASSED", periods=len(rows), full_profit_USD=total)


def episode(scenario, cfg, base=None):
    baseline = base or FrozenBaseline(read_json(cfg["baseline"]) if isinstance(cfg["baseline"], str) else cfg["baseline"])
    return StrategicEpisode(scenario, MatchingParameters.model_validate(cfg["parameters"]).model_dump(),
                            SolverParameters.model_validate(cfg["solver"]).model_dump(),
                            ActionSchema(**cfg["action"]), FeatureScales(**cfg["feature_scales"]), baseline)


def run_record(scenario, cfg, policy, method):
    ep = episode(scenario, cfg)
    obs = ep.reset()
    policy_seconds = 0.
    while True:
        started = time.perf_counter()
        signal = policy(obs)
        policy_seconds += time.perf_counter()-started
        transition = ep.step(signal)
        if transition.terminated:
            break
        obs = transition.observation
    return dict(scenario_id=scenario["id"], scenario_seed=scenario["seed"], method=method,
                ledger=ep.final_ledger, metrics=ep.metrics, policy_seconds=policy_seconds,
                transitions=ep.records)


def run_rh4(scenario, cfg):
    """Original public-information RH4 formulation, no terminal value/oracle."""
    from experiment_core.plugins import resolve
    from model_families.single.runner import matching
    started = time.perf_counter()
    spec = resolve(dict(name="round8 RH4 auxiliary", family="single", framework_id="rolling",
        dataset=dict(dataset_id="round8", revision=identity(scenario)),
        model=dict(id="single_level_matching", parameters=cfg["parameters"]),
        controller=dict(id="rolling_horizon", parameters=dict(planning_horizon=4, commit_periods=1, completion_extension=0)),
        value_function=dict(id="zero"), solver=dict(backend="gurobi", parameters=cfg["solver"]),
        evaluation=dict(accounting_version="single-realized-v1", terminal_policy="cooldown_realized", information_set="online")))
    def check():
        if time.perf_counter()-started > 600:
            raise TimeoutError("RH4 episode timeout")
    metrics, trace = matching(spec, scenario, 0, check)
    if metrics["unfinished_committed"] or any(not all(row["invariants"].values()) for row in trace):
        raise RuntimeError("RH4 execution audit failed")
    if any(w["status"] != "OPTIMAL" for w in metrics["planning_windows"]):
        raise RuntimeError("RH4 solver did not finish optimally")
    result = dict(scenario_id=scenario["id"], scenario_seed=scenario["seed"], method="RH4", metrics=metrics,
                ledger={k: metrics[k] for k in ("realized_profit", "completed_fare", "realized_cost", "unfinished_committed")},
                episode_seconds=time.perf_counter()-started, original_trace=trace)
    replay_rh4(scenario, cfg, result)
    return result


def replay_rh4(scenario, cfg, record):
    sim, orders = matching_state(copy.deepcopy(scenario), MatchingParameters.model_validate(cfg["parameters"]).model_dump(),
                                SolverParameters.model_validate(cfg["solver"]).model_dump())
    rows = []
    for row in record["original_trace"]:
        if row.get("drain"):
            sim.pool = {}
            if identity(state_snapshot(sim)) != identity(row["before"]):
                raise RuntimeError("RH4 cooldown before-state mismatch")
            cooldown_tick(sim)
        else:
            sim.begin_period([o for o in orders.values() if o.book_time == sim.t])
            if identity(state_snapshot(sim)) != identity(row["before"]):
                raise RuntimeError("RH4 before-state mismatch")
            plan = compile_action(sim, action_from_plan(row["plan"]), {})
            if not close(plan[0], row["plan"]["operating_profit"]):
                raise RuntimeError("RH4 integer cash mismatch")
            sim.commit_plan(plan)
            rows.append(row)
        if identity(state_snapshot(sim)) != identity(row["after"]):
            raise RuntimeError("RH4 after-state mismatch")
    ledger = realized_ledger(sim, orders, rows)
    for field, value in record["ledger"].items():
        if not close(ledger[field], value):
            raise RuntimeError("RH4 full ledger mismatch")
    for row in record["original_trace"]:
        reconstructed = realized_ledger(sim, orders, rows, row["period"]+1)
        if reconstructed != row["realized_ledger"]:
            raise RuntimeError("RH4 incremental ledger mismatch")
    return dict(status="PASSED", periods=len(rows))


def engineering(attempt):
    from model_families.single.value.route_milp import ValueRouteMILPPlanner
    from model_families.single.value.providers import MCTableValueProvider
    from model_families.single.value.contracts import ZeroValueProvider
    from model_families.single.rolling.contracts import WindowRequest
    target, manifest = load(attempt)
    summaries = []
    for c in CONDITIONS:
        name, cfg = key(c), run_config(target, c, 29999)
        output = target/"preflight"/name
        output.mkdir(exist_ok=False)
        for scenario in data(target, c, "engineering")["scenarios"]:
            started = time.perf_counter()
            zero = (0.,)*10
            reference_table = read_json(cfg["baseline"])
            for method in ("MC", "M0"):
                local = copy.deepcopy(cfg)
                if method == "M0":
                    local["baseline"] = dict(method="zero")
                ep = episode(scenario, local)
                obs = ep.reset()
                reference, orders = matching_state(copy.deepcopy(scenario), ep._parameters, ep._solver)
                ref_rows, probes = [], []
                provider = MCTableValueProvider(reference_table) if method == "MC" else ZeroValueProvider()
                for t in range(scenario["horizon"]):
                    reference.begin_period([o for o in orders.values() if o.book_time == t])
                    solution = ValueRouteMILPPlanner(provider, dict(T=reference.T)).plan(WindowRequest(
                        observed_state(reference), t, t+1, t+1, 0, ep._check))
                    info = dict(solution.diagnostics)
                    plan = compile_action(reference, solution.actions.get(t, PeriodAction(t)), info)
                    serialized = serialize_plan(plan, t, obs.hash, info)
                    reference.commit_plan(plan)
                    ref_rows.append(dict(period=t, plan=serialized))
                    if method == "MC":
                        baseline_probe = ep.probe(zero)
                        action_changed = coefficient_changed = 0
                        hashes = []
                        for axis in range(10):
                            for sign in (-1., 1.):
                                signal = list(zero)
                                signal[axis] = sign
                                probe = ep.probe(signal)
                                hashes.append(probe["action_hash"])
                                action_changed += probe["action_hash"] != baseline_probe["action_hash"]
                                coefficient_changed += any(abs(row["residual"])>1e-9 for row in probe["coefficients"])
                        probes.append(dict(period=t, action_changed=action_changed,
                            coefficient_changed=coefficient_changed, signal_changed=20, action_hashes=hashes))
                    tr = ep.step(zero)
                    actual = ep.records[-1]
                    if actual["action_hash"] != physical_hash(serialized):
                        raise RuntimeError("zero residual differs from identical parent MILP")
                    for field in ("assignments", "routes", "repositions", "expired", "accounting"):
                        if actual["plan"][field] != serialized[field]:
                            raise RuntimeError("zero residual action/cash differs from parent")
                    if not tr.terminated and actual["after_state_hash"] != identity(state_snapshot(reference)):
                        raise RuntimeError("zero residual state transition differs from parent")
                    obs = tr.observation
                reference.pool = {}
                cap = completion_bound(reference)
                while any(v.phase != "idle" for v in reference.vehs):
                    if reference.t >= cap:
                        raise RuntimeError("reference cooldown failed")
                    cooldown_tick(reference)
                if ep.final_ledger != realized_ledger(reference, orders, ref_rows):
                    raise RuntimeError("zero residual entire ledger differs from parent")
                record = dict(scenario_id=scenario["id"], method=method, ledger=ep.final_ledger,
                              metrics=ep.metrics, transitions=ep.records, probes=probes)
                result = replay(scenario, local, record)
                write_json(output/f"{scenario['seed']}-{method}.json", record)
                summaries.append(dict(condition=name, scenario_seed=scenario["seed"], method=method,
                    action_changes=sum(p["action_changed"] for p in probes),
                    coefficient_changes=sum(p["coefficient_changed"] for p in probes),
                    probes=20*len(probes), elapsed_seconds=time.perf_counter()-started, replay=result))
            hidden = copy.deepcopy(scenario)
            hidden["id"] = "hidden-information-test"
            for order in hidden["orders"]:
                if order["book_time"] > 0:
                    order["revenue"] += 1234567.
                    order["end_time"] += 77
            a, b = episode(scenario, cfg), episode(hidden, cfg)
            if a.reset() != b.reset() or a.probe(zero)["action_hash"] != b.probe(zero)["action_hash"]:
                raise RuntimeError("hidden future leaked into observation/decision")
            print("Engineering passed", name, scenario["seed"], flush=True)
        if not sum(s["action_changes"] for s in summaries if s["condition"] == name):
            raise RuntimeError("no engineering controllability in condition "+name)
    write_json(target/"qa/engineering.json", dict(status="PASSED", at=now(), rows=summaries))
    manifest.update(status="ENGINEERING_PASSED", engineering_at=now())
    write_json(target/"manifest.json", manifest)
    return summaries
