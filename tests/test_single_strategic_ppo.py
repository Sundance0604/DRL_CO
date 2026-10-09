"""Independent synthetic engineering checks; no historical experiment outputs."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from model_families.single.parameters import MatchingParameters
from experiment_core.contracts import SolverParameters
from model_families.single.data import matching_state
from model_families.single.planning import PeriodAction, compile_action
from model_families.single.rolling.contracts import WindowRequest, WindowSolution
from model_families.single.rolling.controller import observed_state
from model_families.single.strategic.baseline import FrozenBaseline
from model_families.single.strategic.contracts import ActionSchema, StrategicSignal, EpisodeCancelled
from model_families.single.strategic.decoder import ReleaseSignalProvider, StrategicDecoder
from model_families.single.strategic.episode import StrategicEpisode
from model_families.single.strategic.observation import FeatureScales, PublicObserver, free_release


def order(oid="late", book=2, fare=100., a="A", b="C", load=1):
    return dict(id=oid, departure=a, destination=b, passenger=load, book_time=book,
                start_time=book, end_time=20, revenue=fare, penalty=1.)


def scenario(split="train", seed=1, fare=100.):
    return dict(id=f"synthetic-{seed}", seed=seed, split=split, horizon=3,
                nodes=["A", "B", "C"], edges=[["A", "B", 2.], ["B", "C", 2.]],
                hubs=["A"], orders=[order(fare=fare)])


def config(updates=1):
    return dict(parameters=dict(speed=1., capacity=2, loaded_cost=2., empty_cost=2.,
                waiting_cost=1., expiry_cost=3., max_delay=0),
                solver=dict(threads=1, mip_gap=0., seed=0),
                action=dict(hubs=["A", "B", "C"], scales=[10., 10., 10.]),
                feature_scales=dict(money=100., distance=4., time=3., load=2.),
                baseline=dict(method="zero"),
                ppo=dict(updates=updates, episodes_per_update=1, epochs=1, hidden_size=8,
                         minibatch_size=3, training_seed=27, reward_scale=100.))


def setup(data=None, baseline=None, decoder=None, cancelled=None):
    cfg = config()
    p = MatchingParameters.model_validate(cfg["parameters"]).model_dump()
    solver = SolverParameters.model_validate(cfg["solver"]).model_dump()
    schema = ActionSchema(**cfg["action"])
    scale = FeatureScales(**cfg["feature_scales"])
    base = baseline or FrozenBaseline(cfg["baseline"])
    return StrategicEpisode(data or scenario(), p, solver, schema, scale, base,
                            decoder=decoder, cancelled=cancelled)


def table():
    return dict(method="table", values={h: {"1": dict(times=[0, 2], values=[3., -4.])}
                                          for h in ("A", "B", "C")})


def dataset():
    return dict(schema_version="dataset/v1", family="single_level_matching", provenance={},
                scenarios=[scenario("train", 101, 100.), scenario("validation", 201, 101.),
                           scenario("test", 301, 102.)])


def test_imports_remain_lazy():
    script = "import sys; import model_families.single.strategic; import model_families.single.learning.ppo.runner; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


def test_signal_freezes_queries_negative_prices_and_terminal():
    base = FrozenBaseline(table())
    schema = ActionSchema(("A", "B", "C"), (10., 10., 10.))
    signal = StrategicSignal((-1., -.5, 0., 0., 0., 0.), 0, 1, "x")
    provider = ReleaseSignalProvider(base, schema, signal, 3)
    assert provider.marginal_release_value(0, "A", 1, {}) == -7.
    assert provider.marginal_release_value(1, "A", 1, {}) == pytest.approx(-.5-10*(2/3+.5/3))
    assert provider.marginal_release_value(1, "A", 1, {}) == provider.marginal_release_value(1, "A", 1, {})
    assert provider.marginal_release_value(3, "A", 1, {}) == 0.
    assert len(provider.queries) == 3
    with pytest.raises(ValueError):
        StrategicSignal((float("nan"),), 0, 0, "")
    with pytest.raises(ValueError):
        StrategicSignal((1.001,), 0, 0, "")
    with pytest.raises(ValueError, match="missing MC"):
        base.validate(schema.hubs, 2)
    original = table()
    frozen = FrozenBaseline(original)
    original["values"]["A"]["1"]["values"][0] = 999
    assert frozen.query(0, "A", 1, 3)[0] == 3


def test_future_information_and_scenario_identity_do_not_affect_current_decision():
    original = scenario()
    changed = copy.deepcopy(original)
    changed.update(id="unrelated", seed=444)
    changed["orders"][0].update(revenue=99999., destination="B", end_time=100)
    changed["orders"].append(order("hidden", book=1, fare=1234.))
    a, b = setup(original), setup(changed)
    oa, ob = a.reset(), b.reset()
    assert oa == ob and oa.hash == ob.hash
    assert a.probe((0.,)*6)["action_hash"] == b.probe((0.,)*6)["action_hash"]
    with pytest.raises(Exception):
        oa.hubs += ("D",)
    assert not hasattr(oa, "all_orders")


def test_zero_signal_matches_parent_table_planner_actions_and_entire_cash_trajectory():
    from model_families.single.value.providers import MCTableValueProvider
    from model_families.single.value.route_milp import ValueRouteMILPPlanner
    from model_families.single.accounting import realized_ledger, cooldown_tick
    from model_families.single.runner import serialize_plan
    data = scenario()
    data["orders"] = [order("now", book=0), order("later", book=2, fare=50., a="C", b="A")]
    cfg = config()
    ep = setup(data, FrozenBaseline(table()))
    sim, orders = matching_state(data, MatchingParameters.model_validate(cfg["parameters"]).model_dump(),
                                SolverParameters.model_validate(cfg["solver"]).model_dump())
    obs = ep.reset()
    rows, profits = [], []
    for t in range(sim.T):
        sim.begin_period([o for o in orders.values() if o.book_time == t])
        solution = ValueRouteMILPPlanner(MCTableValueProvider(table()), {"T": sim.T}).plan(
            WindowRequest(observed_state(sim), t, t+1, t+1, 0, lambda: None))
        info = dict(solution.diagnostics)
        plan = compile_action(sim, solution.actions.get(t, PeriodAction(t)), info)
        serialized = serialize_plan(plan, t, "", info)
        sim.commit_plan(plan)
        rows.append(dict(period=t, plan=serialized))
        tr = ep.step((0.,)*6)
        actual = ep.records[-1]["plan"]
        for key in ("assignments", "routes", "repositions", "expired", "accounting"):
            assert actual[key] == serialized[key]
        assert actual["solver"]["window_objective"] == serialized["solver"]["window_objective"]
        if t < sim.T-1:
            profits.append(realized_ledger(sim, orders, rows)["realized_profit"])
            assert ep.records[-1]["cumulative_profit_USD"] == profits[-1]
    sim.pool = {}
    while any(v.phase != "idle" for v in sim.vehs):
        cooldown_tick(sim)
    assert ep.final_ledger == realized_ledger(sim, orders, rows)


def test_cooldown_reward_telescope_and_reset_are_independent():
    ep = setup()
    ep.reset()
    rewards = []
    for _ in range(3):
        rewards.append(ep.step((0.,)*6).reward)
    assert ep.status == "TERMINATED"
    assert ep.final_ledger["completed_fare"] == 100.
    assert ep.final_ledger["cutoff_exclusive"] > 3
    assert sum(rewards) == pytest.approx(ep.final_ledger["realized_profit"])
    assert ep.final_ledger["unfinished_committed"] == 0
    assert ep.final_ledger["realized_profit"] == 90.  # 4 miles * 2, plus 2 idle periods
    with pytest.raises(RuntimeError, match="not ready"):
        ep.step((0.,)*6)
    old = ep.records
    obs = ep.reset()
    assert obs.period == 0 and ep.records == []
    old[0]["ledger"]["realized_profit"] = 100000
    assert setup().reset() == obs


def test_midroute_release_and_remaining_capacity_are_encoded_without_double_arrival():
    data = scenario()
    data["orders"] = [order("now", book=0)]
    ep = setup(data)
    obs = ep.reset()
    ep.step((0.,)*6)
    tr = ep.step((0.,)*6)
    assert tr.observation.period == 2
    v = ep._sim.vehs[0]
    assert v.loc == "B" and v.route == ["C"] and v.arrive == 2
    assert free_release(observed_state(ep._sim), v) == ("C", 4)
    vehicle = tr.observation.vehicles[0]
    assert len(vehicle.arcs) == 1 and vehicle.arcs[0][-1] == .5
    assert len(vehicle.commitments) == 1


def test_stale_signal_and_failure_do_not_commit_or_terminate():
    ep = setup()
    obs = ep.reset()
    with pytest.raises(RuntimeError, match="stale"):
        ep.step(StrategicSignal((0.,)*6, 0, obs.revision+1, obs.hash))
    assert ep.status == "FAILED" and ep._sim.t == 0
    class Limited:
        def __init__(self, *args):
            pass
        def plan(self, request):
            return WindowSolution({}, {"status": "FEASIBLE_LIMIT"})
    schema = ActionSchema(**config()["action"])
    base = FrozenBaseline({"method": "zero"})
    ep = setup(decoder=StrategicDecoder(schema, base, Limited))
    ep.reset()
    with pytest.raises(RuntimeError, match="OPTIMAL"):
        ep.step((0.,)*6)
    assert ep.status == "FAILED" and ep._sim.t == 0
    cancel = [False]
    ep = setup(cancelled=lambda: cancel[0])
    ep.reset()
    cancel[0] = True
    with pytest.raises(EpisodeCancelled):
        ep.step((0.,)*6)
    assert ep.status == "CANCELLED"


def test_set_encoding_is_invariant_to_vehicle_order_and_keeps_gradients():
    import torch
    from model_families.single.learning.ppo.networks import PolicyNetwork
    data = scenario()
    data["hubs"] = ["A", "B"]
    data["orders"] = [order("one", book=0), order("two", book=0, a="B", b="C")]
    ep = setup(data)
    obs = ep.reset()
    reordered = replace(obs, orders=tuple(reversed(obs.orders)), vehicles=tuple(reversed(obs.vehicles)))
    net = PolicyNetwork(3, 8)
    assert torch.allclose(net.encode(obs), net.encode(reordered), atol=1e-7)
    net([obs])[2].sum().backward()
    assert net.order_encoder.layers[0].weight.grad is not None
    assert torch.isfinite(net.order_encoder.layers[0].weight.grad).all()


def test_squashed_logp_matches_transformed_normal_and_saturated_gradients():
    import torch
    from model_families.single.learning.ppo.networks import squashed_log_probability
    mean, std = torch.tensor([[.2, -.3]]), torch.tensor([[.8, .4]])
    z = torch.tensor([[.7, -.8]])
    dist = torch.distributions.TransformedDistribution(torch.distributions.Normal(mean, std),
                                                       torch.distributions.TanhTransform(cache_size=1))
    assert torch.allclose(squashed_log_probability(mean, std, z), dist.log_prob(z.tanh()).sum(-1))
    extreme = torch.tensor([[100., -100.]], requires_grad=True)
    logp = squashed_log_probability(mean, std, extreme)
    logp.sum().backward()
    assert torch.isfinite(logp).all() and torch.isfinite(extreme.grad).all()


def test_gae_uses_zero_only_at_true_full_episode_end():
    from model_families.single.learning.ppo.backend import advantages
    adv, returns = advantages([1., 2., 10.], [3., 4., 5.], gae_lambda=1.)
    assert returns == [13., 12., 10.] and adv == [10., 8., 5.]


def test_residual_can_change_physical_action_and_probes_do_not_mutate_live_state():
    data = scenario()
    data["orders"] = [order("short", book=0, fare=10., b="B", load=2),
                      order("long", book=0, fare=15., load=2)]
    ep = setup(data)
    obs = ep.reset()
    baseline = ep.probe((0.,)*6)
    assert baseline["plan"]["assignments"][0]["order_id"] == "long"
    correction = ep.probe((0., 0., 1., 1., 0., 0.))
    assert correction["plan"]["assignments"][0]["order_id"] == "short"
    assert baseline["action_hash"] != correction["action_hash"]
    assert ep._sim.t == obs.period and ep._sim.revision == obs.revision
    assert ep.records == [] and len(ep._sim.pool) == 2


def test_checkpoint_training_resume_and_formal_evaluation(tmp_path):
    import torch
    from model_families.single.learning.ppo.runner import train_run, evaluate_run
    from model_families.single.learning.ppo.checkpoints import load_actor, inspect_checkpoint
    from model_families.single.strategic.artifacts import read_json, file_hash
    full, initial, resumed = tmp_path/"full", tmp_path/"initial", tmp_path/"resumed"
    train_run(config(2), dataset(), full)
    train_run(config(1), dataset(), initial)
    train_run(config(2), dataset(), resumed, initial/"update-0001")
    a, *_ = load_actor(full/"last-inference")
    b, *_ = load_actor(resumed/"last-inference")
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert read_json(resumed/"best.json")["checkpoint"] == "best-inference"
    assert not (resumed/"best-inference"/"recovery.pt").exists()
    before = {p.name: file_hash(p) for p in (resumed/"best-inference").iterdir()}
    results = evaluate_run(resumed/"best-inference", dataset(), tmp_path/"evaluation")
    assert len(results) == 1 and results[0]["ledger"]["unfinished_committed"] == 0
    assert before == {p.name: file_hash(p) for p in (resumed/"best-inference").iterdir()}
    from model_families.single.learning.ppo.policy import FrozenPolicy
    policy = FrozenPolicy(resumed/"best-inference")
    obs = setup().reset()
    with pytest.raises(ValueError, match="hub order"):
        policy.act(replace(obs, hubs=tuple(reversed(obs.hubs))))
    manifest = read_json(resumed/"best-inference"/"checkpoint.json")
    manifest["protocols"]["family"] = "legacy"
    (resumed/"best-inference"/"checkpoint.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="family"):
        inspect_checkpoint(resumed/"best-inference")


def test_split_overlap_and_resume_configuration_are_rejected(tmp_path):
    from model_families.single.learning.ppo.runner import prepare, train_run
    from model_families.single.strategic.artifacts import read_json
    data = dataset()
    data["scenarios"][2]["seed"] = data["scenarios"][0]["seed"]
    with pytest.raises(ValueError, match="overlap"):
        prepare(config(), data)
    train_run(config(), dataset(), tmp_path/"initial")
    changed = config(2)
    changed["action"]["hubs"].reverse()
    with pytest.raises(ValueError, match="identity"):
        train_run(changed, dataset(), tmp_path/"bad-resume", tmp_path/"initial"/"update-0001")
    assert read_json(tmp_path/"bad-resume"/"status.json")["status"] == "FAILED"


def test_checkpoint_tamper_and_path_escape_are_rejected(tmp_path):
    from model_families.single.learning.ppo.runner import train_run
    from model_families.single.learning.ppo.checkpoints import inspect_checkpoint
    from model_families.single.strategic.artifacts import contained_file
    train_run(config(), dataset(), tmp_path/"train")
    package = tmp_path/"train"/"last-inference"
    (package/"baseline.json").write_text('{"method": "table"}')
    with pytest.raises(ValueError, match="hash"):
        inspect_checkpoint(package)
    outside = tmp_path/"outside.json"
    outside.write_text("{}")
    with pytest.raises(ValueError, match="escapes"):
        contained_file(package, "../../outside.json")
