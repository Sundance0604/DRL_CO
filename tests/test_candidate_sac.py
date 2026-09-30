import numpy as np
import torch

from drl_co.rl.candidate_sac import CandidateSAC, CandidateScorer
from drl_co.simulation.scenarios import generate_scenario
from experiments.evaluation.evaluate import build_actor_candidate_mask, standardize_action_values
from experiments.training.train_candidate import curriculum_level, solve_supply_counterfactual


def test_candidate_scorer_is_permutation_equivariant():
    torch.manual_seed(0)
    scorer = CandidateScorer(feature_dim=5, hidden_dim=16).eval()
    state = torch.randn(3, 7, 5)
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    original = scorer(state)
    permuted = scorer(state[:, permutation])
    torch.testing.assert_close(permuted, original[:, permutation])


def test_sequential_decisions_reserve_supply_for_later_orders():
    torch.manual_seed(1)
    agent = CandidateSAC("cpu", feature_dim=20, hidden_dim=16, batch_size=2)
    states = np.zeros((2, 3, 20), dtype=np.float32)
    states[:, :, 0] = 0.5
    states[:, :, 1] = [[0.1], [0.2]]
    states[:, :, 6] = 1.0
    mask = np.ones((2, 3), dtype=np.bool_)
    actions, _, _, _, decision_states = agent.take_action_candidates(
        states, mask, explore=False, greedy=True, sequential=True
    )
    assert decision_states[1, actions[0], 6] == 0.5


def test_behavior_cloning_learns_masked_expert_action():
    torch.manual_seed(2)
    agent = CandidateSAC("cpu", feature_dim=4, hidden_dim=16, batch_size=4)
    states = np.zeros((32, 3, 4), dtype=np.float32)
    states[:, 1, 0] = 1.0
    masks = np.ones((32, 3), dtype=np.bool_)
    actions = np.ones(32, dtype=np.int64)
    info = agent.behavior_clone(states, masks, actions, epochs=20, batch_size=16)
    assert info["bc_accuracy"] > 0.95


def test_action_value_standardization_uses_only_valid_actions():
    values = np.asarray([[1.0, 2.0, 100.0], [4.0, 4.0, 9.0]])
    mask = np.asarray([[True, True, False], [True, True, False]])
    standardized = standardize_action_values(values, mask)
    np.testing.assert_allclose(standardized[0], [-1.0, 1.0, 0.0])
    np.testing.assert_allclose(standardized[1], [0.0, 0.0, 0.0])


def test_pressure_curriculum_oversamples_scarce_regimes():
    levels = [curriculum_level(index)[0] for index in range(6)]
    assert levels == ["normal", "moderate", "high", "extreme", "high", "extreme"]


def test_actor_candidate_pruning_keeps_ranked_and_structural_anchors():
    values = np.asarray([[9.0, 8.0, 1.0, 0.0]])
    legal = np.asarray([[True, True, True, True]])
    eligible = np.asarray([[True, True, True, False]])
    pruned, info = build_actor_candidate_mask(
        values,
        legal,
        departures=[3],
        supply_by_city=np.asarray([1.0, 2.0, 10.0, 0.0]),
        eligible_mask=eligible,
        top_k=1,
        entropy_threshold=2.0,
        probability_margin=-1.0,
    )
    np.testing.assert_array_equal(pruned[0], [True, False, True, True])
    assert info["candidate_total"] == 3
    assert info["legal_total"] == 4


def test_candidate_actor_values_expose_logits_without_actions():
    torch.manual_seed(3)
    agent = CandidateSAC("cpu", feature_dim=4, hidden_dim=16, batch_size=2)
    states = np.zeros((2, 3, 4), dtype=np.float32)
    assert agent.candidate_actor_values(states).shape == (2, 3)


def test_supply_counterfactual_does_not_mutate_live_state():
    vehicles, all_orders, graph = generate_scenario(
        91, horizon=1, num_cities=5, num_vehicles=4, orders_per_step=3
    )
    active_orders = {
        order_id: order for order_id, order in all_orders.items()
        if order.start_time == 0
    }
    vehicle_snapshot = {
        vehicle_id: (vehicle.time, vehicle.decision, vehicle.intercity)
        for vehicle_id, vehicle in vehicles.items()
    }
    order_snapshot = {
        order_id: (order.matched, order.virtual_departure)
        for order_id, order in active_orders.items()
    }
    costs = np.tile(np.asarray([10, 1, 3, 10], dtype=float), (len(vehicles), 1))
    result = solve_supply_counterfactual(
        graph, vehicles, active_orders, 0, 7, costs, cancel_penalty=300.0
    )
    assert set(result["outcomes"]) == set(active_orders)
    assert np.isfinite(result["objective"])
    assert vehicle_snapshot == {
        vehicle_id: (vehicle.time, vehicle.decision, vehicle.intercity)
        for vehicle_id, vehicle in vehicles.items()
    }
    assert order_snapshot == {
        order_id: (order.matched, order.virtual_departure)
        for order_id, order in active_orders.items()
    }
