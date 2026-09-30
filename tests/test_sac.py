import random

import numpy as np
import pytest
import torch

from sac_agent import MultiOrderSAC, masked_softmax


def test_masked_softmax_masks_and_normalizes():
    logits = torch.tensor([[1000.0, -1000.0, 3.0]])
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    probabilities = masked_softmax(logits, mask)
    assert probabilities[0, 0].item() == 0.0
    assert probabilities.sum().item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least one valid action"):
        masked_softmax(logits, torch.zeros_like(mask))


def test_dynamic_orders_are_joined_by_id():
    agent = MultiOrderSAC("cpu", 2, 2, 3, HIDDEN_DIM=16, batch_size=2)
    agent.sac_add_order_transitions(
        [0.1, 0.2], [10, 20], [[1.0, 0.0], [0.0, 1.0]],
        [[1, 1, 0], [1, 0, 1]], [1, 2], {10: 1.0, 20: 2.0},
        [0.3, 0.4], [20], [[0.5, 0.5]], [[1, 1, 0]],
    )
    assert len(agent.replay) == 2
    first, second = agent.replay.buf
    assert first[4] == 1.0
    assert second[4] == 0.0
    assert torch.allclose(second[3], torch.tensor([0.3, 0.4, 0.5, 0.5]))


def test_sac_learns_action_dependent_bandit():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    agent = MultiOrderSAC(
        "cpu", 1, 1, 3, HIDDEN_DIM=32, batch_size=32, learning_starts=32,
        actor_lr=3e-3, critic_lr=3e-3, auto_alpha=False,
    )
    with torch.no_grad():
        agent.log_alpha.fill_(-4.0)
    mask = [1, 1, 1]
    state = [0.0, 0.0]
    for index in range(300):
        action = index % 3
        reward = 5.0 if action == 1 else 0.0
        agent.add_transition(state, action, reward, state, True, mask, mask)
    for _ in range(180):
        agent.update_sac()
    actions, _, probabilities, _ = agent.take_action_vehicle([0.0], [[0.0]], [mask], explore=False)
    assert actions == [1]
    assert probabilities[0, 1].item() > 0.8
