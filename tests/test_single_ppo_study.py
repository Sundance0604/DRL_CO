"""Round-eight safety, larger explicit research pool, replay and crossed statistics."""
import copy
import pytest
import numpy as np
from tests.test_single_strategic_ppo import config, dataset, scenario, setup


def test_nonfinite_KL_after_a_successful_minibatch_is_a_run_failure(monkeypatch):
    import torch
    from model_families.single.learning.ppo import backend
    from model_families.single.learning.ppo.config import PPOConfig
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, obs):
            mean = self.p.expand(len(obs), 1)
            return mean, torch.ones_like(mean), self.p.expand(len(obs))
    network = Tiny()
    count = [0]
    def logp(mean, std, latent):
        count[0] += 1
        return mean.sum(-1)*0+(1000. if count[0] == 3 else 0.)
    monkeypatch.setattr(backend, "squashed_log_probability", logp)
    rows = [dict(observation=None, latent=torch.zeros(1), logp=0., advantage=float(i), **{"return": 1.}) for i in range(2)]
    with pytest.raises(FloatingPointError, match="nonfinite PPO KL"):
        backend.update(network, torch.optim.Adam(network.parameters()), rows,
                       PPOConfig(epochs=1, minibatch_size=1), torch.Generator().manual_seed(1))
    assert count[0] == 3


def test_research_pool_256_is_supported_without_changing_platform_upload_limit():
    from model_families.single.learning.ppo.runner import prepare
    from model_families.single.data import validate_payload
    data = dataset()
    data["scenarios"] = [scenario("train", 10000+i, 200.+i) for i in range(256)]+[scenario("validation", 21000, 100.)]
    with pytest.raises(Exception):
        validate_payload(data)
    _, splits, *_ = prepare(config(), data)
    assert len(splits["train"]) == 256 and len(splits["validation"]) == 1


def test_full_public_observation_action_cash_and_cooldown_replay():
    from experiments.cfs_ppo_strategic_control.study_execution import run_record, replay, run_rh4
    cfg = config()
    original = scenario()
    record = run_record(original, cfg, lambda o: (0.,)*6, "MC")
    assert replay(original, cfg, record)["status"] == "PASSED"
    corrupted = copy.deepcopy(record)
    corrupted["transitions"][0]["observation"]["global_features"] = (999.,)
    with pytest.raises(RuntimeError, match="observation/hash"):
        replay(original, cfg, corrupted)
    # The auxiliary original RH path has its own trace protocol and replay.
    rh = run_rh4(original, cfg)
    assert rh["metrics"]["unfinished_committed"] == 0


def test_crossed_bootstrap_preserves_both_axes_and_paired_estimand():
    from experiments.cfs_ppo_strategic_control.study_analysis import bootstrap_crossed
    x = np.full((5, 80), 12.)
    result = bootstrap_crossed(x, replications=2000)
    assert result["mean_USD"] == 12. and result["marginal_95"] == [12., 12.]
    assert result["bonferroni_99_1667"] == [12., 12.] and result["seed_positive_count"] == 5
    models = np.arange(5)[:, None]*np.ones((1, 80))
    result = bootstrap_crossed(models, replications=2000)
    assert result["marginal_95"][0] < 2 < result["marginal_95"][1]
    assert result == bootstrap_crossed(models, replications=2000)
