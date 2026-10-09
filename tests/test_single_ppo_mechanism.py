"""Check faithful public-state restoration, validation constants and fresh tests."""
import copy
from dataclasses import asdict
import numpy as np
import pytest
from experiments.cfs_ppo_strategic_control.mechanism.run import restore_observation, constant_signal, fresh_guard, paired_statistics
from model_families.single.learning.ppo.runner import scenario_content
from tests.test_single_strategic_ppo import setup
from model_families.single.strategic.artifacts import write_json, read_json


def test_saved_public_observation_roundtrip_and_tampering():
    ep = setup()
    obs = ep.reset()
    restored = restore_observation(asdict(obs), obs.hash)
    assert restored == obs
    altered = copy.deepcopy(asdict(obs))
    altered["global_features"] = (999., *altered["global_features"][1:])
    with pytest.raises(RuntimeError, match="hash"):
        restore_observation(altered, obs.hash)


def test_fixed_signal_preserves_each_hub_near_far_and_equal_weights():
    signals = [[.2, -.5, .9, 0.], [.4, .5, -.1, -.6]]
    np.testing.assert_allclose(constant_signal(signals, 4), [.3, 0., .4, -.3])
    assert isinstance(constant_signal(signals, 4), tuple)


def test_fixed_signal_rejects_empty_nonfinite_and_wrong_dimensions():
    for signals, dimensions in (([], 2), ([[float("nan"), 0.]], 2), ([[1.1, 0.]], 2), ([[0.,0.]], 4)):
        with pytest.raises(ValueError):
            constant_signal(signals, dimensions)


def test_fresh_tests_reject_relabelled_prior_content_and_seed():
    original = dict(id="old", seed=22, split="test", payload=[1,2,3])
    used = [dict(id="old",seed=22,content_hash=scenario_content(original))]
    relabelled = dict(original,id="new",seed=999)
    with pytest.raises(RuntimeError, match="overlaps"):
        fresh_guard([relabelled],used)
    new = dict(original,id="new",payload=[4,5,6])
    with pytest.raises(RuntimeError, match="overlaps"):
        fresh_guard([new],used)
    new["seed"] = 23
    fresh_guard([new],used)


def test_mechanism_crossed_interval_recalculation_with_three_comparisons():
    x = np.array([[10.,-5.,30.,15.],[-10.,20.,3.,-7.],[4.,-2.,6.,8.]])
    result = paired_statistics(x,replications=2300)
    assert result["family_size"] == 3
    assert result["mean_USD"] == x.mean()
    assert result["interval_level"] == 1-.05/3
    assert result["simultaneous_interval"][0] <= result["marginal_95"][0]
    assert result["simultaneous_interval"][1] >= result["marginal_95"][1]


def test_mechanism_does_not_count_floating_zero_as_positive_gain():
    result = paired_statistics(np.full((3,8),1e-12),replications=1200)
    assert result["equivalent_fraction"] == 1.
    assert result["seed_positive_count"] == 0


def test_mechanism_result_is_strict_json_serializable(tmp_path):
    result = paired_statistics(np.array([[1.,2.],[-1.,-2.],[0.,1.]]), replications=1500)
    file = tmp_path/"result.json"
    write_json(file, result)
    restored = read_json(file)
    assert type(restored["seed_positive_count"]) is int
    assert restored["seed_positive_count"] == 2
