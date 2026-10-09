"""Independent statistics and held-out guards for the reduced experiment."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.cfs_ppo_strategic_control.laptop.run import independent_bootstrap, check_tests
from experiments.cfs_ppo_strategic_control.study_analysis import bootstrap_crossed
from model_families.single.learning.ppo.runner import physical_signature, scenario_content


def test_reduced_crossed_bootstrap_independent_recalculation():
    x = np.array([[10., -2., 8., 1.], [2., 9., -5., 7.], [-3., 4., 6., 0.]])
    direct = bootstrap_crossed(x, replications=2500, seed=31000)
    count_weighted = independent_bootstrap(x, replications=2500, seed=31000)
    np.testing.assert_allclose(direct["marginal_95"], np.quantile(count_weighted, [.025, .975]), atol=1e-12)
    np.testing.assert_allclose(direct["bonferroni_99_1667"], np.quantile(count_weighted, [.05/12, 1-.05/12]), atol=1e-12)
    assert direct["mean_USD"] == x.mean()


def test_bootstrap_constant_zero_difference():
    assert np.all(independent_bootstrap(np.zeros((3, 12)), 1234) == 0)


def fixture():
    scene = dict(id="test", seed=22, split="test", nodes=["A", "B"],
                 edges=[["A", "B", 10.]], hubs=["A"], horizon=24,
                 orders=[dict(marker="new physical content")])
    train = copy.deepcopy(scene)
    train.update(id="train", seed=20, split="train", orders=[dict(marker="training")])
    policy = SimpleNamespace(environment=dict(physics=physical_signature(train),
        data_splits=dict(train=[dict(id=train["id"], seed=train["seed"], content_hash=scenario_content(train))], validation=[])))
    return scene, train, policy


def test_laptop_test_guard_catches_relabelled_training_content():
    scene, train, policy = fixture()
    check_tests(policy, [scene])
    disguised = copy.deepcopy(train)
    disguised.update(id="new-name", seed=999, split="test")
    with pytest.raises(RuntimeError, match="overlaps"):
        check_tests(policy, [disguised])


def test_laptop_test_guard_catches_changed_physics_and_seed_overlap():
    scene, train, policy = fixture()
    scene["seed"] = train["seed"]
    with pytest.raises(RuntimeError, match="overlaps"):
        check_tests(policy, [scene])
    scene["seed"] = 23
    scene["horizon"] = 25
    with pytest.raises(RuntimeError, match="incompatibility"):
        check_tests(policy, [scene])
