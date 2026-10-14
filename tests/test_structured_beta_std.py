import torch
import pytest
from experiments.cfs_structured_ppo_q.network import build
from experiments.cfs_structured_ppo_beta_std.run import modified
from experiments.cfs_structured_ppo_beta_std.common import configuration
from model_families.single.learning.ppo.networks import squashed_log_probability


def test_only_beta_initial_log_std_changes():
    old=build(30001);new=modified()
    for key,v in old.state_dict().items():
        w=new.state_dict()[key]
        if key=='log_std':
            assert torch.equal(v[:10],w[:10]);assert torch.equal(w[10:],torch.full((10,),-.5))
        else:assert torch.equal(v,w)
    assert configuration()['ppo']['reward_scale']==10000
    assert configuration()['ppo']['updates']==8


def test_common_noise_stream_different_actions_not_forced_trajectory():
    a=torch.Generator().manual_seed(31002);b=torch.Generator().manual_seed(31002)
    e=torch.randn((24,20),generator=a);f=torch.randn((24,20),generator=b)
    assert torch.equal(e,f) and torch.equal(a.get_state(),b.get_state())
    assert not torch.equal(torch.tanh(e*.223),torch.tanh(f*.607))


def test_joint_log_probability_is_sum_with_stable_jacobian():
    mu=torch.zeros((2,20));s=torch.ones_like(mu);z=torch.ones_like(mu)*.3
    full=squashed_log_probability(mu,s,z)
    assert torch.allclose(full,squashed_log_probability(mu[:,:10],s[:,:10],z[:,:10])+squashed_log_probability(mu[:,10:],s[:,10:],z[:,10:]))


def test_beta_variance_ratio_is_registered_not_target_kl_rescaling():
    a=build(30001);b=modified()
    assert (b.log_std[10:].exp()/a.log_std[10:].exp()).tolist()==pytest.approx([2.7182818]*10)
    assert configuration()['ppo']['target_kl']==.03
