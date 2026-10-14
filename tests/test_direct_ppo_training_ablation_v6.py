import copy
import pytest
import torch
from test_direct_ppo_v3 import direct_cfg,scenario
from model_families.single.learning.ppo.config import PPOConfig
from model_families.single.learning.ppo.networks import sample_action,squashed_log_probability
from model_families.single.learning.ppo.backend import update as original_update
from experiments.cfs_direct_ppo_v3.control import episode
from experiments.cfs_direct_ppo_training_ablation_v6.network import build,actor_weights,optimizers,actor_network
from experiments.cfs_direct_ppo_training_ablation_v6.backend import update,full_returns,TimeBaseline


def fixture_rows(net):
    obs=episode(scenario(),direct_cfg()).reset()
    generator=torch.Generator().manual_seed(71)
    rows=[]
    for i in range(12):
        a,z,lp,v=sample_action(net,obs,generator)
        rows.append(dict(observation=obs,latent=z,logp=lp.item(),advantage=float(i-5),**{'return':float(i+1)}))
    return obs,rows


@pytest.mark.parametrize('variant',['A','B','C','D'])
def test_actor_initialization_exactly_matches_original(variant):
    a=build('A',33,3,8);b=build(variant,33,3,8)
    assert all(torch.equal(v,actor_weights(b)[k]) for k,v in actor_weights(a).items())


def test_gradient_logging_preserves_original_update_and_rng():
    a=build('A',33,3,8);b=copy.deepcopy(a);_,rows=fixture_rows(a)
    cfg=PPOConfig(epochs=4,minibatch_size=8,target_kl=.03)
    oa=optimizers(a,cfg.learning_rate);ob=optimizers(b,cfg.learning_rate)
    ga=torch.Generator().manual_seed(999);gb=torch.Generator().manual_seed(999)
    old=original_update(a,oa[0],rows,cfg,ga);new=update(b,ob,rows,cfg,gb,'A')
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert torch.equal(ga.get_state(),gb.get_state())
    for key in old:assert old[key]==pytest.approx(new[key])
    shared=sum(p.numel() for name,p in b.named_parameters() if name.startswith(('order_encoder.','arc_encoder.','vehicle_encoder.','body.')))
    assert new['detailed_gradient']['shared_parameters']==shared


def test_full_returns_include_terminal_cooldown():
    assert full_returns([1,-2,10])==[9,8,10]


def test_time_baseline_uses_previous_batch_only():
    b=TimeBaseline(3);frozen=b.values()
    assert frozen==[0,0,0] and b.episodes==0
    b.observe_after_optimization([[9,8,10],[3,2,1]])
    assert frozen==[0,0,0] and b.values()==[6,5,5.5] and b.episodes==2
    with pytest.raises(ValueError):b.observe_after_optimization([[1]])


@pytest.mark.parametrize('variant',['B','C','D'])
def test_intervention_update_finite_and_correct_paths(variant):
    net=build(variant,35,3,8);obs,rows=fixture_rows(net)
    before=copy.deepcopy(net.state_dict());cfg=PPOConfig(minibatch_size=8)
    opts=optimizers(net,cfg.learning_rate)
    diag=update(net,opts,rows,cfg,torch.Generator().manual_seed(51),variant)
    assert diag['actor_steps']>0 and diag['all_gradients_finite']
    assert any(not torch.equal(v,net.state_dict()[k]) for k,v in before.items())
    if variant=='C':
        assert net.critic is None and diag['value_loss'] is None and diag['critic_steps']==0
    if variant=='D':
        assert len(opts)==2 and diag['detailed_gradient']['shared_cosine'] is None
        assert len(diag['minibatch_records'][0]['preclip_norms'])==2
    assert abs(diag['standardized_advantage_mean'])<1e-6
    assert diag['standardized_advantage_std']==pytest.approx(1.)


def test_decoupled_critic_loss_cannot_reach_actor():
    net=build('D',33,3,8);obs,_=fixture_rows(net)
    actor_ids={id(p) for p in net.actor_model.parameters()}
    assert not actor_ids&{id(p) for p in net.value_model.parameters()}
    _,_,value=net([obs]);(.5*value.square().mean()).backward()
    assert all(p.grad is None for p in net.actor_model.parameters())
    assert any(p.grad is not None for p in net.value_model.parameters())


def test_tanh_jacobian_and_checkpoint_state_dict_exact():
    net=build('D',33,3,8);obs,_=fixture_rows(net)
    a,z,lp,_=sample_action(net,obs,torch.Generator().manual_seed(44))
    mean,std,_=net([obs])
    ref=(torch.distributions.Normal(mean,std).log_prob(z)-(1-torch.tanh(z).square()).log()).sum()
    assert lp.item()==pytest.approx(ref.item(),abs=1e-5)
    b=build('D',34,3,8);b.load_state_dict(net.state_dict())
    assert all(torch.equal(x,y) for x,y in zip(net([obs]),b([obs])))


def test_reward_scaled_divisor_only_affects_learning_units():
    from model_families.single.learning.ppo.environment import PPOEnvironment
    a=episode(scenario(),direct_cfg());b=episode(scenario(),direct_cfg())
    ea,eb=PPOEnvironment(a,1000),PPOEnvironment(b,10000);ea.reset();eb.reset()
    while True:
        _,ra,done,_,_=ea.step([0]*6);_,rb,_,_,_=eb.step([0]*6)
        assert ra==pytest.approx(10*rb)
        if done:break
    assert a.final_ledger==b.final_ledger


@pytest.mark.parametrize('episodes_per_update',[1,4])
def test_smoke_and_formal_baseline_counts_are_config_driven(monkeypatch,episodes_per_update):
    import experiments.cfs_direct_ppo_training_ablation_v6.training as training
    monkeypatch.setattr(training,'physical_replay',lambda *args:None)
    cfg={'ppo':{'reward_scale':1000,'episodes_per_update':episodes_per_update}}
    rec=dict(update=2,transitions=[{'reward_USD':1000} for _ in range(24)],
        learning=[dict(RTG=24-i,value=None,baseline_episodes=episodes_per_update,
            time_baseline=1,advantage=23-i,**{'return':24-i}) for i in range(24)])
    assert training.learning_replay({},cfg,rec,'C')['status']=='PASSED'
    rec['learning'][0]['baseline_episodes']+=episodes_per_update
    with pytest.raises(RuntimeError,match='current batch'):training.learning_replay({},cfg,rec,'C')


def test_completed_registry_result_with_variant_is_not_duplicated(monkeypatch,tmp_path):
    import experiments.cfs_direct_ppo_training_ablation_v6.run as runner
    from model_families.single.strategic.artifacts import read_json as actual_read
    monkeypatch.setattr(runner,'TARGET',tmp_path)
    monkeypatch.setattr(runner,'PARENT',tmp_path/'parent')
    monkeypatch.setattr(runner,'NAMES',('toy',));monkeypatch.setattr(runner,'SEEDS',(30001,))
    monkeypatch.setattr(runner,'VARIANTS',('B',))
    monkeypatch.setattr(runner,'load',lambda:{'source':{'unit_test':True},'status':'PREPARED'})
    class Budget:
        def save(self):pass
    monkeypatch.setattr(runner,'Budget',Budget)
    folder=tmp_path/'training/toy/B/30001';folder.mkdir(parents=True)
    (folder/'status.json').touch()
    (tmp_path/'protocol').mkdir();(tmp_path/'qa').mkdir()
    def fake_read(p):
        if str(p).endswith('smoke.json'):return {'status':'PASSED'}
        if str(p).endswith('status.json'):return dict(status='COMPLETED',variant='B',episodes=64,
            selected_update=16,best_validation_USD=1.)
        return {'scenario_seed':20032}
    monkeypatch.setattr(runner,'read_json',fake_read)
    runner.train()
    row=actual_read(tmp_path/'protocol/frozen-policies.json')['rows'][0]
    assert row['variant']=='B' and row['episodes']==64 and row['seed']==30001

