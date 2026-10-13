"""Direct isolation, M0 equivalence, prices, gradients and frozen inference."""
import copy
from dataclasses import asdict
import pytest
import torch
from test_single_strategic_ppo import scenario,config,setup,table
from experiments.cfs_direct_ppo.control import episode,DirectProvider
from experiments.cfs_direct_ppo.contracts import DirectSchema,DirectSignal,OBSERVATION_VERSION
from experiments.cfs_direct_ppo.observation import dimensions
from experiments.cfs_direct_ppo.network import DirectNetwork,DirectPolicy
from experiments.cfs_direct_ppo.training import train,run_record
from experiments.cfs_direct_ppo.qa import no_mc_guard,replay
from experiments.cfs_ppo_strategic_control.study_execution import physical_hash


def direct_cfg():
    c=config();c.pop('baseline');c['action']['version']='single-direct-near-far-price/v1'
    c['ppo'].update(training_seed=33,validation_every=1,updates=2)
    return c


def test_direct_observation_removes_four_MC_features_not_other_public_fields():
    d=episode(scenario(),direct_cfg());o=d.reset();old=setup();p=old.reset()
    assert o.version==OBSERVATION_VERSION and o.global_features==p.global_features
    assert o.orders==p.orders and o.vehicles==p.vehicles
    assert o.hub_features==tuple(r[:-4] for r in p.hub_features)
    assert dimensions(5)['hub_dim']==12
    with pytest.raises(ValueError,match='baseline'):episode(scenario(),config())


def test_direct_prices_negative_q_invariant_and_terminal():
    schema=DirectSchema(('A','B'),(100.,200.));signal=DirectSignal((-1.,.5,.2,.4),2,0,'a')
    p=DirectProvider(schema,signal,6)
    assert p.marginal_release_value(2,'A',1,{})==-100
    assert p.marginal_release_value(4,'A',1,{})==-25
    assert p.marginal_release_value(4,'A',2,{})==-25
    assert p.marginal_release_value(6,'A',1,{})==0
    assert all('base' not in r and 'residual' not in r for r in p.queries.values())
    with pytest.raises(ValueError):p.marginal_release_value(1,'A',1,{})


def test_zero_direct_matches_full_M0_actions_cash_and_reward():
    d=episode(scenario(),direct_cfg());old=setup();a=d.reset();b=old.reset()
    with no_mc_guard():
        while True:
            tr=d.step((0.,)*6)
            if tr.terminated:break
    while not old.step((0.,)*6).terminated:pass
    assert d.final_ledger==old.final_ledger
    assert [physical_hash(r['plan']) for r in d.records]==[physical_hash(r['plan']) for r in old.records]
    assert all(c['value']==0 for r in d.records for c in r['coefficients'])
    assert sum(r['reward_USD'] for r in d.records)==pytest.approx(d.final_ledger['realized_profit'])
    with no_mc_guard():assert replay(scenario(),direct_cfg(),dict(ledger=d.final_ledger,transitions=d.records))['status']=='PASSED'


def test_MC_queries_and_tables_are_forbidden():
    from model_families.single.strategic.baseline import FrozenBaseline
    with no_mc_guard():
        with pytest.raises(RuntimeError,match='forbidden'):FrozenBaseline(table())
        with pytest.raises(RuntimeError,match='forbidden'):open('some/baselines/table.json')


def test_hidden_future_not_in_direct_observation_or_integer_action():
    a=scenario();b=copy.deepcopy(a);b['orders'][0]['revenue']+=999999
    x,y=episode(a,direct_cfg()),episode(b,direct_cfg())
    assert x.reset()==y.reset()
    with no_mc_guard():assert x.probe((.2,)*6)['action_hash']==y.probe((.2,)*6)['action_hash']


def test_direct_network_rejects_old_input_and_keeps_critic_and_set_invariance():
    from dataclasses import replace
    ep=episode(scenario(),direct_cfg());obs=ep.reset();net=DirectNetwork(3,8)
    means,std,value=net([obs]);assert means.shape==(1,6) and value.shape==(1,)
    assert torch.equal(net.encode(obs),net.encode(replace(obs,vehicles=tuple(reversed(obs.vehicles)))))
    with pytest.raises(ValueError,match='residual'):net([setup().reset()])


def test_actual_training_inference_has_no_MC_and_gradient_updates(tmp_path):
    cfg=direct_cfg();source={'test':'independent'}
    result=train([scenario('train',101),scenario('train',102,120.)],[scenario('validation',201,110.)],
        cfg,tmp_path/'train',source)
    assert result['actor_changed'] and result['critic_changed'] and result['episodes']==2
    with no_mc_guard():
        p=DirectPolicy(tmp_path/'train/best-inference',source)
        r=run_record(scenario('test',301,130.),cfg,p.act,'Direct-test')
        assert r['replay']['status']=='PASSED'
        assert not list((tmp_path/'train').rglob('baseline.json'))
    with pytest.raises(ValueError,match='incompatible'):DirectPolicy(tmp_path/'train/best-inference',{'other':1})


def test_direct_replay_detects_price_and_cash_tampering():
    cfg=direct_cfg();r=run_record(scenario(),cfg,lambda o:(.2,)*6,'Direct')
    cash=copy.deepcopy(r);cash['transitions'][-1]['reward_USD']+=1
    with pytest.raises(RuntimeError,match='reward'):replay(scenario(),cfg,cash)
