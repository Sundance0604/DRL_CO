"""Real 24-period interruption/resume equivalence against immutable v5 math."""
import copy
import json
import os
import shutil
import pytest
import torch
from test_direct_ppo_v3 import direct_cfg,scenario
from experiments.cfs_direct_ppo_training_ablation_v5.training import train as old_train
from experiments.cfs_direct_ppo_training_ablation_v6.training import train as resumed_train
from experiments.cfs_direct_ppo_training_ablation_v6 import io_retry


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def scientific(value):
    if isinstance(value,dict):
        return {k:scientific(v) for k,v in value.items() if k!='runtime' and not any(x in k for x in ('seconds','timestamp','created_at'))}
    if isinstance(value,list):return [scientific(x) for x in value]
    return value


@pytest.mark.parametrize('variant',['B','C','D'])
@pytest.mark.parametrize('validation_every',[1,2])
def test_post_update_resume_bit_exact_against_continuous_v5(tmp_path,variant,validation_every):
    torch.set_num_threads(1)
    cfg=direct_cfg();cfg['ppo'].update(updates=2,episodes_per_update=4,validation_every=validation_every,
        reward_scale=10000. if variant=='B' else 1000.,minibatch_size=64,epochs=2,hidden_size=8)
    data=[scenario(seed=i+1,fare=100+10*i) for i in range(2)]
    for s in data:s['horizon']=24
    # Actual frozen public observation from the toy environment, not fabricated feature arrays.
    from experiments.cfs_direct_ppo_v3.control import episode
    from dataclasses import asdict
    obs=episode(data[0],cfg).reset()
    refs=[dict(observation=asdict(obs),period=0,scenario_seed=1,pending=0,observation_hash=obs.hash)]
    class Unlimited:
        def __call__(self):return False
        def claim(self,*args):pass
    full=tmp_path/'full';broken=tmp_path/'broken'
    source={'unit_test':'identical-source'}
    old_train(data,data[:1],cfg,full,source,variant,refs,Unlimited())
    class Interrupt(Unlimited):
        def __call__(self):
            if (broken/'recovery.pt').exists():raise PermissionError('simulated write interruption after saved update1')
            return False
    with pytest.raises(PermissionError):old_train(data,data[:1],cfg,broken,source,variant,refs,Interrupt())
    snapshot=tmp_path/'immutable-update1.pt';shutil.copyfile(broken/'recovery.pt',snapshot)
    assert torch.load(snapshot,weights_only=False)['update']==1
    resumed_train(data,data[:1],cfg,broken,source,variant,refs,Unlimited(),resume=snapshot)
    equal(torch.load(full/'recovery.pt',weights_only=False),torch.load(broken/'recovery.pt',weights_only=False))
    equal(torch.load(full/'network-0002.pt',weights_only=False),torch.load(broken/'network-0002.pt',weights_only=False))
    equal(torch.load(full/'best-inference/actor.pt',weights_only=False),torch.load(broken/'best-inference/actor.pt',weights_only=False))
    for p in full.glob('*.json'):
        equal(scientific(json.loads(p.read_text(encoding='utf-8'))),scientific(json.loads((broken/p.name).read_text(encoding='utf-8'))))


def test_retry_preserves_target_and_serialization(monkeypatch,tmp_path):
    path=tmp_path/'data.json';path.write_text('old',encoding='utf-8')
    actual=os.replace;count=0;sleeps=[]
    def locked(a,b):
        nonlocal count
        count+=1
        if count<=3:
            assert path.read_text(encoding='utf-8')=='old'
            raise PermissionError('transient lock')
        return actual(a,b)
    monkeypatch.setattr(io_retry.os,'replace',locked);monkeypatch.setattr(io_retry.time,'sleep',sleeps.append)
    value={'中文':3,'a':[1,2]};io_retry.write_json(path,value)
    assert count==4 and sleeps==list(io_retry.DELAYS[:3])
    assert path.read_text(encoding='utf-8')==json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)


def test_persistent_permission_error_is_bounded(monkeypatch,tmp_path):
    path=tmp_path/'data.json';path.write_text('old',encoding='utf-8');count=0
    def locked(*args):
        nonlocal count
        count+=1
        raise PermissionError('persistent lock')
    monkeypatch.setattr(io_retry.os,'replace',locked);monkeypatch.setattr(io_retry.time,'sleep',lambda x:None)
    with pytest.raises(PermissionError):io_retry.write_json(path,{'x':1})
    assert count==8 and path.read_text(encoding='utf-8')=='old'


def test_nonpermission_error_never_retried(monkeypatch,tmp_path):
    count=0
    def failure(*args):
        nonlocal count
        count+=1
        raise OSError('disk error')
    monkeypatch.setattr(io_retry.os,'replace',failure)
    with pytest.raises(OSError):io_retry.write_json(tmp_path/'x.json',{})
    assert count==1
