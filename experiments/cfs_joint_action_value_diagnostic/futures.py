"""Explicit online surrogate distribution; never accepts hidden scenario orders."""
import copy
from types import SimpleNamespace
import numpy as np
from experiment_core.storage import digest,read_json,atomic_json
from .common import OUTPUT,B,sha

def frozen_prior(cid):
    path=OUTPUT/'config/future_priors'/f'{cid}.json'
    if path.exists():return read_json(path)
    marks=[];sources=[];rate=[]
    for seed in range(8300,8360):
        p=B/'raw/datasets'/f'{cid}-s{seed}.dataset.json';s=read_json(p)['scenarios'][0]
        marks.extend({k:v for k,v in o.items() if k not in ['id','book_time','start_time','end_time']}|
            dict(ready_delay=o['start_time']-o['book_time'],
                 slack=o['end_time']-o['start_time']) for o in s['orders'])
        sources.append(dict(path=str(p),sha256=sha(p)));rate.append(len(s['orders'])/s['horizon'])
    prior=dict(marks=marks,rate_per_period=float(np.mean(rate)),source_files=sources,
        distribution='SURROGATE: marked Poisson with empirical construction marks, iid with replacement',
        physical_slack='empirical end-ready includes same network travel; unchanged positive slack distribution')
    prior['hash']=digest(prior);atomic_json(path,prior);return prior

def generate(known_orders,state,prior,condition,state_id,role,replicate):
    assert all(o.book_time<=state.t for o in known_orders.values()),'Hidden future passed into sampler'
    assert role in ['selection','assessment','interaction','fixed_evaluation']
    namespace=dict(condition=condition,state_id=state_id,role=role,replicate=replicate)
    stream=digest(namespace);rng=np.random.default_rng(int(stream[:16],16))
    visible={k:copy.deepcopy(v) for k,v in known_orders.items()};new=[]
    periods=state.T-state.t-1
    n=int(rng.poisson(prior['rate_per_period']*periods)) if periods>0 else 0
    for i in range(n):
        m=prior['marks'][int(rng.integers(len(prior['marks'])))];booking=int(rng.integers(state.t+1,state.T))
        ready=booking+m['ready_delay'];oid=f'future-{stream[:24]}-{i}'
        o=({k:v for k,v in m.items() if k not in ['ready_delay','slack']}|
            dict(id=oid,book_time=booking,start_time=ready,end_time=ready+m['slack']))
        assert oid not in visible;visible[oid]=SimpleNamespace(**o);new.append(o)
    spec=dict(namespace=namespace,stream_hash=stream,prior_hash=prior['hash'],future_orders=new,
        visible_order_hash=digest({k:vars(v) for k,v in sorted(known_orders.items())}),
        distribution=prior['distribution'],future_hash=digest(new),hidden_source_orders_used=False)
    return visible,spec
