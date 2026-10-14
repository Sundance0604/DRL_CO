import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'experiments/cfs_resource_value_state_bucket_exact_accounting'
NAME = 'western_multicluster-n40-k8'
TARGET = ROOT/'reports/cfs_supervised_value/autogluon_pilot_v3'
AG_PYTHON = Path('C:/Users/32599/.conda/envs/pavane/python.exe')
BASIC = ['release_time', 'hub', 'q']
STATE = BASIC + ['decision_time', 'remaining_time', 'release_delay', 'hub_idle', 'total_idle',
    'hub_incoming', 'total_incoming', 'hub_backlog_count', 'total_backlog_count',
    'hub_backlog_load', 'total_backlog_load', 'hub_expiring', 'total_expiring',
    'train_demand_rate', 'hub_imbalance', 'committed_load', 'hub_ready_load']


def read(path):
    p=Path(path)
    return json.loads((gzip.decompress(p.read_bytes()).decode('utf-8')) if p.suffix=='.gz' else p.read_text(encoding='utf-8'))


def write(path, data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(p)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def csv_write(path, rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise ValueError('empty output is not an executed comparison')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def decision_time(release):
    nodes=[0,3,7,12,16,20,23]
    if release not in nodes:raise ValueError('unknown label release time')
    return nodes[max(0,nodes.index(release)-1)]


def check_packet(actual,saved):
    from experiment_core.storage import digest
    if digest({k:v for k,v in actual.items() if k!='J'})!=digest({k:v for k,v in saved.items() if k!='J'}):
        raise ValueError('physical prefix mismatch')
    if not math.isclose(actual['J'],saved['J'],abs_tol=1e-7,rel_tol=1e-10):
        raise ValueError('booked diagnostic mismatch')
    return abs(actual['J']-saved['J'])


def features(state, known, hub, release, q, prior):
    """Only the current public state and TRAINING-only demand prior enter x."""
    from model_families.single.strategic.observation import free_release
    if any(o.book_time>state.t for o in known.values()):raise ValueError('hidden future in features')
    if release<state.t or q<1:raise ValueError('invalid query')
    pool=list(state.pool.values())
    if any(o.book_time>state.t for o in pool):raise ValueError('hidden pool')
    pending=[o for o in pool if o.departure==hub]
    incoming=[free_release(state,v) for v in state.vehs if v.phase!='idle']
    incoming=[(h,t) for h,t in incoming if t<=release]
    idle=sum(v.phase=='idle' and v.loc==hub for v in state.vehs)
    load=sum(o.passenger for o in pending)
    urgent=lambda o:o.end_time-state.t-state.net.tau[o.departure,o.destination]<=1
    row=dict(release_time=int(release),hub=hub,q=int(q),decision_time=state.t,
        remaining_time=state.T-state.t,release_delay=release-state.t,hub_idle=idle,
        total_idle=sum(v.phase=='idle' for v in state.vehs),
        hub_incoming=sum(h==hub for h,t in incoming),total_incoming=len(incoming),
        hub_backlog_count=len(pending),total_backlog_count=len(pool),hub_backlog_load=load,
        total_backlog_load=sum(o.passenger for o in pool),hub_expiring=sum(urgent(o) for o in pending),
        total_expiring=sum(urgent(o) for o in pool),train_demand_rate=prior[hub],
        hub_imbalance=load/state.parameters['capacity']-idle-sum(h==hub for h,t in incoming),
        committed_load=sum(o.passenger for v in state.vehs for o in v.orders.values()),
        hub_ready_load=sum(o.passenger for o in pending if o.start_time<=state.t))
    assert list(row)==STATE
    if any(not math.isfinite(float(v)) for k,v in row.items() if k!='hub'):raise ValueError('nonfinite feature')
    return row


def metric(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if not len(y) or y.shape!=p.shape or not np.isfinite(p).all():raise ValueError('invalid predictions')
    e=p-y;den=float(((y-y.mean())**2).sum())
    return dict(n=len(y),MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt((e**2).mean())),
        R2=float(1-(e**2).sum()/den) if den else None,
        sign_error=float(np.mean(np.sign(np.where(np.abs(y)<=1e-6,0,y))!=np.sign(np.where(np.abs(p)<=1e-6,0,p)))),
        prediction_min=float(p.min()),prediction_max=float(p.max()))
