import copy
from pathlib import Path
from dataclasses import asdict
from model_families.single.strategic.artifacts import read_json,write_json,source_identity
from model_families.single.strategic.contracts import identity,EncodedObservation,VehicleObservation
from experiments.cfs_ppo_strategic_control.study_common import ROOT,sha,now
from experiments.cfs_direct_ppo_v3.run import sources as parent_sources
from experiments.cfs_ppo_signal_action_diagnostic.run import verify_hashes

PARENT=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_no_mc_v3'
TARGET=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v1'
NAMES=('california_corridor-n80-k3','western_multicluster-n40-k8','western_multicluster-n80-k3')
SEEDS=(30001,30002,30003)
PACKAGE=Path(__file__).resolve().parent

def observation(data):
    data=copy.deepcopy(data);data['hubs']=tuple(data['hubs'])
    data['vehicles']=tuple(VehicleObservation(**v) for v in data['vehicles'])
    return EncodedObservation(**data)

def configuration(name):return read_json(PARENT/'protocol'/(name+'.json'))
def scenes(name,split):return read_json(PARENT/('sealed' if split=='test' else 'data')/name/(split+'.json'))['scenarios']
def choice(name,seed):return next(r for r in read_json(PARENT/'protocol/frozen-policies.json')['rows'] if r['condition']==name and r['seed']==seed)
def source_files():
    files=[PACKAGE/n for n in ('__init__.py','common.py','numerics.py','README.md')]+[ROOT/'tests/test_direct_ppo_diagnosis.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}

def verify_parent(full=False):
    m=read_json(PARENT/'final_manifest.json')
    if m['status']!='COMPLETED' or m['source']!=parent_sources():raise RuntimeError('parent/source changed')
    verify_hashes(PARENT,m['input_hashes'])
    if full:
        verify_hashes(PARENT,m['final_artifact_hashes']);verify_hashes(Path('/'),m['history_hashes'])
    return m

def canonical(plan,before):
    """Quotient only genuinely state-equivalent homogeneous vehicles.

    Retain order identities, route internal sequence, commitments and current
    location/phase. Removing arbitrary IDs without these groups is unsound.
    """
    assignments={};routes={};repos={}
    for row in plan['assignments']:assignments.setdefault(str(row['vehicle_id']),[]).append(row['order_id'])
    for row in plan['routes']:routes.setdefault(str(row['vehicle_id']),[]).append({k:v for k,v in row.items() if k!='vehicle_id'})
    for row in plan['repositions']:repos.setdefault(str(row['vehicle_id']),[]).append(row['hub'])
    bundles=[]
    for v in before['vehicles']:
        vid=str(v['id']);signature={k:v for k,v in v.items() if k!='id'}
        bundles.append(dict(state=signature,assigned=sorted(assignments.get(vid,[])),
            routes=sorted(routes.get(vid,[]),key=identity),repositions=sorted(repos.get(vid,[]))))
    return identity(dict(period=plan['period'],vehicles=sorted(bundles,key=identity),expired=sorted(plan['expired'])))

def restore(scenario,cfg,record,period,budget=None):
    """Compile frozen prefix without solving; replay all boundary events once."""
    from experiments.cfs_direct_ppo_v3.control import episode
    from experiments.cfs_ppo_strategic_control.study_execution import action_from_plan
    from model_families.single.planning import compile_action
    from model_families.single.runner import state_snapshot
    ep=episode(scenario,cfg,budget);obs=ep.reset()
    for tr in record['transitions'][:period]:
        if identity(state_snapshot(ep._sim))!=tr['before_state_hash'] or obs.hash!=tr['observation_hash']:
            raise RuntimeError('prefix before mismatch')
        p=tr['plan'];info=copy.deepcopy(p['solver']);plan=compile_action(ep._sim,action_from_plan(p),info)
        if info['accounting']!=p['accounting']:raise RuntimeError('prefix cash mismatch')
        ep._sim.commit_plan(plan);ep._rows.append(dict(period=tr['period'],plan=copy.deepcopy(p)))
        reward,ledger=ep._reward.advance(ep._sim,ep._orders,ep._rows)
        if abs(reward-tr['reward_USD'])>1e-7 or ledger!=tr['ledger']:raise RuntimeError('prefix ledger mismatch')
        ep._records.append(copy.deepcopy(tr));obs=ep._publish()
    tr=record['transitions'][period]
    if obs.hash!=tr['observation_hash'] or identity(state_snapshot(ep._sim))!=tr['before_state_hash']:
        raise RuntimeError('restored state mismatch')
    return ep,obs

def finish(ep,first,policy):
    tr=ep.step(first)
    while not tr.terminated:tr=ep.step(policy.act(tr.observation))
    return dict(scenario_id=ep._scenario['id'],scenario_seed=ep._scenario['seed'],
        transitions=ep.records,ledger=ep.final_ledger,metrics=ep.metrics)

def controls(det,network,obs,state_index):
    import numpy as np
    import torch
    from model_families.single.learning.ppo.networks import sample_action
    rows=[dict(kind='zero',signal=[0.]*10),dict(kind='deterministic',signal=list(det))]
    gen=torch.Generator().manual_seed(51000+state_index)
    for i in range(4):rows.append(dict(kind='stochastic',replicate=i,signal=sample_action(network,obs,gen,False)[0].tolist()))
    direction=np.random.default_rng(52000+state_index).choice([-1.,1.],10)
    # One near axis, one far axis at fixed rotating hub, and a fixed signed direction.
    hub=state_index%5
    directions=[np.eye(10)[2*hub],np.eye(10)[2*hub+1],direction]
    for d_id,d in enumerate(directions):
        for alpha in (.05,.2,.6):rows.append(dict(kind='perturbation',direction=d_id,alpha=alpha,
            signal=np.clip(np.asarray(det)+alpha*d,-1,1).tolist()))
    return rows
