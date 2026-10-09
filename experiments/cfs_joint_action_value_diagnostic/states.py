"""Freeze complete state bank, then complete candidate bank, before any futures."""
import time
from experiment_core.storage import read_json,atomic_json,digest
from model_families.single.data import matching_state
from experiments.cfs_rollout_target_alignment.queries import pack_state
from experiments.cfs_resource_policy_iteration.protocol import restore
from .actions import historical,decide,verify,candidates
from .common import OUTPUT,key,parameters,rho_levels,parallel,sha,now

def collect_job(job):
    path=OUTPUT/'raw/states'/f'{job["stage"]}-{job["condition"]}-s{job["seed"]}-{job["source"]}.json'
    identity=digest(job)
    if path.exists():
        row=read_json(path);assert row['identity']==identity;return dict(path=str(path),sha256=sha(path),state_id=row['state_id'])
    started=time.perf_counter();s,orders=matching_state(job['scenario'],job['parameters'],job['solver'])
    policies,prior=historical(job['condition']);prefix=[]
    for t in range(job['query_time']+1):
        s.begin_period([o for o in orders.values() if o.book_time==t])
        if t==job['query_time']:break
        action=decide(s,policies[job['source']],prior);p,serial=verify(s,action);s.commit_plan(p)
        prefix.append(dict(period=t,plan=serial))
    known=[o for o in job['scenario']['orders'] if o['book_time']<=s.t]
    visible_scenario=job['scenario']|dict(orders=known)
    packet=pack_state(s);state_id=digest(dict(condition=job['condition'],seed=job['seed'],source=job['source'],state=packet))
    row=dict(identity=identity,state_id=state_id,stage=job['stage'],condition=job['condition'],scenario_seed=job['seed'],
        source=job['source'],time=s.t,scenario=visible_scenario,packet=packet,source_prefix=prefix,
        source_policy_hash=policies[job['source']]['policy_hash'],parameters=job['parameters'],solver=job['solver'],
        full_scenario_ref=job['scenario_ref'],full_scenario_sha256=sha(job['scenario_ref']),
        hidden_orders_excluded=True,state_hash=digest(packet),seconds=time.perf_counter()-started,
        numeric_code_hash=read_json(OUTPUT/'manifests/source_manifest.json')['code_hash'])
    atomic_json(path,row);return dict(path=str(path),sha256=sha(path),state_id=state_id)

def restore_row(row):
    return restore(row['scenario'],row['parameters'],row['solver'],row['packet'])

def action_job(ref):
    row=read_json(ref['path']);assert sha(ref['path'])==ref['sha256']
    path=OUTPUT/'raw/actions'/f'{row["state_id"]}.json'
    if path.exists():return dict(path=str(path),sha256=sha(path),state_id=row['state_id'])
    s,_=restore_row(row);actions,missing=candidates(s,row['condition'])
    r=dict(state_ref=ref,actions=actions,missing=missing,state_id=row['state_id'],
        metadata_selection_only=True,future_outcomes_seen=False,created_at=now())
    atomic_json(path,r);return dict(path=str(path),sha256=sha(path),state_id=row['state_id'])

def freeze_bank(cfg,base,data,seeds,stage):
    sp=OUTPUT/'manifests'/f'{stage}_states.json';ap=OUTPUT/'manifests'/f'{stage}_actions.json'
    if ap.exists():
        m=read_json(ap)
        for r in m['actions']:assert sha(r['path'])==r['sha256']
        return m
    assert not list((OUTPUT/'raw/futures').glob(f'{stage}-*')),'Candidates must precede outcomes'
    if sp.exists():refs=read_json(sp)['states']
    else:
        jobs=[]
        for c in cfg['conditions']:
            for i,seed in enumerate(seeds):
                scenario=data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
                for source in cfg['sources']:
                    jobs.append(dict(stage=stage,condition=key(c),seed=seed,source=source,
                        query_time=cfg['query_times'][i%len(cfg['query_times'])],scenario=scenario,
                        scenario_ref=str(OUTPUT/'raw/datasets'/f'{key(c)}-s{seed}.dataset.json'),
                        parameters=parameters(base,rho_levels(scenario,base)['rho_max']),solver=base['solver']))
        refs,wall=parallel(collect_job,jobs,stage+' freeze complete states',cfg['max_parallel'])
        refs.sort(key=lambda r:r['state_id']);atomic_json(sp,dict(status='FROZEN_BEFORE_CANDIDATES',states=refs,
            created_at=now(),wall_seconds=wall,selection='one cyclic time per seed; identical for both sources'))
    for ref in refs:assert sha(ref['path'])==ref['sha256']
    actions,wall=parallel(action_job,refs,stage+' freeze full joint candidates',cfg['max_parallel'])
    actions.sort(key=lambda r:r['state_id']);m=dict(status='FROZEN_BEFORE_FUTURES',states_sha256=sha(sp),
        actions=actions,created_at=now(),wall_seconds=wall,future_outcomes_seen=False)
    atomic_json(ap,m);return m
