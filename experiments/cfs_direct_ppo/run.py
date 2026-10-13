"""Stage-gated Direct PPO experiment, with a separate pure MC process."""
import argparse
import copy
from pathlib import Path
import os
import shutil
import time
from dataclasses import asdict
from model_families.single.strategic.artifacts import read_json,write_json,source_identity
from model_families.single.strategic.contracts import identity
from experiments.cfs_ppo_strategic_control.study_common import ROOT,now,sha,CONDITIONS,key
from experiments.cfs_ppo_signal_action_diagnostic.run import verify_hashes

PACKAGE=Path(__file__).resolve().parent
PREPARED=ROOT/'experiments/cfs_ppo_strategic_control/runs/cfs_first_ppo_v1'
PARENT=ROOT/'experiments/cfs_ppo_strategic_control/runs/cfs_laptop_ppo_v1'
SEEDS=(30001,30002,30003)
SPLITS=dict(train=list(range(20032,20160)),validation=list(range(21008,21024)),test=list(range(22020,22044)))
TASK=Path('D:/download/writing-block (1).md')


def sources():
    paths=[*PACKAGE.glob('*.py'),PACKAGE/'README.md',ROOT/'tests/test_direct_ppo.py']
    return dict(core=source_identity(),files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})


def prepare(target):
    from model_families.single.learning.ppo.runner import scenario_content
    from model_families.single.learning.ppo.config import PPOConfig
    import numpy as np
    if target.exists():raise FileExistsError('new attempt required')
    orig=read_json(PREPARED/'manifest.json');parent=read_json(PARENT/'final_manifest.json')
    if parent['status']!='COMPLETED' or orig['source_identity']!=source_identity():raise RuntimeError('parent source/status mismatch')
    history={str(PARENT/r):h for r,h in parent['final_artifact_hashes'].items()}
    history[str(PARENT/'final_manifest.json')]=sha(PARENT/'final_manifest.json')
    # Preserve every earlier PPO mechanism artifact and its immutable reused raw lineage.
    mech=ROOT/'experiments/cfs_ppo_strategic_control/runs/cfs_dynamic_vs_constant_v2'
    mm=read_json(mech/'final_manifest.json')
    history.update({str(mech/r):h for r,h in mm['final_artifact_hashes'].items()})
    for field in ('recovery_artifacts',):
        for r,h in mm.get(field,{}).items():
            p=Path(r) if Path(r).is_absolute() else (ROOT/r)
            if not p.exists():raise RuntimeError('historical recovery artifact missing: '+str(p))
            history[str(p)]=h
    verify_hashes(PARENT,mm['parent_artifacts'])
    history[str(mech/'final_manifest.json')]=sha(mech/'final_manifest.json')
    verify_hashes(Path('/'),history)
    failed=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_no_mc_v1'
    if failed.exists():
        if read_json(failed/'qa/stop.json')['stage']!='prepare':raise RuntimeError('unexpected earlier failure')
        history.update({str(p):sha(p) for p in failed.rglob('*') if p.is_file()})
    target.mkdir(parents=True)
    for f in ('protocol','data','sealed','engineering','smoke','training','evaluation','mc','analysis','figures','qa'):(target/f).mkdir()
    write_json(target/'manifest.json',dict(status='PREPARING',created_at=now()))
    identities=[];economics=[]
    for c in CONDITIONS:
        name=key(c);items={}
        for split,seeds in SPLITS.items():
            p=PREPARED/('sealed' if split=='test' else 'data')/name/(split+'.json')
            if sha(p)!=orig['input_hashes'][p.relative_to(PREPARED).as_posix()]:raise RuntimeError('prepared input SHA changed')
            history[str(p)]=sha(p)
            data=read_json(p);items[split]=[s for s in data['scenarios'] if s['seed'] in seeds]
            if [s['seed'] for s in items[split]]!=seeds:raise RuntimeError('split coverage mismatch')
            for s in items[split]:identities.append(dict(condition=name,split=split,seed=s['seed'],id=s['id'],content= scenario_content(s)))
        old=[]
        for split in ('train','validation','diagnostic'):
            old+=read_json(PARENT/'data'/name/(split+'.json'))['scenarios']
        old+=read_json(PARENT/'sealed'/name/'test.json')['scenarios']
        for split in ('pilot_train','pilot_validation','engineering'):
            old+=read_json(PREPARED/'data'/name/(split+'.json'))['scenarios']
        old+=read_json(ROOT/'experiments/cfs_ppo_strategic_control/runs/cfs_dynamic_vs_constant_v1/data'/(name+'.json'))['scenarios']
        for directory in ('cfs_ppo_signal_action_diagnostic/runs/signal_action_v1','cfs_ppo_decision_coverage/runs/full_period_v1'):
            p=ROOT/'experiments'/directory/'data'/(name+'.json')
            if p.exists():old+=read_json(p)['scenarios']
        if set(s['seed'] for s in old)&set(sum(SPLITS.values(),[])) or set(scenario_content(s) for s in old)&set(r['content'] for r in identities if r['condition']==name):
            raise RuntimeError('historical actual-use seed/content collision')
        ids=[r['content'] for r in identities if r['condition']==name]
        if len(ids)!=len(set(ids)):raise RuntimeError('cross-split content collision')
        revenues=[o['revenue'] for s in items['train'] for o in s['orders']]
        q75=float(np.quantile(revenues,.75,method='linear'));scale=max(100.,5*q75)
        cfg=read_json(PARENT/'protocol'/(name+'.json'));cfg.pop('baseline')
        cfg['action'].update(scales=[scale]*5,version='single-direct-near-far-price/v1')
        cfg['ppo']=PPOConfig(updates=32,episodes_per_update=4,validation_every=8,training_seed=30001).model_dump()
        economics.append(dict(condition=name,training_order_count=len(revenues),Q75_USD=q75,scale_USD=scale,
            revenue_median_USD=float(np.median(revenues)),rule='max(100,5*Q75 training order revenue); linear quantile'))
        for split,scenes in items.items():
            folder=target/('sealed' if split=='test' else 'data')/name;folder.mkdir(exist_ok=True)
            write_json(folder/(split+'.json'),dict(scenarios=scenes))
        write_json(target/'protocol'/(name+'.json'),cfg)
    history[str(TASK)]=sha(TASK);history[str(PREPARED/'manifest.json')]=sha(PREPARED/'manifest.json')
    source=orig['source'];history[source['path']]=source['sha256']
    protocol=dict(version='direct-ppo-no-mc/v1',created_at=now(),conditions=CONDITIONS,training_seeds=list(SEEDS),
        splits=SPLITS,updates=32,episodes_per_update=4,validation_updates=[8,16,24,32],
        selection='maximum validation mean, earliest tie; no training seed winner selection',
        inference='deterministic tanh(mean)',economics=economics,source=source,
        bootstrap_replications=20000,bootstrap_seed=42000,primary_family=3,multiple_comparison='Bonferroni percentile 98.3333% CI',
        estimand='mean realized profit difference over crossed training seed and paired scenario',
        expected_formal_episodes=2064,maximum_solver_calls=55000,wall_seconds=7200,disk_limit_bytes=8*2**30,
        process_memory_limit_bytes=2*2**30,workers=1,solver_threads=1,
        seed_amendment='Old suggested train20000..20031/validation21000..21007/test22000..22019 used; choose unused prepared intervals',
        direct_no_MC_paths=True,simulated_CFS_days_not_real_observation_dates=True)
    write_json(target/'protocol/preregistration.json',protocol)
    write_json(target/'data/identities.json',identities)
    (target/'experiment_protocol.md').write_text((PACKAGE/'README.md').read_text(encoding='utf-8'),encoding='utf-8')
    shutil.copyfile(TASK,target/'protocol/task.md')
    inputs={p.relative_to(target).as_posix():sha(p) for f in ('protocol','data','sealed') for p in (target/f).rglob('*') if p.is_file()}
    write_json(target/'manifest.json',dict(status='PREPARED',created_at=now(),source=sources(),input_hashes=inputs,
        history_hashes=history,source_CFS=source))
    write_json(target/'qa/data-isolation.json',dict(status='PASSED',scenarios=504,new_test_scenarios=72,
        seed_content_overlap=0,train_scenes_per_condition=128,validation_per_condition=16,test_per_condition=24,
        no_MC_scale=True,economics=economics))
    print('PREPARED Direct:',economics,flush=True)


def load(target):
    m=read_json(target/'manifest.json')
    if m['status'] in ('COMPLETED','INVALID','BUDGET_LIMITED'):raise RuntimeError('immutable terminal attempt')
    if m['source']!=sources():raise RuntimeError('frozen source changed')
    # Sealed test bytes are hashed, not parsed for training decisions.
    verify_hashes(target,m['input_hashes'])
    return m


def cfg(target,c,seed=30001):
    d=read_json(target/'protocol'/(key(c)+'.json'));d['ppo']['training_seed']=seed;return d


def scenes(target,c,split):
    return read_json(target/('sealed' if split=='test' else 'data')/key(c)/(split+'.json'))['scenarios']


def guard(target):
    import psutil
    budget=target/'qa/numerical-start.json'
    if not budget.exists():write_json(budget,dict(started_unix=time.time(),started_at=now(),solver_calls=0))
    state=read_json(budget)
    def check():
        if time.time()-state['started_unix']>7200:raise TimeoutError('Direct 120-minute numerical budget')
        if psutil.Process().memory_info().rss>2*2**30:raise MemoryError('Direct 2 GiB process limit')
        if shutil.disk_usage(target).free<10*2**30:raise RuntimeError('less than 10 GiB free disk')
        return False
    def claim(count=1):
        check()
        if state['solver_calls']+count>55000:raise TimeoutError('55000 solver call budget')
        state['solver_calls']+=count;write_json(budget,state)
    check.claim=claim
    return check


def engineering(target):
    from .control import episode
    from .qa import no_mc_guard
    from .training import run_record
    from experiments.cfs_ppo_strategic_control.study_execution import run_record as legacy_record
    from experiments.cfs_ppo_decision_coverage.run import decision_identity
    m=load(target)
    if (target/'qa/engineering.json').exists():raise RuntimeError('do not rerun engineering')
    checks=[];budget=guard(target)
    for c in CONDITIONS:
        name=key(c);configuration=cfg(target,c);s=scenes(target,c,'train')[0];zero=(0.,)*10
        with no_mc_guard():direct=run_record(s,configuration,lambda o:zero,'Direct-zero',budget)
        old=copy.deepcopy(configuration);old['action'].pop('version');old['baseline']=dict(method='zero')
        budget.claim(24);legacy=legacy_record(s,old,lambda o:zero,'M0')
        if direct['ledger']!=legacy['ledger'] or any(decision_identity(a['plan'])!=decision_identity(b['plan'])
            for a,b in zip(direct['transitions'],legacy['transitions'],strict=True)):
            raise RuntimeError('Direct zero differs from M0')
        for a,b in zip(direct['transitions'],legacy['transitions']):
            if any(q['value']!=0 for q in a['coefficients']) or a['plan']['operating_profit']!=b['plan']['operating_profit']:
                raise RuntimeError('Direct zero objective/cash mismatch')
        write_json(target/'engineering'/(name+'-zero.json'),direct)
        write_json(target/'engineering'/(name+'-M0.json'),legacy)
        probes=[];responsive=0;tested=0;ep=episode(s,configuration,budget);obs=ep.reset()
        with no_mc_guard():
            while True:
                if ep._sim.pool and tested<6 and responsive==0:
                    ref=ep.probe(zero);tested+=1
                    for axis in range(10):
                        for sign in (-1.,1.):
                            a=[0.]*10;a[axis]=sign;p=ep.probe(a)
                            changed=decision_identity(p['plan'])!=decision_identity(ref['plan'])
                            responsive+=changed;probes.append(dict(period=obs.period,axis=axis,sign=sign,changed=changed,probe=p))
                tr=ep.step(zero)
                if tr.terminated:break
                obs=tr.observation
        write_json(target/'engineering'/(name+'-controls.json'),probes)
        checks.append(dict(condition=name,zero_matches_M0=True,full_profit_USD=direct['ledger']['realized_profit'],
            nonzero_price_changed_probes=responsive,probed_states=tested,probe_count=len(probes),no_mc_guard=True))
    write_json(target/'qa/engineering.json',dict(status='PASSED',conditions=checks,
        action_interface_sanity=any(r['nonzero_price_changed_probes'] for r in checks)))
    if not any(r['nonzero_price_changed_probes'] for r in checks):raise RuntimeError('no nonzero price changed any training action')
    print('ENGINEERING PASSED',checks,flush=True)


def smoke(target):
    from .training import train
    load(target)
    if read_json(target/'qa/engineering.json')['status']!='PASSED':raise RuntimeError('engineering required')
    if (target/'qa/smoke.json').exists():raise RuntimeError('do not rerun smoke')
    c=CONDITIONS[-1];configuration=cfg(target,c,29999);configuration['ppo'].update(updates=2,episodes_per_update=2,validation_every=1)
    result=train(scenes(target,c,'train')[:4],scenes(target,c,'validation')[:1],configuration,target/'smoke/model',sources(),guard(target))
    estimate=result['wall_seconds']/6*2064
    write_json(target/'qa/smoke.json',dict(status='PASSED',episodes=6,configuration=configuration,result=result,
        projected_formal_seconds=estimate,conservative_formal_seconds=2*estimate,scale_or_hyperparameter_changes=0))
    if estimate*2>7200:raise TimeoutError('smoke conservative projection exceeds registered numerical budget')
    print('SMOKE PASSED; projection minutes',estimate/60,flush=True)


def train(target):
    from .training import train as training
    m=load(target)
    if read_json(target/'qa/smoke.json')['status']!='PASSED':raise RuntimeError('smoke required')
    if (target/'protocol/frozen-policies.json').exists():raise RuntimeError('training already frozen')
    rows=[];check=guard(target)
    for c in CONDITIONS:
        for seed in SEEDS:
            load(target);output=target/'training'/key(c)/str(seed)
            if output.exists():raise RuntimeError('existing training must not overwrite/resume silently')
            result=training(scenes(target,c,'train'),scenes(target,c,'validation'),cfg(target,c,seed),output,m['source'],check)
            rows.append(dict(condition=key(c),seed=seed,checkpoint=(output/'best-inference').relative_to(target).as_posix(),**result))
            if result['episodes']!=128:raise RuntimeError('wrong formal training budget')
    hashes={p.relative_to(target).as_posix():sha(p) for p in (target/'training').rglob('*') if p.is_file()}
    write_json(target/'protocol/frozen-policies.json',dict(rows=rows,hashes=hashes,frozen_before_tests=now()))
    write_json(target/'qa/training.json',dict(status='PASSED',models=9,training_episodes=1152,validation_episodes=576,
        all_no_mc_guard=True,all_replayed=True,rows=rows))
    m.update(status='MODELS_FROZEN',models_frozen_at=now());write_json(target/'manifest.json',m)


def mc(target):
    from experiments.cfs_ppo_strategic_control.study_execution import run_record,replay
    m=load(target)
    if m['status']!='MODELS_FROZEN':raise RuntimeError('all models must freeze before opening test')
    if (target/'qa/mc.json').exists():raise RuntimeError('MC complete, do not rerun')
    count=0;check=guard(target)
    for c in CONDITIONS:
        name=key(c);configuration=read_json(PARENT/'protocol'/(name+'.json'))
        configuration['baseline']=str(PARENT/configuration['baseline'])
        for split in ('validation','test'):
            for s in scenes(target,c,split):
                check.claim(24)
                r=run_record(s,configuration,lambda o:(0.,)*10,'Pure-MC');r['replay']=replay(s,configuration,r)
                write_json(target/'mc'/f'{name}-{split}-s{s["seed"]}.json',r);count+=1
    write_json(target/'qa/mc.json',dict(status='PASSED',episodes=count,actor_loaded=False,separate_MC_process=True))


def evaluate(target):
    from .network import DirectPolicy
    from .training import run_record
    from .qa import no_mc_guard
    m=load(target)
    if m['status']!='MODELS_FROZEN':raise RuntimeError('all selected policies must freeze')
    if (target/'qa/evaluation.json').exists():raise RuntimeError('evaluation complete')
    frozen=read_json(target/'protocol/frozen-policies.json');verify_hashes(target,frozen['hashes'])
    check=guard(target);count=0
    for c in CONDITIONS:
        name=key(c);tests=scenes(target,c,'test')
        for seed in SEEDS:
            row=next(r for r in frozen['rows'] if r['condition']==name and r['seed']==seed)
            with no_mc_guard():
                policy=DirectPolicy(target/row['checkpoint'],m['source'])
                for s in tests:
                    r=run_record(s,cfg(target,c,seed),policy.act,'Direct-PPO',check)
                    write_json(target/'evaluation'/f'{name}-{seed}-s{s["seed"]}.json',r);count+=1
            print('TEST Direct',name,seed,'completed',count,flush=True)
    write_json(target/'qa/evaluation.json',dict(status='PASSED',episodes=count,no_mc_guard_passed=True,
        deterministic=True,all_replayed=True,full_cooldown=True))
    m.update(status='NUMERICAL_COMPLETE_QA_PENDING',numerical_completed_at=now());write_json(target/'manifest.json',m)


def replay(target):
    from .qa import replay as direct_replay,no_mc_guard
    load(target);count=0
    for c in CONDITIONS:
        name=key(c);data={s['seed']:s for split in ('train','validation','test') for s in scenes(target,c,split)}
        for seed in SEEDS:
            configuration=cfg(target,c,seed);folder=target/'training'/name/str(seed)
            for p in [*folder.glob('training-episode-*.json'),*folder.glob('validation-*-s*.json'),*target.glob(f'evaluation/{name}-{seed}-s*.json')]:
                r=read_json(p)
                with no_mc_guard():direct_replay(data[r['scenario_seed']],configuration,r)
                count+=1
    write_json(target/'qa/final-replay.json',dict(status='PASSED',direct_episodes=count,expected=1944,
        full_independent_cash_price_GAE_and_physics=True))
    if count!=1944:raise RuntimeError('final Direct replay count mismatch')
    print('FINAL REPLAY',count,'PASSED',flush=True)


def freeze(target):
    m=load(target)
    if m['status']!='NUMERICAL_COMPLETE_QA_PENDING':raise RuntimeError('not numerically complete')
    for name in ('data-isolation','engineering','smoke','training','mc','evaluation','final-replay','analysis','tests','visual','integrity'):
        if read_json(target/'qa'/(name+'.json'))['status']!='PASSED':raise RuntimeError('missing final gate '+name)
    verify_hashes(Path('/'),m['history_hashes']);verify_hashes(target,read_json(target/'protocol/frozen-policies.json')['hashes'])
    hashes={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file() and p.name not in ('manifest.json','final_manifest.json')}
    # Checkpoint manifests are included; exclude only the top-level scientific manifests.
    for p in (target/'training').rglob('manifest.json'):hashes[p.relative_to(target).as_posix()]=sha(p)
    for p in (target/'smoke').rglob('manifest.json'):hashes[p.relative_to(target).as_posix()]=sha(p)
    m.update(status='COMPLETED',completed_at=now(),final_artifact_hashes=hashes)
    write_json(target/'manifest.json',m);write_json(target/'final_manifest.json',m);verify_hashes(target,hashes)
    print('COMPLETED',len(hashes),'files',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','engineering','smoke','train','mc','evaluate','replay','freeze'));p.add_argument('--attempt',required=True)
    a=p.parse_args();target=Path(a.attempt).resolve()
    from experiments.cfs_ppo_strategic_control.study_run import ensure_environment
    ensure_environment()
    import torch,psutil
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    try:globals()[a.stage](target)
    except BaseException as exc:
        if target.exists() and (target/'manifest.json').exists() and a.stage!='freeze':
            m=read_json(target/'manifest.json')
            if m['status'] not in ('COMPLETED','INVALID','BUDGET_LIMITED'):
                status='BUDGET_LIMITED' if isinstance(exc,(TimeoutError,MemoryError)) else 'INVALID'
                m.update(status=status,stopped_at=now());write_json(target/'manifest.json',m)
                write_json(target/'qa/stop.json',dict(status=status,stage=a.stage,error=type(exc).__name__,message=str(exc)))
        raise


if __name__=='__main__':main()
