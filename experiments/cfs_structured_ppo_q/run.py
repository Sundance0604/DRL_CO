import argparse
import copy
import json
from pathlib import Path
import torch
import numpy as np
from .common import *


def pilot():
    from experiments.cfs_direct_ppo_diagnosis.common import restore,canonical,finish
    from experiments.cfs_direct_ppo_v3.training import run_record as run_old
    from experiments.cfs_direct_ppo_v3.qa import no_mc_guard,replay as replay_old
    from model_families.single.runner import state_snapshot
    from .control import upgrade,replay
    from .training import run_record
    m=load();budget=Budget()
    if (TARGET/'qa/pilot.json').exists():return
    refs=read_json(TARGET/'protocol/references.json');switches=[];nest_days=0;ranks=[];coverage=[]
    try:
        with no_mc_guard():
            for name in NAMES:
                c=cfg(name);p=old_policy(name);first=None
                for scene in scenes(name,'validation')[:2]:
                    old=run_old(scene,c,p.act,'old-nesting',budget)
                    new=run_record(scene,c,lambda o:list(p.act(o))+[0.]*10,'q-zero-nesting',budget)
                    if old['ledger']!=new['ledger']:raise RuntimeError('beta0 fullcash differs')
                    for a,b in zip(old['transitions'],new['transitions']):
                        if (a['action_hash']!=b['action_hash'] or a['plan']['solver']['window_objective']!=b['plan']['solver']['window_objective'] or
                            [(r['time'],r['hub'],r['supply_slot'],r['value']) for r in a['coefficients']]!=
                            [(r['time'],r['hub'],r['supply_slot'],r['value']) for r in b['coefficients']]):
                            raise RuntimeError('beta0 full coefficient/action/objective mismatch')
                    write_json(TARGET/'pilot'/f'{name}-nesting-s{scene["seed"]}.json',dict(old=old,new=new));nest_days+=1
                for spec in [r for r in refs if r['condition']==name]:
                    scene=next(s for s in scenes(name,'validation') if s['seed']==spec['scenario_seed'])
                    record=read_json(spec['source']);ep,o=restore(scene,c,record,spec['period'],budget);ep=upgrade(ep)
                    before=state_snapshot(ep._sim);alpha=list(p.act(o));ep._decoder.rank_audit=True
                    zero=ep.probe(alpha+[0.]*10);ep._decoder.rank_audit=False
                    rank=zero['plan']['solver'].get('coefficient_rank',dict(alpha=0,expanded=0))
                    ranks.append(dict(condition=name,seed=scene['seed'],period=spec['period'],**rank))
                    base=canonical(zero['plan'],before);rows=[dict(direction='zero',slope=0.,signal=alpha+[0.]*10,probe=zero,canonical=base)]
                    for direction in range(6):
                        d=np.zeros(10)
                        if direction<5:d[2*direction:2*direction+2]=1
                        else:d[:]=1
                        for beta in (-.6,-.2,.2,.6):
                            signal=alpha+(beta*d).tolist();probe=ep.probe(signal)
                            canon=canonical(probe['plan'],before)
                            row=dict(direction=direction,slope=beta,signal=signal,probe=probe,canonical=canon);rows.append(row)
                            if first is None and canon!=base:
                                first=(spec,scene,record,alpha,row)
                    if state_snapshot(ep._sim)!=before:raise RuntimeError('probe changed physical state')
                    occupancy=zero['plan']['solver']['slot_occupancy']
                    occupied=[r for r in occupancy if r['occupied']]
                    coverage.append(dict(condition=name,seed=scene['seed'],period=spec['period'],queries=len(occupancy),
                        q_ge2_queries=sum(r['q']>=2 for r in occupancy),occupied_slots=len(occupied),
                        occupied_q_ge2=sum(r['q']>=2 for r in occupied),
                        changed=sum(r['canonical']!=base for r in rows[1:]),probes=len(rows)-1,
                        ever_occupied_q_ge2=any(s['occupied'] and s['q']>=2 for r in rows for s in r['probe']['plan']['solver']['slot_occupancy'])))
                    write_json(TARGET/'pilot'/f'{name}-s{scene["seed"]}-t{spec["period"]}.json',dict(spec=spec,state=before,rows=rows,rank=rank))
                if first:
                    spec,scene,record,alpha,row=first;profits=[]
                    for label,signal in (('zero',alpha+[0.]*10),('changed',row['signal'])):
                        ep,o=restore(scene,c,record,spec['period'],budget);ep=upgrade(ep)
                        # Original10 continuation is embedded with beta0, keeping one action schema.
                        class Continuation:
                            def act(self,o):return list(p.act(o))+[0.]*10
                        r=finish(ep,signal,Continuation());r['replay']=replay(scene,c,r)
                        write_json(TARGET/'pilot'/f'{name}-cf-{label}.json',r);profits.append(r['ledger']['realized_profit'])
                    switches.append(dict(condition=name,period=spec['period'],scenario_seed=scene['seed'],direction=row['direction'],
                        slope=row['slope'],difference_USD=profits[1]-profits[0],selection='first changed; NOT profit selected'))
                else:switches.append(dict(condition=name,reason='no canonical switch'))
        gate=any(r['expanded']>r['alpha'] for r in ranks) and any(r['ever_occupied_q_ge2'] and r['changed'] for r in coverage)
        write_json(TARGET/'qa/pilot.json',dict(status='PASSED',promote=gate,nesting_days=nest_days,
            coverage=coverage,ranks=ranks,cases=switches,probe_states=18,probe_solves=450))
        m['status']='PILOT_PASSED' if gate else 'NOT_PROMOTED_ANALYSIS_PENDING';write_json(TARGET/'manifest.json',m)
        print('PILOT promote',gate,'canonical changed',sum(r['changed'] for r in coverage),flush=True)
    finally:budget.save()


def train():
    from .training import fit
    m=load()
    if not read_json(TARGET/'qa/pilot.json')['promote']:return
    if (TARGET/'protocol/models-frozen.json').exists():return
    budget=Budget();choices=[]
    try:
        for name in NAMES:
            for seed in SEEDS:
                folder=TARGET/'training'/name/str(seed)
                if folder.exists():raise RuntimeError('partial training requires independent explicit recovery')
                choices.append(fit(name,seed,scenes(name,'train'),scenes(name,'validation'),cfg(name,seed),folder,m['source'],budget))
        hashes={str(p):sha(p) for p in (TARGET/'training').rglob('*') if p.is_file()}
        write_json(TARGET/'protocol/models-frozen.json',dict(rows=choices,hashes=hashes,frozen_at=now(),test_opened=False))
        m['status']='MODELS_FROZEN';write_json(TARGET/'manifest.json',m)
    finally:budget.save()


def new_tests():
    """Only after all policies freeze. Check actual historical scenario content."""
    from model_families.single.learning.ppo.runner import scenario_content
    from experiments.cfs_ppo_strategic_control.study_common import CONDITIONS,key
    import re
    # Historic scenario datasets (not trajectories) are enough to detect used seeds/content.
    used=set();contents=set();files=[];executed=set()
    for root in (ROOT/'reports/cfs_ppo_strategic_control',ROOT/'experiments/cfs_ppo_strategic_control/runs'):
        for p in root.rglob('*-s*.json'):
            if p.is_relative_to(TARGET):continue
            found=re.search(r'-s(\d+)\.json$',p.name)
            if found:executed.add(int(found.group(1)))
    if set(TESTS)&executed:raise RuntimeError('test seed previously executed')
    for root in (ROOT/'reports/cfs_ppo_strategic_control',ROOT/'experiments/cfs_ppo_strategic_control/runs'):
        for p in root.rglob('*.json'):
            if p.is_relative_to(TARGET):continue
            if p.name not in ('train.json','validation.json','test.json','scenario.json','scenarios.json'):continue
            d=read_json(p);items=d.get('scenarios',[]) if isinstance(d,dict) else []
            if items:
                files.append(dict(path=str(p),sha256=sha(p)))
                for s in items:
                    used.add(s['seed'])
                    if s['split']!='test' or s['seed'] in executed:contents.add(scenario_content(s))
    # Existing prepared-but-unused broad pools are not tests actually opened. They
    # are still checked for content duplication and reported rather than called leakage.
    results={}
    for c in CONDITIONS:
        name=key(c)
        pool=ROOT/'experiments/cfs_ppo_strategic_control/runs/cfs_first_ppo_v1/sealed'/name/'test.json'
        items=[s for s in read_json(pool)['scenarios'] if s['seed'] in TESTS]
        if [s['seed'] for s in items]!=TESTS:raise RuntimeError('unused sealed prepared pool coverage')
        for s in items:
            if scenario_content(s) in contents:raise RuntimeError('historical scenario content collision')
        results[name]=items
    write_json(TARGET/'qa/test-isolation.json',dict(status='PASSED',datasets=files,seeds=TESTS,
        exact_content_disjoint=True,seeds_in_any_generated_dataset=sorted(set(TESTS)&used),
        executed_seed_disjoint=True,executed_file_seed_inventory=sorted(executed),
        provenance='previously generated but unexecuted sealed CFS pool; not newly generated'))
    for name,items in results.items():write_json(TARGET/'evaluation/scenarios'/f'{name}.json',dict(scenarios=items))


def evaluate():
    from .training import Policy,run_record
    from experiments.cfs_direct_ppo_v3.training import run_record as old_run
    from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
    from experiments.cfs_ppo_strategic_control.study_execution import run_record as base_run,replay as base_replay
    from experiments.cfs_direct_ppo_v3.run import PARENT as mc
    m=load()
    if not read_json(TARGET/'qa/pilot.json')['promote']:return
    if (TARGET/'qa/evaluation.json').exists():return
    if m['status']!='MODELS_FROZEN':raise RuntimeError('all models first')
    verify_hashes(Path('/'),read_json(TARGET/'protocol/models-frozen.json')['hashes'])
    new_tests();budget=Budget();count=0
    try:
        for name in NAMES:
            for seed in SEEDS:
                c=cfg(name,seed)
                with no_mc_guard():
                    old=old_policy(name,seed);new=Policy(TARGET/'training'/name/str(seed)/'inference-0008')
                    for s in scenes(name,'test'):
                        for method,fun,pol in (('PPO10',old_run,old),('PPO20',run_record,new)):
                            r=fun(s,c,pol.act,method,budget);r['training_seed']=seed
                            write_json(TARGET/'evaluation'/f'{name}-{method}-{seed}-s{s["seed"]}.json',r);count+=1
            baseline=read_json(mc/'protocol'/(name+'.json'));baseline['baseline']=str(mc/baseline['baseline'])
            for method in ('MC','M0'):
                c=copy.deepcopy(baseline)
                if method=='M0':c['baseline']=dict(method='zero')
                for s in scenes(name,'test'):
                    budget.claim(24);r=base_run(s,c,lambda o:(0.,)*10,method);r['replay']=base_replay(s,c,r)
                    write_json(TARGET/'evaluation'/f'{name}-{method}-s{s["seed"]}.json',r);count+=1
            print('EVALUATED',name,count,flush=True)
        write_json(TARGET/'qa/evaluation.json',dict(status='PASSED',days=count,expected=192))
        m['status']='NUMERICAL_COMPLETE_QA_PENDING';write_json(TARGET/'manifest.json',m)
    finally:budget.save()


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','pilot','train','evaluate','all'));stage=p.parse_args().stage
    torch.set_num_threads(1)
    try:
        if stage=='prepare':prepare()
        elif stage=='all':pilot();train();evaluate()
        else:globals()[stage]()
    except Exception as e:
        if TARGET.exists():
            write_json(TARGET/'qa/stop.json',dict(error=type(e).__name__,message=str(e),stage=stage,at=now()))
            m=read_json(TARGET/'manifest.json');m['status']='BUDGET_LIMITED' if isinstance(e,TimeoutError) else 'FAILED';write_json(TARGET/'manifest.json',m)
        raise


if __name__=='__main__':main()
