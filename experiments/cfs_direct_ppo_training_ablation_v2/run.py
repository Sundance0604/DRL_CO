"""Guarded stages. No test may open until all27 new policies are frozen."""
import argparse
import copy
import os
import time
import torch
from .common import *


def smoke():
    from .training import train
    from .network import DirectPolicy
    from experiments.cfs_direct_ppo_v3.training import run_record
    from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
    m=load();rows=[];budget=Budget()
    if (TARGET/'qa/smoke.json').exists():raise RuntimeError('smoke must not repeat')
    try:
        for name in NAMES:
            for variant in ('A','B','C','D'):
                cfg=configuration(name,variant,29998)
                cfg['ppo'].update(updates=2,episodes_per_update=1,validation_every=1)
                folder=TARGET/'smoke'/name/variant
                result=train(scenes(name,'train')[:4],scenes(name,'validation')[:1],cfg,folder,
                             m['source'],variant,references(name),budget)
                with no_mc_guard():
                    p=DirectPolicy(folder/'best-inference',m['source'])
                    rec=run_record(scenes(name,'validation')[0],cfg,p.act,'smoke-reload',budget)
                original=read_json(folder/f'validation-{result["selected_update"]:04d}-s21008.json')
                if original['ledger']!=rec['ledger'] or [t['action_hash'] for t in original['transitions']]!=[t['action_hash'] for t in rec['transitions']]:
                    raise RuntimeError('loaded checkpoint inference differs')
                write_json(folder/'reload.json',rec)
                size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
                rows.append(dict(condition=name,variant=variant,seconds=result['wall_seconds'],bytes=size,
                                 episodes=5,checkpoint_exact=True))
                print('SMOKE',name,variant,'PASSED',result['wall_seconds'],flush=True)
        # Conservative per-episode extrapolation, extra allowance for mechanisms/replays.
        projected=sum(r['seconds']/4*576 for r in rows if r['variant']!='A')
        # 576 per condition/variant = 3models*(128training+64validation).
        projected+=1152*max(r['seconds']/4 for r in rows)
        estimated_bytes=sum(r['bytes']/5*576 for r in rows if r['variant']!='A')*1.25
        conservative=projected*1.5
        calls=budget.state['calls']+24*(3456+1728+1008+144)+432+576
        gate=dict(status='PASSED',rows=rows,projected_numerical_seconds=projected,
            conservative_seconds=conservative,projected_bytes=estimated_bytes,
            projected_max_calls=calls,limits=dict(seconds=14400,bytes=15*2**30,calls=180000))
        write_json(TARGET/'qa/smoke.json',gate)
        if conservative>14400 or estimated_bytes>15*2**30 or calls>180000:
            raise TimeoutError('smoke projection exceeds frozen budget; no formal training started')
        print('SMOKE BUDGET PASSED',conservative/60,'conservative minutes',flush=True)
    finally:budget.save()


def train():
    from .training import train as fit
    m=load()
    if read_json(TARGET/'qa/smoke.json')['status']!='PASSED':raise RuntimeError('smoke required')
    if (TARGET/'protocol/frozen-policies.json').exists():raise RuntimeError('training already frozen')
    budget=Budget();choices=[]
    try:
        for name in NAMES:
            for variant in VARIANTS:
                for seed in SEEDS:
                    folder=TARGET/'training'/name/variant/str(seed)
                    if (folder/'status.json').exists() and read_json(folder/'status.json')['status']=='COMPLETED':
                        result=read_json(folder/'status.json')
                    else:
                        if folder.exists():raise RuntimeError('partial model requires explicit evidence-led recovery; do not overwrite')
                        result=fit(scenes(name,'train'),scenes(name,'validation'),configuration(name,variant,seed),
                                   folder,m['source'],variant,references(name),budget)
                    if result['episodes']!=128:raise RuntimeError('training budget mismatch')
                    # All schemes use exactly A's preregistered scene-sampling schedule.
                    for i in range(1,129):
                        a=read_json(PARENT/'training'/name/str(seed)/f'training-episode-{i:06d}.json')
                        b=read_json(folder/f'training-episode-{i:06d}.json')
                        if a['scenario_seed']!=b['scenario_seed']:raise RuntimeError('training scene schedule changed')
                    choices.append(dict(condition=name,variant=variant,seed=seed,
                                        checkpoint=(folder/'best-inference').relative_to(TARGET).as_posix(),**result))
        winners={name:max(VARIANTS,key=lambda v:sum(r['best_validation_USD'] for r in choices
                     if r['condition']==name and r['variant']==v)/3) for name in NAMES}
        hashes={p.relative_to(TARGET).as_posix():sha(p) for p in (TARGET/'training').rglob('*') if p.is_file()}
        write_json(TARGET/'protocol/frozen-policies.json',dict(rows=choices,hashes=hashes,
            validation_selected_variant=winners,frozen_before_test_at=now()))
        write_json(TARGET/'qa/training.json',dict(status='PASSED',models=27,training_episodes=3456,
            deterministic_validation_episodes=1728,same_A_scene_schedule=True))
        m.update(status='MODELS_FROZEN',models_frozen_at=now());write_json(TARGET/'manifest.json',m)
    finally:budget.save()


def policy(name,variant,seed,m):
    from experiments.cfs_direct_ppo_v3.network import DirectPolicy
    if variant=='A':
        cp=PARENT/parent_choice(name,seed)['checkpoint'];source=read_json(PARENT/'manifest.json')['source']
    else:
        frozen=read_json(TARGET/'protocol/frozen-policies.json')
        row=next(r for r in frozen['rows'] if (r['condition'],r['variant'],r['seed'])==(name,variant,seed))
        cp=TARGET/row['checkpoint'];source=m['source']
    return DirectPolicy(cp,source)


def evaluate():
    from experiments.cfs_direct_ppo_v3.training import run_record
    from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
    from model_families.single.learning.ppo.networks import sample_action
    m=load()
    if m['status']!='MODELS_FROZEN':raise RuntimeError('all27 choices freeze before tests')
    if (TARGET/'qa/evaluation.json').exists():raise RuntimeError('do not repeat evaluation')
    frozen=read_json(TARGET/'protocol/frozen-policies.json');verify_hashes(TARGET,frozen['hashes'])
    budget=Budget();count=0;stoch=0
    try:
        for name in NAMES:
            # Test file parsed outside the Direct guard only after all choices frozen.
            tests=scenes(name,'test');vals=scenes(name,'validation')[:4]
            for variant in ('A','B','C','D'):
                for seed in SEEDS:
                    cfg=configuration(name,variant,seed)
                    with no_mc_guard():
                        p=policy(name,variant,seed,m)
                        for s in tests:
                            r=run_record(s,cfg,p.act,variant+'-test',budget)
                            r.update(variant=variant,training_seed=seed)
                            write_json(TARGET/'evaluation'/f'{name}-{variant}-{seed}-s{s["seed"]}.json',r);count+=1
                        for s in vals:
                            # Reset same latent noise for matched model/scene across all variants.
                            gen=torch.Generator().manual_seed(62000+seed+s['seed'])
                            r=run_record(s,cfg,lambda o:sample_action(p.network,o,gen,False)[0].tolist(),
                                         variant+'-stochastic-validation',budget)
                            r.update(variant=variant,training_seed=seed,policy_seed=62000+seed+s['seed'])
                            write_json(TARGET/'validation'/f'{name}-{variant}-{seed}-s{s["seed"]}.json',r);stoch+=1
                    print('TEST',name,variant,seed,'total',count,flush=True)
        if (count,stoch)!=(864,144):raise RuntimeError('evaluation coverage')
        write_json(TARGET/'qa/evaluation.json',dict(status='PASSED',Direct_test_days=count,
                                                  stochastic_validation_days=stoch,all_replayed=True))
    finally:budget.save()


def baselines():
    from experiments.cfs_ppo_strategic_control.study_execution import run_record,replay
    from experiments.cfs_direct_ppo_v3.run import PARENT as MC_PARENT
    m=load()
    if m['status']!='MODELS_FROZEN':raise RuntimeError('all27 choices must freeze')
    if (TARGET/'qa/baselines.json').exists():raise RuntimeError('do not repeat baselines')
    budget=Budget();count=0
    try:
        for name in NAMES:
            base=read_json(MC_PARENT/'protocol'/(name+'.json'))
            base['baseline']=str(MC_PARENT/base['baseline'])
            for method in ('MC','M0'):
                cfg=copy.deepcopy(base)
                if method=='M0':cfg['baseline']=dict(method='zero')
                for s in scenes(name,'test'):
                    budget.claim(24);r=run_record(s,cfg,lambda o:(0.,)*10,method)
                    r['replay']=replay(s,cfg,r)
                    write_json(TARGET/'evaluation'/f'{name}-{method}-s{s["seed"]}.json',r);count+=1
        write_json(TARGET/'qa/baselines.json',dict(status='PASSED',days=count,MC_actor=False,all_replayed=True))
    finally:budget.save()


def numerical_complete():
    m=load()
    for stage in ('training','evaluation','baselines','mechanism'):
        if read_json(TARGET/'qa'/(stage+'.json'))['status']!='PASSED':raise RuntimeError('pending '+stage)
    m.update(status='NUMERICAL_COMPLETE_QA_PENDING',numerical_completed_at=now())
    write_json(TARGET/'manifest.json',m)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','smoke','train','evaluate','baselines','mechanism','all'))
    a=p.parse_args()
    from experiments.cfs_ppo_strategic_control.study_run import ensure_environment
    ensure_environment();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import psutil
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    try:
        if a.stage=='prepare':prepare()
        elif a.stage=='all':
            for name,fn in (('smoke',smoke),('training',train),('evaluation',evaluate),('baselines',baselines)):
                if not (TARGET/'qa'/(name+'.json')).exists():fn()
            from .mechanism import run
            if not (TARGET/'qa/mechanism.json').exists():run()
            numerical_complete()
        elif a.stage=='mechanism':
            from .mechanism import run
            run();numerical_complete()
        else:globals()[a.stage]()
    except BaseException as exc:
        if (TARGET/'manifest.json').exists():
            m=read_json(TARGET/'manifest.json')
            if m['status'] not in ('COMPLETED','FAILED','BUDGET_LIMITED'):
                status='BUDGET_LIMITED' if isinstance(exc,(TimeoutError,MemoryError)) else 'FAILED'
                m.update(status=status,stopped_at=now());write_json(TARGET/'manifest.json',m)
                write_json(TARGET/'qa/stop.json',dict(status=status,stage=a.stage,error=type(exc).__name__,message=str(exc)))
        raise


if __name__=='__main__':main()
