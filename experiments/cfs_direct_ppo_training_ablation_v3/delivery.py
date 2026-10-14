"""Non-numerical delivery gates; never alters frozen training or selection."""
import argparse
import copy
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from .common import *


def analysis():
    from .training import sensitivity
    from .network import DirectPolicy
    from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
    load()
    if (TARGET/'qa/analysis.json').exists():raise RuntimeError('analysis already passed; do not overwrite report')
    analysis_path=PACKAGE/'analysis.py';freeze=TARGET/'qa/analysis-source-freeze.json'
    if freeze.exists():
        if read_json(freeze)['sha256']!=sha(analysis_path):raise RuntimeError('analysis source changed')
    else:write_json(freeze,dict(sha256=sha(analysis_path),frozen_at=now()))
    # Shared reference observations were frozen before every new trained model.
    out=TARGET/'analysis/A-sensitivity';out.mkdir(exist_ok=True)
    source=read_json(PARENT/'manifest.json')['source']
    for name in NAMES:
        for seed in SEEDS:
            for u in (8,16,24,32):
                p=out/f'{name}-{seed}-{u:04d}.json'
                if p.exists():continue
                with no_mc_guard():
                    policy=DirectPolicy(PARENT/'training'/name/str(seed)/f'inference-{u:04d}',source)
                    result=sensitivity(policy.network,references(name),configuration(name)['action']['scales'])
                write_json(p,result)
    marker=TARGET/'qa/CSV-create-marker.json'
    if not marker.exists():
        runtime='C:/Users/32599/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
        script='C:/Users/32599/.codex/plugins/cache/openai-primary-runtime/spreadsheets/26.1011.11113/skills/spreadsheets/container_tools/mark_artifact_operation_started.mjs'
        subprocess.run([runtime,script,'--operation-kind','create','--expected-output-count','8','--output-format','csv'],check=True)
        write_json(marker,dict(status='EXECUTED',expected_count=8,at=now()))
    python='C:/Users/32599/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    subprocess.run([python,'-m','experiments.cfs_direct_ppo_training_ablation_v3.analysis'],check=True,cwd=ROOT)
    from .figures import run
    configs=run()
    write_json(TARGET/'qa/figures.json',dict(status='GENERATED_TECHNICAL_PENDING_VISUAL',groups=5,configs=configs))


def replay():
    from .training import learning_replay
    from .backend import TimeBaseline
    from .run import policy
    from experiments.cfs_direct_ppo_v3.qa import replay as physical,no_mc_guard
    from experiments.cfs_ppo_strategic_control.study_execution import replay as baseline_replay
    from experiments.cfs_direct_ppo_v3.run import PARENT as MC_PARENT
    from experiments.cfs_direct_ppo_diagnosis.common import observation
    started=time.monotonic();m=load();count=0;deterministic_steps=0
    if (TARGET/'qa/replay.json').exists():raise RuntimeError('replay already passed')
    verify_hashes(Path('/'),m['history_hashes'])
    frozen=read_json(TARGET/'protocol/frozen-policies.json');verify_hashes(TARGET,frozen['hashes'])
    for name in NAMES:
        data={s['seed']:s for split in ('train','validation','test') for s in scenes(name,split)}
        for variant in VARIANTS:
            for seed in SEEDS:
                cfg=configuration(name,variant,seed);folder=TARGET/'training'/name/variant/str(seed)
                baseline=TimeBaseline();validation_means=[]
                for u in range(1,33):
                    hist=baseline.values();future=[]
                    log=read_json(folder/f'update-{u:04d}.json')
                    for i in range(4*(u-1)+1,4*u+1):
                        r=read_json(folder/f'training-episode-{i:06d}.json')
                        with no_mc_guard():learning_replay(data[r['scenario_seed']],cfg,r,variant)
                        if variant=='C':
                            if [t['time_baseline'] for t in r['learning']]!=hist:raise RuntimeError('independent baseline history differs')
                            future.append([t['RTG'] for t in r['learning']])
                        count+=1
                    if variant=='C':
                        baseline.observe_after_optimization(future)
                        if baseline.values()!=log['baseline_after']:raise RuntimeError('C after-baseline differs')
                    if u in (8,16,24,32):
                        with no_mc_guard():p=policy_checkpoint(folder/f'inference-{u:04d}',m['source'])
                        profits=[]
                        for s in scenes(name,'validation'):
                            r=read_json(folder/f'validation-{u:04d}-s{s["seed"]}.json')
                            deterministic_steps+=check_policy_record(r,p,cfg,s);count+=1
                            profits.append(r['ledger']['realized_profit'])
                        expected=sum(profits)/16
                        if expected!=read_json(folder/f'validation-{u:04d}.json')['mean_profit_USD']:raise RuntimeError('validation mean mismatch')
                        validation_means.append((expected,u))
                chosen=next(r for r in frozen['rows'] if (r['condition'],r['variant'],r['seed'])==(name,variant,seed))
                if max(validation_means,key=lambda x:(x[0],-x[1]))[1]!=chosen['selected_update']:raise RuntimeError('selected checkpoint')
        for variant in ('A','B','C','D'):
            for seed in SEEDS:
                cfg=configuration(name,variant,seed)
                with no_mc_guard():p=policy(name,variant,seed,m)
                for s in scenes(name,'test'):
                    r=read_json(TARGET/'evaluation'/f'{name}-{variant}-{seed}-s{s["seed"]}.json')
                    deterministic_steps+=check_policy_record(r,p,cfg,s);count+=1
                for s in scenes(name,'validation')[:4]:
                    r=read_json(TARGET/'validation'/f'{name}-{variant}-{seed}-s{s["seed"]}.json')
                    with no_mc_guard():physical(s,cfg,r)
                    check_stochastic_record(r,p,cfg);count+=1
        old=read_json(MC_PARENT/'protocol'/(name+'.json'));old['baseline']=str(MC_PARENT/old['baseline'])
        for method in ('MC','M0'):
            cfg=copy.deepcopy(old)
            if method=='M0':cfg['baseline']=dict(method='zero')
            for s in scenes(name,'test'):
                r=read_json(TARGET/'evaluation'/f'{name}-{method}-s{s["seed"]}.json')
                baseline_replay(s,cfg,r);count+=1
    if count!=6336:raise RuntimeError('independent replay day coverage '+str(count))
    counterfactual_calls=0
    cases=read_json(TARGET/'mechanism/cases.json')
    for case in cases:
        if 'difference_USD' in case:counterfactual_calls+=2*(24-case['period'])
    for p in (TARGET/'mechanism').glob('*counterfactual-*.json'):
        r=read_json(p);name=next(n for n in NAMES if p.name.startswith(n))
        s=next(s for s in scenes(name,'validation') if s['seed']==r['scenario_seed'])
        with no_mc_guard():physical(s,configuration(name),r)
    smoke_count=0
    for name in NAMES:
        for variant in ('A','B','C','D'):
            cfg=configuration(name,variant,29998)
            cfg['ppo'].update(updates=2,episodes_per_update=4,validation_every=2)
            base=TARGET/'smoke'/name/variant
            for p in base.glob('training-episode-*.json'):
                r=read_json(p);s=data_scene(name,r['scenario_seed'],'train')
                with no_mc_guard():learning_replay(s,cfg,r,variant)
                smoke_count+=1
            for p in [*base.glob('validation-*-s*.json'),base/'reload.json']:
                r=read_json(p);s=data_scene(name,r['scenario_seed'],'validation')
                with no_mc_guard():physical(s,cfg,r)
                smoke_count+=1
    if smoke_count!=300:raise RuntimeError('smoke replay denominator')
    probe_count=audit_probes()
    expected_calls=(6336+300)*24+probe_count+counterfactual_calls
    if read_json(TARGET/'qa/numerical-budget.json')['calls']!=expected_calls:
        raise RuntimeError('MILP accounting differs from actual complete record/branch coverage')
    # Every raw record and checkpoint is included, not only report summaries.
    index={p.relative_to(TARGET).as_posix():sha(p) for d in ('smoke','training','validation','evaluation','mechanism')
           for p in (TARGET/d).rglob('*') if p.is_file()}
    write_json(TARGET/'qa/numeric-index.json',dict(status='PASSED',hashes=index))
    write_json(TARGET/'qa/replay.json',dict(status='PASSED',days=count,deterministic_actor_steps=deterministic_steps,
        history_files=len(m['history_hashes']),C_baseline_independently_recomputed=True,
        additional_smoke_days=smoke_count,probe_count=probe_count,actual_solver_calls=expected_calls,
        numeric_files=len(index),seconds=time.monotonic()-started))
    print('REPLAY PASSED',count,'days',flush=True)


def data_scene(name,seed,split):
    return next(s for s in scenes(name,split) if s['seed']==seed)


def audit_probes():
    from experiments.cfs_direct_ppo_diagnosis.common import restore,canonical
    from experiments.cfs_ppo_strategic_control.study_execution import action_from_plan
    from model_families.single.planning import compile_action
    from model_families.single.runner import state_snapshot
    from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
    count=0
    for p in (TARGET/'mechanism').glob('*-t*.json'):
        r=read_json(p);cfg=configuration(r['condition'],r['variant'],r['seed'])
        s=data_scene(r['condition'],r['spec']['scenario_seed'],'validation')
        original=read_json(r['spec']['source'])
        with no_mc_guard():ep,o=restore(s,cfg,original,r['spec']['period'])
        if state_snapshot(ep._sim)!=r['state']:raise RuntimeError('probe source/state chain')
        for row in r['rows']:
            plan=row['probe']['plan'];info=copy.deepcopy(plan['solver'])
            compiled=compile_action(ep._sim,action_from_plan(plan),info)
            if info['accounting']!=plan['accounting'] or info['status']!='OPTIMAL':raise RuntimeError('probe integer accounting/status')
            if canonical(plan,r['state'])!=row['canonical']:raise RuntimeError('canonical probe classification')
            for q in row['probe']['coefficients']:
                hub=cfg['action']['hubs'].index(q['hub']);ell=(q['time']-o.period)/(24-o.period)
                expected=cfg['action']['scales'][hub]*((1-ell)*row['signal'][2*hub]+ell*row['signal'][2*hub+1])
                if q['time']>=24 or abs(q['value']-expected)>1e-8:raise RuntimeError('independent probe price')
            count+=1
    if count!=432:raise RuntimeError('probe audit coverage')
    return count


def policy_checkpoint(path,source):
    from experiments.cfs_direct_ppo_v3.network import DirectPolicy
    return DirectPolicy(path,source)


def check_policy_record(r,p,cfg,scene):
    from experiments.cfs_direct_ppo_diagnosis.common import observation
    from experiments.cfs_direct_ppo_v3.qa import no_mc_guard,replay
    with no_mc_guard():
        replay(scene,cfg,r)
        for tr in r['transitions']:
            expected=p.act(observation(tr['observation']))
            if tuple(tr['signal']['values'])!=expected:raise RuntimeError('deterministic Actor mismatch')
    return len(r['transitions'])


def check_stochastic_record(r,p,cfg):
    from model_families.single.learning.ppo.networks import sample_action
    from experiments.cfs_direct_ppo_diagnosis.common import observation
    gen=torch.Generator().manual_seed(r['policy_seed'])
    for t in r['transitions']:
        expected=sample_action(p.network,observation(t['observation']),gen,False)[0].tolist()
        if expected!=t['signal']['values']:raise RuntimeError('stochastic replay noise/Actor')


def tests():
    load();p=TARGET/'qa/pytest-final.xml';suites=list(ET.parse(p).getroot().iter('testsuite'))
    counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts['tests']<79 or any(counts[k] for k in ('failures','errors','skipped')):raise RuntimeError('final related tests not passed')
    write_json(TARGET/'qa/tests.json',dict(status='PASSED',**counts,XML_sha256=sha(p),
        scope='ablation-v3, Direct-v3/Direct, PPO, classical value, rolling; not full-platform'))


def visual(reviewed):
    if not reviewed:raise RuntimeError('actual five plot review required')
    from PIL import Image
    from .figures import STEMS
    load();checks=[]
    for stem in STEMS:
        p=TARGET/'figures'/(stem+'.png');im=Image.open(p)
        if abs(im.info['dpi'][0]-600)>.1:raise RuntimeError('PNG dpi')
        ET.parse(TARGET/'figures'/(stem+'.svg'))
        checks.append(dict(stem=stem,pixels=list(im.size),dpi=im.info['dpi']))
    write_json(TARGET/'qa/visual.json',dict(status='PASSED',actually_reviewed=True,groups=5,checks=checks,
        hashes={p.relative_to(TARGET).as_posix():sha(p) for p in (TARGET/'figures').glob('*')}))


def freeze():
    m=load()
    if m['status']!='NUMERICAL_COMPLETE_QA_PENDING':raise RuntimeError('numerical not complete')
    for name in ('prepare','smoke','training','evaluation','baselines','mechanism','analysis','replay','tests','visual'):
        if read_json(TARGET/'qa'/(name+'.json'))['status']!='PASSED':raise RuntimeError('missing '+name)
    verify_hashes(Path('/'),m['history_hashes']);verify_hashes(TARGET,read_json(TARGET/'qa/numeric-index.json')['hashes'])
    verify_hashes(TARGET,read_json(TARGET/'qa/visual.json')['hashes'])
    if read_json(TARGET/'qa/analysis-source-freeze.json')['sha256']!=sha(PACKAGE/'analysis.py'):raise RuntimeError('analysis changed')
    b=read_json(TARGET/'qa/numerical-budget.json')
    size=sum(p.stat().st_size for p in TARGET.rglob('*') if p.is_file())
    if b['calls']>180000 or b['seconds']>14400 or size>15*2**30:raise RuntimeError('budget exceeded')
    hashes={p.relative_to(TARGET).as_posix():sha(p) for p in TARGET.rglob('*') if p.is_file() and
            p not in (TARGET/'manifest.json',TARGET/'final_manifest.json')}
    m.update(status='COMPLETED',completed_at=now(),final_artifact_hashes=hashes,
        delivery_sources={p.relative_to(ROOT).as_posix():sha(p) for p in (PACKAGE/'analysis.py',PACKAGE/'figures.py',PACKAGE/'delivery.py')})
    write_json(TARGET/'manifest.json',m);write_json(TARGET/'final_manifest.json',m);verify_hashes(TARGET,hashes)
    print('COMPLETED',len(hashes),'files',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('analysis','replay','tests','visual','freeze'))
    p.add_argument('--reviewed',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.stage=='visual':visual(a.reviewed)
    else:globals()[a.stage]()


if __name__=='__main__':main()
