"""Independent attempt, immutable input/source lineage and numerical resource caps."""
import copy
import random
import shutil
import time
from pathlib import Path
from model_families.single.strategic.artifacts import read_json,write_json,source_identity
from experiments.cfs_ppo_strategic_control.study_common import ROOT,CONDITIONS,key,sha,now
from experiments.cfs_ppo_signal_action_diagnostic.run import verify_hashes

PACKAGE=Path(__file__).resolve().parent
TARGET=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v4'
PARENT=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_no_mc_v3'
DIAG=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v2'
PREPARED=ROOT/'experiments/cfs_ppo_strategic_control/runs/cfs_first_ppo_v1'
SEEDS=(30001,30002,30003)
NAMES=tuple(key(c) for c in CONDITIONS)
TESTS=list(range(22044,22068))
VARIANTS=('B','C','D')
SMOKE_PARENT=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v3'
TASK=Path('D:/download/writing-block (3).md')


def sources():
    files=[PACKAGE/n for n in ('__init__.py','common.py','network.py','backend.py','training.py','run.py','mechanism.py','analysis.py','README.md')]
    files += [ROOT/'tests/test_direct_ppo_training_ablation_v4.py']
    files += list((ROOT/'experiments/cfs_direct_ppo_v3').glob('*.py'))
    files += [ROOT/'experiments/cfs_direct_ppo_diagnosis/common.py']
    return dict(core=source_identity(),files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)})


def load():
    m=read_json(TARGET/'manifest.json')
    if m['status'] in ('COMPLETED','FAILED','BUDGET_LIMITED'):
        raise RuntimeError('immutable terminal attempt')
    if m['source']!=sources():raise RuntimeError('frozen numerical source changed')
    verify_hashes(TARGET,m['input_hashes'])
    return m


def configuration(name,variant='A',seed=30001):
    c=read_json(TARGET/'protocol'/(name+'.json'))
    c['ppo']['training_seed']=seed
    if variant=='B':c['ppo']['reward_scale']=10000.
    return c


def scenes(name,split):
    return read_json(TARGET/('sealed' if split=='test' else 'data')/name/(split+'.json'))['scenarios']


def parent_choice(name,seed):
    old=next(r for r in read_json(PARENT/'protocol/frozen-policies.json')['rows']
             if r['condition']==name and r['seed']==seed)
    scores=[(read_json(PARENT/'training'/name/str(seed)/f'validation-{u:04d}.json')['mean_profit_USD'],u) for u in (8,16)]
    score,u=max(scores,key=lambda r:(r[0],-r[1]))
    return dict(old,selected_update=u,best_validation_USD=score,episodes=64,
                checkpoint=f'training/{name}/{seed}/inference-{u:04d}',
                A_budget='first16update prefix; choose only8/16, never32')


def references(name):
    return [r for r in read_json(TARGET/'protocol/reference-states.json') if r['condition']==name]


class Budget:
    def __init__(self):
        import psutil
        self.process=psutil.Process()
        self.path=TARGET/'qa/numerical-budget.json'
        self.state=read_json(self.path) if self.path.exists() else dict(calls=0,seconds=0.,started_at=now())
        self.started=time.monotonic()
        self.base=self.state['seconds']
        self.last_disk_check=0

    def __call__(self):
        if self.base+time.monotonic()-self.started>14400:raise TimeoutError('240 minute numerical budget')
        if self.process.memory_info().rss>2*2**30:raise MemoryError('2GiB process budget')
        if shutil.disk_usage(TARGET).free<5*2**30:raise RuntimeError('less than 5GiB free disk')
        if self.state['calls']-self.last_disk_check>=3072:
            size=sum(p.stat().st_size for p in TARGET.rglob('*') if p.is_file())
            if size>15*2**30:raise TimeoutError('15GiB output budget')
            self.state['output_bytes']=size;self.last_disk_check=self.state['calls']
        return False

    def claim(self,count=1):
        self()
        if self.state['calls']+count>180000:raise TimeoutError('180000 MILP call budget')
        self.state['calls']+=count
        self.save()

    def save(self):
        self.state['seconds']=self.base+time.monotonic()-self.started
        write_json(self.path,self.state)


def prepare():
    from model_families.single.learning.ppo.runner import scenario_content
    from experiments.cfs_direct_ppo_v3.run import sources as old_sources
    from experiments.cfs_direct_ppo_diagnosis.common import observation
    if TARGET.exists():raise FileExistsError('new attempt required')
    parent=read_json(PARENT/'final_manifest.json');diag=read_json(DIAG/'final_manifest.json')
    if parent['status']!='COMPLETED' or diag['status']!='COMPLETED' or parent['source']!=old_sources():
        raise RuntimeError('immutable parents/source invalid')
    history=dict(parent['history_hashes'])
    for folder,m in ((PARENT,parent),(DIAG,diag)):
        history.update({str(folder/r):h for r,h in m['final_artifact_hashes'].items()})
        history[str(folder/'final_manifest.json')]=sha(folder/'final_manifest.json')
    history[str(TASK)]=sha(TASK)
    failed=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v1'
    if read_json(failed/'qa/stop.json')['stage']!='smoke':raise RuntimeError('expected immutable smoke stop')
    fm=read_json(failed/'manifest.json');verify_hashes(ROOT,fm['source']['files'])
    history.update({str(ROOT/r):h for r,h in fm['source']['files'].items()})
    history.update({str(p):sha(p) for p in failed.rglob('*') if p.is_file()})
    stopped=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v2'
    sm=read_json(stopped/'manifest.json')
    if sm['status']!='BUDGET_LIMITED':raise RuntimeError('expected immutable v2 budget stop')
    verify_hashes(ROOT,sm['source']['files'])
    history.update({str(ROOT/r):h for r,h in sm['source']['files'].items()})
    history.update({str(p):sha(p) for p in stopped.rglob('*') if p.is_file()})
    sm=read_json(SMOKE_PARENT/'manifest.json')
    gate=read_json(SMOKE_PARENT/'qa/smoke.json')
    if len(gate['rows'])!=12 or gate['status']!='PASSED' or any((SMOKE_PARENT/'training').iterdir()):
        raise RuntimeError('require complete engineering-only v3; no formal models')
    verify_hashes(ROOT,sm['source']['files'])
    history.update({str(ROOT/r):h for r,h in sm['source']['files'].items()})
    history.update({str(p):sha(p) for p in SMOKE_PARENT.rglob('*') if p.is_file()})
    for n in ('network.py','backend.py','training.py'):
        if sha(PACKAGE/n)!=sha(ROOT/'experiments/cfs_direct_ppo_training_ablation_v3'/n):
            raise RuntimeError('engineering reuse requires byte-identical training math')
    verify_hashes(Path('/'),history)
    # Existing published numeric indexes represent actual use, not unused prepared pools.
    used_seeds=set();used_contents=set();records=0
    directories=[ROOT/'reports/cfs_ppo_strategic_control',ROOT/'experiments/cfs_ppo_strategic_control/runs',
                 ROOT/'experiments/cfs_ppo_signal_action_diagnostic/runs',ROOT/'experiments/cfs_ppo_decision_coverage/runs']
    for directory in directories:
        for p in directory.rglob('*.json'):
            if any(x in p.parts for x in ('data','sealed','protocol','figures','qa','analysis')):continue
            # Only actual operational records; never count unused preparation as evaluation.
            if not (p.name.startswith(('training-episode-','validation-','evaluation-','MC-','M0-','Pure-MC-'))
                    or (p.parent.name in ('m0','zero','mc','evaluation'))):continue
            r=read_json(p)
            if isinstance(r,dict) and 'transitions' in r and 'scenario_seed' in r:
                used_seeds.add(r['scenario_seed']);records+=1
    if set(TESTS)&used_seeds:raise RuntimeError('new test seeds previously executed')
    TARGET.mkdir(parents=True)
    for d in ('protocol','data','sealed','smoke','training','validation','evaluation','mechanism','analysis','figures','qa'):
        (TARGET/d).mkdir()
    refs=[];identities=[]
    original=read_json(PREPARED/'manifest.json')
    for name in NAMES:
        cfg=read_json(PARENT/'protocol'/(name+'.json'))
        cfg['ppo'].update(updates=16,validation_every=8)
        write_json(TARGET/'protocol'/(name+'.json'),cfg)
        parent_data={split:read_json(PARENT/'data'/name/(split+'.json')) for split in ('train','validation')}
        sealed=PREPARED/'sealed'/name/'test.json'
        if sha(sealed)!=original['input_hashes'][sealed.relative_to(PREPARED).as_posix()]:raise RuntimeError('prepared pool SHA')
        all_tests=read_json(sealed)['scenarios'];new=[s for s in all_tests if s['seed'] in TESTS]
        if [s['seed'] for s in new]!=TESTS:raise RuntimeError('new test pool incomplete')
        old_content=set(scenario_content(s) for data in parent_data.values() for s in data['scenarios'])
        old_content.update(scenario_content(s) for s in read_json(PARENT/'sealed'/name/'test.json')['scenarios'])
        legacy=ROOT/'experiments/cfs_ppo_strategic_control/runs/cfs_laptop_ppo_v1'
        for split in ('train','validation','diagnostic','test'):
            p=legacy/('sealed' if split=='test' else 'data')/name/(split+'.json')
            if p.exists():old_content.update(scenario_content(s) for s in read_json(p)['scenarios'])
        new_content=[scenario_content(s) for s in new]
        if set(new_content)&old_content or len(set(new_content))!=24:raise RuntimeError('test content collision')
        for split,data in [*parent_data.items(),('test',dict(scenarios=new))]:
            folder=TARGET/('sealed' if split=='test' else 'data')/name;folder.mkdir(exist_ok=True)
            write_json(folder/(split+'.json'),data)
            identities.extend(dict(condition=name,split=split,seed=s['seed'],content=scenario_content(s)) for s in data['scenarios'])
        ch=parent_choice(name,30001)
        for scene in parent_data['validation']['scenarios'][:4]:
            path=PARENT/'training'/name/'30001'/f'validation-{ch["selected_update"]:04d}-s{scene["seed"]}.json'
            r=read_json(path)
            for t in (4,12,20):
                o=observation(r['transitions'][t]['observation'])
                refs.append(dict(condition=name,scenario_seed=scene['seed'],period=t,pending=len(o.orders),
                    observation=r['transitions'][t]['observation'],observation_hash=o.hash,source=str(path),source_sha256=sha(path)))
    write_json(TARGET/'protocol/reference-states.json',refs)
    write_json(TARGET/'protocol/A-prefix-policies.json',dict(rows=[dict(parent_choice(n,s),condition=n,seed=s) for n in NAMES for s in SEEDS],rule='only8/16 frozen checkpoints; earliest tie'))
    write_json(TARGET/'data/identities.json',identities)
    protocol=dict(version='direct-ppo-training-ablation/v1',created_at=now(),variants=['A','B','C','D'],
        training_seeds=list(SEEDS),conditions=list(NAMES),train_seeds=list(range(20032,20160)),
        validation_seeds=list(range(21008,21024)),test_seeds=TESTS,updates=16,episodes_per_update=4,
        validation_updates=[8,16],checkpoint_selection='maximum deterministic validation mean; earliest tie',
        validation_method_winner='largest mean best-validation over three seeds among B,C,D; B,C,D tie order',
        primary_family=9,secondary_MC_family=12,bootstrap_replicates=20000,bootstrap_seed=61000,
        bootstrap='crossed seed and scenario, matched corresponding seeds; percentile Bonferroni intervals',
        reward_divisors=dict(A=1000,B=10000,C=1000,D=1000),C_baseline='all previously completed own training episodes; update AFTER optimization',
        D='independent encoders/backbones, Adam and clipping; actor init exactly aligned; critic seed+2001',
        A_historical_gradient='missing; no diagnostic reproduction added',references_per_condition=12,
        stochastic_validation_seeds=[21008,21009,21010,21011],stochastic_policy_seed=62000,
        probes='three predetermined t4/12/20 states per condition, all4 variants x3 seeds x4controls(zero,det,2sample)',
        counterfactual='first canonically changed control per condition/variant, at most12cases; same A30001 continuation',
        probe_seed=63000,maximum_calls=180000,wall_seconds=14400,output_bytes_limit=15*2**30,
        process_bytes_limit=2*2**30,workers=1,solver_threads=1,
        expected_training_episodes=1728,expected_validation_episodes=864,expected_test_episodes=1008,
        expected_stochastic_validation_episodes=144,full_cooldown=True,no_MC_for_Direct=True,
        user_amendment='2026-10-13: maintain240minute limit, explicitly halve experiment;16updates for B/C/D and A first16 prefix, same3conditions/3seeds/test24; MC unchanged',
        engineering_reuse='300 successful v3 smoke days, byte-identical network/backend/training; read-only reuse and charge actual7200calls/time')
    write_json(TARGET/'protocol/preregistration.json',protocol)
    shutil.copyfile(TASK,TARGET/'protocol/task.md')
    shutil.copyfile(PACKAGE/'README.md',TARGET/'experiment_protocol.md')
    inputs={p.relative_to(TARGET).as_posix():sha(p) for d in ('protocol','data','sealed') for p in (TARGET/d).rglob('*') if p.is_file()}
    write_json(TARGET/'manifest.json',dict(status='PREPARED',created_at=now(),source=sources(),
        input_hashes=inputs,history_hashes=history,parent_manifest_sha256=sha(PARENT/'final_manifest.json')))
    write_json(TARGET/'qa/prepare.json',dict(status='PASSED',history_files=len(history),
        historical_operational_records_checked=records,new_test_seed_overlap=0,new_test_content_overlap=0,
        reference_states=len(refs),new_test_scenarios=72,all_parent_hashes_checked=True))
    print('PREPARED',len(history),'history files,',records,'actual previous records; sealed72new tests',flush=True)
