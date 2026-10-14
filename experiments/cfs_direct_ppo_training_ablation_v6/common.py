"""Independent attempt, immutable input/source lineage and numerical resource caps."""
import copy
import random
import shutil
import time
from pathlib import Path
from model_families.single.strategic.artifacts import read_json,source_identity
from .io_retry import write_json
from experiments.cfs_ppo_strategic_control.study_common import ROOT,CONDITIONS,key,sha,now
from experiments.cfs_ppo_signal_action_diagnostic.run import verify_hashes

PACKAGE=Path(__file__).resolve().parent
TARGET=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v6'
PARENT=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_no_mc_v3'
DIAG=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v2'
PREPARED=ROOT/'experiments/cfs_ppo_strategic_control/runs/cfs_first_ppo_v1'
SEEDS=(30001,30002,30003)
NAMES=tuple(key(c) for c in CONDITIONS)
TESTS=list(range(22044,22068))
VARIANTS=('B','C','D')
FAILED_PARENT=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v4'
SMOKE_PARENT=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v3'
TASK=Path('D:/download/writing-block (3).md')


def sources():
    files=[PACKAGE/n for n in ('__init__.py','common.py','network.py','backend.py','training.py','run.py','mechanism.py','analysis.py','io_retry.py','README.md')]
    files += [ROOT/'tests/test_direct_ppo_training_ablation_v6.py',ROOT/'tests/test_direct_ppo_training_ablation_v6_recovery.py']
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
        from .io_retry import RETRY_EVENTS
        self.state['current_process_IO_retry_events']=list(RETRY_EVENTS)
        write_json(self.path,self.state)


RECOVERY_PARENT=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v5'

def prepare():
    """Read-only v5 recovery. No scenario generation, checkpoint or outcome selection."""
    if TARGET.exists():raise FileExistsError('new attempt required')
    old=read_json(RECOVERY_PARENT/'manifest.json')
    stop=read_json(RECOVERY_PARENT/'qa/stop.json')
    if old['status']!='FAILED' or stop['error']!='PermissionError' or 'numerical-budget.json' not in stop['message']:
        raise RuntimeError('only authorized atomic-write failure may resume')
    verify_hashes(ROOT,old['source']['files'])
    verify_hashes(RECOVERY_PARENT,old['input_hashes'])
    verify_hashes(Path('/'),old['history_hashes'])
    if (RECOVERY_PARENT/'protocol/frozen-policies.json').exists() or any((RECOVERY_PARENT/'evaluation').iterdir()):
        raise RuntimeError('sealed test was already opened')
    # Core math and analysis must be unchanged; only package path differs in analysis.
    prior=ROOT/'experiments/cfs_direct_ppo_training_ablation_v5'
    for n in ('network.py','backend.py','mechanism.py'):
        if sha(PACKAGE/n)!=sha(prior/n):raise RuntimeError('numerical math changed '+n)
    if (PACKAGE/'analysis.py').read_text(encoding='utf-8').replace('ablation_v6','ablation_v5')!= (prior/'analysis.py').read_text(encoding='utf-8'):
        raise RuntimeError('statistical math changed')
    evidence=ROOT/'workspace/ablation_v6_recovery_preflight.xml'
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(evidence).getroot().iter('testsuite'))
    if sum(int(r.get('tests',0)) for r in suites)<90 or any(int(r.get(k,0)) for r in suites for k in ('failures','errors','skipped')):
        raise RuntimeError('real recovery preflight required')
    history=dict(old['history_hashes'])
    history.update({str(ROOT/r):h for r,h in old['source']['files'].items()})
    history.update({str(p):sha(p) for p in RECOVERY_PARENT.rglob('*') if p.is_file()})
    verify_hashes(Path('/'),history)
    TARGET.mkdir()
    for d in ('protocol','data','sealed'):
        shutil.copytree(RECOVERY_PARENT/d,TARGET/d)
    for d in ('smoke','training','validation','evaluation','mechanism','analysis','figures','qa'):
        (TARGET/d).mkdir()
    copied=[];previous=read_json(RECOVERY_PARENT/'protocol/reused-models.json')['rows']
    for name in NAMES:
        for variant in VARIANTS:
            for seed in SEEDS:
                folder=RECOVERY_PARENT/'training'/name/variant/str(seed)
                if not folder.exists():continue
                status=read_json(folder/'status.json')
                if status['status']!='COMPLETED':continue
                if status['episodes']!=64:raise RuntimeError('completed prefix coverage')
                destination=TARGET/'training'/name/variant/str(seed)
                shutil.copytree(folder,destination)
                hashes={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}
                verify_hashes(destination,hashes)
                record=next((r for r in previous if (r['condition'],r['variant'],r['seed'])==(name,variant,seed)),None)
                copied.append(dict(condition=name,variant=variant,seed=seed,source=record['source'] if record else old['source'],
                    original_folder=str(folder),copied_folder=destination.relative_to(TARGET).as_posix(),hashes=hashes))
    if len(copied)!=4:raise RuntimeError('expected four completed models')
    write_json(TARGET/'protocol/reused-models.json',dict(rows=copied))
    partial=RECOVERY_PARENT/'training'/NAMES[0]/'C'/'30002'
    import torch
    cfg=configuration(NAMES[0],'C',30002)
    saved=torch.load(partial/'recovery.pt',map_location='cpu',weights_only=False)
    if saved['update']!=1 or saved['baseline_episodes']!=4 or saved['config']!=cfg:
        raise RuntimeError('resume checkpoint does not match saved update1 configuration')
    prefix=[partial/'initialization.json',partial/'update-0001.json',*[partial/f'training-episode-{i:06d}.json' for i in range(1,5)]]
    if len(list(partial.glob('training-episode-*.json')))!=4:raise RuntimeError('partial day boundary')
    destination=TARGET/'training'/NAMES[0]/'C'/'30002';destination.mkdir(parents=True)
    for p in prefix:shutil.copyfile(p,destination/p.name)
    snap=TARGET/'protocol/resume';snap.mkdir()
    shutil.copyfile(partial/'recovery.pt',snap/'post-update-0001.pt')
    prefix_hashes={p.name:sha(p) for p in prefix};verify_hashes(destination,prefix_hashes)
    write_json(destination/'status.json',dict(status='RESUME_READY',variant='C',episodes=4))
    write_json(TARGET/'protocol/resume-model.json',dict(condition=NAMES[0],variant='C',seed=30002,
        checkpoint='protocol/resume/post-update-0001.pt',checkpoint_sha256=sha(snap/'post-update-0001.pt'),
        prefix_hashes=prefix_hashes,original_folder=str(partial),previous_source=old['source'],
        update=1,completed_days=4,all_three_RNG_optimizer_network_baseline=True))
    inherited=read_json(RECOVERY_PARENT/'qa/numerical-budget.json')
    if inherited['calls']!=16535 or inherited['calls']-(7200+4*96*24+4*24)!=23:
        raise RuntimeError('failed reservation cost chain')
    write_json(TARGET/'protocol/reused-budget.json',inherited)
    write_json(TARGET/'protocol/recovery-amendment.json',dict(
        authorized='user agreed bounded atomic-write retries and post-update checkpoint recovery',
        failed_parent=str(RECOVERY_PARENT),old_status='FAILED',completed_models=4,completed_partial_days=4,
        orphan_reserved_calls=23,orphan_calls_not_independently_verified_solver_executions=True,
        cumulative_budget_seconds=inherited['seconds'],cumulative_reserved_calls=inherited['calls'],
        unchanged_updates=16,unchanged_limits=dict(seconds=14400,calls=180000,bytes=15*2**30),
        retry_scope='isolated JSON writer; PermissionError only; eight attempts; same serialization; no RNG',
        interrupted_rollout='not a sample; restore saved post-update1 RNGs, replay update2 schedule',
        preflight_XML_sha256=sha(evidence)))
    shutil.copyfile(evidence,TARGET/'qa/pytest-recovery-preflight.xml')
    shutil.copyfile(PACKAGE/'README.md',TARGET/'experiment_protocol.md')
    inputs={p.relative_to(TARGET).as_posix():sha(p) for d in ('protocol','data','sealed') for p in (TARGET/d).rglob('*') if p.is_file()}
    write_json(TARGET/'manifest.json',dict(status='PREPARED',created_at=now(),source=sources(),
        input_hashes=inputs,history_hashes=history,parent_manifest_sha256=sha(PARENT/'final_manifest.json')))
    write_json(TARGET/'qa/prepare.json',dict(status='PASSED',history_files=len(history),
        all_parent_hashes_checked=True,completed_reused_models=4,resume_update=1,sealed_test_not_opened=True))
    write_json(TARGET/'qa/recovery.json',dict(status='PASSED',real_preflight_XML_sha256=sha(evidence),
        bounded_IO_and_B_C_D_resume_equality_tested=True,completed_models_copied_byte_identically=4,
        partial_prefix_hashes=prefix_hashes,orphan_reserved_calls=23))
    print('PREPARED recovery',len(history),'historical files, four completed models and update1 prefix',flush=True)


def model_source(name,variant,seed,current_source):
    records=read_json(TARGET/'protocol/reused-models.json')['rows']
    row=next((r for r in records if (r['condition'],r['variant'],r['seed'])==(name,variant,seed)),None)
    return row['source'] if row else current_source

