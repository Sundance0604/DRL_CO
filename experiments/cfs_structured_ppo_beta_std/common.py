import json
import os
import re
import time
from pathlib import Path
from model_families.single.strategic.artifacts import read_json,source_identity
from model_families.single.strategic.contracts import identity
from experiments.cfs_direct_ppo_training_ablation_v6.io_retry import write_json
from experiments.cfs_ppo_strategic_control.study_common import ROOT,sha,now
from experiments.cfs_ppo_signal_action_diagnostic.run import verify_hashes

PACKAGE=Path(__file__).resolve().parent
TARGET=ROOT/'reports/cfs_ppo_strategic_control/structured_ppo_beta_std_pilot_v1'
OLD=ROOT/'reports/cfs_ppo_strategic_control/structured_ppo_q_light_v2'
MECHANISM=ROOT/'reports/cfs_ppo_strategic_control/structured_ppo_q_mechanism_v1'
ABLATION=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v6'
W40='western_multicluster-n40-k8'
OLD_TRAIN=OLD/'training'/W40/'30001'
POOL=ROOT/'experiments/cfs_ppo_strategic_control/runs/cfs_first_ppo_v1/data'/W40/'validation.json'
TASK=Path('D:/download/writing-block (6).md')


def configuration():return read_json(OLD_TRAIN/'inference-0008/manifest.json')['config']
def source():
    files=[*PACKAGE.glob('*.py'),PACKAGE/'README.md',ROOT/'tests/test_structured_beta_std.py']
    files=[p for p in files if p.name not in ('delivery.py','qa.py')]
    parent=read_json(MECHANISM/'final_manifest.json')['source']
    return dict(core=source_identity(),files={str(p):sha(p) for p in files},parent=parent)


def load():
    m=read_json(TARGET/'manifest.json')
    if m['status'] in ('COMPLETED','FAILED','BUDGET_LIMITED'):raise RuntimeError('immutable terminal attempt')
    if m['source']!=source():raise RuntimeError('frozen source differs')
    if sha(TARGET/'protocol.json')!=m['protocol_sha256']:raise RuntimeError('protocol differs')
    return m


class Budget:
    def __init__(self):
        self.path=TARGET/'qa/budget.json';self.state=read_json(self.path)
        self.base=self.state['seconds'];self.started=time.monotonic()
    def __call__(self):
        if self.base+time.monotonic()-self.started>=600:raise TimeoutError('600 seconds hard numerical cap')
        return False
    def claim(self):
        self()
        if self.state['calls']>=1000:raise TimeoutError('1000 new solver claims cap')
        self.state['calls']+=1
        self.save()
    def save(self):
        self.state['seconds']=self.base+time.monotonic()-self.started;write_json(self.path,self.state)
        if sum(p.stat().st_size for p in TARGET.rglob('*') if p.is_file())>512*2**20:raise TimeoutError('512MiB storage cap')


def prepare():
    if TARGET.exists():raise FileExistsError('independent output already exists')
    parent=read_json(MECHANISM/'final_manifest.json')
    if parent['status']!='COMPLETED' or sha(MECHANISM/'final_manifest.json')!='f5ed86ef5f59645184532cb292bbb1f10ae2a83b18e70fcb12ec7d010a59bc92':raise RuntimeError('mechanism completion differs')
    verify_hashes(MECHANISM,parent['artifact_hashes']);verify_hashes(Path('/'),parent['history_hashes'])
    if parent['source']['core']!=source_identity():raise RuntimeError('physical source changed')
    verify_hashes(Path('/'),parent['source']['files'])
    history=dict(parent['history_hashes']);history.update({str(MECHANISM/'final_manifest.json'):sha(MECHANISM/'final_manifest.json'),str(TASK):sha(TASK)})
    # Original trained checkpoint metadata was frozen through its recovery parent,
    # even though the original inventory omitted nested filenames manifest.json.
    om=read_json(OLD/'final_manifest.json');am=read_json(ABLATION/'final_manifest.json')
    used=[POOL,ABLATION/'data'/W40/'train.json',MECHANISM/'states-frozen.json',
          OLD/'structured_ppo_q_vs_mc_report.md',ABLATION/'direct_ppo_training_ablation_report.md']
    used += [OLD_TRAIN/f'training-{i:04d}.json' for i in range(1,33)]
    used += [OLD_TRAIN/f'update-{u:04d}.json' for u in range(1,9)]
    used += list((OLD_TRAIN/'inference-0008').iterdir())
    used += list((MECHANISM/'raw/kl').glob(f'PPO20-{W40}-30001-*.json'))
    used += list((MECHANISM/'raw/probes').glob('*-m30001.json'))
    original_pool=POOL.parents[2];pool_manifest=read_json(original_pool/'manifest.json')
    if pool_manifest['input_hashes'].get(POOL.relative_to(original_pool).as_posix())!=sha(POOL):raise RuntimeError('frozen prepared development pool changed')
    history[str(original_pool/'manifest.json')]=sha(original_pool/'manifest.json')
    for p in used:
        h=sha(p)
        if p.is_relative_to(OLD):
            rel=p.relative_to(OLD).as_posix();expected=om['artifact_hashes'].get(rel) or om['history_hashes'].get(str(Path(om['recovery_parent'])/rel))
            if expected!=h:raise RuntimeError('frozen baseline input differs '+str(p))
        if p.is_relative_to(ABLATION) and am['final_artifact_hashes'].get(p.relative_to(ABLATION).as_posix())!=h:raise RuntimeError('train pool differs')
        history[str(p)]=h
    # Do not inspect previous test profits. Dataset identities and trajectory
    # filenames/immutable inventory keys identify prior uses without outcomes.
    used_seeds=set();used_content=set();catalog=[];executed_names=[]
    from model_families.single.learning.ppo.runner import scenario_content
    for base in (ROOT/'reports/cfs_ppo_strategic_control',ROOT/'experiments/cfs_ppo_strategic_control/runs'):
        for directory,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d not in ('__pycache__','baselines','tables') and not re.match(r'^(update-\d+|inference-\d+|best-inference|last-inference)$',d)]
            for name in files:
                p=Path(directory)/name
                match=re.search(r'(?:-s|scenario-)(\d+)\.json$',name)
                if match:used_seeds.add(int(match[1]));executed_names.append(str(p))
                if name=='final_manifest.json':
                    fm=read_json(p)
                    for rel in fm.get('artifact_hashes',fm.get('final_artifact_hashes',{})):
                        match=re.search(r'(?:-s|scenario-)(\d+)\.json$',rel)
                        if match:used_seeds.add(int(match[1]));executed_names.append(str(p.parent/rel))
                if name in ('train.json','validation.json','test.json','diagnostic.json','pilot_train.json','pilot_validation.json','engineering.json'):
                    data=read_json(p);scenes=data.get('scenarios',[])
                    if scenes:catalog.append((p,scenes));history[str(p)]=sha(p)
    for p,scenes in catalog:
        for s in scenes:
            if s['seed'] in used_seeds or s['split']=='train':used_content.add(scenario_content(s))
    available={s['seed']:s for s in read_json(POOL)['scenarios']};picked=[];skipped=[]
    candidate=21024
    while len(picked)<4:
        if candidate not in available:raise RuntimeError('unused four scenes unavailable in prepared pool; register generation before training')
        s=available[candidate]
        if candidate in used_seeds or scenario_content(s) in used_content:skipped.append(candidate)
        else:picked.append(s)
        candidate+=1
    scenes=read_json(ABLATION/'data'/W40/'train.json')['scenarios'];train_by_seed={s['seed']:s for s in scenes}
    schedule=[read_json(OLD_TRAIN/f'training-{i:04d}.json')['scenario_seed'] for i in range(1,33)]
    if {s['seed'] for s in picked}&set(schedule):raise RuntimeError('development/train overlap')
    cost=sum(read_json(OLD_TRAIN/f'training-{i:04d}.json')['metrics']['episode_seconds'] for i in range(1,33))
    # Episode timings are measured historical wall time; multiply conservatively
    # for updates/instrumentation/IO, not an asserted new runtime measurement.
    predicted=cost/32*40*4+30
    if predicted>600:raise TimeoutError('historical conservative runtime forecast exceeds cap')
    TARGET.mkdir(parents=True)
    for d in ('qa','training','training/kl','snapshots','probes','evaluation','figures','data'):(TARGET/d).mkdir(exist_ok=True)
    protocol=dict(condition=W40,training_seed=30001,updates=8,episodes_per_update=4,
        change='ONLY beta initial log_std -1.5 to -.5; alpha remains -.5',same_initial_tensors_except_beta_log_std=True,
        data_sampling='original random.Random(31004), same pool/order; exact32 historical seeds',training_schedule=schedule,
        sampling_rng=31002,minibatch_rng=31003,standard_Gaussian_noise='same sampling stream, 24period/day; actual actions/states may differ; on-policy retained',
        development_seeds=[s['seed'] for s in picked],skipped_development_seeds=skipped,
        development_provenance='previously prepared but not executed CFS pool; new evaluation, NOT newly collected real dates',
        checkpoint='fixed update8, no best checkpoint/seed selection',q_states='frozen7active+2only from previous mechanism; reused observed development set',
        old_q_results='read-only frozen A_learned/B_beta0 for model30001; only18 new probes',
        inference='deterministic tanh(mu)',no_MC=True,no_formal_test=True,
        caps=dict(seconds=600,calls=1000,bytes=512*2**20),planned_calls=978,
        no_extra_smoke_solves=True,forecast_seconds=predicted,forecast_source='historical32episode walltime x40/32 x4 +30s',
        inference_profit='four paired development days; no significance test',
        diagnostics='actual KL checks; mean/std split; beta mean absolute and within-state dispersion; H1/H2/H3 separate')
    write_json(TARGET/'protocol.json',protocol);write_json(TARGET/'data/development.json',dict(scenarios=picked))
    write_json(TARGET/'qa/isolation.json',dict(status='PASSED',chosen=protocol['development_seeds'],skipped=skipped,
        used_seed_inventory=sorted(used_seeds),executed_filename_inventory=executed_names,
        content_hashes=[scenario_content(s) for s in picked],dataset_catalog=[str(p) for p,_ in catalog],
        forbidden_formal_intervals=[[22020,22043],[22044,22067],[22068,22075]],no_outcome_based_selection=True))
    write_json(TARGET/'qa/budget.json',dict(calls=0,seconds=0.))
    write_json(TARGET/'manifest.json',dict(status='PREPARED',created_at=now(),source=source(),history_hashes=history,
        protocol_sha256=sha(TARGET/'protocol.json'),development_sha256=sha(TARGET/'data/development.json')))
    print('PREPARED',protocol['development_seeds'],'forecast seconds',predicted,flush=True)
