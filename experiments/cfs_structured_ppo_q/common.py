import copy
import time
from pathlib import Path
from model_families.single.strategic.artifacts import read_json, source_identity
from experiments.cfs_direct_ppo_training_ablation_v6.io_retry import write_json
from experiments.cfs_ppo_strategic_control.study_common import ROOT, sha, now
from experiments.cfs_ppo_signal_action_diagnostic.run import verify_hashes

PACKAGE=Path(__file__).resolve().parent
OLD=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v6'
TARGET=ROOT/'reports/cfs_ppo_strategic_control/structured_ppo_q_light_v1'
NAMES=('california_corridor-n80-k3','western_multicluster-n40-k8','western_multicluster-n80-k3')
SEEDS=(30001,30002,30003)
LABELS=('CA80/3','W40/8','W80/3')
TESTS=list(range(22068,22076))


def cfg(name,seed=30001):
    c=read_json(OLD/'protocol'/(name+'.json'))
    c['ppo'].update(reward_scale=10000.,updates=8,validation_every=8,training_seed=seed)
    return c


def scenes(name,split):
    if split=='test':return read_json(TARGET/'evaluation/scenarios'/f'{name}.json')['scenarios']
    s=read_json(OLD/'data'/name/(split+'.json'))['scenarios']
    return s[:4] if split=='validation' else s


def sources():
    paths=[PACKAGE/n for n in ('__init__.py','control.py','network.py','common.py','run.py','training.py','analysis.py','README.md')]
    paths+=[ROOT/'tests/test_structured_ppo_q.py']
    paths+=list((ROOT/'experiments/cfs_direct_ppo_v3').glob('*.py'))
    paths += [ROOT/'experiments/cfs_direct_ppo_training_ablation_v6'/n for n in ('backend.py','network.py','io_retry.py')]
    paths += [ROOT/'experiments/cfs_direct_ppo_diagnosis/common.py']
    return dict(core=source_identity(),files={str(p):sha(p) for p in paths})


def load():
    m=read_json(TARGET/'manifest.json')
    if m['status'] in ('COMPLETED','FAILED','BUDGET_LIMITED'):raise RuntimeError('immutable terminal attempt')
    if sources()!=m['source']:raise RuntimeError('frozen code changed')
    verify_hashes(Path('/'),m['history_hashes'])
    return m


class Budget:
    def __init__(self):
        self.path=TARGET/'qa/budget.json'
        self.state=read_json(self.path) if self.path.exists() else dict(calls=0,seconds=0.)
        self.base=self.state['seconds'];self.start=time.monotonic()
    def __call__(self):
        if self.base+time.monotonic()-self.start>1800:raise TimeoutError('30minute numerical cap')
        if self.state['calls']%512==0:
            if sum(p.stat().st_size for p in TARGET.rglob('*') if p.is_file())>2*2**30:
                raise TimeoutError('2GiB result cap')
        return False
    def claim(self,count=1):
        self()
        if self.state['calls']+count>20000:raise TimeoutError('20000 solver cap')
        self.state['calls']+=count
        self.save()
    def save(self):
        self.state['seconds']=self.base+time.monotonic()-self.start;write_json(self.path,self.state)


def old_policy(name,seed=30001):
    from experiments.cfs_direct_ppo_v3.network import DirectPolicy
    return DirectPolicy(OLD/'training'/name/'B'/str(seed)/'inference-0008')


def prepare():
    if TARGET.exists():raise FileExistsError('independent attempt already exists')
    fm=read_json(OLD/'final_manifest.json')
    if fm['status']!='COMPLETED' or sha(OLD/'final_manifest.json')!='42249ce18e53a40dc8974baab604f6f3dc6cffd44a07775d001c0bd27d192a9e':
        raise RuntimeError('old completion identity mismatch')
    history={str(OLD/'final_manifest.json'):sha(OLD/'final_manifest.json')}
    # Verify only used old files against their immutable completion inventory.
    def use(p):
        h=sha(p)
        if p.is_relative_to(OLD) and p!=OLD/'final_manifest.json':
            r=p.relative_to(OLD).as_posix()
            if fm['final_artifact_hashes'].get(r)!=h:raise RuntimeError('old frozen artifact differs: '+r)
        history[str(p)]=h
    refs=read_json(OLD/'protocol/reference-states.json')
    chosen=[r for r in refs if r['scenario_seed'] in (21008,21009) and r['period'] in (4,12,20)]
    if len(chosen)!=18:raise RuntimeError('reference coverage')
    for name in NAMES:
        for split in ('train','validation'):use(OLD/'data'/name/(split+'.json'))
        use(OLD/'protocol'/(name+'.json'))
        for seed in SEEDS:
            cp=OLD/'training'/name/'B'/str(seed)/'inference-0008'
            for p in cp.iterdir():use(p)
            for i in range(1,33):use(OLD/'training'/name/'B'/str(seed)/f'training-episode-{i:06d}.json')
    for r in chosen:
        p=Path(r['source'])
        if sha(p)!=r['source_sha256']:raise RuntimeError('reference SHA differs')
        history[str(p)]=sha(p)
    for p in (OLD/'qa/tests.json',OLD/'qa/replay.json',OLD/'qa/exports.json',OLD/'protocol/reference-states.json'):
        use(p)
    use(Path('D:/download/writing-block (4).md'))
    # Record the actual MC baseline input; never used by Direct training.
    from experiments.cfs_direct_ppo_v3.run import PARENT as mc
    for name in NAMES:
        p=mc/'protocol'/(name+'.json');history[str(p)]=sha(p)
        c=read_json(p);bp=mc/c['baseline']
        if bp.is_dir():
            for f in bp.rglob('*'):
                if f.is_file():history[str(f)]=sha(f)
        else:history[str(bp)]=sha(bp)
    TARGET.mkdir(parents=True)
    for d in ('qa','protocol','pilot','training','validation','evaluation','figures'):(TARGET/d).mkdir()
    write_json(TARGET/'protocol/references.json',chosen)
    protocol=dict(kind='lightweight exploratory; user overrides16update24test',updates=8,
        episodes_per_update=4,validation_seeds=list(range(21008,21012)),test_seeds=TESTS,
        training_seeds=SEEDS,periods=[4,12,20],reference_seeds=[21008,21009],
        slopes=[-.6,-.2,0.,.2,.6],directions='each hub near+far; all hubs near+far',
        primary='PPO20-PPO10/B8',secondary='PPO20-PureMC; separate family',
        bootstrap=20000,bootstrap_seed=71000,adjustment='Bonferroni3 within family',
        cap_seconds=1800,cap_calls=20000,cap_bytes=2*2**30,
        gate='all correctness; equality-quotient rank added; occupiedq>=2; canonical switch',
        selection='only update8; no checkpoint search',formula='alpha+beta*psi; no clipping; max2Sc')
    write_json(TARGET/'protocol/preregistration.json',protocol)
    write_json(TARGET/'manifest.json',dict(status='PREPARED',created_at=now(),source=sources(),history_hashes=history,
        preregistration_sha256=sha(TARGET/'protocol/preregistration.json')))
