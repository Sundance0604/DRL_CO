import time
from pathlib import Path
from model_families.single.strategic.artifacts import read_json,source_identity
from experiments.cfs_direct_ppo_training_ablation_v6.io_retry import write_json
from experiments.cfs_ppo_strategic_control.study_common import ROOT,sha,now
from experiments.cfs_ppo_signal_action_diagnostic.run import verify_hashes

PACKAGE=Path(__file__).resolve().parent
P20=ROOT/'reports/cfs_ppo_strategic_control/structured_ppo_q_light_v2'
P10=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v6'
TARGET=ROOT/'reports/cfs_ppo_strategic_control/structured_ppo_q_mechanism_v1'
NAMES=('california_corridor-n80-k3','western_multicluster-n40-k8','western_multicluster-n80-k3')
SEEDS=(30001,30002,30003)
W40=NAMES[1]


def cfg(name,seed=30001):
    c=read_json(P10/'protocol'/(name+'.json'));c['ppo'].update(reward_scale=10000.,training_seed=seed)
    return c


def sources():
    paths=[PACKAGE/n for n in ('__init__.py','common.py','kl.py','states.py','run.py','README.md')]
    paths += [ROOT/'tests/test_structured_q_mechanism.py']
    paths += list((ROOT/'experiments/cfs_structured_ppo_q').glob('*.py'))
    paths += [ROOT/'experiments/cfs_direct_ppo_training_ablation_v6'/n for n in ('backend.py','network.py','io_retry.py')]
    paths += list((ROOT/'experiments/cfs_direct_ppo_v3').glob('*.py'))
    paths += [ROOT/'experiments/cfs_direct_ppo_diagnosis/common.py']
    return dict(core=source_identity(),files={str(p):sha(p) for p in paths})


def load():
    m=read_json(TARGET/'manifest.json')
    if m['status'] in ('COMPLETED','FAILED','BUDGET_LIMITED'):raise RuntimeError('immutable terminal experiment')
    if m['source']!=sources():raise RuntimeError('diagnostic source changed')
    if sha(TARGET/'protocol.json')!=m['protocol_sha256']:raise RuntimeError('protocol changed')
    return m


class Budget:
    def __init__(self):
        self.path=TARGET/'qa/budget.json'
        self.state=read_json(self.path) if self.path.exists() else dict(seconds=0.,calls=0)
        self.base=self.state['seconds'];self.start=time.monotonic()
    def __call__(self):
        if self.base+time.monotonic()-self.start>600:raise TimeoutError('10minute numerical budget')
        return False
    def claim(self,count=1):
        self()
        if self.state['calls']+count>500:raise TimeoutError('500 solver calls')
        self.state['calls']+=count
    def save(self):
        self.state['seconds']=self.base+time.monotonic()-self.start;write_json(self.path,self.state)


def prepare():
    if TARGET.exists():raise FileExistsError('new independent attempt required')
    history={}
    parents={P20:'e234f8a384760175a8b7f0805c4445342ff31a1136818a32806898ac5e7dd1c5',
             P10:'42249ce18e53a40dc8974baab604f6f3dc6cffd44a07775d001c0bd27d192a9e'}
    for root,expected in parents.items():
        f=root/'final_manifest.json';m=read_json(f)
        if m['status']!='COMPLETED' or sha(f)!=expected:raise RuntimeError('historical completion differs')
        history[str(f)]=expected
        inventory=m.get('artifact_hashes',m.get('final_artifact_hashes'))
        paths=[root/'qa/replay.json',root/'qa/tests.json']
        if root==P20:paths += [root/'structured_ppo_q_vs_mc_report.md',root/'protocol/preregistration.json']
        for name in NAMES:
            for seed in SEEDS:
                folder=root/'training'/name/str(seed) if root==P20 else root/'training'/name/'B'/str(seed)
                paths += [folder/f'update-{u:04d}.json' for u in range(1,9)]
                paths += [folder/(f'training-{i:04d}.json' if root==P20 else f'training-episode-{i:06d}.json') for i in range(1,33)]
                cp=folder/'inference-0008';paths += list(cp.iterdir())
                paths += [folder/'recovery.pt' if root==P20 else folder/'network-0008.pt']
        if root==P20:
            paths += list((root/'training'/W40/'30001').glob('validation-s*.json'))
        else:paths += [root/'data'/W40/'validation.json',root/'protocol'/(W40+'.json')]
        for p in paths:
            h=sha(p)
            rel=p.relative_to(root).as_posix();expected_hash=inventory.get(rel)
            # P20's completion excludes filenames manifest.json, including nested
            # inference metadata. Their immutable identities are in its full
            # recovered-parent history. Require that provenance, never self-hash.
            if expected_hash is None and root==P20:
                expected_hash=m['history_hashes'].get(str(Path(m['recovery_parent'])/rel))
            if expected_hash!=h:raise RuntimeError('historical used input differs '+str(p))
            history[str(p)]=h
    history['D:/download/writing-block (5).md']=sha('D:/download/writing-block (5).md')
    TARGET.mkdir(parents=True)
    for d in ('qa','raw/kl','raw/selection','raw/probes','raw/counterfactual','snapshots','figures'):(TARGET/d).mkdir(parents=True)
    protocol=dict(kind='mechanism diagnosis, NOT new training or formal test',caps=dict(seconds=600,calls=500,bytes=512*2**20),
        KL_models='both10/20,3conditions,3seeds,8updates; saved rollout reconstruction',
        KL_capture='Python frame at actual backend stopping check; no parameter/RNG mutation',
        state_pool=dict(scenario_seeds=[21008,21009,21010,21011],periods=[4,8,12,16,20],source='frozenPPO20/30001 validation'),
        selection='first8 provablyq-active + first2 q-only ordered byseed,time; retain shortages, no profit selection',
        qactive_certificate='copy original MILP constraints; maximize sum ordered q2 slots; OPTIMAL0 proves q-only',
        controls=['learned20','beta0','old10','fixed single witness-hub slope','fixed all-hub slope'],
        slopes=[-.6,-.2,.2,.6],seeds=SEEDS,
        counterfactual='first2 distinct states with canonical switch, state→seed→fixed control order; no profit selection',
        continuation='frozenPPO10/B8/30001 from next period',
        no_new_formal_days=True,no_hyperparameter_change=True,minimum_elapsed_claim='offline includes update reconstruction; does not fit new data')
    write_json(TARGET/'protocol.json',protocol)
    write_json(TARGET/'manifest.json',dict(status='PREPARED',created_at=now(),source=sources(),history_hashes=history,
        protocol_sha256=sha(TARGET/'protocol.json')))
