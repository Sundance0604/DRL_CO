import argparse,time
from pathlib import Path
from experiments.cfs_direct_ppo_diagnosis import common as c,numerics as n

OLD=c.TARGET
TARGET=c.ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v2'
PACKAGE=Path(__file__).resolve().parent
ORIGINAL_SOURCES=c.source_files

def sources():
    return ORIGINAL_SOURCES()|{p.relative_to(c.ROOT).as_posix():c.sha(p) for p in [PACKAGE/'__init__.py',PACKAGE/'run.py',PACKAGE/'README.md']}

def setup():
    # Full bytes are checked BEFORE entering the policy no-MC/no-sealed guard.
    # Cached metadata contains no table values or hidden scenario orders.
    parent=c.verify_parent(True)
    n.TARGET=TARGET;c.TARGET=TARGET;n.source_files=sources
    n.verify_parent=lambda:parent
    return parent

def prepare():
    if TARGET.exists():raise FileExistsError('new recovery required')
    old=c.read_json(OLD/'manifest.json')
    if old['status']!='FAILED' or c.read_json(OLD/'qa/m0.json')['status']!='PASSED' or list((OLD/'pilot').glob('*')):
        raise RuntimeError('expected pre-diagnostic guard failure only')
    c.verify_hashes(c.ROOT,old['source_files'])
    history={str(p):c.sha(p) for p in OLD.rglob('*') if p.is_file()}
    parent=setup()
    TARGET.mkdir(parents=True)
    for f in ('protocol','pilot','diagnostics','sensitivity','counterfactual','analysis','figures','qa'):(TARGET/f).mkdir()
    protocol=c.read_json(OLD/'protocol/preregistration.json')
    c.write_json(TARGET/'protocol/preregistration.json',protocol)
    (TARGET/'protocol/task.md').write_text((OLD/'protocol/task.md').read_text(encoding='utf-8'),encoding='utf-8')
    (TARGET/'experiment_protocol.md').write_text((c.PACKAGE/'README.md').read_text(encoding='utf-8')+'\n'+(PACKAGE/'README.md').read_text(encoding='utf-8'),encoding='utf-8')
    c.write_json(TARGET/'qa/m0.json',c.read_json(OLD/'qa/m0.json')|dict(read_only_reused=True,recovery_source=str(OLD)))
    c.write_json(TARGET/'manifest.json',dict(status='PREPARED',created_at=c.now(),source_files=sources(),
        parent_manifest_sha256=c.sha(c.PARENT/'final_manifest.json'),parent_artifacts=parent['final_artifact_hashes'],
        recovered_history=history,M0_directory=str(OLD/'m0'),zero_directory=str(OLD/'zero'),
        core=c.source_identity(),inputs={p.relative_to(TARGET).as_posix():c.sha(p) for p in (TARGET/'protocol').glob('*')}))
    c.write_json(TARGET/'qa/prepare.json',dict(status='PASSED',guard_placement_only=True,no_numeric_sample_protocol_changes=True,
        old_attempt_files=len(history),reused_M0_days=72,reused_zero_days=72))
    print('RECOVERY PREPARED; 144 frozen days reused',flush=True)

def run():
    import torch,psutil,os
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    parent=setup();m=c.read_json(TARGET/'manifest.json')
    if m['status']!='PREPARED' or m['source_files']!=sources():raise RuntimeError('recovery source/status mismatch')
    c.verify_hashes(Path('/'),m['recovered_history'])
    budget=n.Budget();old=c.read_json(OLD/'qa/m0.json');budget.calls=old['calls'];budget.started-=old['seconds']
    try:n.pilot(budget);n.diagnose(budget)
    except BaseException as exc:
        c.write_json(TARGET/'qa/stop.json',dict(status='FAILED',message=str(exc),error=type(exc).__name__))
        m=c.read_json(TARGET/'manifest.json');m.update(status='FAILED',stopped_at=c.now());c.write_json(TARGET/'manifest.json',m)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','run'));a=p.parse_args();globals()[a.stage]()
