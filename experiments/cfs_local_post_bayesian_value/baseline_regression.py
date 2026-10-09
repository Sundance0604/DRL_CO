"""Phase 0: independent re-execution of all four frozen zero-value regimes."""
from pathlib import Path
from experiment_core.storage import atomic_json, read_json, digest
from experiments.cfs_classical_value.common import ROOT, key, parameters, map_jobs
from experiments.cfs_classical_value.simulation import simulate
from model_families.single.data import matching_state

OUTPUT = ROOT / 'experiments/cfs_local_post_bayesian_value'

def physical(rows):
    return [{k:r[k] for k in ('period','before','after','drain') if k in r} |
            {'action':{k:r['plan'][k] for k in ('assignments','routes','repositions','expired')}} for r in rows]

def job(task):
    cid, label, policy = task
    source = ROOT / 'results/cfs_classical_value/raw/runs' / f'formal-{cid}-s3000-{label}'
    spec = read_json(source/'spec.json')
    sim, orders = matching_state(spec['scenario'],spec['parameters'],spec['solver'])
    metrics, trace, _ = simulate(sim,orders,policy,trace=True)
    prior = read_json(source/'summary.json')
    oldtrace = read_json(source/'trace.json')
    difference = metrics['realized_profit']-prior['realized_profit']
    identical = physical(trace)==physical(oldtrace)
    from .simulation import simulate as fourth_simulate
    s2,o2=matching_state(spec['scenario'],spec['parameters'],spec['solver'])
    m2,tr2,_=fourth_simulate(s2,o2,policy,trace=True)
    assert abs(m2['realized_profit']-prior['realized_profit'])<1e-6 and physical(tr2)==physical(oldtrace)
    record = dict(condition=cid,policy=policy,seed=3000,profit_difference=difference,
                  physical_trace_identical=identical,old_physical_hash=digest(physical(oldtrace)),
                  reproduced_physical_hash=digest(physical(trace)),fourth_physical_hash=digest(physical(tr2)),tolerance=1e-6)
    assert abs(difference)<1e-6 and identical, record
    return record

def main():
    cfg = read_json(ROOT/'experiments/cfs_classical_value/configs/main.json')
    tasks=[(key(c),label,policy) for c in cfg['conditions'] for label,policy in
           [('P0-Myopic','myopic'),('P6-RH4','RH4'),('P10-RH8','RH8')]]
    rows, seconds = map_jobs(job,tasks,'Phase 0',3)
    atomic_json(OUTPUT/'manifests/baseline_regression.json',dict(status='PASSED',runs=12,
                  rows=sorted(rows,key=lambda r:(r['condition'],r['policy'])),seconds=seconds))

if __name__=='__main__':
    main()
