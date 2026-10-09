"""Read-only historical replay comparison, including W20 protective regression."""
from experiment_core.storage import read_json,atomic_json
from model_families.single.data import matching_state
from experiments.cfs_rollout_target_alignment.protocol import read_compressed,assert_replay_snapshot
from .common import OUTPUT,B,key,parallel,sha,now
from .actions import historical,physical_action
from .worlds import evaluate

def regression_job(c):
    cid=key(c);rows=[];policies,prior=historical(cid)
    from experiments.cfs_resource_value_state_bucket_exact_accounting.policies import load
    for label in ['B0','B1','B2']:
        path=B/'raw/runs'/f'formal-{cid}-s9200-{label}'
        spec=read_json(path/'spec.json');summary=read_json(path/'summary.json');previous=read_compressed(B/summary['trace_ref'])
        state,orders=matching_state(spec['scenario'],spec['parameters'],spec['solver'])
        state.begin_period([o for o in orders.values() if o.book_time==0])
        result=evaluate(state,orders,load(label,cid),prior)
        assert abs(result['total']-previous['metrics']['realized_profit'])<=1e-6
        assert len(result['trace'])==len(previous['trace'])
        for a,b in zip(result['trace'],previous['trace']):
            assert_replay_snapshot(a['before'],b['before']);assert_replay_snapshot(a['after'],b['after'])
            if not a.get('drain'):assert physical_action(a['plan'])==physical_action(b['plan'])
        rows.append(dict(condition=cid,policy=label,scenario_seed=9200,profit_difference_USD=result['total']-summary['realized_profit'],
            exact_integer_cash=True,all_period_physics_identical=True,full_action_sequence_identical=True,
            old_trace_sha256=sha(B/summary['trace_ref']),new_replay=result['replay'],world_ref=summary['trace_ref']))
    return rows

def run(cfg):
    path=OUTPUT/'qa/historical_regression.json'
    if path.exists():assert read_json(path)['status']=='PASSED';return
    conditions=read_json(B/'config/main.json')['conditions']
    items,wall=parallel(regression_job,conditions,'Historical exact B regression + W20',cfg['max_parallel'])
    atomic_json(path,dict(status='PASSED',rows=[r for item in items for r in item],runs=12,
        wall_seconds=wall,checked_at=now(),historical_writes=False))
