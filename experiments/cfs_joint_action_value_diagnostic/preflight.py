"""Historical-state feasibility smoke test; never part of new scientific samples."""
import argparse
import time
from experiment_core.storage import read_json,atomic_json,digest
from experiments.cfs_rollout_target_alignment.protocol import read_compressed
from experiments.cfs_resource_policy_iteration.protocol import restore
from .common import OUTPUT,B,key,config,sha,now
from .actions import candidates,historical
from .futures import frozen_prior,generate
from .worlds import evaluate

def run():
    rows=[]
    for c in config()['conditions']:
        cid=key(c);folder=B/'raw/runs'/f'formal-{cid}-s9200-B0'
        spec=read_json(folder/'spec.json');old=read_compressed(B/read_json(folder/'summary.json')['trace_ref'])
        state,all_orders=restore(spec['scenario'],spec['parameters'],spec['solver'],old['state_packets']['3'])
        before=digest(old['state_packets']['3']);aa,missing=candidates(state,cid)
        policies,prior=historical(cid);fp=frozen_prior(cid)
        known={k:v for k,v in all_orders.items() if v.book_time<=state.t}
        orders,stream=generate(known,state,fp,cid,'historical-smoke-'+before,'selection',0)
        started=time.perf_counter();outcomes=[]
        for a in aa:
            result=evaluate(state,orders,policies['M0'],prior,a['serial'])
            outcomes.append(dict(action_hash=a['action_hash'],labels=a['labels'],replay=result['replay'],
                forced_action_hash=result['current_action_hash'],total_USD=result['total'],seconds=result['wall_seconds']))
        rows.append(dict(condition=cid,candidates=len(aa),missing=missing,outcomes=outcomes,
            future_stream_hash=stream['stream_hash'],seconds=time.perf_counter()-started,
            historical_sample_not_reused_for_new_inference=True))
    atomic_json(OUTPUT/'qa/historical_state_smoke.json',dict(status='PASSED',rows=rows,at=now()))
    print([(r['condition'],r['candidates'],r['seconds']) for r in rows],flush=True)

if __name__=='__main__':run()
