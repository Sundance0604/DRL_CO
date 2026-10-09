"""B's expanded context preserves baseline physics and base/fallback actions."""
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from experiments.cfs_local_post_bayesian_value.baseline_regression import physical as historical_physical
from experiments.cfs_resource_policy_iteration.action_analysis import canonical
from .common import OUTPUT,HISTORY,key,map_jobs,now
from .protocol import continuation,read_compressed,physical_identity,replay_world
from .simulation import simulate
from .pressure import expected_prior
import copy

def physical_without_booked_cash(trace):
    """Exclude only the deliberately corrected cash, not any physical field."""
    records=copy.deepcopy(historical_physical(trace))
    for row in records:
        for boundary in ['before','after']:
            if boundary in row:row[boundary].pop('profit',None)
    return records

def job(c):
    cid=key(c);rows=[]
    bank=[read_json(HISTORY/'raw/datasets'/f'{cid}-s{seed}.dataset.json')['scenarios'][0] for seed in range(8000,8060)]
    context=dict(base_prior=read_json(HISTORY/'tables/priors.json')[cid],pressure_prior=expected_prior(bank),thresholds=dict(c33=0.,c67=1.))
    for label,oldlabel in [('M0','P0'),('base','P1'),('all_unsupported_bucket','P1'),('RH4','R0'),('RH8','R1')]:
        root=HISTORY/'raw/formal_runs'/f'formal-{cid}-s9000-{oldlabel}'
        spec=read_json(root/'spec.json');previous=read_compressed(HISTORY/read_json(root/'summary.json')['trace_ref'])
        s,orders=matching_state(spec['scenario'],spec['parameters'],spec['solver'])
        s.begin_period([o for o in orders.values() if o.book_time==0]);initial=s.clone()
        if label in ['RH4','RH8']:
            metrics,trace,_=simulate(s,orders,label,prior=context,trace=True,first_begun=True)
            replay_world(initial,orders,dict(trace=trace,total=metrics['realized_profit'],physical_final_hash=physical_identity(s)))
        else:
            table=spec['policy']['table']
            if label=='all_unsupported_bucket':
                measured=sorted({1,2,3,4,6,8}&set(range(1,c['fleet']+1))|{c['fleet']})
                support={h:{str(b):{str(q):{str(t):0 for t in rec['1']['times']} for q in measured} for b in range(3)} for h,rec in table['values'].items()}
                table=dict(method='bucket_table',base=table,buckets={str(b):table for b in range(3)},
                    measured_q=measured,observed_max_q=c['fleet'],support=support)
            p=dict(table=table,policy_hash=digest(dict(label=label,cid=cid)),table_hash=None if table is None else digest(table))
            result=continuation(s,orders,p,context);metrics=result['metrics'];trace=result['trace']
            if label=='all_unsupported_bucket':
                active=[q for q in result['lookup_queries'] if q['release_time']<s.T]
                assert active and result['fallback_calls']==len(active) and all(q['fallback'] for q in active)
                assert all(q['value_USD']==0 and not q['fallback'] for q in result['lookup_queries'] if q['release_time']>=s.T)
        assert abs(metrics['realized_profit']-previous['metrics']['realized_profit'])<1e-6
        assert physical_without_booked_cash(trace)==physical_without_booked_cash(previous['trace'])
        assert [canonical(r['plan']) for r in trace if not r.get('drain')]==[canonical(r['plan']) for r in previous['trace'] if not r.get('drain')]
        rows.append(dict(condition=cid,label=label,old_policy=oldlabel,scenario_seed=9000,
            profit_difference=metrics['realized_profit']-previous['metrics']['realized_profit'],
            full_physical_actions_identical=True,
            booked_profit_recomputed_from_integer_actions=label=='M0',
            maximum_booked_cash_difference_USD=max(abs(a[b]['profit']-p[b]['profit'])
                for a,p in zip(trace,previous['trace']) for b in ['before','after']),
            excluded_comparison_fields=['before.profit','after.profit'],
            purpose='regression with historical base surface; unsupported bucket is QA-only, not B2 formal policy'))
    return rows

def run(cfg):
    path=OUTPUT/'qa/baseline_regression.json'
    if path.exists():assert read_json(path)['status']=='PASSED';return
    items,wall=map_jobs(job,cfg['conditions'],'B independent baseline/fallback regression',cfg['max_parallel'])
    atomic_json(path,dict(status='PASSED',rows=[r for item in items for r in item],
        runs=20,wall_seconds=wall,checked_at=now(),historical_writes=False))
