"""Read-only historical reaggregation plus genuinely independent resolutions."""
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from experiments.cfs_resource_policy_iteration.targets import profiled_table_model
from experiments.cfs_resource_policy_iteration.protocol import (
    continuation,read_compressed,restore,resource_world,pair_audit,action_hash)
from .common import OUTPUT,HISTORY,key,parameters,rho_levels,map_jobs,file_hash,now

def regression_job(c):
    cid=key(c);rows=[];pairs=[]
    oldbase=read_json(HISTORY/'manifest.json')['baseline_config']
    for n in range(5):
        spec=read_json(HISTORY/'raw/formal_runs'/f'formal-{cid}-s9000-P{n}'/'spec.json')
        previous=read_compressed(HISTORY/read_json(HISTORY/'raw/formal_runs'/spec['id']/'summary.json')['trace_ref'])
        s,orders=matching_state(spec['scenario'],spec['parameters'],spec['solver'])
        s.begin_period([o for o in orders.values() if o.book_time==0])
        result=continuation(s,orders,spec['policy'],spec['prior'])
        assert result['final_hash']==previous['final_hash']
        assert abs(result['metrics']['realized_profit']-previous['metrics']['realized_profit'])<1e-6
        assert [action_hash(r['plan']) for r in result['trace'] if not r.get('drain')]==[action_hash(r['plan']) for r in previous['trace'] if not r.get('drain')]
        rows.append(dict(condition=cid,iteration=n,scenario_seed=9000,
            profit_difference=result['metrics']['realized_profit']-previous['metrics']['realized_profit'],
            full_actions_identical=True,physical_final_identical=True,old_trace_sha256=file_hash(HISTORY/read_json(HISTORY/'raw/formal_runs'/spec['id']/'summary.json')['trace_ref'])))
        if n==4:continue
        target=next(r for r in read_json(HISTORY/'targets'/f'iter{n}'/'targets.json') if r['condition']==cid and not r['terminal_anchor'])
        seed=target['scenario_seed'];t=target['time'];h=target['hub'];q=target['q']
        scenario=read_json(HISTORY/'raw/datasets'/f'{cid}-s{seed}.dataset.json')['scenarios'][0]
        item=read_json(HISTORY/'targets'/f'iter{n}'/'raw'/f'{cid}-s{seed}.json')
        packet=read_compressed(HISTORY/item['parent_trace'])['state_packets'][str(t)]
        policyrow=read_json(HISTORY/'policies'/f'pi{n}.json')[cid]
        policy=policyrow|dict(table=None if n==0 else read_json(HISTORY/policyrow['table_ref']))
        state,orders=restore(scenario,parameters(oldbase,rho_levels(scenario,oldbase)['rho_max']),oldbase['solver'],packet)
        a,b=resource_world(state,h,q-1),resource_world(state,h,q);pair_audit(a,b,h)
        prior=read_json(HISTORY/'tables/priors.json')[cid]
        low,high=continuation(a,orders,policy,prior),continuation(b,orders,policy,prior)
        value=high['metrics']['realized_profit']-low['metrics']['realized_profit']
        assert abs(value-target['target'])<1e-6
        pairs.append(dict(condition=cid,iteration=n,scenario_seed=seed,time=t,hub=h,q=q,
            reproduced_target=value,old_target=target['target'],difference=value-target['target']))
    return dict(runs=rows,pairs=pairs)

def run(cfg):
    path=OUTPUT/'qa/baseline_regression.json'
    if path.exists():assert read_json(path)['status']=='PASSED';return read_json(path)
    final=read_json(HISTORY/'manifests/final_manifest.json')
    for name,sha in final['deliverable_sha256'].items(): assert file_hash(HISTORY/name)==sha,name
    recon=[]
    for n in range(4):
        targets=read_json(HISTORY/'targets'/f'iter{n}'/'targets.json')
        policies=read_json(HISTORY/'policies'/f'pi{n}.json')
        for c in cfg['conditions']:
            cid=key(c);rr=[r for r in targets if r['condition']==cid]
            assert all(r['continuation_policy_hash']==policies[cid]['policy_hash'] for r in rr)
            model,_,_=profiled_table_model(rr,2*c['fleet'])
            original=read_json(HISTORY/'tables'/f'V{n}'/(cid+'.json'))
            assert model['values']==original['values']
            recon.append(dict(condition=cid,iteration=n,rows=len(rr),exact_values=True,
                target_sha256=file_hash(HISTORY/'targets'/f'iter{n}'/'targets.json'),
                table_sha256=file_hash(HISTORY/'tables'/f'V{n}'/(cid+'.json'))))
    result,wall=map_jobs(regression_job,cfg['conditions'],'A2 independent history regression',cfg['max_parallel'])
    report=dict(status='PASSED',checked_at=now(),historical_deliverables=223,
        reconstructed_tables=recon,independent_runs=[r for item in result for r in item['runs']],
        independent_target_pairs=[r for item in result for r in item['pairs']],wall_seconds=wall,
        reuse='V0/V1 reused only after exact target reaggregation and independent source policy/target resolutions',
        historical_writes=False)
    atomic_json(path,report);return report
