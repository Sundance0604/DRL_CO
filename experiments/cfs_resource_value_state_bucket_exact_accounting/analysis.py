"""Frozen, scenario-paired inference; cells and solver queries are not replications."""
import itertools
from collections import defaultdict
import numpy as np
from experiment_core.storage import atomic_json,read_json
from experiments.cfs_resource_policy_iteration.analysis import describe,holm
from .common import OUTPUT,key,write_csv,file_hash
from .protocol import read_compressed,restore
from .pressure import pressure,bucket
from .provider import BucketValueProvider,corners
from .policies import LABELS

def profits(cfg,stage):
    runs=read_json(OUTPUT/'processed'/f'{stage}_runs.json');seeds=cfg[f'{stage}_seeds']
    lookup={(r['condition'],r['scenario_seed'],r['policy']):r for r in runs}
    assert len(lookup)==len(runs)==len(cfg['conditions'])*len(seeds)*len(LABELS)
    summaries=[];paired=[];differences=[]
    for c in cfg['conditions']:
        cid=key(c)
        for seed in seeds:assert len({lookup[cid,seed,p]['scenario_hash'] for p in LABELS})==1
        for p in LABELS:
            rr=[lookup[cid,s,p] for s in seeds]
            summaries.append(dict(stage=stage,condition=cid,policy=p,
                **describe([r['realized_profit'] for r in rr],cfg),
                mean_service=float(np.mean([r['delivered']/c['orders'] for r in rr])),
                mean_online_seconds=float(np.mean([r['daily_wall_seconds'] for r in rr]))))
        for a,b in cfg['comparisons']:
            dd=[lookup[cid,s,a]['realized_profit']-lookup[cid,s,b]['realized_profit'] for s in seeds]
            family='primary_4' if [a,b]==['B2','B1'] else 'preview_descriptive' if a=='D' else 'secondary_16'
            paired.append(dict(stage=stage,condition=cid,comparison=f'{a}-{b}',policy_a=a,policy_b=b,
                family=family,reference_mean=float(np.mean([lookup[cid,s,b]['realized_profit'] for s in seeds])),**describe(dd,cfg)))
            differences.extend(dict(stage=stage,condition=cid,scenario_seed=s,comparison=f'{a}-{b}',difference_USD=v) for s,v in zip(seeds,dd))
    for family in ['primary_4','secondary_16']:holm([r for r in paired if r['family']==family])
    write_csv(OUTPUT/'processed'/f'{stage}_policy_summary.csv',summaries)
    write_csv(OUTPUT/'processed'/f'{stage}_paired_policy_differences.csv',paired)
    write_csv(OUTPUT/'processed'/f'{stage}_paired_scenario_differences.csv',differences)
    atomic_json(OUTPUT/'processed'/f'{stage}_statistics.json',dict(summary=summaries,paired=paired))
    if stage=='formal':write_csv(OUTPUT/'processed/paired_policy_differences.csv',paired)
    return summaries,paired

def action_analysis(cfg,stage):
    rows=read_json(OUTPUT/'processed'/f'{stage}_action_disagreement.json');groups=defaultdict(list)
    for r in rows:groups[r['condition'],r['scenario_seed'],r['policy_b']+'-'+r['policy_a']].append(r)
    fields=['full_action_disagree','routes_disagree','assignments_disagree','wait_reposition_disagree','release_hub_disagree']
    daily=[]
    for (cid,seed,comparison),rr in groups.items():
        assert len(rr)==24
        daily.append(dict(stage=stage,condition=cid,scenario_seed=seed,comparison=comparison,
            **{f:float(np.mean([r[f] for r in rr])) for f in fields}))
    gg=defaultdict(list)
    for r in daily:gg[r['condition'],r['comparison']].append(r)
    ss=[]
    for (cid,comparison),rr in gg.items():
        for field in fields:ss.append(dict(stage=stage,condition=cid,comparison=comparison,component=field,
            **describe([r[field] for r in rr],cfg)))
    write_csv(OUTPUT/'processed'/f'{stage}_action_daily.csv',daily)
    write_csv(OUTPUT/'processed'/f'{stage}_action_summary.csv',ss)
    atomic_json(OUTPUT/'processed'/f'{stage}_action_summary.json',ss)
    if stage=='formal':write_csv(OUTPUT/'processed/action_disagreement.csv',rows)
    return ss

def variance_analysis(cfg):
    """Decomposition is descriptive. Confirmatory residuals use NO held-out fitting."""
    train=read_json(OUTPUT/'targets/construction/targets.json')
    diagnostic=read_json(OUTPUT/'targets/diagnostic/targets.json')
    variance=[];contrasts=[];loss_daily=[];prediction_rows=[]
    for c in cfg['conditions']:
        cid=key(c);model=read_json(OUTPUT/'tables/bucket_tuqb'/(cid+'.json'))
        provider=BucketValueProvider(model)
        gg=defaultdict(list)
        for r in train:
            if r['condition']==cid and not r['terminal_anchor']:gg[r['time'],r['hub'],r['q']].append(r)
        for (t,h,q),rr in gg.items():
            yy=np.array([r['target'] for r in rr]);mean=float(yy.mean());total=float(np.var(yy))
            within=0.;between=0.
            for b in range(3):
                bb=[r['target'] for r in rr if r['bucket']==b];n=len(bb)
                bm=float(np.mean(bb)) if n else None
                if n:
                    within+=n/len(rr)*float(np.var(bb));between+=n/len(rr)*(bm-mean)**2
                val=provider.marginal_release_value(t,h,q,dict(T=24,period=t,bucket={h:b},pressure={h:{'R':None}}))
                query=provider.query_log[-1]
                contrasts.append(dict(condition=cid,time=t,hub=h,q=q,bucket=b,sample_count=n,
                    raw_conditional_mean_USD=bm,MC_SE=float(np.std(bb,ddof=1)/np.sqrt(n)) if n>1 else None,
                    deployed_value_USD=val,base_value_USD=mean,unsupported_flag=n<10,
                    fallback=query['fallback'],fallback_reason=query['fallback_reason']))
            assert abs(total-within-between)<=1e-6*max(1,total)
            variance.append(dict(condition=cid,time=t,hub=h,q=q,sample_count=len(rr),
                unconditional_variance_USD2=total,weighted_within_bucket_variance_USD2=within,
                between_bucket_variance_USD2=between,variance_reduction_USD2=total-within,
                fraction_reduction=(total-within)/total if total>0 else None,
                denominator='population decomposition n, exact law of total variance',
                inferential_status='construction descriptive; partitioning mechanically cannot increase weighted within variance'))
        daily=defaultdict(list)
        for r in diagnostic:
            if r['condition']!=cid or r['terminal_anchor']:continue
            t,h,q,b=r['time'],r['hub'],r['q'],r['bucket']
            context=dict(T=24,period=t,bucket={h:b},pressure={h:{'R':r['pre_injection_pressure']['R']}})
            base=provider.base.marginal_release_value(t,h,q,context)
            pred=provider.marginal_release_value(t,h,q,context);y=r['target']
            row=dict(condition=cid,scenario_seed=r['scenario_seed'],time=t,hub=h,q=q,bucket=b,
                target_USD=y,base_prediction_USD=base,bucket_prediction_USD=pred,
                base_squared_error_USD2=(y-base)**2,bucket_squared_error_USD2=(y-pred)**2,
                paired_MSE_improvement_USD2=(y-base)**2-(y-pred)**2,
                fallback=provider.query_log[-1]['fallback'],training_targets_sha256=model['base_and_bucket_same_targets_sha256'])
            daily[r['scenario_seed']].append(row);prediction_rows.append(row)
        for seed,rr in daily.items():
            loss_daily.append(dict(condition=cid,scenario_seed=seed,query_count=len(rr),
                base_MSE_USD2=float(np.mean([r['base_squared_error_USD2'] for r in rr])),
                bucket_MSE_USD2=float(np.mean([r['bucket_squared_error_USD2'] for r in rr])),
                paired_MSE_improvement_USD2=float(np.mean([r['paired_MSE_improvement_USD2'] for r in rr])),
                fallback_rate=float(np.mean([r['fallback'] for r in rr]))))
    corroboration=[]
    for c in cfg['conditions']:
        cid=key(c);rr=[r for r in loss_daily if r['condition']==cid]
        assert len(rr)==len(cfg['diagnostic_seeds'])
        corroboration.append(dict(condition=cid,**describe([r['paired_MSE_improvement_USD2'] for r in rr],cfg),
            base_MSE_USD2=float(np.mean([r['base_MSE_USD2'] for r in rr])),
            bucket_MSE_USD2=float(np.mean([r['bucket_MSE_USD2'] for r in rr])),unit='independent diagnostic scenario; all cells averaged within scenario'))
    holm(corroboration)
    for r in corroboration:r['corroborated']=r['mean']>0 and r['CI95_lower']>0 and r['Holm_p']<=.05
    for name,rr in [('within_cell_variance',variance),('bucket_contrast',contrasts),
                    ('diagnostic_target_predictions',prediction_rows),('diagnostic_MSE_daily',loss_daily),('variance_corroboration',corroboration)]:
        write_csv(OUTPUT/'processed'/(name+'.csv'),rr);atomic_json(OUTPUT/'processed'/(name+'.json'),rr)
    return corroboration

def support_provenance(cfg):
    """Enumerate actual lookup support; interpolated points do not get fake MC SE."""
    rows=[]
    for c in cfg['conditions']:
        cid=key(c);path=OUTPUT/'tables/bucket_tuqb'/(cid+'.json');model=read_json(path);p=BucketValueProvider(model)
        for t,h,q,b in itertools.product(range(25),sorted(model['base']['values']),range(1,2*c['fleet']+1),range(3)):
            v=p.marginal_release_value(t,h,q,dict(T=24,period=t,bucket={h:b},pressure={h:{'R':None}}))
            info=p.query_log[-1];source=info['source_corners'];times=model['base']['values'][h][str(q)]['times']
            rows.append(dict(condition=cid,time=t,hub=h,q=q,bucket=b,value_USD=v,
                cell_count=len(source),sample_count=min([s['count'] for s in source],default=0),
                source_corners=source,MC_SE=None,MC_SE_reason='source marginal SE in support_audit.csv; interpolated covariance not estimated',
                interpolated_flag=t not in times or q not in model['measured_q'],
                unsupported_flag=info['fallback'],fallback=info['fallback'],fallback_reason=info['fallback_reason'],
                terminal_anchor=t>=24,table_sha256=file_hash(path)))
    write_csv(OUTPUT/'processed/deployment_query_provenance.csv',rows)
    atomic_json(OUTPUT/'processed/deployment_query_provenance.json',rows);return rows

def state_and_fallback(cfg,stage):
    priors=read_json(OUTPUT/'tables/priors.json');runs=read_json(OUTPUT/'processed'/f'{stage}_runs.json')
    observations=[];queries=[];runtime=[]
    for r in runs:
        trace=read_compressed(OUTPUT/r['trace_ref']);spec=read_json(OUTPUT/'raw/runs'/r['id']/'spec.json')
        for tt,packet in trace['state_packets'].items():
            state,_=restore(spec['scenario'],spec['parameters'],spec['solver'],packet)
            pp=pressure(state,priors[r['condition']]['pressure_prior'])
            for h,p in pp.items():
                observations.append(dict(stage=stage,condition=r['condition'],scenario_seed=r['scenario_seed'],policy=r['policy'],
                    time=int(tt),hub=h,bucket=bucket(p['R'],priors[r['condition']]['thresholds']),**p))
        for query in trace.get('lookup_queries',[]):
            queries.append(dict(stage=stage,condition=r['condition'],scenario_seed=r['scenario_seed'],policy=r['policy'],**query))
        runtime.append(dict(stage=stage,condition=r['condition'],scenario_seed=r['scenario_seed'],policy=r['policy'],
            online_seconds=r['daily_wall_seconds'],pipeline_seconds=r['pipeline_seconds'],
            **{k:r.get(k) for k in ['daily_solver_seconds','provider_lookup_seconds','context_preparation_seconds',
                'build_seconds','state_transition_seconds','realized_accounting_seconds','value_calls','fallback_calls']}))
    gg=defaultdict(list)
    for r in observations:gg[r['condition'],r['scenario_seed'],r['policy']].append(r)
    daily=[]
    for (cid,seed,p),rr in gg.items():
        assert len(rr)==120
        for b in range(3):daily.append(dict(stage=stage,condition=cid,scenario_seed=seed,policy=p,bucket=b,
            fraction=float(np.mean([r['bucket']==b for r in rr])),denominator=120))
    ss=[]
    for c,p,b in itertools.product(cfg['conditions'],LABELS,range(3)):
        rr=[r['fraction'] for r in daily if r['condition']==key(c) and r['policy']==p and r['bucket']==b]
        ss.append(dict(stage=stage,condition=key(c),policy=p,bucket=b,**describe(rr,cfg)))
    fg=defaultdict(list)
    for r in queries:fg[r['condition'],r['scenario_seed'],r['bucket']].append(r)
    fallback=[]
    for (cid,seed,b),rr in fg.items():
        # Only B2 emits this provenance log. Denominator is ALL price queries,
        # not only binary slots chosen in the final MILP solution.
        active=[r for r in rr if r['release_time']<24]
        fallback.append(dict(stage=stage,condition=cid,scenario_seed=seed,bucket=b,calls=len(rr),
            fallback_calls=sum(r['fallback'] for r in rr),fallback_rate=float(np.mean([r['fallback'] for r in rr])),
            nonterminal_calls=len(active),nonterminal_fallback_calls=sum(r['fallback'] for r in active),
            nonterminal_fallback_rate=float(np.mean([r['fallback'] for r in active])) if active else None,
            below_10_calls=sum(r['fallback_reason']=='sample_count_below_10' for r in rr),
            unsupported_q_calls=sum(r['fallback_reason']=='unsupported_q_endpoint' for r in rr)))
    aggregate=[]
    for c in cfg['conditions']:
        cid=key(c);rr=[r for r in runs if r['condition']==cid and r['policy']=='B2']
        qq=[r for r in queries if r['condition']==cid]
        assert sum(r['value_calls'] for r in rr)==len(qq)
        assert sum(r['fallback_calls'] for r in rr)==sum(r['fallback'] for r in qq)
        rates=[r['fallback_calls']/r['value_calls'] if r['value_calls'] else 0. for r in rr]
        ac=[r for r in qq if r['release_time']<24]
        aggregate.append(dict(stage=stage,condition=cid,**describe(rates,cfg),calls=len(qq),
            pooled_rate=sum(r['fallback'] for r in qq)/len(qq) if qq else 0.,
            nonterminal_pooled_rate=sum(r['fallback'] for r in ac)/len(ac) if ac else 0.,
            zero_query_days=sum(r['value_calls']==0 for r in rr),denominator='all B2 pre-solve coefficient queries; n=scenario'))
    for name,rr in [('state_observations',observations),('bucket_occupancy_daily',daily),('bucket_occupancy',ss),
                    ('fallback_queries',queries),('fallback_audit',fallback),('fallback_summary',aggregate),('runtime',runtime)]:
        write_csv(OUTPUT/'processed'/f'{stage}_{name}.csv',rr)
        if name not in ['fallback_queries','state_observations']:atomic_json(OUTPUT/'processed'/f'{stage}_{name}.json',rr)
        if stage=='formal' and name in ['bucket_occupancy','fallback_audit','runtime']:write_csv(OUTPUT/'processed'/(name+'.csv'),rr)
    return aggregate

def finalize_analysis(cfg):
    assert read_json(OUTPUT/'manifests/policy_freeze.json')['formal_seen'] is False
    assert read_json(OUTPUT/'manifests/table_freeze.json')['diagnostic_seen'] is False
    support_provenance(cfg);variance_analysis(cfg)
    for stage in ['diagnostic','formal']:
        profits(cfg,stage);action_analysis(cfg,stage);state_and_fallback(cfg,stage)
    # No alpha or bucket/threshold retuning occurs in any analysis function.
