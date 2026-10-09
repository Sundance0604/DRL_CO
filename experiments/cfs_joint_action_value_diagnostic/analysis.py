"""Preregistered scenario statistics. No selector, threshold, or model mutation."""
import itertools
import numpy as np
from scipy import stats
from experiment_core.storage import read_json,atomic_json
from .common import OUTPUT,A,config,csv_output,sha,now

def statistics(values,seed=20261008):
    x=np.asarray(values,float);n=len(x);assert n and np.isfinite(x).all()
    mean=float(x.mean());sd=float(x.std(ddof=1)) if n>1 else None
    se=sd/np.sqrt(n) if n>1 else None
    margin=float(stats.t.ppf(.975,n-1)*se) if n>1 else None
    p=float(2*stats.t.sf(abs(mean/se),n-1)) if se and se>0 else (1. if mean==0 else 0.) if n>1 else None
    boot=np.random.default_rng(seed).choice(x,(10000,n)).mean(axis=1)
    lo,hi=np.quantile(boot,[.025,.975])
    return dict(n=n,mean_USD=mean,SD_USD=sd,median_USD=float(np.median(x)),SE_USD=se,
        t95_low_USD=None if margin is None else mean-margin,t95_high_USD=None if margin is None else mean+margin,
        bootstrap95_low_USD=float(lo),bootstrap95_high_USD=float(hi),p_two_sided=p,
        interval_multiplicity='unadjusted',unit='scenario',bootstrap_replications=10000)

def holm(rows):
    good=sorted((r for r in rows if r['p_two_sided'] is not None),key=lambda r:r['p_two_sided']);carry=0.
    for i,r in enumerate(good):
        carry=max(carry,min(1.,(len(good)-i)*r['p_two_sided']));r['p_Holm']=carry
    return rows

def load_rows(stage):
    m=read_json(OUTPUT/'manifests'/f'{stage}_joint.json');rows=[]
    for r in m['results']:assert sha(r['path'])==r['sha256'];rows.append(read_json(r['path']))
    return rows

def run():
    prereg=read_json(OUTPUT/'manifests/analysis_preregistration.json')
    assert prereg['analysis_sha256']==sha(OUTPUT/'analysis.py'),'Frozen analysis source changed'
    cfg=config();rows=load_rows('diagnostic')+load_rows('formal')
    query=[];align=[];fallback=[];scores=[];selection=[];assess=[];diffs=[];interactions=[];residual=[];fixed=[];runtime=[]
    for r in rows:
        actions=read_json(r['action_ref']['path'])['actions'];common={k:r[k] for k in ['state_id','stage','condition','source','scenario_seed','time']}
        qmeans={h:float(np.mean(v)) for h,v in r['assessment_returns'].items()}
        pointbest=max(qmeans.values());qties={h for h,v in qmeans.items() if pointbest-v<=1e-8}
        pval={a['action_hash']:a['scores']['pi1']['L_V_USD'] for a in actions}
        pties={h for h,v in pval.items() if max(pval.values())-v<=1e-8}
        strict=[];uncertain=0
        for a,b in itertools.combinations(pval,2):
            d=np.asarray(r['assessment_returns'][a])-np.asarray(r['assessment_returns'][b])
            se=float(np.std(d,ddof=1)/np.sqrt(len(d)));ci=float(stats.t.ppf(.975,len(d)-1)*se)
            uncertain+=abs(float(np.mean(d)))<=ci
            pd=pval[a]-pval[b];qd=qmeans[a]-qmeans[b]
            if abs(pd)>1e-8 and abs(qd)>1e-8:strict.append(int(pd*qd>0))
        for a in actions:
            h=a['action_hash'];v=np.asarray(r['assessment_returns'][h]);se=float(v.std(ddof=1)/np.sqrt(len(v)))
            score=a['scores']['pi1']
            scores.append(common|dict(action_hash=h,labels=a['labels'],**score,
                Q_assessment_mean_USD=qmeans[h],Q_MC_SE_USD=se,
                strict_point_ranking_agreement=None if not strict else float(np.mean(strict)),
                uncertain_pair_count=uncertain,tie_set_jaccard=len(qties&pties)/len(qties|pties),
                point_top_disagreement=int(r['selections']['price']['action_hash'] not in qties)))
            assess.append(common|dict(action_hash=h,paths=len(v),Q_mean_USD=qmeans[h],MC_SE_USD=se,
                candidate_subset_regret_biased_USD=pointbest-qmeans[h],finite_sample_maximum_bias=True,
                returns_USD=v.tolist(),selection_not_recomputed=True))
        selection.append(common|dict(choices=r['selections'],assessment_seen_at_choice=False,
            selection_sha256=r['selection_sha256'],missing=r['missing'],candidate_count=len(actions)))
        for left,right in [('select32','price'),('select32','M0'),('select32','select8'),('select32','bucket_price')]:
            lh=r['selections'][left]['action_hash'];rh=r['selections'][right]['action_hash']
            v=np.asarray(r['assessment_returns'][lh])-np.asarray(r['assessment_returns'][rh])
            mcse=float(v.std(ddof=1)/np.sqrt(len(v)));margin=float(stats.t.ppf(.975,len(v)-1)*mcse)
            d=dict(comparison=left+'-'+right,left_action=lh,right_action=rh,
                mean_difference_USD=float(v.mean()),paired_MC_SE_USD=mcse,
                path_t95_low_USD=float(v.mean()-margin),path_t95_high_USD=float(v.mean()+margin),
                paired_path_differences_USD=v.tolist(),path_unit_not_independent_scenarios=True,
                changed_action=lh!=rh,continuation='M0')
            diffs.append(common|d)
            lefts=next(a for a in actions if a['action_hash']==lh)['scores']['pi1']
            rights=next(a for a in actions if a['action_hash']==rh)['scores']['pi1']
            residual.append(common|dict(comparison=d['comparison'],Q_difference_USD=d['mean_difference_USD'],
                L_difference_USD=lefts['L_V_USD']-rights['L_V_USD'],
                mixed_residual_USD=d['mean_difference_USD']-lefts['L_V_USD']+rights['L_V_USD'],
                g_difference_USD=lefts['g_booked_USD']-rights['g_booked_USD'],
                bonus_difference_USD=lefts['value_bonus_USD']-rights['value_bonus_USD'],
                releases_left=lefts['release_events'],releases_right=rights['release_events'],
                MC_SE_USD=mcse,not_pure_nonseparability=True))
        for i,x in enumerate(r['interactions']):interactions.append(common|dict(pair_index=i,**x))
        for x in r['fixed']:
            # Same-state re-evaluation versus the historical on-policy projected table surface.
            version={'M0':'V0','pi1':'V1','pi3':'V3'}[x['continuation']]
            table=read_json(A/'tables'/version/f'{r["condition"]}.json')
            v=float(np.interp(r['time'],table['values'][x['hub']]['1']['times'],table['values'][x['hub']]['1']['values']))
            fixed.append(common|x|dict(natural_projected_table_USD=v,table_sha256=sha(A/'tables'/version/f'{r["condition"]}.json'),
                fixed_minus_natural_USD=x['mean_USD']-v,projection_distribution_mixed_with_sampler_difference=True,
                not_a_Bellman_residual=True))
        query.extend(common|x for x in r['query_semantics']);align.extend(common|x for x in r['bucket_base_alignment'])
        fallback.extend(common|x for x in r['fallback_audit'])
        runtime.append(common|dict(pipeline_inclusive_seconds=r['seconds'],world_reference_count=len(r['worlds']),
            simulation_including_replay_seconds=sum(w['simulation_seconds'] for w in r['worlds']),
            replay_seconds=sum(w['replay_seconds'] for w in r['worlds']),
            solver_seconds=sum(w['solver_seconds'] for w in r['worlds']),inclusive_times_not_additive=True))
    comparisons=[]
    for (stage,condition,source,comp) in sorted({(r['stage'],r['condition'],r['source'],r['comparison']) for r in diffs}):
        vals=[r['mean_difference_USD'] for r in diffs if (r['stage'],r['condition'],r['source'],r['comparison'])==(stage,condition,source,comp)]
        comparisons.append(dict(stage=stage,condition=condition,source=source,comparison=comp,**statistics(vals)))
    primary=[r for r in comparisons if r['stage']=='formal' and r['comparison']=='select32-price']
    assert len(primary)==6 and all(r['n']==40 for r in primary)
    holm(primary)
    bankdata=read_json(OUTPUT/'manifests/banks_deployment.json')['runs'];bankstats=[]
    for cid in sorted({r['condition'] for r in bankdata}):
        baseline={r['scenario_seed']:r['profit_USD'] for r in bankdata if r['condition']==cid and r['policy']=='M0'}
        for bank in range(1,5):
            rr=[r for r in bankdata if r['condition']==cid and r['policy']==f'bank{bank}'];assert len(rr)==40
            vals=[r['profit_USD']-baseline[r['scenario_seed']] for r in rr]
            bankstats.append(dict(condition=cid,bank=bank,**statistics(vals),
                uncertainty='conditional on frozen model; shared scenarios across four banks; training variation descriptive'))
    exports=dict(query_semantics=query,bucket_base_alignment=align,fallback_action_audit=fallback,
        joint_action_scores=scores,selection_manifest=selection,independent_assessment=assess,
        paired_selector_differences=diffs,resource_interactions=interactions,action_score_residuals=residual,
        independent_bank_results=bankdata,fixed_state_evaluation=fixed,runtime=runtime)
    for name,rr in exports.items():
        if rr:csv_output(name,rr)
        else:atomic_json(OUTPUT/'processed'/f'{name}_not_completed.json',dict(status='NOT_COMPLETED',reason='No eligible observations'))
    result=dict(created_at=now(),primary=primary,all_comparisons=comparisons,bank_statistics=bankstats,
        study_distribution='SURROGATE',primary_independent_unit='scenario',missing=[r for r in selection if r['missing']],
        cross_time_interactions='NOT_COMPLETED_UNSUPPORTED',analysis_sha256=sha(OUTPUT/'analysis.py'))
    atomic_json(OUTPUT/'processed/analysis.json',result)
    return result
