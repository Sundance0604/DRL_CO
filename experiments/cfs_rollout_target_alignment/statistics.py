"""Scenario-level inference, predeclared comparisons, solve-after diagnostics."""
import itertools
import numpy as np
from scipy.stats import t as student_t,ttest_1samp,skew
from experiment_core.storage import atomic_json,read_json,digest
from .common import OUTPUT,FOURTH,key,parameters,rho_levels,write_csv,file_hash,now
from .policies import LADDER
from .protocol import read_compressed,state_identity
from .fitting import prediction_scores
from .diagnostics import correlation

COMPARISONS=[('P4','P3','ACTIONminusOLD'),('P6','P5','ACTIONminusOLD'),('P8','P7','ACTIONminusOLD'),('P10','P9','ACTIONminusOLD'),
    ('P6','P2','ACTIONminusTable'),('P8','P2','ACTIONminusTable'),('P10','P2','ACTIONminusTable')] + [
    (p,'P0','VminusZero') for p in ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10']] + [
    ('R1','R0','RHreference'),('P6','P8','exploratoryMethod'),('P10','P6','exploratoryMethod'),('P10','P8','exploratoryMethod')]

def preregister_formal():
    p=OUTPUT/'manifests/formal_preregistration.json'
    if p.exists():return read_json(p)
    assert not (OUTPUT/'processed/formal_policies.json').exists()
    r=dict(created_at=now(),formal_outcomes_seen=False,comparisons=COMPARISONS,unit='independent scenario/seed',n=40,
        CI='paired differences t95; 10000 paired-scenario bootstrap; unadjusted intervals labeled',
        multiplicity='Holm two-sided paired t p-values across all conditions within each predeclared comparison family',
        heldout='12 independent scenarios; whole-scenario paired bootstrap and per-scenario ranking/MAE differences, not candidate-row SE',
        decision_claim='Same target + better ranking + formal paired profit + direction across conditions required; no RMSE-only wins')
    atomic_json(p,r);return r

def interval(x,cfg):
    x=np.asarray(x,float);n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1));se=sd/np.sqrt(n)
    half=float(student_t.ppf(.975,n-1)*se)
    rng=np.random.default_rng(cfg['bootstrap_seed']);boot=x[rng.integers(0,n,(cfg['bootstrap_replications'],n))].mean(axis=1)
    return dict(n=n,mean=mean,SD=sd,median=float(np.median(x)),CI95_lower=mean-half,CI95_upper=mean+half,
        bootstrap_CI95_lower=float(np.quantile(boot,.025)),bootstrap_CI95_upper=float(np.quantile(boot,.975)),
        skewness=float(skew(x,bias=False)) if sd else 0.,p_two_sided=float(ttest_1samp(x,0).pvalue) if sd else (1. if mean==0 else 0.))

def holm(rows):
    for family in sorted({r['comparison_family'] for r in rows}):
        group=sorted([r for r in rows if r['comparison_family']==family],key=lambda r:r['p_two_sided']);running=0.
        for i,r in enumerate(group):
            running=max(running,min(1.,(len(group)-i)*r['p_two_sided']));r['holm_p']=running
            r['CI_multiplicity_adjusted']=False

def policy_statistics(cfg):
    rows=read_json(OUTPUT/'processed/formal_policies.json');summary=[];pairs=[];differences=[];runtime=[];gaps=[]
    for c in cfg['conditions']:
        cid=key(c);lookup={(r['seed'],r['policy']):r for r in rows if r['condition']==cid}
        for label,policy,method in LADDER:
            runs=[lookup[seed,label] for seed in cfg['formal_seeds']]
            stats=interval([r['realized_profit'] for r in runs],cfg)
            summary.append(dict(condition=cid,policy=label,method=method or policy,**stats,
                mean_service_rate=float(np.mean([r['delivered']/c['orders'] for r in runs])),
                mean_realized_cost=float(np.mean([r['realized_cost'] for r in runs])),
                mean_wall_seconds=float(np.mean([r['daily_wall_seconds'] for r in runs]))))
            for r in runs:
                runtime.append(dict(condition=cid,seed=r['seed'],policy=label,
                    context_preparation_seconds=r.get('online_feature_seconds',0),provider_evaluation_seconds=r.get('online_provider_seconds',0),
                    slot_evaluation_seconds=r.get('slot_value_seconds',0),MILP_build_seconds=r.get('exclusive_milp_build_seconds',r.get('build_seconds',0)),
                    MILP_solve_seconds=r['daily_solver_seconds'],daily_wall_seconds=r['daily_wall_seconds'],
                    replay_and_storage_wall_seconds=r['wall_seconds'],note='provider inclusive of slot timer; build field exclusive; not all timers additive'))
                oracle=lookup[r['seed'],'D']['realized_profit'];zero=lookup[r['seed'],'P0']['realized_profit'];denom=oracle-zero
                gaps.append(dict(condition=cid,seed=r['seed'],policy=label,oracle_preview_profit=oracle,
                    realized_profit=r['realized_profit'],oracle_preview_difference=oracle-r['realized_profit'],
                    preview_gap_capture=(r['realized_profit']-zero)/denom if denom>1e-6 else None,
                    label='OraclePreview restricted H8 diagnostic; NOT a certified global bound'))
        for a,b,family in COMPARISONS:
            values=[]
            for seed in cfg['formal_seeds']:
                ra,rb=lookup[seed,a],lookup[seed,b]
                assert ra['scenario_hash']==rb['scenario_hash']
                d=ra['realized_profit']-rb['realized_profit'];values.append(d)
                differences.append(dict(condition=cid,seed=seed,algorithm=a,reference=b,comparison_family=family,
                    paired_difference=d,scenario_hash=ra['scenario_hash']))
            pairs.append(dict(condition=cid,algorithm=a,reference=b,comparison_family=family,**interval(values,cfg)))
    holm(pairs)
    write_csv(OUTPUT/'processed/policy_summary.csv',summary)
    write_csv(OUTPUT/'processed/paired_policy_differences.csv',pairs)
    write_csv(OUTPUT/'processed/paired_policy_samples.csv',differences)
    write_csv(OUTPUT/'processed/direct_method_comparisons.csv',[r for r in pairs if r['comparison_family']=='exploratoryMethod'])
    write_csv(OUTPUT/'processed/runtime_breakdown.csv',runtime)
    write_csv(OUTPUT/'processed/oracle_preview_gap.csv',gaps)
    atomic_json(OUTPUT/'processed/policy_statistics.json',dict(summary=summary,comparisons=pairs))
    return dict(summary=summary,comparisons=pairs)

def prediction_statistics(cfg):
    rows=read_json(OUTPUT/'targets/heldout_prediction_rows.json');paired=[];scenario_stats=[];allranks=[];pooled_intervals=[]
    for cid,method in sorted({(r['condition'],r['method']) for r in rows}):
        selected=[r for r in rows if r['condition']==cid and r['method']==method]
        moments=[]
        for seed in sorted({r['scenario_seed'] for r in selected}):
            rr=[r for r in selected if r['scenario_seed']==seed]
            moments.append([len(rr),sum((r['target']-r['prediction'])**2 for r in rr)])
        moments=np.array(moments,float);rng=np.random.default_rng(cfg['bootstrap_seed'])
        draws=rng.integers(0,len(moments),(cfg['bootstrap_replications'],len(moments)))
        sums=moments[draws].sum(axis=1);boot=np.sqrt(sums[:,1]/sums[:,0])
        pooled_intervals.append(dict(condition=cid,method=method,n_scenarios=len(moments),
            pooled_RMSE=float(np.sqrt(moments[:,1].sum()/moments[:,0].sum())),
            bootstrap_CI95_lower=float(np.quantile(boot,.025)),bootstrap_CI95_upper=float(np.quantile(boot,.975)),
            uncertainty='10000 whole-scenario bootstrap; pooled candidate RMSE, not independent-row bootstrap'))
    for c in cfg['conditions']:
        cid=key(c)
        for method in ['linear','chebyshev','spline','bayesian']:
            fs_stats={}
            for fs in ['OLD','ACTION']:
                rr=[r for r in rows if r['condition']==cid and r['method']==method+'-'+fs]
                fs_stats[fs]={}
                for seed in cfg['heldout_value_seeds']:
                    selected=[r for r in rr if r['scenario_seed']==seed]
                    if not selected:continue
                    metrics=prediction_scores(selected,[r['prediction'] for r in selected])
                    fs_stats[fs][seed]=metrics
                    scenario_stats.append(dict(condition=cid,method=method,feature_set=fs,seed=seed,**metrics))
                from .diagnostics import ranking
                allranks.extend(r|dict(method=method,feature_set=fs) for r in ranking(rr,'target','prediction'))
            for metric in ['MAE','RMSE','pairwise_ranking_accuracy','top1_accuracy']:
                seeds=[s for s in cfg['heldout_value_seeds'] if s in fs_stats['OLD'] and s in fs_stats['ACTION'] and
                    fs_stats['OLD'][s][metric] is not None and fs_stats['ACTION'][s][metric] is not None]
                if len(seeds)<2:continue
                delta=[fs_stats['ACTION'][s][metric]-fs_stats['OLD'][s][metric] for s in seeds]
                paired.append(dict(condition=cid,method=method,metric=metric,comparison_family='heldout-'+metric,
                    direction='negative better' if metric in ['MAE','RMSE'] else 'positive better',**interval(delta,cfg)))
    holm(paired)
    write_csv(OUTPUT/'processed/heldout_scenario_metrics.csv',scenario_stats)
    write_csv(OUTPUT/'processed/heldout_paired_metrics.csv',paired)
    write_csv(OUTPUT/'processed/heldout_candidate_ranking_metrics.csv',allranks)
    write_csv(OUTPUT/'processed/value_predictions.csv',rows)
    write_csv(OUTPUT/'processed/heldout_pooled_RMSE_intervals.csv',pooled_intervals)
    return paired

def deployment_support_audit(cfg):
    """Extra diagnostics only: target anchors, terminal gate, and g+V ranking."""
    rows=read_json(OUTPUT/'targets/heldout_prediction_rows.json');support=[];decisions=[]
    for cid,method in sorted({(r['condition'],r['method']) for r in rows}):
        rr=[r for r in rows if r['condition']==cid and r['method']==method]
        terminal=[r for r in rr if r['terminal_release']]
        baseline=[r for r in rr if r['baseline']]
        assert all(abs(r['target'])<1e-6 for r in baseline)
        support.append(dict(condition=cid,method=method,n_candidates=len(rr),
            n_scenarios=len({r['scenario_seed'] for r in rr}),terminal_candidates=len(terminal),
            terminal_nonzero_targets=sum(abs(r['target'])>1e-6 for r in terminal),
            terminal_mean_absolute_target=float(np.mean([abs(r['target']) for r in terminal])) if terminal else None,
            baseline_candidates=len(baseline),baseline_mean_absolute_prediction=float(np.mean([abs(r['prediction']) for r in baseline])) if baseline else None,
            online_terminal_gate_preserved=True,target_artificial_zero_imputation=False,
            interpretation='System action-advantage label differs from absolute slot value; original additive gate/anchors are not exact action-value identities'))
        scored=[r|dict(target=r['target']+r['candidate_current_stage']) for r in rr]
        scores=prediction_scores(scored,[r['prediction']+r['candidate_current_stage'] for r in rr])
        decisions.append(dict(condition=cid,method=method,scope='g_current_joint_action + relative_future_advantage',**scores,
            interpretation='Offline candidate utility diagnostic; not an exact online multi-vehicle additive MILP ranking'))
    write_csv(OUTPUT/'processed/target_support_audit.csv',support)
    write_csv(OUTPUT/'processed/heldout_g_plus_value_ranking.csv',decisions)
    return dict(support=support,decision_ranking=decisions)

def donor_dependency_audit(cfg):
    """Post-hoc structural sensitivity, chosen for independence, never outcome wins.

    Averaging own+offset-donor targets reuses underlying scenarios across parent
    clusters. The primary parent-block intervals therefore ignore donor-node
    dependence. Reevaluate fixed predictions on rep0 (own future) only: all
    remaining outcomes then use independent original scenario seeds.
    """
    raw=read_json(OUTPUT/'targets/heldout.json')
    own={r['parent_query_id']:r for r in raw if r['mc_replication']==0}
    assert all(r['donor_seed']==r['scenario_seed'] for r in own.values())
    predictions=read_json(OUTPUT/'targets/heldout_prediction_rows.json')
    rows=[r|dict(target=own[r['parent_query_id']]['Y_roll'],
        diagnostic='own-future-only; no shared donor path; fixed model/prediction') for r in predictions]
    paired=[];summaries=[];scenario=[]
    for c in cfg['conditions']:
        cid=key(c)
        for method in ['linear','chebyshev','spline','bayesian']:
            by_feature={}
            for fs in ['OLD','ACTION']:
                rr=[r for r in rows if r['condition']==cid and r['method']==method+'-'+fs]
                summaries.append(dict(condition=cid,method=method,feature_set=fs,
                    **prediction_scores(rr,[r['prediction'] for r in rr])))
                by_feature[fs]={}
                for seed in cfg['heldout_value_seeds']:
                    selected=[r for r in rr if r['scenario_seed']==seed]
                    if not selected:continue
                    metrics=prediction_scores(selected,[r['prediction'] for r in selected])
                    by_feature[fs][seed]=metrics
                    scenario.append(dict(condition=cid,method=method,feature_set=fs,seed=seed,**metrics))
            for metric in ['MAE','RMSE','pairwise_ranking_accuracy','top1_accuracy']:
                seeds=[s for s in cfg['heldout_value_seeds'] if s in by_feature['OLD'] and s in by_feature['ACTION']
                    and by_feature['OLD'][s][metric] is not None and by_feature['ACTION'][s][metric] is not None]
                if len(seeds)<2:continue
                delta=[by_feature['ACTION'][s][metric]-by_feature['OLD'][s][metric] for s in seeds]
                paired.append(dict(condition=cid,method=method,metric=metric,
                    comparison_family='own-future-posthoc-'+metric,post_hoc=True,**interval(delta,cfg)))
    holm(paired)
    write_csv(OUTPUT/'processed/heldout_own_future_summary.csv',summaries)
    write_csv(OUTPUT/'processed/heldout_own_future_paired_metrics.csv',paired)
    write_csv(OUTPUT/'processed/heldout_own_future_scenario_metrics.csv',scenario)
    result=dict(status='COMPLETED',post_hoc=True,model_selection_updated=False,
        primary_heldout_dependence='Own+offset7 donor averaging shares original scenario sources across parent clusters; parent-only bootstrap/t95 omit donor-node dependence',
        primary_intervals_scope='conditional parent-block approximation; not unconditional independent-cluster confirmation',
        own_future_diagnostic='rep0 only, fixed selected models and predictions; original scenario seed is independent unit; fewer future paths and noisier labels',
        no_effect_on='Phase1 own-scenario identification and formal40 paired profit inference',summaries=summaries,paired=paired)
    atomic_json(OUTPUT/'qa/heldout_donor_dependency_audit.json',result)
    return result

def post_state_audit(cfg):
    from types import SimpleNamespace
    from model_families.single.data import matching_state
    from model_families.single.rolling.audit import incoming_supply
    rows=read_json(OUTPUT/'targets/heldout_prediction_rows.json');out=[];cache={};networks={}
    for r in rows:
        if r['method'] not in ['chebyshev-ACTION','spline-ACTION','bayesian-ACTION']:continue
        ref=r['candidate_roll_ref']
        if ref not in cache:cache[ref]=read_compressed(OUTPUT/ref)['world']['trace'][0]['after']
        post=cache[ref];hub=r['release_hub'];cap=40
        network_key=(r['condition'],r['scenario_seed'])
        if network_key not in networks:
            packet=read_compressed(OUTPUT/f'raw/query_states/{network_key[0]}-{network_key[1]}.json.gz')
            networks[network_key]=matching_state(packet['scenario'],packet['parameters'],packet['solver'])[0].net
        # Reconstruct only committed physical state; no solver and no future-demand reads.
        audit_state=SimpleNamespace(t=post['period'],net=networks[network_key],vehs=[
            SimpleNamespace(id=v['id'],loc=v['hub'],phase=v['phase'],route=v['route'],arrive=v['arrive'],
                hold=v['hold'],orders={oid:None for oid in v['assigned_orders']}) for v in post['vehicles']])
        incoming=[e for e in incoming_supply(audit_state) if e['idle_release'] and e['hub']==hub]
        backlog=sum(o['passenger'] for o in post['pool'] if o['departure']==hub)/cap
        idle=sum(v['phase']=='idle' and v['hub']==hub for v in post['vehicles'])
        out.append(dict(condition=r['condition'],scenario_seed=r['scenario_seed'],time=r['time'],vehicle_id=r['vehicle_id'],
            query_id=r['parent_query_id'],method=r['method'],target=r['target'],prediction=r['prediction'],
            residual=r['target']-r['prediction'],actual_post_backlog=backlog,actual_post_idle=idle,
            actual_post_incoming=sum(post['period']<e['eta']<=r['release_time']+4 for e in incoming),
            actual_post_committed_vehicles=sum(v['phase']!='idle' for v in post['vehicles']),actual_post_imbalance=backlog-idle,
            release_hub=hub,diagnostic_only=True,used_by_online_provider=False))
    write_csv(OUTPUT/'processed/post_state_residual_audit.csv',out)
    summary=[]
    for cid,method in sorted({(r['condition'],r['method']) for r in out}):
        rr=[r for r in out if r['condition']==cid and r['method']==method]
        for field in ['actual_post_backlog','actual_post_idle','actual_post_incoming','actual_post_imbalance']:
            summary.append(dict(condition=cid,method=method,field=field,Spearman=correlation([r[field] for r in rr],[r['residual'] for r in rr]),
                n_scenarios=len({r['scenario_seed'] for r in rr}),claim='descriptive correlation only; cannot identify benefit of full joint V'))
    write_csv(OUTPUT/'processed/post_state_residual_summary.csv',summary)
    return summary

def historical_target_regression():
    """Independent fourth construction-label recreation, never writing old outputs."""
    from model_families.single.data import matching_state
    from experiments.cfs_classical_value.simulation import simulate
    from experiments.cfs_classical_value.build_mc_targets import resource_world,future_orders
    from experiments.cfs_local_post_bayesian_value.targets import interventions,project_intervention
    cfg=read_json(FOURTH/'config/main.json');base=read_json(OUTPUT/'manifest.json')['baseline_config']
    cid='western_multicluster-n40-k8';seed=4000
    sc=read_json(FOURTH/'raw/datasets'/f'{cid}-s{seed}.dataset.json')['scenarios'][0]
    saved=read_json(FOURTH/'targets/construction.json')
    bs=[r for r in saved['base'] if r['condition']==cid and r['scenario_seed']==seed and r.get('mc_replication')==0]
    t=min(r['decision_time'] for r in bs);bs=[r for r in bs if r['decision_time']==t]
    acts=[r for r in saved['actions'] if r['condition']==cid and r['scenario_seed']==seed and r['decision_time']==t and r['mc_replication']==0]
    sim,orders=matching_state(sc,parameters(base,rho_levels(sc,base)['rho_max']),base['solver'])
    _,_,states=simulate(sim,orders,collect_times=[t]);state=states[t];future=future_orders(orders,orders,t,0)
    checks=[];cache={}
    def marginal(s,hub,q):
        q=int(q)
        def value(n):
            k=(state_identity(s),hub,n)
            if k not in cache:cache[k]=simulate(resource_world(s,hub,n),future,first_begun=True)[0]['realized_profit']
            return cache[k]
        return value(q)-value(q-1)
    for r in bs:
        new=marginal(state,r['hub'],r['q']);assert abs(new-r['target'])<1e-6
        checks.append(dict(kind='normal_rolling_MC_slot',time=t,hub=r['hub'],q=r['q'],difference=new-r['target']))
    choices=interventions(state,seed,0)
    for r in acts:
        ctx,action=next((ctx,a) for ctx,a in choices if ctx['candidate_id']==r['candidate_id'])
        projected,_=project_intervention(state,action,future,ctx['release_time'])
        new=marginal(projected,ctx['release_hub'],r['q']) if ctx['release_time']<state.T else 0.
        assert abs(new-r['target'])<1e-6
        checks.append(dict(kind='original_fourth_fixed_coalition_release_marginal',time=t,q=r['q'],difference=new-r['target']))
    result=dict(status='PASSED',checks=checks,seed=seed,condition=cid,no_historical_writes=True)
    atomic_json(OUTPUT/'qa/historical_target_regression.json',result);return result

def main():
    cfg=read_json(OUTPUT/'config/main.json')
    result=dict(policy=policy_statistics(cfg),heldout=prediction_statistics(cfg),post_state=post_state_audit(cfg),
        support=deployment_support_audit(cfg),donor_dependency=donor_dependency_audit(cfg),historical=historical_target_regression())
    atomic_json(OUTPUT/'processed/statistical_summary.json',result)

if __name__=='__main__':main()
