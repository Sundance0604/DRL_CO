"""Read-only post-freeze diagnostics and scientific exports (never model selection)."""
import itertools,json,logging,time
from pathlib import Path
import numpy as np
from scipy.stats import norm,t,skew,spearmanr
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.local_value.providers import design,FEATURES
from model_families.single.value.providers import MCTableValueProvider,load_provider
from .common import OUTPUT,OLD,ROOT,key,file_hash,write_csv,now
from .targets import averaged_actions
from .fit_models import residualize,scores

METHODS=['zero','legacy_fluid_reference','table','linear','chebyshev','spline','bayesian']

def value_diagnostics(cfg):
    predictions=[];metrics=[];cal=[];shrink=[];audit=[];bayesqa=[];density_metrics=[];conditional_cal=[]
    heldout=read_json(OUTPUT/'targets/heldout.json')['actions']
    construction=read_json(OUTPUT/'targets/construction.json')['actions']
    for c in cfg['conditions']:
        cid=key(c);table=read_json(OUTPUT/'models/table'/f'{cid}.json')
        rows=residualize(averaged_actions([r for r in heldout if r['condition']==cid]),table)
        active=[r for r in rows if not r['terminal_anchor']]
        train=residualize(averaged_actions([r for r in construction if r['condition']==cid and not r['terminal_anchor']]),table)
        for method in METHODS:
            if method=='zero':pred=np.zeros(len(active))
            elif method in ('legacy_fluid_reference','table'):
                p=load_provider(OUTPUT/'models'/method/f'{cid}.json')
                pred=np.array([p.marginal_release_value(r['time'],r['hub'],r['q'],{'T':24}) for r in active])
            else:
                m=read_json(OUTPUT/'models'/method/f'{cid}.json');X=design(active,m,True)
                pred=X@np.array(m['coefficients'])+np.array([r['base_slot_value'] for r in active])
            conditional_pred=(design(active,m,False)@np.array(m['coefficients'])+np.array([r['base_slot_value'] for r in active])) if method in ['linear','chebyshev','spline','bayesian'] else pred
            metric=scores(active,pred)
            strict=[];groups={}
            for i,a in enumerate(active):groups.setdefault((a['scenario_seed'],a['decision_time'],a['time'],a['q']),[]).append(i)
            for ids in groups.values():
                for i,j in itertools.combinations(ids,2):
                    a,b=active[i],active[j]
                    if a['hub']==b['hub']:continue
                    d=a['target']-b['target'];v=pred[i]-pred[j]
                    if abs(d)>1e-6:strict.append(.5 if abs(v)<1e-6 else float(d*v>0))
            metric['same_q_matched_release_ranking']=float(np.mean(strict)) if strict else None
            metric['same_q_matched_release_pairs']=len(strict)
            residual_metric=scores([r|{'target':r['residual']} for r in active],pred-np.array([r['base_slot_value'] for r in active]))
            metrics.append(dict(condition=cid,method=method,**metric,
                conditional_offline_RMSE=scores(active,conditional_pred)['RMSE'],
                residual_RMSE=residual_metric['RMSE'],residual_rank=residual_metric['pairwise_ranking_accuracy'],
                terminal_rows=sum(r['terminal_anchor'] for r in rows),terminal_prediction=0.))
            counts={h:sum(r['hub']==h for r in train) for h in {r['hub'] for r in train}}
            density_cut=float(np.median(list(counts.values())))
            qs=sorted({r['q'] for r in train});qcut=float(np.median(qs))
            for group,ids in [('sparse',[i for i,r in enumerate(active) if counts.get(r['hub'],0)<=density_cut]),
                              ('dense',[i for i,r in enumerate(active) if counts.get(r['hub'],0)>density_cut])]+[
                              (f'q={q}',[i for i,r in enumerate(active) if r['q']==q]) for q in qs]:
                if ids:
                    err=np.array([active[i]['target'] for i in ids])-pred[ids]
                    density_metrics.append(dict(condition=cid,method=method,group=group,n=len(ids),
                        RMSE=float(np.sqrt(np.mean(err**2))),MAE=float(np.mean(abs(err))),
                        conditional_offline_RMSE=float(np.sqrt(np.mean((np.array([active[i]['target'] for i in ids])-conditional_pred[ids])**2))),
                        density_definition='construction hub count <= median is sparse; ties all sparse'))
            for r,p,cp in zip(active,pred,conditional_pred):
                predictions.append(dict(condition=cid,method=method,scenario_seed=r['scenario_seed'],decision_time=r['decision_time'],
                    time=r['time'],hub=r['hub'],q=r['q'],candidate_id=r['candidate_id'],target=r['target'],
                    base_slot_value=r['base_slot_value'],residual=r['residual'],prediction=float(p),error=r['target']-float(p),conditional_offline_prediction=float(cp),
                    actual_post_backlog=r['actual_post_backlog'],actual_post_idle=r['actual_post_idle'],
                    actual_release_backlog=r['actual_release_backlog'],actual_release_supply=r['actual_release_supply'],
                    actual_release_imbalance=r['actual_release_imbalance'],local_origin_idle=r['local_origin_idle'],
                    actual_post_origin_idle=r['actual_post_origin_idle']))
            if method=='bayesian':
                C=np.array(m['posterior_covariance']);var=np.einsum('ij,jk,ik->i',X,C,X)+m['predictive_noise_variance']
                Xc=design(active,m,False);cv=np.einsum('ij,jk,ik->i',Xc,C,Xc)+m['predictive_noise_variance']
                for record,variance in zip(predictions[-len(active):],var):
                    record['posterior_predictive_variance']=float(variance)
                    for level in [.5,.8,.95]:
                        half=float(norm.ppf((1+level)/2)*np.sqrt(variance))
                        record[f'predictive_{int(level*100)}_lower']=record['prediction']-half
                        record[f'predictive_{int(level*100)}_upper']=record['prediction']+half
                eigen=float(np.linalg.eigvalsh(C).min());assert eigen>=-1e-6 and np.min(var)>=0
                count={h:sum(r['hub']==h for r in train) for h in m['hubs']};cut=float(np.median(list(count.values())))
                for group,ids in [('all',np.arange(len(active)))]+[(f'q={q}',np.array([i for i,r in enumerate(active) if r['q']==q])) for q in m['qs']]+[
                        (density,np.array([i for i,r in enumerate(active) if (count.get(r['hub'],0)<=cut)==(density=='sparse')])) for density in ['sparse','dense']]:
                    if not len(ids):continue
                    for level in [.5,.8,.95]:
                        width=2*norm.ppf((1+level)/2)*np.sqrt(var[ids]);error=np.array([active[i]['target'] for i in ids])-pred[ids]
                        cal.append(dict(condition=cid,group=group,nominal=level,n=len(ids),coverage=float(np.mean(abs(error)<=width/2)),mean_width=float(np.mean(width))))
                        cwidth=2*norm.ppf((1+level)/2)*np.sqrt(cv[ids]);ce=np.array([active[i]['target'] for i in ids])-conditional_pred[ids]
                        conditional_cal.append(dict(condition=cid,group=group,nominal=level,n=len(ids),coverage=float(np.mean(abs(ce)<=cwidth/2)),mean_width=float(np.mean(cwidth)),
                            scope='offline conditional q posterior, not deployed fixed-q-mixture correction'))
                beta=np.array(m['coefficients']);Xt=design(train,m,False);nh=len(m['hubs'])
                for j,h in enumerate(m['hubs']):
                    ids=np.array([i for i,r in enumerate(train) if r['hub']==h]);offset=Xt[ids]@beta-Xt[ids,1+j]*beta[1+j]
                    raw=np.array([train[i]['residual'] for i in ids])
                    unpooled=float((raw-offset).mean());factor=len(ids)*m['prior_variances'][1+j]/(m['predictive_noise_variance']+len(ids)*m['prior_variances'][1+j])
                    assert abs(beta[1+j]-factor*unpooled)<1e-4
                    shrink.append(dict(condition=cid,hub=h,n=len(ids),raw_hub_residual_mean=float(raw.mean()),
                        raw_adjusted_hub_effect=unpooled,posterior_hub_effect=float(beta[1+j]),posterior_hub_sd=float(np.sqrt(C[1+j,1+j])),
                        conditional_shrinkage_factor=float(factor),shrinkage_identity_residual=float(beta[1+j]-factor*unpooled)))
                for j,q in enumerate(m['qs']):
                    ids=np.array([i for i,r in enumerate(train) if r['q']==q]);col=1+nh+j
                    offset=Xt[ids]@beta-Xt[ids,col]*beta[col]
                    unpooled=float(np.mean([train[i]['residual'] for i in ids])-offset.mean());factor=len(ids)*m['prior_variances'][col]/(m['predictive_noise_variance']+len(ids)*m['prior_variances'][col])
                    assert abs(beta[col]-factor*unpooled)<1e-4
                    shrink.append(dict(condition=cid,hub=f'q={q}',n=len(ids),raw_hub_residual_mean=float(np.mean([train[i]['residual'] for i in ids])),
                        raw_adjusted_hub_effect=unpooled,posterior_hub_effect=float(beta[col]),posterior_hub_sd=float(np.sqrt(C[col,col])),
                        conditional_shrinkage_factor=float(factor),shrinkage_identity_residual=float(beta[col]-factor*unpooled)))
                bayesqa.append(dict(condition=cid,min_covariance_eigenvalue=eigen,min_predictive_variance=float(var.min()),
                    evidence_success=m['evidence_optimizer_success'],evidence_message=m['evidence_message'],MCMC=False))
    write_csv(OUTPUT/'processed/value_predictions.csv',predictions);write_csv(OUTPUT/'processed/value_metrics.csv',metrics)
    write_csv(OUTPUT/'processed/bayesian_calibration.csv',cal);write_csv(OUTPUT/'processed/bayesian_shrinkage.csv',shrink)
    write_csv(OUTPUT/'processed/value_density_q_metrics.csv',density_metrics)
    write_csv(OUTPUT/'processed/bayesian_conditional_calibration.csv',conditional_cal)
    atomic_json(OUTPUT/'qa/bayesian_audit.json',dict(status='PASSED',models=bayesqa,
        caveat='PSD and finite variance do not imply empirical predictive calibration. Hyperparameter and base-table uncertainty excluded.'))
    return dict(rows=len(predictions),active_targets=sum(r['n'] for r in metrics if r['method']=='table'),models=28,
        formal_tuning=False,heldout_target_hash=file_hash(OUTPUT/'targets/heldout.json'))

def ci(values,cfg):
    x=np.array(values,float);n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1));half=float(t.ppf(.975,n-1)*sd/np.sqrt(n))
    shape=float(skew(x,bias=False)) if sd>1e-12*max(1,float(np.max(abs(x)))) else 0.
    result=dict(n=n,mean=mean,sd=sd,median=float(np.median(x)),ci_lower=mean-half,ci_upper=mean+half,skew=shape)
    if abs(result['skew'])>1:
        rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.mean(rng.choice(x,(cfg['bootstrap_replications'],n)),axis=1)
        result.update(bootstrap_lower=float(np.quantile(boot,.025)),bootstrap_upper=float(np.quantile(boot,.975)))
    return result

def value_paired_comparisons(cfg):
    rows=read_csv('value_predictions.csv');out=[]
    for c in cfg['conditions']:
        cid=key(c);groups={}
        for r in rows:
            if r['condition']==cid:groups.setdefault(r['method'],{}).setdefault(int(r['scenario_seed']),[]).append(float(r['error']))
        pairs=[(m,'table') for m in ['linear','chebyshev','spline','bayesian']]+[('bayesian','chebyshev'),('bayesian','spline'),('chebyshev','spline')]
        for left,right in pairs:
            assert set(groups[left])==set(groups[right]);seeds=sorted(groups[left]);a=np.array([np.mean(np.square(groups[left][s])) for s in seeds]);b=np.array([np.mean(np.square(groups[right][s])) for s in seeds])
            rng=np.random.default_rng(cfg['bootstrap_seed']);idx=rng.integers(0,len(seeds),(10000,len(seeds)))
            boot=np.sqrt(a[idx].mean(axis=1))-np.sqrt(b[idx].mean(axis=1))
            out.append(dict(condition=cid,left=left,right=right,scenario_clusters=len(seeds),macro_RMSE_left=float(np.sqrt(a.mean())),
                macro_RMSE_right=float(np.sqrt(b.mean())),macro_RMSE_difference=float(np.sqrt(a.mean())-np.sqrt(b.mean())),
                cluster_bootstrap_ci_lower=float(np.quantile(boot,.025)),cluster_bootstrap_ci_upper=float(np.quantile(boot,.975)),
                interpretation='equal-scenario MSE then square root; negative favors left; q/candidates are not independent samples'))
    write_csv(OUTPUT/'processed/value_prediction_paired_comparisons.csv',out)
    return out

def old_summary_value_job(task):
    from model_families.single.data import matching_state
    from model_families.single.runner import state_snapshot
    from experiments.cfs_classical_value.simulation import simulate,summary
    from .common import nodes
    cid,seed=task;job=read_json(OUTPUT/'raw/runs'/f'formal-{cid}-{seed}-P0'/'spec.json')
    sim,orders=matching_state(job['scenario'],job['parameters'],job['solver'])
    _,_,states=simulate(sim,orders,collect_times=nodes(sim.T))
    prior=read_json(OLD/'models/priors.json')[cid]
    rows=averaged_actions([r for r in read_json(OUTPUT/'targets/heldout.json')['actions'] if r['condition']==cid and r['scenario_seed']==seed and not r['terminal_anchor']])
    out=[]
    for method in ['chebyshev','spline']:
        provider=load_provider(OLD/'models'/f'{cid}-{method}.json')
        for r in rows:
            s=states[r['decision_time']];assert digest(state_snapshot(s))==r['parent_state_hash']
            pred=provider.marginal_release_value(r['time'],r['hub'],r['q'],summary(s,prior))
            out.append(dict(condition=cid,method=method,scenario_seed=seed,decision_time=r['decision_time'],time=r['time'],q=r['q'],hub=r['hub'],
                candidate_id=r['candidate_id'],target=r['target'],old_summary_prediction=pred,error=r['target']-pred,
                note='frozen third-study coefficients and priors on identical fourth-study action-conditioned Y; no refitting'))
    return out

def old_summary_value_comparison(cfg):
    from .common import map_jobs
    tasks=[(key(c),s) for c in cfg['conditions'] for s in cfg['heldout_value_seeds']]
    saved,wall=map_jobs(old_summary_value_job,tasks,'Frozen old-summary value diagnostic',cfg['max_parallel'])
    rows=[r for a in saved for r in a];write_csv(OUTPUT/'processed/old_summary_value_predictions.csv',rows)
    local=read_csv('value_predictions.csv');out=[]
    for cid in [key(c) for c in cfg['conditions']]:
        for method in ['chebyshev','spline']:
            old=[r for r in rows if r['condition']==cid and r['method']==method];new=[r for r in local if r['condition']==cid and r['method']==method]
            old_keys={(r['scenario_seed'],r['candidate_id'],r['q']) for r in old};new_keys={(int(r['scenario_seed']),r['candidate_id'],int(r['q'])) for r in new}
            assert old_keys==new_keys
            old_error=np.array([r['error'] for r in old]);new_error=np.array([float(r['error']) for r in new])
            out.append(dict(condition=cid,method=method,n=len(old),old_RMSE=float(np.sqrt(np.mean(old_error**2))),new_RMSE=float(np.sqrt(np.mean(new_error**2))),
                old_MAE=float(np.mean(abs(old_error))),new_MAE=float(np.mean(abs(new_error))),
                interpretation='whole frozen old vs new pipeline on matched Y; target/base/features/fitting all differ, not isolated causal feature ablation'))
    write_csv(OUTPUT/'processed/old_summary_value_comparison.csv',out)
    return dict(jobs=len(tasks),rows=len(rows),wall_seconds=wall,old_models_changed=False)

def policy_statistics(cfg):
    runs=read_json(OUTPUT/'processed/formal_policies.json');summary=[];paired=[];direct=[];ratios=[];runtime=[]
    ablation=read_json(OUTPUT/'processed/ablation_policies.json');abl=[]
    for c in cfg['conditions']:
        cid=key(c);a=[r for r in runs if r['condition']==cid]
        by={p:{r['seed']:r for r in a if r['policy']==p} for p in sorted({r['policy'] for r in a})}
        def diff(left,right):
            assert set(by[left])==set(by[right])
            for seed in by[left]:assert by[left][seed]['scenario_hash']==by[right][seed]['scenario_hash']
            values=np.array([by[left][s]['realized_profit']-by[right][s]['realized_profit'] for s in sorted(by[left])])
            # Same tolerance as ledger replay; do not turn floating-point noise
            # into a nominally positive denominator or huge meaningless ratio.
            return np.where(abs(values)<1e-6,0.,values)
        for p,data in by.items():
            summary.append(dict(condition=cid,policy=p,**ci([r['realized_profit'] for r in data.values()],cfg)))
            for measure in ['daily_wall_seconds','online_feature_seconds','online_provider_seconds','slot_value_seconds','exclusive_milp_build_seconds','daily_solver_seconds']:
                runtime.append(dict(condition=cid,policy=p,component=measure,mean_seconds=float(np.mean([r.get(measure,0) for r in data.values()])),
                    p95_seconds=float(np.quantile([r.get(measure,0) for r in data.values()],.95))))
        for base,policies in [('P0',[f'P{i}' for i in range(1,7)]),('P7',[f'P{i}' for i in range(8,14)])]:
            for p in policies:
                paired.append(dict(condition=cid,left=p,right=base,**ci(diff(p,base),cfg)))
                oracle=diff('D',base);den=ci(oracle,cfg)
                ratios.append(dict(condition=cid,kind='oracle_gap',policy=p,denominator_mean=den['mean'],denominator_ci_lower=den['ci_lower'],
                    ratio=float(diff(p,base).mean()/oracle.mean()) if den['ci_lower']>0 else None,eligible=den['ci_lower']>0))
        d=diff('P14','P7');den=ci(d,cfg)
        for p in ['P8','P9','P10','P11','P12','P13']:
            ratios.append(dict(condition=cid,kind='compression',policy=p,denominator_mean=den['mean'],denominator_ci_lower=den['ci_lower'],
                ratio=float(diff(p,'P7').mean()/d.mean()) if den['ci_lower']>0 else None,eligible=den['ci_lower']>0))
        for left,right in [('P6','P4'),('P6','P5'),('P6','P2'),('P4','P5'),('P2','P4'),
                           ('P13','P11'),('P13','P12'),('P13','P9'),('P11','P12'),('P9','P11')]:
            direct.append(dict(condition=cid,left=left,right=right,**ci(diff(left,right),cfg),multiple_correction='none; predeclared'))
        for old,new in [('OldM-Cheb','P4'),('OldM-Spline','P5'),('OldRH4-Cheb','P11'),('OldRH4-Spline','P12')]:
            oldruns={r['seed']:r for r in ablation if r['condition']==cid and r['policy']==old};values=[]
            for s in sorted(oldruns):
                assert oldruns[s]['scenario_hash']==by[new][s]['scenario_hash']
                values.append(by[new][s]['realized_profit']-oldruns[s]['realized_profit'])
            abl.append(dict(condition=cid,new=new,old=old,**ci(values,cfg),
                caveat='feature, target, base-table and fit paradigm change together; not isolated causal feature ablation'))
    for name,rows in [('policy_summary',summary),('paired_policy_differences',paired),('direct_method_comparisons',direct),
                      ('performance_ratios',ratios),('runtime_breakdown',runtime),('state_ablation',abl)]:write_csv(OUTPUT/'processed'/f'{name}.csv',rows)
    atomic_json(OUTPUT/'processed/statistics.json',dict(summary=summary,paired=paired,direct=direct,ratios=ratios,runtime=runtime,ablation=abl,numerical_zero_tolerance=1e-6))
    return runs,dict(summary=summary,paired=paired,direct=direct,ratios=ratios,runtime=runtime,ablation=abl)

def post_audit(cfg):
    import csv
    with (OUTPUT/'processed/value_predictions.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    output=[];rng=np.random.default_rng(cfg['bootstrap_seed'])
    for c in cfg['conditions']:
        cid=key(c)
        for method in ['table','linear','chebyshev','spline','bayesian']:
            a=[r for r in rows if r['condition']==cid and r['method']==method];y=np.array([float(r['error']) for r in a])
            fields=['actual_release_backlog','actual_release_supply','actual_release_imbalance']
            Z=np.array([[float(r[f]) for f in fields] for r in a]);X=np.c_[np.ones(len(a)),Z[:,:2]]
            beta=np.linalg.lstsq(X,y,rcond=None)[0];fit=X@beta
            r2=1-np.sum((y-fit)**2)/np.sum((y-y.mean())**2)
            seed=np.array([int(r['scenario_seed']) for r in a]);groups=np.unique(seed);cv=np.empty(len(y))
            for g in groups:
                b=np.linalg.lstsq(X[seed!=g],y[seed!=g],rcond=None)[0];cv[seed==g]=X[seed==g]@b
            cv_r2=1-np.sum((y-cv)**2)/np.sum((y-y.mean())**2)
            # Cluster bootstrap keeps correlated q/candidates/futures within each scenario together.
            for j,f in enumerate(fields):
                corr=float(spearmanr(Z[:,j],y).statistic) if np.ptp(Z[:,j])>1e-12 and np.ptp(y)>1e-12 else None
                boot=[]
                for _ in range(1000):
                    selected=rng.choice(groups,len(groups));idx=np.concatenate([np.flatnonzero(seed==g) for g in selected])
                    if np.ptp(Z[idx,j])>1e-12 and np.ptp(y[idx])>1e-12:boot.append(float(spearmanr(Z[idx,j],y[idx]).statistic))
                output.append(dict(condition=cid,method=method,omitted_feature=f,n=len(a),clusters=len(groups),Spearman=corr,
                    cluster_ci_lower=float(np.quantile(boot,.025)) if boot else None,cluster_ci_upper=float(np.quantile(boot,.975)) if boot else None,
                    valid_bootstrap_resamples=len(boot),undefined_constant_resamples=1000-len(boot),
                    joint_in_sample_R2=float(r2),leave_seed_out_R2=float(cv_r2),joint_design_rank=int(np.linalg.matrix_rank(X)),
                    regression_backlog_coefficient=float(beta[1]),regression_supply_coefficient=float(beta[2]),
                    note='imbalance=backlog-supply; omitted from joint regression to avoid exact collinearity; diagnostic associations, not causal or online features'))
    write_csv(OUTPUT/'processed/post_decision_audit.csv',output);return output

def joint_audit_job(task):
    """Offline labels on REAL joint solved actions; never feed them back online."""
    from model_families.single.data import matching_state
    from model_families.single.runner import state_snapshot
    from model_families.single.planning import PeriodAction,compile_action
    from experiments.cfs_dynamic_lookahead.case_study import replay_plan
    from experiments.cfs_classical_value.build_mc_targets import resource_world,future_orders
    from experiments.cfs_classical_value.simulation import simulate,summary
    cid,seed,policy=task;rid=f'formal-{cid}-{seed}-{policy}';path=OUTPUT/'raw/runs'/rid
    job=read_json(path/'spec.json');steps=read_json(path/'trace.json')
    identity=dict(trace_hash=file_hash(path/'trace.json'),model_hash=job['model_hash'],periods=[3,12,20],
        q=1,version='joint-post-action-audit/v1',diagnostic_only=True)
    cache=OUTPUT/'raw/joint_post_audit'/f'{rid}.json'
    if cache.exists():
        saved=read_json(cache);assert saved['identity']==identity;return saved
    sim,orders=matching_state(job['scenario'],job['parameters'],job['solver']);out=[];skipped=0
    donor_seed=5000+(seed-5000+7)%40
    donor_s=read_json(OUTPUT/'raw/datasets'/f'{cid}-s{donor_seed}.dataset.json')['scenarios'][0]
    donor_orders=matching_state(donor_s,job['parameters'],job['solver'])[1]
    slot=MCTableValueProvider(read_json(OUTPUT/'models/table'/f'{cid}.json'))
    for step in steps:
        period=step['period']
        if period>=24:break
        sim.begin_period([o for o in orders.values() if o.book_time==period]);parent_hash=digest(state_snapshot(sim))
        sim.commit_plan(replay_plan(sim,step))
        if period not in [3,12,20]:continue
        items=[a for a in step['plan']['solver'].get('selected_local_contexts',[])
               if a['context']['departure_time']==period and a['context']['release_time']<sim.T]
        if not items:skipped+=1;continue
        # Fixed lexicographic candidate rule, not selected by target/error magnitude.
        item=min(items,key=lambda a:a['context']['candidate_id']);ctx=item['context'];release=ctx['release_time'];hub=ctx['release_hub']
        post=summary(sim,job['prior']);targets=[];futurehashes=[];statehashes=[]
        for rep,donor in enumerate([orders,donor_orders]):
            future=future_orders(orders,donor,period,rep);s=sim.clone()
            while s.t<release:
                s.begin_period([o for o in future.values() if o.book_time==s.t]);info={}
                s.commit_plan(compile_action(s,PeriodAction(s.t),info))
            s.begin_period([o for o in future.values() if o.book_time==s.t])
            profits=[simulate(resource_world(s,hub,q),future,first_begun=True)[0]['realized_profit'] for q in [0,1]]
            targets.append(profits[1]-profits[0]);futurehashes.append(digest([vars(o) for o in future.values()]));statehashes.append(digest(state_snapshot(s)))
        Y=float(np.mean(targets));base=slot.marginal_release_value(release,hub,1,{'T':sim.T});pred=base+item['correction']
        served=step['plan']['assignments']
        out.append(dict(condition=cid,policy=policy,scenario_seed=seed,period=period,candidate_id=ctx['candidate_id'],
            release_time=release,hub=hub,q=1,target=Y,base_slot_value=base,correction=item['correction'],prediction=pred,error=Y-pred,
            actual_post_backlog=post['backlog'][hub],actual_post_idle=post['idle'][hub],
            actual_post_imbalance=post['backlog'][hub]-post['idle'][hub],
            actual_served_count=len(served),actual_served_load=sum(orders[a['order_id']].passenger for a in served),
            actual_post_backlog_by_hub=post['backlog'],actual_post_idle_by_hub=post['idle'],actual_pipeline=post['incoming'],
            local_origin_idle=item['features']['origin_idle'],actual_origin_idle=post['idle'][ctx['vehicle_origin_hub']],
            local_release_idle=item['features']['release_idle'],local_compatible_load=item['features']['compatible_load'],
            context_hash=digest(ctx),parent_state_hash=parent_hash,actual_post_state_hash=digest(state_snapshot(sim)),
            projected_state_hashes=statehashes,future_hashes=futurehashes,
            summary_semantics='actual joint assignment at current period; commitments-only projection to candidate release; Myopic V0 paired continuation'))
    saved=dict(identity=identity,rows=out,skipped_no_current_executed_sink=skipped);atomic_json(cache,saved);return saved

def actual_joint_post_audit(cfg):
    from .common import map_jobs
    tasks=[(key(c),seed,p) for c in cfg['conditions'] for seed in cfg['heldout_value_seeds'] for p in ['P4','P6','P11','P13']]
    saved,wall=map_jobs(joint_audit_job,tasks,'Actual joint post-state diagnostic',cfg['max_parallel'])
    rows=[r for s in saved for r in s['rows']];write_csv(OUTPUT/'processed/joint_post_action_targets.csv',rows)
    result=[];rng=np.random.default_rng(cfg['bootstrap_seed'])
    for c in cfg['conditions']:
        cid=key(c)
        for p in ['P4','P6','P11','P13']:
            a=[r for r in rows if r['condition']==cid and r['policy']==p]
            if len(a)<4:
                result.append(dict(condition=cid,policy=p,n=len(a),status='insufficient_current_sink_actions'));continue
            y=np.array([r['error'] for r in a]);Z=np.array([[r['actual_post_backlog'],r['actual_post_idle'],r['actual_post_imbalance']] for r in a])
            seed=np.array([r['scenario_seed'] for r in a]);groups=np.unique(seed);X=np.c_[np.ones(len(a)),Z[:,:2]]
            beta=np.linalg.lstsq(X,y,rcond=None)[0];pred=X@beta
            R2=1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2) if np.std(y) else None
            cv=np.empty(len(y))
            for s in groups:cv[seed==s]=X[seed==s]@np.linalg.lstsq(X[seed!=s],y[seed!=s],rcond=None)[0]
            cvR2=1-np.sum((y-cv)**2)/np.sum((y-y.mean())**2) if np.std(y) else None
            for j,f in enumerate(['actual_post_backlog','actual_post_idle','actual_post_imbalance']):
                rho=float(spearmanr(Z[:,j],y).statistic) if np.ptp(Z[:,j])>1e-12 and np.ptp(y)>1e-12 else None;boot=[]
                for _ in range(1000):
                    idx=np.concatenate([np.flatnonzero(seed==s) for s in rng.choice(groups,len(groups))])
                    if np.ptp(Z[idx,j])>1e-12 and np.ptp(y[idx])>1e-12:boot.append(float(spearmanr(Z[idx,j],y[idx]).statistic))
                result.append(dict(condition=cid,policy=p,n=len(a),clusters=len(groups),omitted_feature=f,Spearman=rho,
                    cluster_ci_lower=float(np.quantile(boot,.025)) if boot else None,cluster_ci_upper=float(np.quantile(boot,.975)) if boot else None,
                    valid_bootstrap_resamples=len(boot),undefined_constant_resamples=1000-len(boot),
                    joint_in_sample_R2=R2,leave_seed_out_R2=cvR2,design_rank=int(np.linalg.matrix_rank(X)),status='evaluated',
                    interpretation='actual solved joint action, not hypothetical unexecuted RH suffix; descriptive diagnostic only'))
    write_csv(OUTPUT/'processed/joint_post_decision_audit.csv',result)
    atomic_json(OUTPUT/'qa/joint_post_audit.json',dict(status='PASSED',jobs=len(tasks),targets=len(rows),
        skipped_unexecuted_sink=sum(s['skipped_no_current_executed_sink'] for s in saved),wall_seconds=wall,
        target_seed_range=[5000,5011],model_tuning=False,original_coefficients_unchanged=True))
    return rows,result

def timing_reference_job(task):
    """Memory-only timing hooks; identical values/actions checked independently.

Formal fields time context construction and include feature extraction inside
provider evaluation. This reference replay additionally splits summary creation,
feature extraction, numerical basis evaluation and slot coefficient evaluation.
No frozen source or model is edited and no coefficient is changed.
"""
    from unittest.mock import patch
    from model_families.single.data import matching_state
    import model_families.single.local_value.providers as lp
    import model_families.single.value.providers as vp
    from . import simulation as sm
    from .baseline_regression import physical
    cid,policy=task;path=OUTPUT/'raw/runs'/f'formal-{cid}-5000-{policy}'
    job=read_json(path/'spec.json');old=read_json(path/'summary.json')
    counters=dict(summary=0.,features=0.,correction=0.,slot=0.,myopic_non_solve=0.,summary_calls=0,correction_active=False)
    original_summary=sm.summary;original_features=lp.features;original_correction=lp.LocalValueProvider.corrections;original_slot=vp.MCTableValueProvider.marginal_release_value
    def timed_summary(*args,**kwargs):
        ts=time.perf_counter();result=original_summary(*args,**kwargs);elapsed=time.perf_counter()-ts
        if job['model'] and counters['summary_calls']%2==0:counters['summary']+=elapsed
        counters['summary_calls']+=1;return result
    def timed_features(*args,**kwargs):
        ts=time.perf_counter();result=original_features(*args,**kwargs)
        if counters['correction_active']:counters['features']+=time.perf_counter()-ts
        return result
    def timed_correction(*args,**kwargs):
        ts=time.perf_counter();counters['correction_active']=True
        try:return original_correction(*args,**kwargs)
        finally:counters['correction_active']=False;counters['correction']+=time.perf_counter()-ts
    def timed_slot(*args,**kwargs):
        ts=time.perf_counter();result=original_slot(*args,**kwargs);counters['slot']+=time.perf_counter()-ts;return result
    sim,orders=matching_state(job['scenario'],job['parameters'],job['solver'])
    original_solve_plan=sim.solve_plan
    def timed_current_solve(*args,**kwargs):
        ts=time.perf_counter();result=original_solve_plan(*args,**kwargs)
        info=kwargs.get('solve_info',args[-1] if args and isinstance(args[-1],dict) else {})
        counters['myopic_non_solve']+=max(0,time.perf_counter()-ts-(info or {}).get('runtime',0))
        return result
    provider=lp.load_local(Path(job['model'])) if job['model'] else None
    with patch.object(sim,'solve_plan',timed_current_solve),patch.object(sm,'summary',timed_summary),patch.object(lp,'features',timed_features),patch.object(lp.LocalValueProvider,'corrections',timed_correction),patch.object(vp.MCTableValueProvider,'marginal_release_value',timed_slot):
        metrics,trace,_=sm.simulate(sim,orders,job['policy'],provider,job['prior'],trace=True)
    assert abs(metrics['realized_profit']-old['realized_profit'])<1e-6
    assert physical(trace)==physical(read_json(path/'trace.json'))
    return dict(condition=cid,policy=policy,seed=5000,status='PASSED',profit_difference=metrics['realized_profit']-old['realized_profit'],
        summary_preparation_seconds=counters['summary'],context_preparation_seconds=metrics['online_feature_seconds'],
        numeric_feature_extraction_seconds=counters['features'],
        correction_basis_and_prediction_seconds=max(0,counters['correction']-counters['features']),
        slot_coefficient_evaluation_seconds=counters['slot'],
        slot_variable_constraint_seconds=max(0,metrics['slot_value_seconds']-counters['slot']),
        exclusive_milp_build_seconds=metrics['exclusive_milp_build_seconds'],
        myopic_model_build_and_solution_extraction_seconds=counters['myopic_non_solve'],
        milp_solve_seconds=metrics['daily_solver_seconds'],daily_wall_seconds=metrics['daily_wall_seconds'],
        profiling_scope='instrumented deterministic reference run, not reconstructed formal timing',physical_trace_identical=True)

def runtime_reference(cfg):
    from .common import map_jobs
    from .run_policies import LADDER
    tasks=[(key(c),p) for c in cfg['conditions'] for p,_,_ in LADDER]
    rows,wall=map_jobs(timing_reference_job,tasks,'Runtime component replay',cfg['max_parallel'])
    write_csv(OUTPUT/'processed/runtime_component_audit.csv',rows)
    result=dict(status='PASSED',runs=len(rows),wall_seconds=wall,identical_realized_profit_and_physical_trace=True,
                source_and_models_modified=False,timing_scope='reference scenario 5000; includes instrumentation overhead')
    atomic_json(OUTPUT/'qa/runtime_component_audit.json',result);return result

def replay_audit(runs):
    from model_families.single.data import matching_state
    from model_families.single.runner import state_snapshot
    from model_families.single.accounting import cooldown_tick,realized_ledger
    from experiments.cfs_dynamic_lookahead.case_study import replay_plan,action_of
    from model_families.single.planning import compile_action
    checks=[];contexts=0;postrows=[]
    for i,r in enumerate(runs):
        path=OUTPUT/'raw/runs'/r['id'];job=read_json(path/'spec.json')
        assert digest(job['scenario'])==r['scenario_hash'] and file_hash(path/'trace.json')==r['trace_hash']
        sim,orders=matching_state(job['scenario'],job['parameters'],job['solver']);decision=[];seen=set()
        steps=read_json(path/'trace.json')
        for step in steps:
            period=step['period']; plan=step['plan'];meta=plan['solver']
            if step.get('drain'):
                sim.pool={};assert not plan['assignments'] and not plan['routes'] and not plan['repositions']
                assert digest(state_snapshot(sim))==plan['state_hash'];cooldown_tick(sim)
            else:
                sim.begin_period([o for o in orders.values() if o.book_time==period])
                assert digest(state_snapshot(sim))==plan['state_hash']
                for a in plan['assignments']:
                    assert a['order_id'] not in seen and orders[a['order_id']].book_time<=period;seen.add(a['order_id'])
                preview=meta.get('window',{}).get('preview_order_ids',[])
                if job['policy']!='OracleH8':assert not preview
                else:assert all(period<orders[o].book_time<=period+8 for o in preview)
                if 'context_hash_before' in meta:
                    assert meta['context_hash_before']==meta['context_hash_after']
                    assert meta['general_constraints']==meta['quadratic_constraints']==meta['quadratic_terms']==0
                    contexts+=meta['local_correction_count']
                verified={};compile_action(sim,action_of(plan),verified)
                assert set(verified['accounting'])==set(plan['accounting'])
                assert all(abs(verified['accounting'][k]-plan['accounting'][k])<1e-6 for k in verified['accounting'])
                sim.commit_plan(replay_plan(sim,step));decision.append(step)
                post=plan.get('post_decision_audit')
                if post:
                    remaining={h:sum(o.passenger for o in sim.pool.values() if o.departure==h)/sim.parameters['capacity'] for h in sim.net.H}
                    idle={h:sum(v.phase=='idle' and v.loc==h for v in sim.vehs) for h in sim.net.H}
                    assert remaining==post['remaining_backlog'] and idle==post['idle_supply']
                    for item in meta.get('selected_local_contexts',[]):
                        h=item['context']['release_hub'];origin=item['context']['vehicle_origin_hub']
                        postrows.append(dict(run_id=r['id'],condition=r['condition'],policy=r['policy'],seed=r['seed'],period=period,
                            candidate_id=item['context']['candidate_id'],correction=item['correction'],
                            local_origin_idle=item['features']['origin_idle'],actual_origin_idle=idle[origin],
                            actual_backlog=remaining[h],actual_idle=idle[h],actual_served_count=post['served_count'],actual_served_load=post['served_load']))
            assert state_snapshot(sim)==step['after']
        ledger=realized_ledger(sim,orders,decision)
        assert abs(ledger['realized_profit']-r['realized_profit'])<1e-6 and not ledger['unfinished_committed']
        for step in steps:
            at_t=realized_ledger(sim,orders,decision,step['period']+1);recorded=step['realized_ledger']
            assert set(at_t)==set(recorded)
            for k,v in at_t.items():
                assert abs(v-recorded[k])<1e-6 if isinstance(v,(int,float)) else v==recorded[k]
        checks.append(dict(id=r['id'],steps=len(steps),status='PASSED'))
        if (i+1)%100==0:print('Independent replay',i+1,'/',len(runs),flush=True)
    if postrows:write_csv(OUTPUT/'processed/actual_post_decision_differences.csv',postrows)
    result=dict(status='PASSED',runs=len(runs),steps=sum(c['steps'] for c in checks),checks=checks)
    atomic_json(OUTPUT/'qa/trace_audit.json',result)
    atomic_json(OUTPUT/'qa/information_boundary_audit.json',dict(status='PASSED',contexts_sealed=contexts,
        future_bookings='deployable summary has no future schedule; candidate guard rejects future book_time; Oracle independent',
        selected_contexts='fixed before solve; scalar coefficients only; post-solve audit never read by provider'))
    atomic_json(OUTPUT/'qa/value_provider_audit.json',dict(status='PASSED',linear_objectives=True,slot_prefix_tests=True,
        legacy_none_compatible=True,once_per_sink_correction=True))
    return result

def read_csv(name):
    import csv
    with (OUTPUT/'processed'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def matched_continuation_sensitivity(cfg):
    primary=read_json(OUTPUT/'targets/construction.json')['base'];secondary=read_json(OUTPUT/'targets/sensitivity.json')['base']
    def identity(r):return tuple(r.get(k) for k in ['condition','scenario_seed','time','hub','q','mc_replication','state_hash','future_hash'])
    lookup={identity(r):r for r in primary if not r.get('terminal_anchor')};out=[]
    for r in secondary:
        if r.get('terminal_anchor') or identity(r) not in lookup:continue
        p=lookup[identity(r)]
        out.append(dict(condition=r['condition'],scenario_seed=r['scenario_seed'],time=r['time'],hub=r['hub'],q=r['q'],
            target_myopic=p['target'],target_RH4=r['target'],difference=r['target']-p['target'],
            mc_replication=r['mc_replication'],state_hash=r['state_hash'],future_hash=r['future_hash'],
            interpretation='strictly matched state, hub, q and future path; diagnostic only, no refitting'))
    assert out,'no exact continuation-sensitivity pairs'
    write_csv(OUTPUT/'targets/continuation_sensitivity.csv',out)
    summary=[]
    for q in sorted({r['q'] for r in out}):
        a=[r for r in out if r['q']==q];summary.append(dict(q=q,**ci([r['difference'] for r in a],cfg)))
    write_csv(OUTPUT/'processed/continuation_sensitivity_summary.csv',summary)
    atomic_json(OUTPUT/'qa/continuation_sensitivity.json',dict(status='PASSED',matched_rows=len(out),
        independent_seeds=len({r['scenario_seed'] for r in out}),periods=sorted({r['time'] for r in out}),
        unmatched_secondary_rows=len(secondary)-len(out),note='unmatched queries/futures excluded from paired diagnostic, retained in raw secondary dataset; no outcome-driven filtering'))
    return out

def publication_figures(cfg,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','legend.fontsize':6})
    conditions=[key(c) for c in cfg['conditions']]
    titles=dict(zip(conditions,['California N80/K3','Western N40/K8','Western N80/K3','Western N20/K8']))
    colors=['#777777','#000000','#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']
    markers=['o','x','s','^','D','v','P'];color=dict(zip(METHODS,colors));marker=dict(zip(METHODS,markers));configs=[]
    def panels(height=5.4):return plt.subplots(2,2,figsize=(7.2,height))
    def save(fig,name,data,symlog=None):
        path=OUTPUT/'processed'/f'{name}_source.csv';write_csv(path,data)
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(OUTPUT/'figures'/f'{name}.{ext}',dpi=600,bbox_inches='tight')
        configs.append(dict(name=name,size_inches=list(fig.get_size_inches()),dpi=600,font='DejaVu Serif',
            source_csv=str(path.relative_to(OUTPUT)),source_sha256=file_hash(path),symlog=symlog,
            clipping='none; all extremes retained',pdf_fonttype=42,colors=colors,markers=markers))
        plt.close(fig)
    # 1. Discrete slot surface; no arbitrary regression profile.
    fig,axes=panels();data=[]
    for ax,cid in zip(axes.flat,conditions):
        model=read_json(OUTPUT/'models/table'/f'{cid}.json');hubs=sorted(model['values']);ts=model['values'][hubs[0]]['1']['times']
        Z=np.array([model['values'][h]['1']['values'] for h in hubs]);bound=max(abs(Z).max(),100)
        im=ax.pcolormesh(np.arange(len(ts)+1),np.arange(len(hubs)+1),Z,cmap='RdBu_r',norm=SymLogNorm(linthresh=100,vmin=-bound,vmax=bound),rasterized=False)
        ax.set_xticks(np.arange(len(ts))+.5,ts);ax.set_yticks(np.arange(len(hubs))+.5,hubs);ax.invert_yaxis();ax.set_title(titles[cid]);ax.set_xlabel('Release period (q=1)')
        cb=fig.colorbar(im,ax=ax,label='USD; symlog 100')
        if cb.solids is not None:cb.solids.set_rasterized(False)
        for i in range(len(hubs)):
            for j in range(len(ts)):
                text=f'{Z[i,j]/1000:.1f}k' if abs(Z[i,j])>=1000 else f'{Z[i,j]:.0f}'
                ax.text(j+.5,i+.5,text,ha='center',va='center',fontsize=4.5,color='white' if abs(Z[i,j])>bound/2 else 'black')
        data.extend(dict(condition=cid,hub=h,time=t,value=float(Z[i,j]),q=1) for i,h in enumerate(hubs) for j,t in enumerate(ts))
    save(fig,'01_mc_slot_heatmap',data,{'color_linthresh':100})
    # 2. Every correction observation including extreme residual predictions.
    predictions=read_csv('value_predictions.csv');local=['linear','chebyshev','spline','bayesian']
    fig,axes=panels();data=[];rng=np.random.default_rng(77)
    for ax,cid in zip(axes.flat,conditions):
        for j,m in enumerate(local):
            a=[r for r in predictions if r['condition']==cid and r['method']==m];v=np.array([float(r['prediction'])-float(r['base_slot_value']) for r in a])
            ax.scatter(j+rng.uniform(-.12,.12,len(v)),v,s=2,alpha=.18,color=color[m],marker=marker[m])
            ax.plot([j-.18,j+.18],[np.median(v)]*2,color='black',lw=1)
            data.extend(dict(condition=cid,method=m,seed=r['scenario_seed'],candidate_id=r['candidate_id'],q=r['q'],correction=float(x)) for r,x in zip(a,v))
        ax.set_xticks(range(4),['Linear','Cheb','Spline','Bayes']);ax.set_yscale('symlog',linthresh=100);ax.set_ylabel('Correction (USD)');ax.set_title(titles[cid]);ax.axhline(0,c='grey',lw=.5)
    save(fig,'02_local_correction_distribution',data,{'y_linthresh':100})
    # 3. Individual coefficient blocks; not a causal smoothness measure.
    fig,axes=panels();data=[]
    for ax,cid in zip(axes.flat,conditions):
        m=read_json(OUTPUT/'models/chebyshev'/f'{cid}.json');beta=np.array(m['coefficients']);offset=1+len(m['hubs'])+len(m['qs'])
        for j,f in enumerate(FEATURES):
            n=m['bases'][f].get('degree',1);a=beta[offset:offset+n];offset+=n
            ax.plot(range(1,n+1),abs(a),marker=markers[j%7],color=colors[j%7],ls='-' if j<7 else '--',ms=2,lw=.7,label=f)
            data.extend(dict(condition=cid,feature=f,degree=k,coefficient=float(x)) for k,x in enumerate(a,1))
        ax.set_yscale('symlog',linthresh=100);ax.set_ylim(bottom=0);ax.set_xticks(range(1,max(b.get('degree',1) for b in m['bases'].values())+1))
        upper=ax.get_ylim()[1];ticks=[v for v in [0,10,100,1000,10000,100000] if v<=upper]
        ax.set_yticks(ticks,[str(v) for v in ticks]);ax.minorticks_off()
        ax.set_xlabel('Additive basis degree');ax.set_ylabel('|Coefficient| (USD)');ax.set_title(titles[cid])
    axes[0,0].legend(ncol=2,fontsize=4.5);save(fig,'03_chebyshev_coefficient_decay',data,{'y_linthresh':100})
    # 4. Centered spline partial differences on common frozen profiles.
    fig,axes=plt.subplots(4,3,figsize=(7.2,9));data=[]
    train=averaged_actions(read_json(OUTPUT/'targets/construction.json')['actions'])
    for ax,f in zip(axes.flat,FEATURES):
        for j,cid in enumerate(conditions):
            m=read_json(OUTPUT/'models/spline'/f'{cid}.json');a=[r for r in train if r['condition']==cid and not r['terminal_anchor']]
            profile={name:float(np.median([r[name] for r in a])) for name in FEATURES};profile.update(hub=m['hubs'][0],q=1)
            xs=np.linspace(min(r[f] for r in a),max(r[f] for r in a),101)
            vals=(design([profile|{f:float(x)} for x in xs],m,True)-design([profile],m,True))@np.array(m['coefficients'])
            ax.plot(xs,vals,color=colors[j+2],ls=['-','--',':','-.'][j],label=titles[cid],lw=.8)
            data.extend(dict(condition=cid,feature=f,x=float(x),partial_difference=float(y)) for x,y in zip(xs,vals))
        ax.set_title(f);ax.set_yscale('symlog',linthresh=100);ax.set_ylabel('Partial C (USD)',fontsize=6)
    axes[0,0].legend(fontsize=4.5);save(fig,'04_spline_components',data,{'y_linthresh':100})
    # 5. Adjusted group effects, raw residual mean remains in source table.
    shrink=read_csv('bayesian_shrinkage.csv');fig,axes=panels()
    for ax,cid in zip(axes.flat,conditions):
        a=[r for r in shrink if r['condition']==cid];x=np.array([float(r['raw_adjusted_hub_effect']) for r in a]);y=np.array([float(r['posterior_hub_effect']) for r in a])
        for r,u,v in zip(a,x,y):ax.scatter(u,v,s=8+8*np.log1p(int(r['n'])),marker='^' if r['hub'].startswith('q=') else 'o',alpha=.6,color='#0072B2')
        lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],c='grey',ls='--',lw=.6)
        ax.set_xscale('symlog',linthresh=100);ax.set_yscale('symlog',linthresh=100);ax.set_title(titles[cid]);ax.set_xlabel('Unpooled adjusted effect (USD)');ax.set_ylabel('Posterior group effect (USD)')
    save(fig,'05_bayesian_shrinkage',shrink,{'x_linthresh':100,'y_linthresh':100})
    # 6. Conditional predictive coverage, not policy confidence intervals.
    cal=read_csv('bayesian_calibration.csv');fig,axes=panels()
    for ax,cid in zip(axes.flat,conditions):
        for group,col,ls in [('all','#000000','-'),('sparse','#0072B2','--'),('dense','#D55E00',':')]:
            a=sorted([r for r in cal if r['condition']==cid and r['group']==group],key=lambda r:float(r['nominal']))
            ax.plot([float(r['nominal']) for r in a],[float(r['coverage']) for r in a],color=col,ls=ls,marker='o',label=group)
        ax.plot([0,1],[0,1],c='grey',lw=.6);ax.set_xlim(.4,1);ax.set_ylim(0,1);ax.set_title(titles[cid]);ax.set_xlabel('Nominal coverage');ax.set_ylabel('Empirical coverage');ax.legend()
    save(fig,'06_bayesian_calibration',cal)
    # 7. All methods, including frozen legacy extremes, use symmetric log axes.
    fig,axes=panels()
    for ax,cid in zip(axes.flat,conditions):
        for m in METHODS:
            a=[r for r in predictions if r['condition']==cid and r['method']==m]
            ax.scatter([float(r['target']) for r in a],[float(r['prediction']) for r in a],s=3,alpha=.18,color=color[m],marker=marker[m],label=m)
        ax.set_xscale('symlog',linthresh=100);ax.set_yscale('symlog',linthresh=100);ax.set_title(titles[cid]);ax.set_xlabel('Paired target Y (USD)');ax.set_ylabel('Deployment prediction (USD)')
        limits=ax.get_xlim();ax.plot(limits,limits,c='grey',ls='--',lw=.6)
    axes[0,0].legend(fontsize=4.5);save(fig,'07_heldout_prediction_scatter',predictions,{'x_linthresh':100,'y_linthresh':100})
    # 8. Distinguish action-spatial ranking and strict matched-release ranking.
    vm=read_csv('value_metrics.csv');fig,axes=panels()
    for ax,cid in zip(axes.flat,conditions):
        for j,m in enumerate(METHODS):
            r=next(r for r in vm if r['condition']==cid and r['method']==m)
            if r['spatial_same_q_ranking_accuracy']:ax.scatter(j,float(r['spatial_same_q_ranking_accuracy']),color=color[m],marker=marker[m],s=24)
            if r['same_q_matched_release_ranking']:ax.scatter(j,float(r['same_q_matched_release_ranking']),color=color[m],marker='x',s=35)
        ax.axhline(.5,color='grey',ls='--',lw=.6);ax.set_ylim(-.03,1.03);ax.set_xticks(range(7),['0','Fluid','Table','Lin','Cheb','Spl','Bayes'],fontsize=6)
        ax.set_title(titles[cid]);ax.set_ylabel('Same-q rank accuracy');ax.text(.02,.04,'x: same release; blank if no pairs',transform=ax.transAxes,fontsize=5)
    save(fig,'08_same_q_spatial_ranking',vm)
    # 9. Formal scenario means and t95 intervals; paired results are separate.
    fig,axes=panels();data=stats['summary'];order=[f'P{i}' for i in range(15)]+['D']
    for ax,cid in zip(axes.flat,conditions):
        a=[next(r for r in data if r['condition']==cid and r['policy']==p) for p in order]
        ax.errorbar(range(16),[r['mean'] for r in a],yerr=[r['ci_upper']-r['mean'] for r in a],fmt='o',ms=2,capsize=2,color='#0072B2',lw=.6)
        ax.set_xticks(range(16),order,rotation=90,fontsize=5);ax.set_title(titles[cid]);ax.set_ylabel('Full-cooldown profit (USD)')
    save(fig,'09_policy_ladder_profit',data)
    # 10. Paired RH4 improvements, alongside paired RH8 gain.
    fig,axes=panels();data=[]
    runs=read_json(OUTPUT/'processed/formal_policies.json')
    for ax,cid in zip(axes.flat,conditions):
        vals={p:{r['seed']:r['realized_profit'] for r in runs if r['condition']==cid and r['policy']==p} for p in ['P7','P14']}
        d=ci([vals['P14'][s]-vals['P7'][s] for s in vals['P7']],cfg);ax.axhline(d['mean'],color='black',ls='--',label='RH8 - RH4')
        ax.axhspan(d['ci_lower'],d['ci_upper'],color='grey',alpha=.15)
        data.append(dict(condition=cid,policy='P14',**d))
        for j,p in enumerate(['P8','P9','P10','P11','P12','P13']):
            r=next(r for r in stats['paired'] if r['condition']==cid and r['left']==p)
            ax.errorbar(j,r['mean'],yerr=[[r['mean']-r['ci_lower']],[r['ci_upper']-r['mean']]],fmt=markers[j+1],color=colors[j+1],capsize=2);data.append(r|{'policy':p})
        ax.set_xticks(range(6),['Fluid','Table','Lin','Cheb','Spline','Bayes'],fontsize=6);ax.set_yscale('symlog',linthresh=100);ax.set_ylabel('Paired gain over RH4 (USD)');ax.set_title(titles[cid]);ax.legend()
    save(fig,'10_rh4_terminal_vs_rh8',data,{'y_linthresh':100})
    # 11. Descriptive gap ratios, inadmissible denominators shown as missing.
    fig,axes=panels();data=[r for r in stats['ratios'] if r['kind']=='oracle_gap']
    for ax,cid in zip(axes.flat,conditions):
        for base,pols,ls in [('Myopic',['P1','P2','P3','P4','P5','P6'],'-'),('RH4',['P8','P9','P10','P11','P12','P13'],'--')]:
            a=[next(r for r in data if r['condition']==cid and r['policy']==p) for p in pols]
            ax.plot(range(6),[r['ratio'] if r['ratio'] is not None else np.nan for r in a],ls=ls,marker='o',label=base)
        ax.axhline(0,c='grey',lw=.5);ax.axhline(1,c='grey',ls=':',lw=.5);ax.set_yscale('symlog',linthresh=1)
        ax.set_xticks(range(6),['Fluid','Table','Lin','Cheb','Spline','Bayes'],fontsize=6);ax.set_ylabel('Performance gap ratio');ax.set_title(titles[cid]);ax.legend()
    save(fig,'11_oracle_gap_capture',data,{'y_linthresh':1})
    # 12. Online daily execution excludes all offline fits.
    fig,axes=panels();data=[]
    for ax,cid in zip(axes.flat,conditions):
        for j,p in enumerate(order):
            profit=next(r for r in stats['summary'] if r['condition']==cid and r['policy']==p)
            runtime=next(r for r in stats['runtime'] if r['condition']==cid and r['policy']==p and r['component']=='daily_wall_seconds')
            method_idx=0 if p in ['P0','P7','P14','D'] else (int(p[1:]) if int(p[1:])<7 else int(p[1:])-7)
            marker_= 'o' if p.startswith('P') and int(p[1:])<7 else 's' if p.startswith('P') and 7<=int(p[1:])<=13 else '^'
            ax.scatter(runtime['mean_seconds'],profit['mean'],marker=marker_,color=colors[method_idx],s=24,label=METHODS[method_idx] if j<7 else None)
            data.append(dict(condition=cid,policy=p,mean_daily_seconds=runtime['mean_seconds'],mean_profit=profit['mean']))
        ax.set_xscale('log');ax.set_title(titles[cid]);ax.set_xlabel('Online daily wall time (s, log)');ax.set_ylabel('Mean realized profit (USD)')
        ax.text(.02,.03,'o Myopic; s RH4; ^ RH8 / Oracle',transform=ax.transAxes,fontsize=5)
    axes[0,0].legend(fontsize=4.5,loc='lower right');save(fig,'12_runtime_profit_frontier',data)
    # 13. Actual solved joint assignment audit; never a hypothetical RH suffix.
    fig,axes=panels();data=[];joint_targets=read_csv('joint_post_action_targets.csv')
    for ax,cid in zip(axes.flat,conditions):
        for p,m in [('P4','chebyshev'),('P6','bayesian'),('P11','chebyshev'),('P13','bayesian')]:
            a=[r for r in joint_targets if r['condition']==cid and r['policy']==p]
            if a:ax.scatter([float(r['actual_post_imbalance']) for r in a],[float(r['error']) for r in a],color=color[m],marker='o' if p in ['P4','P6'] else 's',s=9,alpha=.45,label=p);data.extend(a)
        ax.set_xscale('symlog',linthresh=1);ax.set_yscale('symlog',linthresh=100);ax.set_title(titles[cid]);ax.set_xlabel('Actual joint post-action imbalance');ax.set_ylabel('Y - original deployment value (USD)');ax.legend()
    save(fig,'13_omitted_post_state_residual',data,{'x_linthresh':1,'y_linthresh':100})
    atomic_json(OUTPUT/'figures/plotting_config.json',dict(figures=configs,statistical_unit='independent scenario seed',
        panel_size='double-column 7.2 inch; component figure 4x3 facets',black_white='markers and line types supplement color',
        value_metrics_sha256=file_hash(OUTPUT/'processed/value_metrics.csv'),policy_statistics_sha256=file_hash(OUTPUT/'processed/statistics.json')))
    return configs

def independent_reproduction(cfg):
    from model_families.single.data import matching_state
    from model_families.single.local_value.providers import load_local
    from .simulation import simulate
    from .baseline_regression import physical
    cid='western_multicluster-n40-k8';rid=f'formal-{cid}-5000-P13';p=OUTPUT/'raw/runs'/rid
    job=read_json(p/'spec.json');old=read_json(p/'summary.json');sim,orders=matching_state(job['scenario'],job['parameters'],job['solver'])
    metrics,trace,_=simulate(sim,orders,job['policy'],load_local(OUTPUT/'models/bayesian'/f'{cid}.json'),job['prior'],trace=True)
    original=read_json(p/'trace.json');delta=metrics['realized_profit']-old['realized_profit']
    assert abs(delta)<1e-6 and physical(trace)==physical(original)
    from model_families.single.local_value.providers import LegacyAdapter
    from experiments.cfs_classical_value.simulation import summary
    compatibility=0
    for c in cfg['conditions']:
        condition=key(c);j=read_json(OUTPUT/'raw/runs'/f'formal-{condition}-5000-P0'/'spec.json')
        s,o=matching_state(j['scenario'],j['parameters'],j['solver']);s.begin_period([v for v in o.values() if v.book_time==0])
        online=summary(s,read_json(OLD/'models/priors.json')[condition])
        for method in ['table','fluid','linear','chebyshev','spline']:
            oldp=load_provider(OLD/'models'/f'{condition}-{method}.json');adapter=LegacyAdapter(oldp)
            for when,hub,q in itertools.product([0,3,12,23,24],online['hubs'],range(1,c['fleet']+1)):
                expected=oldp.marginal_release_value(when,hub,q,online)
                assert adapter.marginal_release_value(when,hub,q,online,None)==expected
                compatibility+=1
    result=dict(status='PASSED',run_id=rid,profit_difference=delta,physical_trace_identical=True,
        legacy_none_exact_checks=compatibility,
        original_physical_hash=digest(physical(original)),reproduced_physical_hash=digest(physical(trace)))
    atomic_json(OUTPUT/'qa/independent_reproduction.json',result);return result

def main():
    cfg=read_json(OUTPUT/'config/main.json');manifest=read_json(OUTPUT/'manifest.json')
    assert manifest['status']=='NUMERICAL_RUNS_COMPLETED'
    freeze=read_json(OUTPUT/'manifests/model_freeze.json')
    assert all(file_hash(OUTPUT/name)==sha for name,sha in freeze['models'].items())
    value_diagnostics(cfg);value_paired_comparisons(cfg);old_summary_value_comparison(cfg)
    runs,stats=policy_statistics(cfg)
    audit=replay_audit(runs+read_json(OUTPUT/'processed/ablation_policies.json')+read_json(OUTPUT/'processed/pilot_policies.json'))
    post=post_audit(cfg);jointrows,joint=actual_joint_post_audit(cfg);independent=independent_reproduction(cfg);runtime_reference(cfg)
    figures=publication_figures(cfg,stats)
    # Canonical deliverables are scalar/JSON-safe aliases, while raw split tables remain intact.
    targets=read_json(OUTPUT/'targets/construction.json')
    write_csv(OUTPUT/'targets/mc_slot_targets.csv',targets['base']);write_csv(OUTPUT/'targets/action_context_targets.csv',targets['actions'])
    matched_continuation_sensitivity(cfg)
    atomic_json(OUTPUT/'qa/analysis_summary.json',dict(status='PASSED',trace=audit,independent=independent,
        post_audit_rows=len(post),joint_post_target_rows=len(jointrows),joint_post_audit_rows=len(joint),figures=len(figures),analysis_source_sha256=file_hash(OUTPUT/'analysis.py'),
        models_unchanged=True,formal_count=len(runs),created_at=now()))
    print('Analysis, independent replay and 13 figure groups completed',flush=True)

if __name__=='__main__':main()
