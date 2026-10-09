"""Scenario-paired inference and restricted-class stabilization diagnostics."""
import csv,itertools,time
from collections import defaultdict
import numpy as np
from scipy.stats import t as student_t,skew,spearmanr,wasserstein_distance
from experiment_core.storage import read_json,atomic_json,digest
from experiments.cfs_classical_value.simulation import summary
from .common import OUTPUT,key,write_csv,COMPARISONS
from .protocol import read_compressed,restore

def numeric(x):return float(x) if np.isfinite(x) else None
def oscillation_metrics(values):
    """Native JSON scalar counts; numerical comparisons are unchanged."""
    return dict(sign_flip_count=int(sum(np.sign(a)!=np.sign(b) for a,b in zip(values,values[1:]))),
        range_USD=float(max(values)-min(values)),
        update_direction_reversal_count=int(sum(a*b<0 for a,b in zip(np.diff(values),np.diff(values)[1:]))))

def describe(values,cfg,bootstrap=True):
    x=np.asarray(values,float);n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1)) if n>1 else 0
    half=float(student_t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else 0
    if sd>1e-12:p=float(2*student_t.sf(abs(mean)/(sd/np.sqrt(n)),n-1))
    else:p=0. if abs(mean)>1e-12 else 1.
    out=dict(n=n,mean=mean,SD=sd,median=float(np.median(x)),CI95_lower=mean-half,CI95_upper=mean+half,
        skewness=numeric(skew(x,bias=False)) if n>2 and sd>1e-12 else 0.,p_two_sided=p)
    if bootstrap:
        rng=np.random.default_rng(cfg['bootstrap_seed']);means=x[rng.integers(0,n,(cfg['bootstrap_replications'],n))].mean(axis=1)
        out.update(bootstrap95_lower=float(np.quantile(means,.025)),bootstrap95_upper=float(np.quantile(means,.975)),
            bootstrap_replications=cfg['bootstrap_replications'])
    return out

def holm(rows):
    sortedrows=sorted(rows,key=lambda r:r['p_two_sided']);last=0
    for i,r in enumerate(sortedrows):
        last=max(last,min(1,(len(rows)-i)*r['p_two_sided']));r['Holm_p']=last

def profits(cfg):
    runs=read_json(OUTPUT/'processed/formal_runs.json');summaries=[];comparisons=[];differences=[]
    for c in cfg['conditions']:
        cid=key(c);rr=[r for r in runs if r['condition']==cid];lookup={(r['scenario_seed'],r['policy']):r for r in rr}
        assert len(lookup)==40*8
        for seed in cfg['formal_seeds']:
            assert len({lookup[seed,p]['scenario_hash'] for p in ['P0','P1','P2','P3','P4','R0','R1','D']})==1
        for policy in ['P0','P1','P2','P3','P4','R0','R1','D']:
            a=[lookup[s,policy] for s in cfg['formal_seeds']]
            summaries.append(dict(condition=cid,policy=policy,**describe([r['realized_profit'] for r in a],cfg),
                mean_online_seconds=float(np.mean([r['daily_wall_seconds'] for r in a])),
                mean_service=float(np.mean([r['delivered']/c['orders'] for r in a]))))
        for a,b in COMPARISONS:
            x=[lookup[s,a]['realized_profit']-lookup[s,b]['realized_profit'] for s in cfg['formal_seeds']]
            for seed,value in zip(cfg['formal_seeds'],x):differences.append(dict(condition=cid,scenario_seed=seed,policy_a=a,policy_b=b,difference_USD=value))
            comparisons.append(dict(condition=cid,policy_a=a,policy_b=b,comparison=f'{a}-{b}',
                family='oracle_descriptive' if a=='D' else 'main_36',**describe(x,cfg),
                reference_mean=float(np.mean([lookup[s,b]['realized_profit'] for s in cfg['formal_seeds']]))))
    holm([r for r in comparisons if r['family']=='main_36'])
    for r in comparisons:
        if r['family']=='oracle_descriptive':r['Holm_p']=None
    write_csv(OUTPUT/'processed/formal_policy_summary.csv',summaries)
    write_csv(OUTPUT/'processed/paired_policy_improvements.csv',comparisons)
    write_csv(OUTPUT/'processed/paired_scenario_differences.csv',differences)
    atomic_json(OUTPUT/'processed/policy_statistics.json',dict(summary=summaries,comparisons=comparisons,
        sample_unit='40 independent scenarios per condition; same scenario across policies',formal_runs=len(runs),
        Holm_family='36 declared main comparisons; Oracle gap descriptive, not an attainable bound'))
    return summaries,comparisons

def csvrows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def query_support(model,when,hub,q,fleet,slots,horizon):
    """Provenance only, never a replacement value provider or a new estimator."""
    if when>=horizon:return dict(support_status='terminal_exact_zero',source_time_left=horizon,
        source_time_right=horizon,time_weight_right=0.,source_q_left=None,source_q_right=None,q_weight_right=0.)
    times=model['values'][hub][str(q)]['times'];nodes=sorted(set(int(v) for v in slots if int(v)<=fleet)|{fleet})
    left=max((v for v in times if v<=when),default=times[0]);right=min((v for v in times if v>=when),default=times[-1])
    qleft=max((v for v in nodes if v<=q),default=nodes[0]);qright=min((v for v in nodes if v>=q),default=nodes[-1])
    flags=[]
    if when not in times:flags.append('interpolated_time')
    if q>fleet:flags.append('unsupported_q_endpoint')
    elif q not in nodes:flags.append('interpolated_q')
    return dict(support_status='+'.join(flags) if flags else 'observed',source_time_left=left,source_time_right=right,
        time_weight_right=0. if left==right else float((when-left)/(right-left)),source_q_left=qleft,source_q_right=qright,
        q_weight_right=0. if qleft==qright else float((q-qleft)/(qright-qleft)))

def deployment_query_provenance(cfg):
    """All integer online/counterfactual query keys, without fake interpolation SE."""
    from model_families.single.value.providers import MCTableValueProvider
    from .common import file_hash
    rows=[]
    for n in range(4):
        raw={(r['condition'],int(r['time']),r['hub'],int(r['q'])):r for r in csvrows(OUTPUT/'tables'/f'V{n}'/'cells.csv')}
        for c in cfg['conditions']:
            cid=key(c);path=OUTPUT/'tables'/f'V{n}'/(cid+'.json');model=read_json(path);provider=MCTableValueProvider(model)
            horizon=24;sha=file_hash(path)
            for t,h,q in itertools.product(range(horizon+1),sorted(model['values']),range(1,2*c['fleet']+1)):
                info=query_support(model,t,h,q,c['fleet'],cfg['target_supply_slots'],horizon)
                value=provider.marginal_release_value(t,h,q,dict(T=horizon));source=raw.get((cid,t,h,q))
                if source is not None:assert abs(value-float(source['value']))<1e-9
                direct=info['support_status']=='observed'
                rows.append(dict(condition=cid,iteration=n,time=t,hub=h,q=q,value_USD=value,table_hash=sha,**info,
                    directly_measured_source_scenarios=int(source['independent_scenarios']) if direct and source else 0,
                    MC_SE_USD=float(source['MC_SE']) if direct and source and source['MC_SE'] else None,
                    uncertainty='terminal deterministic zero' if t==horizon else 'raw-cell conditional descriptive MC_SE' if direct else
                        'interpolated/extrapolated value: no invented SE or new independent samples; source corners and rules recorded'))
    write_csv(OUTPUT/'processed/deployment_query_provenance.csv',rows)
    atomic_json(OUTPUT/'qa/deployment_query_provenance_audit.json',dict(status='PASSED',integer_query_keys=len(rows),
        stored_nodes_match_provider=True,all_integer_times_and_q_to_2K=True,primary_table_metrics_unchanged=True))

def table_diagnostics(cfg):
    cells=[];lookup={};byiter={}
    for n in range(4):
        rr=csvrows(OUTPUT/'tables'/f'V{n}'/'cells.csv');byiter[n]=rr
        for r in rr:
            for name in ['time','q','independent_scenarios','iteration']:r[name]=int(r[name])
            for name in ['value','MC_SE','MC_SD']:r[name]=float(r[name]) if r[name] else None
            lookup[r['condition'],r['time'],r['hub'],r['q'],n]=r;cells.append(r)
    changes=[];monotonic=[];negative=[];oscillations=[];rank=[];qrank=[]
    for c in cfg['conditions']:
        cid=key(c)
        for n in range(3):
            for scope in ['observed_shared','deployment_grid']:
                rr=[r for r in byiter[n] if r['condition']==cid and r['time']<24 and r['q']<=c['fleet'] and
                    (scope!='observed_shared' or r['support_status']=='observed')]
                for segment in ['all','early','middle','late']:
                    subset=[r for r in rr if segment=='all' or (segment=='early' and r['time']<8) or
                        (segment=='middle' and 8<=r['time']<16) or (segment=='late' and r['time']>=16)]
                    a=np.array([r['value'] for r in subset]);b=np.array([lookup[cid,r['time'],r['hub'],r['q'],n+1]['value'] for r in subset]);d=b-a
                    changes.append(dict(condition=cid,from_V=n,to_V=n+1,scope=scope,segment=segment,cells=len(a),
                        RMSE=float(np.sqrt(np.mean(d*d))),relative_L2=float(np.linalg.norm(d)/(np.linalg.norm(a)+1e-12)),
                        Spearman=numeric(spearmanr(a,b).statistic) if np.ptp(a)>1e-9 and np.ptp(b)>1e-9 else None,
                        sign_flip_rate=float(np.mean(np.sign(a)!=np.sign(b))),strict_sign_flip_rate=float(np.mean(a*b<0)),
                        note='same cell support, different current-policy state distributions; joint operator and distribution change'))
            times=sorted({r['time'] for r in byiter[n] if r['condition']==cid and r['time']<24})
            hubs=sorted({r['hub'] for r in byiter[n] if r['condition']==cid})
            for t,q in itertools.product(times,range(1,c['fleet']+1)):
                a=np.array([lookup[cid,t,h,q,n]['value'] for h in hubs]);b=np.array([lookup[cid,t,h,q,n+1]['value'] for h in hubs])
                rank.append(dict(condition=cid,from_V=n,to_V=n+1,time=t,q=q,hub_rank_Spearman=numeric(spearmanr(a,b).statistic)
                    if np.ptp(a)>1e-9 and np.ptp(b)>1e-9 else None,top_hub_changed=int(hubs[int(np.argmax(a))]!=hubs[int(np.argmax(b))])))
            for t,h in itertools.product(times,hubs):
                a=np.array([lookup[cid,t,h,q,n]['value'] for q in range(1,c['fleet']+1)])
                b=np.array([lookup[cid,t,h,q,n+1]['value'] for q in range(1,c['fleet']+1)])
                qrank.append(dict(condition=cid,from_V=n,to_V=n+1,time=t,hub=h,
                    q_order_Spearman=numeric(spearmanr(a,b).statistic) if np.ptp(a)>1e-9 and np.ptp(b)>1e-9 else None))
        for n in range(4):
            for r in byiter[n]:
                if r['condition']!=cid or r['time']>=24 or r['q']>c['fleet']:continue
                if r['value']<0:
                    se=r['MC_SE'];negative.append(dict(condition=cid,iteration=n,time=r['time'],hub=r['hub'],q=r['q'],value=r['value'],
                        support_status=r['support_status'],MC_SE=se,negative_two_SE_support=se is not None and r['value']+2*se<0,
                        persistence_count=sum(lookup[cid,r['time'],r['hub'],r['q'],i]['value']<0 for i in range(4)),
                        interpretation='policy-induced marginal can be negative; costs and action switches are real; MC noise/support also matter, no clipping'))
                if r['q']<c['fleet']:
                    b=lookup[cid,r['time'],r['hub'],r['q']+1,n];diff=b['value']-r['value']
                    monotonic.append(dict(condition=cid,iteration=n,time=r['time'],hub=r['hub'],q=r['q'],next_q=r['q']+1,
                        violation=int(diff>1e-6),magnitude_USD=max(0,diff),value_q=r['value'],value_next=b['value'],
                        support_q=r['support_status'],support_next=b['support_status'],SE_q=r['MC_SE'],SE_next=b['MC_SE'],
                        persistence_count=sum(lookup[cid,r['time'],r['hub'],r['q']+1,i]['value']>lookup[cid,r['time'],r['hub'],r['q'],i]['value']+1e-6 for i in range(4)),
                        restriction='no isotonic repair; differences between dependent q estimates not independent-SE test'))
        for r in byiter[0]:
            if r['condition']!=cid or r['time']>=24 or r['q']>c['fleet']:continue
            values=[lookup[cid,r['time'],r['hub'],r['q'],i]['value'] for i in range(4)]
            oscillations.append(dict(condition=cid,time=r['time'],hub=r['hub'],q=r['q'],**{f'V{i}':v for i,v in enumerate(values)},
                **oscillation_metrics(values),
                support_status=r['support_status'],occupancy_frequency=None))
    write_csv(OUTPUT/'processed/table_change_metrics.csv',changes);write_csv(OUTPUT/'processed/hub_ranking_changes.csv',rank)
    write_csv(OUTPUT/'processed/q_ranking_changes.csv',qrank)
    write_csv(OUTPUT/'processed/q_monotonicity_audit.csv',monotonic);write_csv(OUTPUT/'processed/negative_value_audit.csv',negative)
    # Negative output may be empty: store an explicit zero-count summary too.
    atomic_json(OUTPUT/'processed/table_diagnostics.json',dict(changes=changes,monotonicity=monotonic,negative=negative,oscillations=oscillations,
        negative_count=len(negative),negative_are_not_errors=True))
    return changes,monotonic,negative,oscillations

def state_distributions(cfg,oscillations):
    runs=read_json(OUTPUT/'processed/diagnostic_runs.json');rows=[];releases=[]
    for r in runs:
        j=read_json(OUTPUT/'raw/runs'/r['id']/'spec.json');data=read_compressed(OUTPUT/r['trace_ref'])
        for t,packet in data['state_packets'].items():
            s,_=restore(j['scenario'],j['parameters'],j['solver'],packet);online=summary(s,j['prior'])
            for hub in sorted(s.net.H):
                incoming=[item for item in online['incoming'] if item['hub']==hub]
                releases.extend(dict(condition=r['condition'],policy=r['policy'],iteration=int(r['policy'][1:]),
                    scenario_seed=r['scenario_seed'],time=int(t),hub=hub,incoming_index=i,
                    context_release_time=item['time'],context_release_delay=item['time']-s.t,
                    provenance='frozen summary incoming context; deterministic existing-commitment projection, no unbooked demand')
                    for i,item in enumerate(incoming))
                rows.append(dict(condition=r['condition'],policy=r['policy'],iteration=int(r['policy'][1:]),scenario_seed=r['scenario_seed'],time=int(t),hub=hub,
                    idle=float(online['idle'][hub]),incoming=float(len(incoming)),backlog=float(online['backlog'][hub]),
                    committed=float(sum(v.phase!='idle' for v in s.vehs)),fleet_utilization=sum(v.phase!='idle' for v in s.vehs)/len(s.vehs),
                    release_delay_mean=float(np.mean([a['time']-s.t for a in incoming])) if incoming else 0.,
                    state_hash=digest(packet)))
    write_csv(OUTPUT/'processed/state_distribution_observations.csv',rows)
    write_csv(OUTPUT/'processed/release_time_observations.csv',releases)
    releasegroups=defaultdict(list)
    for r in releases:releasegroups[r['condition'],r['iteration'],r['time'],r['hub']].append(r)
    releasehist=[]
    for group,items in releasegroups.items():
        cid,n,t,h=group;delays=sorted({r['context_release_delay'] for r in items})
        for delay in delays:
            count=sum(r['context_release_delay']==delay for r in items)
            releasehist.append(dict(condition=cid,iteration=n,time=t,hub=h,context_release_delay=delay,
                incoming_observations=count,total_incoming_observations=len(items),fraction=count/len(items),
                independent_scenarios=20,aggregation='pooled descriptive incoming-resource histogram, not independent statistical replicates'))
    write_csv(OUTPUT/'processed/release_time_histograms.csv',releasehist)
    grouped=defaultdict(list)
    for r in rows:grouped[r['condition'],r['time'],r['hub'],r['iteration']].append(r)
    shifts=[];coverage=[];features=['idle','incoming','backlog','committed','fleet_utilization','release_delay_mean']
    for c in cfg['conditions']:
        cid=key(c);hubs=sorted({r['hub'] for r in rows if r['condition']==cid})
        for n,t,h in itertools.product(range(4),range(24),hubs):
            a=grouped[cid,t,h,n];b=grouped[cid,t,h,n+1];assert len(a)==len(b)==20
            for feature in features:
                x=np.array([r[feature] for r in a]);y=np.array([r[feature] for r in b]);scale=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
                smd=(y.mean()-x.mean())/scale if scale>1e-12 else (0 if x.mean()==y.mean() else None)
                shifts.append(dict(condition=cid,from_pi=n,to_pi=n+1,time=t,hub=h,feature=feature,n=20,
                    mean_before=float(x.mean()),mean_after=float(y.mean()),SMD=None if smd is None else float(smd),
                    absolute_SMD=None if smd is None else float(abs(smd)),Wasserstein=float(wasserstein_distance(x,y)),
                    support_outside_before_range=float(np.mean((y<x.min())|(y>x.max()))),
                    note='descriptive distribution diagnostic; repeated time/hub observations are not independent samples'))
        for n in range(5):
            subset=[r for r in rows if r['condition']==cid and r['iteration']==n]
            coverage.append(dict(condition=cid,iteration=n,rows=len(subset),independent_scenarios=20,
                zero_idle_fraction=float(np.mean([r['idle']==0 for r in subset])),max_idle=max(r['idle'] for r in subset),
                fraction_idle_with_observed_q=float(np.mean([r['idle'] in cfg['target_supply_slots'] or r['idle']==c['fleet'] for r in subset if r['idle']>0])),
                meaning='q support is current-idle occupancy, not a count of provider lookups or chosen solver slots'))
    for r in oscillations:
        if r['time']==24:continue
        fractions=[]
        for n in range(4):
            rr=grouped[r['condition'],r['time'],r['hub'],n]
            frac=sum(x['idle']==r['q'] for x in rr)/len(rr);r[f'occupancy_pi{n}']=frac;fractions.append(frac)
        r['occupancy_frequency']=float(np.mean(fractions))
    write_csv(OUTPUT/'processed/state_distribution_shift.csv',shifts)
    write_csv(OUTPUT/'processed/state_support_coverage.csv',coverage)
    write_csv(OUTPUT/'processed/oscillation_cells.csv',oscillations)
    atomic_json(OUTPUT/'processed/state_diagnostics.json',dict(shifts=shifts,coverage=coverage))
    actions=read_json(OUTPUT/'processed/policy_action_disagreement.json');groups=defaultdict(list)
    for r in actions:groups[r['condition'],r['scenario_seed'],r['iteration']].append(r)
    daily=[]
    for (cid,seed,n),rr in groups.items():
        daily.append(dict(condition=cid,scenario_seed=seed,iteration=n,periods=len(rr),
            **{field:float(np.mean([r[field] for r in rr])) for field in ['full_action_disagree','routes_disagree','assignments_disagree','wait_reposition_disagree','release_hub_disagree']}))
    write_csv(OUTPUT/'processed/policy_action_disagreement_daily.csv',daily)
    action_summary=[]
    for c,n in itertools.product(cfg['conditions'],range(1,5)):
        rr=[r for r in daily if r['condition']==key(c) and r['iteration']==n]
        action_summary.append(dict(condition=key(c),iteration=n,**describe([r['full_action_disagree'] for r in rr],cfg),
            estimand='fraction of pi0-bank SAME-state periods with different full canonical action'))
    write_csv(OUTPUT/'processed/policy_action_disagreement_summary.csv',action_summary)
    atomic_json(OUTPUT/'processed/action_summary.json',action_summary)
    # Formal corroboration has its own independent 40-scenario sample unit.
    formal=read_json(OUTPUT/'processed/formal_policy_action_disagreement.json');fg=defaultdict(list)
    for r in formal:fg[r['condition'],r['scenario_seed'],r['iteration']].append(r)
    fd=[]
    for (cid,seed,n),rr in fg.items():
        assert len(rr)==24
        fd.append(dict(condition=cid,scenario_seed=seed,iteration=n,periods=len(rr),
            **{field:float(np.mean([r[field] for r in rr])) for field in ['full_action_disagree','routes_disagree','assignments_disagree','wait_reposition_disagree','release_hub_disagree']}))
    fs=[]
    for c,n in itertools.product(cfg['conditions'],range(1,5)):
        rr=[r for r in fd if r['condition']==key(c) and r['iteration']==n];assert len(rr)==40
        fs.append(dict(condition=key(c),iteration=n,**describe([r['full_action_disagree'] for r in rr],cfg),
            estimand='fraction of pi0-bank SAME-state periods with different canonical full action; formal corroboration, not tuning'))
    write_csv(OUTPUT/'processed/formal_policy_action_disagreement_daily.csv',fd)
    write_csv(OUTPUT/'processed/formal_policy_action_disagreement_summary.csv',fs)
    atomic_json(OUTPUT/'processed/formal_action_summary.json',fs)
    return shifts,coverage,action_summary

def negative_components(cfg):
    """Observed negative cell prices traced to realized fare/cost and actions."""
    tab=read_json(OUTPUT/'processed/table_diagnostics.json');details=[];summary_rows=[]
    base=read_json(OUTPUT/'manifest.json')['baseline_config']
    alltargets={n:read_json(OUTPUT/'targets'/f'iter{n}'/'targets.json') for n in range(4)}
    grouped=defaultdict(list)
    for n,rows in alltargets.items():
        for r in rows:
            if not r['terminal_anchor']:grouped[n,r['condition'],r['time'],r['hub'],r['q']].append(r)
    for cell in tab['negative']:
        n,cid,t,h,q=cell['iteration'],cell['condition'],cell['time'],cell['hub'],cell['q']
        source=grouped[n,cid,t,h,q];one=[]
        for r in source:
            seed=r['scenario_seed'];prefix=f'{cid}-s{seed}-t{t}-h{h}'
            a=read_compressed(OUTPUT/'raw/worlds'/f'iter{n}'/(prefix+f'-q{q-1}.json.gz'))
            b=read_compressed(OUTPUT/'raw/worlds'/f'iter{n}'/(prefix+f'-q{q}.json.gz'))
            ma,mb=a['metrics'],b['metrics'];d=lambda key:mb[key]-ma[key]
            amounts=dict(fare_increment=d('completed_fare'),realized_cost_increment=d('realized_cost'),
                loaded_cost_increment=d('actual_loaded_distance')*base['loaded_cost'],
                empty_cost_increment=d('actual_empty_distance')*base['empty_cost'],
                waiting_cost_increment=(d('actual_vehicle_waiting')-t)*base['waiting_cost'],
                backlog_cost_increment=d('backlog_penalty'),expiry_cost_increment=d('cancellation_penalty'))
            reconstructed=amounts['fare_increment']-sum(amounts[k] for k in ['loaded_cost_increment','empty_cost_increment',
                'waiting_cost_increment','backlog_cost_increment','expiry_cost_increment'])
            assert abs(reconstructed-r['target'])<1e-6
            firsta=next(row['plan'] for row in a['trace'] if not row.get('drain'))
            firstb=next(row['plan'] for row in b['trace'] if not row.get('drain'))
            from .action_analysis import canonical
            record=dict(condition=cid,iteration=n,scenario_seed=seed,time=t,hub=h,q=q,
                target_USD=r['target'],reconstructed_target_USD=reconstructed,**amounts,
                first_period_system_action_changed=canonical(firsta)!=canonical(firstb),
                first_period_business_gain=firstb['operating_profit']-firsta['operating_profit'],
                continuation_policy_hash=r['continuation_policy_hash'])
            details.append(record);one.append(record)
        means={field:float(np.mean([r[field] for r in one])) for field in ['fare_increment','realized_cost_increment','loaded_cost_increment',
            'empty_cost_increment','backlog_cost_increment','expiry_cost_increment','first_period_business_gain']} if one else {}
        summary_rows.append(cell|dict(observed_counterfactual_samples=len(one),component_means=means,
            negative_sample_fraction=float(np.mean([r['target_USD']<0 for r in one])) if one else None,
            first_action_changed_fraction=float(np.mean([r['first_period_system_action_changed'] for r in one])) if one else None,
            source_explanation='paired fare/cost identity reconstructed; not a unique causal attribution' if one else
                'negative interpolated q price, no directly measured label at this q; do not fabricate component attribution',
            uncertainty_note='construction-cell dispersion conditional on empirical parent policy; no uncertainty for re-estimating whole PI chain'))
    write_csv(OUTPUT/'processed/negative_value_components.csv',details)
    write_csv(OUTPUT/'processed/negative_value_audit.csv',summary_rows)
    atomic_json(OUTPUT/'qa/negative_value_component_audit.json',dict(status='PASSED',negative_cells=len(summary_rows),
        reconstructed_pairs=len(details),profit_identity_tolerance_USD=1e-6,interpolated_cells_without_direct_label=sum(not r['observed_counterfactual_samples'] for r in summary_rows)))

def runtime(cfg):
    rows=[];costs=[];stats=read_json(OUTPUT/'processed/policy_statistics.json')
    for n in range(4):
        m=read_json(OUTPUT/'manifests'/f'iteration_{n}_manifest.json')
        rows.append(dict(stage='offline_evaluation',iteration=n,condition='all',policy=f'pi{n}',
            wall_seconds=m['wall_seconds'],worker_seconds=m['worker_seconds'],parent_policy_seconds=m['parent_policy_worker_seconds'],
            paired_world_simulation_seconds=m['simulation_worker_seconds'],world_serialization_seconds=m['serialization_worker_seconds'],
            aggregation_interpolation_serialization_seconds=m['aggregate_interpolate_serialize_seconds'],worlds=m['worlds'],
            table_aggregation_seconds=m['table_aggregation_seconds'],table_interpolation_seconds=m['table_interpolation_seconds'],
            table_serialization_seconds=m['table_serialization_seconds'],paired_world_QA_worker_seconds=m['paired_world_QA_worker_seconds'],
            target_collection_seconds=m['target_collection_seconds'],target_serialization_seconds=m['target_serialization_seconds'],
            timer_note='worker time includes simulation/replay/serialization; nested timers not additive with worker or wall'))
        for c in cfg['conditions']:
            cid=key(c);items=[read_json(OUTPUT/'targets'/f'iter{n}'/'raw'/f'{cid}-s{s}.json') for s in cfg['construction_seeds']]
            worker=sum(i['seconds'] for i in items)
            gain=next(r for r in stats['comparisons'] if r['condition']==cid and r['comparison']==f'P{n+1}-P{n}')
            costs.append(dict(condition=cid,evaluation_iteration=n,improved_policy=f'pi{n+1}',evaluation_worker_seconds=worker,
                paired_mean_increment_USD=gain['mean'],gain_CI_lower=gain['CI95_lower'],gain_CI_upper=gain['CI95_upper'],
                mean_daily_increment_USD_per_offline_worker_hour=gain['mean']/(worker/3600),
                interpretation='descriptive amortization diagnostic, not cash ROI; negative gains retained; 3-worker wall not allocated per condition'))
    for r in read_json(OUTPUT/'processed/formal_runs.json'):
        rows.append(dict(stage='online_day',iteration=int(r['policy'][1:]) if r['policy'].startswith('P') else None,condition=r['condition'],
            policy=r['policy'],scenario_seed=r['scenario_seed'],wall_seconds=r['daily_wall_seconds'],
            context_preparation_seconds=r['context_preparation_seconds'],slot_lookup_seconds=r['provider_lookup_seconds'],
            release_slot_assembly_seconds=r['release_slot_assembly_seconds'],build_inclusive_value_seconds=r['build_seconds'],
            build_exclusive_lookup_seconds=r['build_exclusive_lookup_seconds'],solver_seconds=r['daily_solver_seconds'],
            state_transition_seconds=r['state_transition_seconds'],cooldown_transition_subset_seconds=r['cooldown_transition_seconds'],
            plan_postprocessing_seconds=r['plan_postprocessing_seconds'],trace_preparation_seconds=r['trace_preparation_seconds'],
            realized_accounting_seconds=r['realized_accounting_seconds'],
            planner_internal_overhead_seconds=max(0,r['planning_wall_seconds']-r['build_seconds']-r['daily_solver_seconds']),
            pipeline_replay_serialization_seconds=r['pipeline_seconds'],
            timer_note='lookup subset of build; release-slot assembly includes lookup; cooldown subset of transition; native M0 planning overhead combines uninstrumented native build/extraction, not pure build'))
    write_csv(OUTPUT/'processed/runtime_breakdown.csv',rows);atomic_json(OUTPUT/'processed/runtime_breakdown.json',rows)
    write_csv(OUTPUT/'processed/offline_evaluation_gain_efficiency.csv',costs);atomic_json(OUTPUT/'processed/offline_evaluation_gain_efficiency.json',costs)

def terminal_estimand_distinction():
    """Historical action-advantage diagnostic only; never used in E or G."""
    from experiments.cfs_rollout_target_alignment.toys import fixture,order,find
    from experiments.cfs_rollout_target_alignment.queries import choose_candidates
    from experiments.cfs_rollout_target_alignment.protocol import world,paired_target,replay_world
    from .protocol import resource_world,pair_audit,continuation
    from .phase0 import toy_policy
    s,orders=fixture([order('short','A','B',20,deadline=1),order('long','A','C',10,deadline=2),
        order('future','B','A',100,book=1,deadline=2)],horizon=2)
    candidates,_,_=choose_candidates(s);candidate=find(candidates,'C')
    assert candidate['release_time']>=s.T
    baseline=world(s,orders);forced=world(s,orders,candidate)
    assert baseline['total']==120 and baseline['current_stage']==20
    assert forced['total']==10 and forced['current_stage']==10
    advantage=paired_target(forced,baseline);assert advantage==-100
    for w in [baseline,forced]:replay_world(s,orders,w)
    # At T there is no new dispatch; only the extra FREE resource differs.
    terminal=s.clone();terminal.t=terminal.T;terminal.pool={}
    a,b=resource_world(terminal,'A',0),resource_world(terminal,'A',1)
    pair_audit(a,b,'A');p0=toy_policy(None,0)
    low,high=continuation(a,orders,p0,dict(rate={})),continuation(b,orders,p0,dict(rate={}))
    marginal=high['metrics']['realized_profit']-low['metrics']['realized_profit'];assert marginal==0
    return dict(status='PASSED',horizon=2,candidate_release_time=candidate['release_time'],
        baseline_J=120,baseline_g=20,candidate_J=10,candidate_g=10,terminal_candidate_Yroll=advantage,
        free_resource_at_T_marginal=marginal,independent_physical_replays=4,
        meaning='Yroll changes the current action and removes a B opportunity; Vres at T adds only free supply after all dispatch decisions. Different estimands.',
        used_in_policy_evaluation=False,used_in_policy_improvement=False)

def preregister_analysis():
    """Analysis plan/source record before diagnostic or formal outcomes exist."""
    from .common import file_hash,now
    path=OUTPUT/'manifests/analysis_preregistration.json'
    if path.exists():return
    assert not (OUTPUT/'processed/diagnostic_runs.json').exists() and not (OUTPUT/'processed/formal_runs.json').exists()
    names=['analysis.py','action_analysis.py','figures.py','delivery.py','pdf_qa.py']
    atomic_json(path,dict(created_at=now(),diagnostic_seen=False,formal_seen=False,
        source_sha256={n:file_hash(OUTPUT/n) for n in names},
        primary_stability_bank='pi0 diagnostic n20; formal pi0 n40 separate corroboration',
        stabilization='frozen main.json joint thresholds and nonincreasing last two steps; descriptive only',
        categorical_interpretation='H-A-compatible requires joint stability and positive P4-P0 Holm<=.05; H-C-compatible requires stability and descriptive preview-gap t95 lower>0, with magnitude separately reported (not automatically large); H-B-compatible deterioration requires a negative adjacent gain with Holm<=.05; alternation counts descriptive, no cycle theorem',
        formal_choices='fixed four improvements, all40 formal seeds, all36 primary comparisons; no stopping, alpha or model selection',
        negative_values='retain all; observed pair cost/fare reconstruction and first action change; interpolated q without direct labels explicitly missing',
        plots='16 specified groups, full source/config; oscillation plot largest range5 per condition, all cells in output',
        extra_terminal_toy='historical J-g versus terminal-free-resource estimand distinction only, never E/G'))

def main():
    cfg=read_json(OUTPUT/'config/main.json')
    deployment_query_provenance(cfg)
    atomic_json(OUTPUT/'qa/terminal_estimand_distinction.json',terminal_estimand_distinction())
    from .action_analysis import run as canonical_action_analysis
    canonical_action_analysis(cfg)
    profits(cfg);_,_,_,osc=table_diagnostics(cfg);state_distributions(cfg,osc);runtime(cfg)
    tab=read_json(OUTPUT/'processed/table_diagnostics.json');tab['oscillations']=osc
    atomic_json(OUTPUT/'processed/table_diagnostics.json',tab)
    negative_components(cfg)
    print('Analysis tables complete.',flush=True)

if __name__=='__main__':main()
