"""Predeclared diagnostic selection and scenario-level formal inference."""
from collections import defaultdict
import itertools
import numpy as np
from scipy.stats import wasserstein_distance
from experiment_core.storage import atomic_json,read_json,digest
from experiments.cfs_resource_policy_iteration.analysis import describe,holm,oscillation_metrics
from experiments.cfs_resource_policy_iteration.protocol import read_compressed,restore
from experiments.cfs_classical_value.simulation import summary
from model_families.single.rolling.audit import incoming_supply
from .common import OUTPUT,HISTORY,ALPHAS,key,write_csv,now,file_hash

def table_model(cid,n,alpha,deployed=True):
    if alpha=='undamped':return read_json(HISTORY/'tables'/f'V{n}'/(cid+'.json'))
    return read_json(OUTPUT/'tables'/f'alpha_{alpha}'/f'{"deployed" if deployed else "raw"}_V{n}'/(cid+'.json'))

def table_metrics(cfg):
    rows=[];osc=[]
    for c in cfg['conditions']:
        cid=key(c)
        for alpha in ['undamped']+list(ALPHAS):
            models=[table_model(cid,n,alpha) for n in range(4)]
            for kind in ['deployed','raw'] if alpha!='undamped' else ['deployed']:
                mm=models if kind=='deployed' else [table_model(cid,n,alpha,False) for n in range(4)]
                for n in range(1,4):
                    aa=[];bb=[]
                    for h,qs in mm[n]['values'].items():
                        for q,rec in qs.items():
                            if int(q)>c['fleet']:continue
                            for i,t in enumerate(rec['times']):
                                if t>=24:continue
                                aa.append(mm[n-1]['values'][h][q]['values'][i]);bb.append(rec['values'][i])
                    a,b=np.asarray(aa),np.asarray(bb);d=b-a
                    rows.append(dict(condition=cid,alpha=alpha,kind=kind,iteration=n+1,cells=len(a),
                        RMSE_USD=float(np.sqrt(np.mean(d*d))),relative_L2=float(np.linalg.norm(d)/(np.linalg.norm(a)+1e-12)),
                        sign_flip_rate=float(np.mean(np.sign(a)!=np.sign(b))),
                        strict_sign_flip_rate=float(np.mean(a*b<0))))
            raw=[table_model(cid,n,alpha,False) for n in range(4)]
            for h,qs in models[0]['values'].items():
                for q,rec in qs.items():
                    if int(q)>c['fleet']:continue
                    for i,t in enumerate(rec['times']):
                        if t>=24:continue
                        vals=[m['values'][h][q]['values'][i] for m in models]
                        rawvals=[m['values'][h][q]['values'][i] for m in raw]
                        metrics=oscillation_metrics(vals)
                        osc.append(dict(condition=cid,alpha=alpha,time=t,hub=h,q=int(q),
                            **{f'raw_V{n}':v for n,v in enumerate(rawvals)},
                            **{f'damped_V{n}':v for n,v in enumerate(vals)},
                            **metrics,direction_reversal_count=metrics['update_direction_reversal_count'],
                            raw_range_USD=float(max(rawvals)-min(rawvals)),
                            occupancy_frequency=None))
    write_csv(OUTPUT/'processed/table_change_metrics.csv',rows)
    atomic_json(OUTPUT/'processed/table_metrics.json',rows)
    atomic_json(OUTPUT/'processed/oscillations.json',osc)
    return rows

def action_summary(cfg,stage):
    rows=read_json(OUTPUT/'processed'/f'{stage}_action_disagreement.json');groups=defaultdict(list)
    for r in rows:groups[r['condition'],r['alpha'],r['iteration'],r['kind'],r['scenario_seed']].append(r)
    daily=[]
    fields=['full_action_disagree','routes_disagree','assignments_disagree','wait_reposition_disagree','release_hub_disagree']
    for (cid,a,n,kind,seed),rr in groups.items():
        assert len(rr)==24
        daily.append(dict(condition=cid,alpha=a,iteration=n,kind=kind,scenario_seed=seed,
            **{f:float(np.mean([r[f] for r in rr])) for f in fields}))
    grouped=defaultdict(list)
    for r in daily:grouped[r['condition'],r['alpha'],r['iteration'],r['kind']].append(r)
    out=[]
    for (cid,a,n,kind),rr in grouped.items():
        out.append(dict(condition=cid,alpha=a,iteration=n,kind=kind,**describe([r['full_action_disagree'] for r in rr],cfg),
            **{f+'_mean':float(np.mean([r[f] for r in rr])) for f in fields[1:]}))
    write_csv(OUTPUT/'processed'/f'{stage}_action_daily.csv',daily)
    write_csv(OUTPUT/'processed'/f'{stage}_action_summary.csv',out)
    atomic_json(OUTPUT/'processed'/f'{stage}_action_summary.json',out)
    return out

def lookup_runs(stage):
    rows=read_json(OUTPUT/'processed'/f'{stage}_runs.json')
    lookup={(r['condition'],r['scenario_seed'],r['policy'],r['alpha']):r for r in rows}
    assert len(lookup)==len(rows)
    return rows,lookup

def run_for(lookup,cid,seed,label,alpha):
    return lookup[cid,seed,label,None if label.startswith('U') else alpha]

def select_alpha(cfg):
    frozen=OUTPUT/'manifests/alpha_freeze.json'
    if frozen.exists():
        result=read_json(frozen)
        assert result['selection_config_hash']==digest(cfg['selection'])
        assert file_hash(OUTPUT/'processed/diagnostic_runs.json')==result['diagnostic_results_hash']
        return result['selected_alpha']
    assert not (OUTPUT/'processed/formal_runs.json').exists() and not (OUTPUT/'processed/pilot_runs.json').exists()
    tables=table_metrics(cfg);actions=action_summary(cfg,'diagnostic')
    _,lookup=lookup_runs('diagnostic');seeds=cfg['diagnostic_seeds'];records=[]
    primary=[key(c) for c in cfg['conditions'] if c['region']=='california_corridor' or c['orders']==40]
    keyregimes=[key(c) for c in cfg['conditions'] if c['orders']!=20]
    for alpha in ALPHAS:
        summaries=[];table_bad=[];action_bad=[];collapse=[]
        for cid in keyregimes:
            one=[run_for(lookup,cid,s,'D1',alpha)['realized_profit'] for s in seeds]
            four=[run_for(lookup,cid,s,'D4',alpha)['realized_profit'] for s in seeds]
            delta=describe(np.array(four)-np.array(one),cfg)
            collapse.append(delta['mean']<-.05*abs(np.mean(one)) and delta['CI95_upper']<0)
            summaries.append(dict(condition=cid,alpha=alpha,D4_D1_mean=delta['mean'],D4_mean=float(np.mean(four))))
        aa=[];tt=[];pp=[];scale=[]
        for cid in primary:
            tr=[next(r for r in tables if r['condition']==cid and r['alpha']==alpha and r['kind']=='deployed' and r['iteration']==n)['relative_L2'] for n in range(2,5)]
            ac=[next(r for r in actions if r['condition']==cid and r['alpha']==alpha and r['kind']=='adjacent' and r['iteration']==n)['mean'] for n in range(1,5)]
            table_bad.append(tr[-1]>1.10*tr[0]+1e-12);action_bad.append(ac[-1]>ac[1]+.02)
            aa.append(np.mean(ac[-2:]));tt.append(np.mean(tr[-2:]))
            pp.append(np.mean([run_for(lookup,cid,s,'D4',alpha)['realized_profit'] for s in seeds]))
            scale.append(abs(np.mean([run_for(lookup,cid,s,'U0',alpha)['realized_profit'] for s in seeds])))
        records.append(dict(alpha=alpha,alpha_value=ALPHAS[alpha],collapse=any(collapse),
            table_increase=any(table_bad),action_increase=any(action_bad),
            last_two_action=float(np.mean(aa)),last_two_relative_L2=float(np.mean(tt)),
            diagnostic_mean_profit=float(np.mean(pp)),profit_scale=float(np.mean(scale)),
            failed_screens=int(any(collapse))+int(any(table_bad))+int(any(action_bad)),conditions=summaries))
    eligible=[r for r in records if r['failed_screens']==0]
    exploratory=not eligible
    if not eligible:eligible=[r for r in records if r['failed_screens']==min(x['failed_screens'] for x in records)]
    best=min(r['last_two_action'] for r in eligible)
    eligible=[r for r in eligible if r['last_two_action']<=best+cfg['selection']['primary_action_tie']]
    best=min(r['last_two_relative_L2'] for r in eligible)
    eligible=[r for r in eligible if r['last_two_relative_L2']<=best+cfg['selection']['secondary_table_tie']]
    best=max(r['diagnostic_mean_profit'] for r in eligible)
    eligible=[r for r in eligible if r['diagnostic_mean_profit']>=best-cfg['selection']['profit_tie_fraction']*r['profit_scale']]
    selected=min(eligible,key=lambda r:cfg['selection']['tie_preference'].index(r['alpha']))['alpha']
    write_csv(OUTPUT/'processed/alpha_selection.csv',records)
    result=dict(status='FROZEN',selected_alpha=selected,alpha_value=ALPHAS[selected],
        exploratory_selection=exploratory,records=records,created_at=now(),formal_seen=False,pilot_seen=False,
        selection_config_hash=digest(cfg['selection']),
        diagnostic_results_hash=file_hash(OUTPUT/'processed/diagnostic_runs.json'),
        diagnostic_action_hash=file_hash(OUTPUT/'processed/diagnostic_action_disagreement.json'),
        selection_code_hash=file_hash(OUTPUT/'analysis.py'),rule=cfg['selection'])
    atomic_json(frozen,result)
    print(f'GLOBAL ALPHA FROZEN: {ALPHAS[selected]}, exploratory={exploratory}; formal unseen.',flush=True)
    return selected

def profits(cfg,stage,alpha):
    runs,lookup=lookup_runs(stage);seeds=cfg[f'{stage}_seeds'];out=[];paired=[];differences=[]
    labels=[f'U{n}' for n in range(5)]+[f'D{n}' for n in range(1,5)]
    for c in cfg['conditions']:
        cid=key(c)
        for seed in seeds:
            assert len({run_for(lookup,cid,seed,l,alpha)['scenario_hash'] for l in labels})==1
            assert run_for(lookup,cid,seed,'U1',alpha)['realized_profit']==run_for(lookup,cid,seed,'D1',alpha)['realized_profit']
        for label in labels:
            rr=[run_for(lookup,cid,s,label,alpha) for s in seeds]
            out.append(dict(stage=stage,condition=cid,alpha=alpha,policy=label,
                **describe([r['realized_profit'] for r in rr],cfg),
                mean_online_seconds=float(np.mean([r['daily_wall_seconds'] for r in rr])),
                mean_service=float(np.mean([r['delivered']/c['orders'] for r in rr]))))
        for a,b in cfg['comparisons']:
            x=[run_for(lookup,cid,s,a,alpha)['realized_profit']-run_for(lookup,cid,s,b,alpha)['realized_profit'] for s in seeds]
            family='primary_4' if [a,b]==['D4','U4'] else 'exploratory_8' if b.startswith('U') and a!='U4' else 'supporting_20'
            paired.append(dict(stage=stage,condition=cid,alpha=alpha,comparison=f'{a}-{b}',policy_a=a,policy_b=b,
                family=family,reference_mean=float(np.mean([run_for(lookup,cid,s,b,alpha)['realized_profit'] for s in seeds])),**describe(x,cfg)))
            differences.extend(dict(stage=stage,condition=cid,alpha=alpha,scenario_seed=s,comparison=f'{a}-{b}',difference_USD=v) for s,v in zip(seeds,x))
    for family in ['primary_4','supporting_20','exploratory_8']:holm([r for r in paired if r['family']==family])
    write_csv(OUTPUT/'processed'/f'{stage}_policy_summary.csv',out)
    write_csv(OUTPUT/'processed'/f'{stage}_paired_policy_differences.csv',paired)
    write_csv(OUTPUT/'processed'/f'{stage}_paired_scenario_differences.csv',differences)
    atomic_json(OUTPUT/'processed'/f'{stage}_statistics.json',dict(summary=out,paired=paired))
    return out,paired

def state_analysis(cfg,stage,alphas):
    runs,_=lookup_runs(stage);observations=[]
    for r in runs:
        spec=read_json(OUTPUT/'raw/runs'/r['id']/'spec.json')
        packets=read_compressed(OUTPUT/r['trace_ref'])['state_packets']
        path='undamped' if r['policy'].startswith('U') else r['alpha']
        n=int(r['policy'][1:])
        for t,packet in packets.items():
            s,_=restore(spec['scenario'],spec['parameters'],spec['solver'],packet);online=summary(s,spec['prior'])
            # Postprocessing only: do not change the frozen numerical operator.
            # Distinguish actual locked idle-release ETA from the historical
            # summary's last-delivery timestamps; retain the latter for audit.
            releases=[e for e in incoming_supply(s) if e['idle_release']]
            for h in sorted(s.net.H):
                incoming=[a for a in releases if a['hub']==h]
                legacy=[a for a in online['incoming'] if a['hub']==h]
                observations.append(dict(stage=stage,condition=r['condition'],alpha=path,iteration=n,
                    scenario_seed=r['scenario_seed'],time=int(t),hub=h,idle=float(online['idle'][h]),
                    incoming=float(len(incoming)),backlog=float(online['backlog'][h]),
                    committed=float(sum(v.phase!='idle' for v in s.vehs)),
                    utilization=float(sum(v.phase!='idle' for v in s.vehs)/len(s.vehs)),
                    release_delay_mean=float(np.mean([a['eta']-s.t for a in incoming])) if incoming else 0.,
                    legacy_incoming_timestamp_offset_mean=float(np.mean([a['time']-s.t for a in legacy])) if legacy else 0.,
                    release_source='readonly rolling.audit final idle_release; postprocessing only'))
    grouped=defaultdict(list)
    for r in observations:grouped[r['condition'],r['alpha'],r['iteration'],r['time'],r['hub']].append(r)
    shifts=[]
    for c in cfg['conditions']:
        cid=key(c);hubs=sorted({r['hub'] for r in observations if r['condition']==cid})
        for a,n,t,h in itertools.product(['undamped']+alphas,range(1,5),range(24),hubs):
            before=grouped[cid,'undamped' if a!='undamped' and n==1 else a,n-1,t,h]
            after=grouped[cid,a,n,t,h]
            assert len(before)==len(after)==len(cfg[f'{stage}_seeds'])
            for feature in ['idle','incoming','backlog','committed','utilization','release_delay_mean']:
                x=np.array([r[feature] for r in before]);y=np.array([r[feature] for r in after])
                scale=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
                smd=float((y.mean()-x.mean())/scale) if scale>1e-12 else 0. if x.mean()==y.mean() else None
                shifts.append(dict(stage=stage,condition=cid,alpha=a,iteration=n,time=t,hub=h,feature=feature,
                    n=len(x),SMD=smd,absolute_SMD=None if smd is None else abs(smd),
                    Wasserstein=float(wasserstein_distance(x,y)),
                    support_outside_before_range=float(np.mean((y<x.min())|(y>x.max())))))
    write_csv(OUTPUT/'processed'/f'{stage}_state_observations.csv',observations)
    write_csv(OUTPUT/'processed'/f'{stage}_state_distribution_shift.csv',shifts)
    atomic_json(OUTPUT/'processed'/f'{stage}_state_shifts.json',shifts)
    if stage=='diagnostic':
        osc=read_json(OUTPUT/'processed/oscillations.json')
        for r in osc:
            frequencies=[]
            for n in range(4):
                bank=grouped[r['condition'],'undamped' if n==0 else r['alpha'],n,r['time'],r['hub']]
                frequencies.append(float(np.mean([x['idle']==r['q'] for x in bank])))
            r['occupancy_frequency']=float(np.mean(frequencies))
        write_csv(OUTPUT/'processed/oscillation_cells.csv',osc)
        atomic_json(OUTPUT/'processed/oscillations.json',osc)
    return shifts

def finalize_analysis(cfg):
    selected=read_json(OUTPUT/'manifests/alpha_freeze.json')['selected_alpha']
    table_metrics(cfg);action_summary(cfg,'diagnostic');action_summary(cfg,'formal')
    for stage in ['diagnostic','formal']:
        profits(cfg,stage,selected);state_analysis(cfg,stage,list(ALPHAS) if stage=='diagnostic' else [selected])
    runtime=[]
    for stage in ['diagnostic','pilot','formal']:
        for r in read_json(OUTPUT/'processed'/f'{stage}_runs.json'):
            runtime.append(dict(stage=stage,condition=r['condition'],alpha=r['alpha'] or 'undamped',policy=r['policy'],
                scenario_seed=r['scenario_seed'],online_seconds=r['daily_wall_seconds'],pipeline_seconds=r['pipeline_seconds'],
                solver_seconds=r['daily_solver_seconds'],lookup_seconds=r['provider_lookup_seconds'],
                build_seconds=r['build_seconds'],context_seconds=r['context_preparation_seconds'],
                transition_seconds=r['state_transition_seconds'],ledger_seconds=r['realized_accounting_seconds'],alias_of=r['alias_of']))
    for a in ALPHAS:
        for n in [2,3]:
            r=read_json(OUTPUT/'manifests'/f'alpha_{a}_raw_V{n}.json')
            runtime.append(dict(stage='offline',condition='all',alpha=a,policy=f'D{n}',iteration=n,
                wall_seconds=r['wall_seconds'],worker_seconds=r['worker_seconds'],worlds=r['worlds'],
                simulation_seconds=r['simulation_worker_seconds'],parent_seconds=r['parent_worker_seconds'],
                serialization_seconds=r['serialization_worker_seconds'],QA_seconds=r['QA_worker_seconds']))
    write_csv(OUTPUT/'processed/runtime.csv',runtime)
    atomic_json(OUTPUT/'processed/runtime.json',runtime)
    # Stable task-document names, alongside explicit diagnostic/formal names.
    write_csv(OUTPUT/'processed/paired_policy_differences.csv',read_json(OUTPUT/'processed/formal_statistics.json')['paired'])
    write_csv(OUTPUT/'processed/action_disagreement.csv',read_json(OUTPUT/'processed/formal_action_disagreement.json'))
    write_csv(OUTPUT/'processed/state_distribution_shift.csv',read_json(OUTPUT/'processed/formal_state_shifts.json'))
    return selected
