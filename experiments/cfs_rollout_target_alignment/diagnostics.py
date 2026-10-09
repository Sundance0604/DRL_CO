"""Predeclared Phase-1 statistics; scenarios, not candidate rows, are samples."""
import itertools
import numpy as np
from scipy.stats import pearsonr,spearmanr,t as student_t
from experiment_core.storage import atomic_json,read_json
from .common import OUTPUT,write_csv,now

PREREGISTRATION=dict(
    primary_scalar_population='nonbaseline candidate rows; all rows reported separately; baseline is exactly zero by construction',
    candidate_group='condition, scenario_seed, decision time, deciding vehicle (not unrelated vehicles pooled together)',
    sign='three-way sign with USD 1e-6 tie zone; inclusive changes and strict opposite-nonzero changes both reported',
    ranking='strict inversion / pairs strictly ordered under BOTH labels; tie-change rate separately; no tie becomes an invented inversion',
    top1='deterministic smallest candidate hash among values within USD 1e-6 of maximum; tie-set-disjoint also reported',
    duration_gate='pooled Pearson(duration, absolute mismatch)>=0.30 with scenario-cluster bootstrap lower 95% bound >0; association not causality',
    subgroup_gate='>=10 independent scenarios; within-subgroup strict pair inversion >=10%; no row-level p values',
    uncertainty='20 independent scenarios; equal-scenario absolute mismatch means with t95 CI; duration cluster bootstrap 10000',
    missing='keep all predeclared times and all deciding vehicles; missing alternatives never replaced',
    decision_rule='T1 if any predeclared strong mismatch criterion; T0 only if all small-mismatch criteria and no subgroup/duration trigger; otherwise exploratory gray')

def preregister(cfg):
    p=OUTPUT/'manifests/phase1_preregistration.json'
    r=dict(protocol=PREREGISTRATION,gating=cfg['gating'],created_at=now(),outcomes_seen=False)
    if not p.exists():atomic_json(p,r)
    else:assert read_json(p)['protocol']==PREREGISTRATION and read_json(p)['gating']==cfg['gating']

def numeric(value):return float(value) if np.isfinite(value) else None

def correlation(x,y,kind='spearman'):
    if len(x)<3 or np.ptp(x)<1e-10 or np.ptp(y)<1e-10:return None
    return numeric((spearmanr if kind=='spearman' else pearsonr)(x,y).statistic)

def sign(x,eps):return np.where(x>eps,1,np.where(x<-eps,-1,0))

def ranking(rows,xfield='Y_freeze',yfield='Y_roll',eps=1e-6):
    groups={}
    for r in rows:groups.setdefault((r['condition'],r['scenario_seed'],r['time'],r['vehicle_id']),[]).append(r)
    output=[]
    for group,items in groups.items():
        if len(items)<2:continue
        comparable=flips=tiechanges=allpairs=0
        for a,b in itertools.combinations(items,2):
            dx=a[xfield]-b[xfield];dy=a[yfield]-b[yfield];allpairs+=1
            sx=int(sign(np.array([dx]),eps)[0]);sy=int(sign(np.array([dy]),eps)[0])
            if sx and sy:comparable+=1;flips+=int(sx!=sy)
            tiechanges+=int((sx==0)!=(sy==0))
        def top(field):
            best=max(a[field] for a in items)
            return {a['candidate_hash'] for a in items if abs(a[field]-best)<=eps}
        topx,topy=top(xfield),top(yfield)
        output.append(dict(condition=group[0],scenario_seed=group[1],time=group[2],vehicle_id=group[3],
            candidates=len(items),all_pairs=allpairs,comparable_pairs=comparable,ranking_flips=flips,
            pair_flip_rate=flips/comparable if comparable else None,pair_flip_all_pairs=flips/allpairs,
            tie_change_pairs=tiechanges,tie_change_rate=tiechanges/allpairs,
            top1_disagreement=int(min(topx)!=min(topy)),top1_disjoint=int(not topx&topy),
            top1_freeze_ties=len(topx),top1_roll_ties=len(topy)))
    return output

def aggregate(rows,eps=1e-6):
    if not rows:return dict(rows=0,scenarios=0)
    a=np.array([r['Y_freeze'] for r in rows]);b=np.array([r['Y_roll'] for r in rows]);d=b-a
    ss,st=sign(a,eps),sign(b,eps)
    ranks=ranking(rows,eps=eps);comp=sum(r['comparable_pairs'] for r in ranks);flips=sum(r['ranking_flips'] for r in ranks)
    return dict(rows=len(rows),scenarios=len({r['scenario_seed'] for r in rows}),
        mean_absolute_difference=float(np.abs(d).mean()),RMSE=float(np.sqrt(np.mean(d*d))),
        mean_difference=float(d.mean()),Pearson=correlation(a,b,'pearson'),Spearman=correlation(a,b),
        sign_flip=float(np.mean(ss!=st)),strict_sign_opposition=float(np.mean(ss*st==-1)),
        freeze_zero_fraction=float(np.mean(ss==0)),roll_zero_fraction=float(np.mean(st==0)),
        pair_flip_rate=flips/comp if comp else None,ranking_flip_count=flips,comparable_pairs=comp,
        candidate_groups=len(ranks),top1_disagreement=float(np.mean([r['top1_disagreement'] for r in ranks])) if ranks else None,
        top1_disjoint=float(np.mean([r['top1_disjoint'] for r in ranks])) if ranks else None)

def bucket(x,bounds):
    for i,b in enumerate(bounds):
        if x<=b:return ('<= '+str(b)) if not i else f'({bounds[i-1]}, {b}]'
    return '> '+str(bounds[-1])

def group_label(r,field):
    c=r['context'];f=r['metadata_features']
    values=dict(time=str(r['time']),remaining_horizon=str(r['remaining_horizon']),
        route_duration=bucket(c['route_duration'],[1,4,8,12]),release_hub=r['release_hub'],
        same_cross=r['route_type'],wait_active=c['kind'],deadhead=bucket(c['deadhead_distance'],[0,100,300,600]),
        compatible_load=bucket(f['compatible_load'],[0,.5,1,2]),expiring_load=bucket(f['expiring_load'],[0,.5,1]),
        fleet_tightness=bucket(r['fleet_tightness'],[.25,.5,.75]),backlog=bucket(r['current_backlog_load'],[0,1,3,6]),
        release_current_supply=bucket(r['release_hub_current_supply'],[0,1,2]))
    return values[field]

GROUP_FIELDS=['time','remaining_horizon','route_duration','release_hub','same_cross','wait_active',
    'deadhead','compatible_load','expiring_load','fleet_tightness','backlog','release_current_supply']

def duration_association(rows,cfg):
    seeds=sorted({r['scenario_seed'] for r in rows});moments=[]
    for seed in seeds:
        rr=[r for r in rows if r['scenario_seed']==seed]
        x=np.array([r['context']['route_duration'] for r in rr],float)
        y=np.array([abs(r['Y_roll']-r['Y_freeze']) for r in rr],float)
        moments.append([len(x),x.sum(),y.sum(),x@x,y@y,x@y])
    def rho(m):
        n,sx,sy,xx,yy,xy=m;vx=xx-sx*sx/n;vy=yy-sy*sy/n
        return (xy-sx*sy/n)/np.sqrt(vx*vy) if vx>0 and vy>0 else np.nan
    m=np.array(moments);rng=np.random.default_rng(cfg['bootstrap_seed'])
    draws=rng.integers(0,len(seeds),(cfg['bootstrap_replications'],len(seeds)))
    boot=np.array([rho(a) for a in m[draws].sum(axis=1)]);boot=boot[np.isfinite(boot)]
    x=[r['context']['route_duration'] for r in rows];y=[abs(r['Y_roll']-r['Y_freeze']) for r in rows]
    return dict(Pearson=numeric(rho(m.sum(axis=0))),Spearman=correlation(x,y),
        cluster_bootstrap_lower=float(np.quantile(boot,.025)) if len(boot) else None,
        cluster_bootstrap_upper=float(np.quantile(boot,.975)) if len(boot) else None,
        independent_scenarios=len(seeds),replications=len(boot),interpretation='association; time/network/fleet confounding not controlled')

def analyze_phase1(cfg):
    rows=read_json(OUTPUT/'targets/phase1.json');eps=cfg['tie_tolerance_usd']
    nonbase=[r for r in rows if not r['baseline']]
    primary=aggregate(nonbase,eps);allrows=aggregate(rows,eps)
    ranks=ranking(rows,eps=eps)
    # Gate scalar differences on nonbaseline rows, but ranking includes the zero baseline action.
    comparison=aggregate(rows,eps)
    for k in ['pair_flip_rate','ranking_flip_count','comparable_pairs','candidate_groups','top1_disagreement','top1_disjoint']:
        primary[k]=comparison[k]
    grouped=[]
    for field in GROUP_FIELDS:
        labels=sorted({group_label(r,field) for r in nonbase})
        for label in labels:
            rr=[r for r in nonbase if group_label(r,field)==label]
            grouped.append(dict(dimension=field,group=label,**aggregate(rr,eps)))
    scenario=[]
    for seed in cfg['phase1_seeds']:
        rr=[r for r in nonbase if r['scenario_seed']==seed]
        if rr:scenario.append(dict(scenario_seed=seed,**aggregate(rr,eps)))
    x=np.array([r['mean_absolute_difference'] for r in scenario]);se=x.std(ddof=1)/np.sqrt(len(x))
    ci=float(student_t.ppf(.975,len(x)-1)*se)
    clustered=dict(mean=float(x.mean()),SD=float(x.std(ddof=1)),CI95_lower=float(x.mean()-ci),CI95_upper=float(x.mean()+ci),n=len(x))
    duration=duration_association(nonbase,cfg);g=cfg['gating'];triggers=[]
    tests=[('Spearman below 0.90',primary['Spearman'] is not None and primary['Spearman']<g['T1_spearman_below']),
        ('strict ranking inversion >=10%',primary['pair_flip_rate'] is not None and primary['pair_flip_rate']>=g['T1_pair_flip_min']),
        ('canonical top1 disagreement >=10%',primary['top1_disagreement'] is not None and primary['top1_disagreement']>=g['T1_top1_min']),
        ('inclusive sign change >=10%',primary['sign_flip']>=g['T1_sign_flip_min']),
        ('duration association',duration['Pearson'] is not None and duration['Pearson']>=g['duration_association_min'] and
            duration['cluster_bootstrap_lower'] is not None and duration['cluster_bootstrap_lower']>0)]
    for name,hit in tests:
        if hit:triggers.append(name)
    regimes=[r for r in grouped if r['dimension'] in ['route_duration','same_cross','release_hub','time'] and
        r['scenarios']>=g['subgroup_min_scenarios'] and r.get('pair_flip_rate') is not None and
        r['pair_flip_rate']>=g['subgroup_strict_inversion_min']]
    if regimes:triggers.append('predeclared regime strict inversion')
    t0=all([primary['Spearman'] is not None and primary['Spearman']>=g['T0_spearman_min'],
        primary['pair_flip_rate'] is not None and primary['pair_flip_rate']<=g['T0_pair_flip_max'],
        primary['top1_disagreement'] is not None and primary['top1_disagreement']<=g['T0_top1_max'],
        primary['sign_flip']<=g['T0_sign_flip_max'],not regimes,not triggers])
    gate='T1' if triggers else 'T0' if t0 else 'GRAY_EXPLORATORY'
    result=dict(status='COMPLETED',gate=gate,triggers=triggers,primary=primary,including_baseline=allrows,
        scenario_equal_weighted_absolute_difference=clustered,duration=duration,
        primary_estimand='strict matched J-g advantage; NOT fourth release resource marginal',
        phase2_required=gate!='T0',predeclared_rule_hash=__import__('experiment_core.storage',fromlist=['digest']).digest(PREREGISTRATION))
    write_csv(OUTPUT/'processed/candidate_ranking_metrics.csv',ranks)
    write_csv(OUTPUT/'processed/target_mismatch_by_group.csv',grouped)
    write_csv(OUTPUT/'processed/phase1_scenario_statistics.csv',scenario)
    atomic_json(OUTPUT/'processed/phase1_summary.json',result)
    atomic_json(OUTPUT/'manifests/phase1_gating_decision.json',result)
    return result
