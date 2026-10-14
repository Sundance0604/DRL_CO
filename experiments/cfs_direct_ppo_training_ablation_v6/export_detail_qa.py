"""Independent remaining five CSV-to-raw checks, no scientific-output rewrite."""
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v6'
PARENT=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_no_mc_v3'

@lru_cache(maxsize=256)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(stem):
    with (TARGET/(stem+'.csv')).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def check(value,expected):
    if expected is None:
        if value!='':raise RuntimeError('missing CSV value became nonmissing')
    elif isinstance(expected,(list,dict)):
        if json.loads(value)!=expected:raise RuntimeError('structured CSV source mismatch')
    elif isinstance(expected,bool):
        if value!=str(expected):raise RuntimeError('CSV bool mismatch')
    elif isinstance(expected,(float,int,np.number)):
        if not np.isclose(float(value),expected,atol=1e-8,rtol=1e-12):raise RuntimeError('CSV numeric source mismatch')
    elif value!=expected:raise RuntimeError('CSV text source mismatch')
def folder(r):
    return PARENT/'training'/r['condition']/r['training_seed'] if r['variant']=='A' else TARGET/'training'/r['condition']/r['variant']/r['training_seed']


@lru_cache(maxsize=64)
def prediction_arrays(base,u,divisor):
    predicted=[];cash=[];targets=[];periods=[]
    for i in range(4*(u-1)+1,4*u+1):
        record=read(str(Path(base)/f'training-episode-{i:06d}.json'))
        rtg=np.cumsum([t['reward_USD'] for t in record['transitions']][::-1])[::-1]
        for t,row in enumerate(record['learning']):
            predicted.append(row['value']*divisor);cash.append(rtg[t]);targets.append(row['return']*divisor);periods.append(t)
    return tuple(np.asarray(x) for x in (predicted,cash,targets,periods))


def run():
    if read(str(TARGET/'manifest.json'))['status']!='NUMERICAL_COMPLETE_QA_PENDING':raise RuntimeError('numerical completion required')
    counts={}
    data=rows('critic_diagnostics');counts['critic_diagnostics']=len(data)
    for r in data:
        if r['variant']=='C':
            if r['target']!='NA' or any(r[k]!='' for k in ('EV','RMSE_USD','MAE_USD')):raise RuntimeError('C missing Critic became zero')
            continue
        p,rtg,gae,t=prediction_arrays(str(folder(r)),int(r['update']),10000 if r['variant']=='B' else 1000)
        mask=np.ones(t.shape,dtype=bool) if r['period']=='-1' else t==int(r['period'])
        p=p[mask];y=(rtg if r['target']=='MC_RTG' else gae)[mask]
        expected=dict(states=len(p),predicted_mean_USD=p.mean(),target_mean_USD=y.mean(),RMSE_USD=np.sqrt(np.mean((p-y)**2)),
            MAE_USD=np.abs(p-y).mean(),EV=1-np.var(y-p)/np.var(y) if np.var(y)>0 else None)
        for k,value in expected.items():check(r[k],value)
    data=rows('gradient_diagnostics');counts['gradient_diagnostics']=len(data)
    for r in data:
        log=read(str(folder(r)/f'update-{int(r["update"]):04d}.json'));d=log.get('detailed_gradient') or {};m=log.get('minibatch_records',[])
        for column,key in (('shared_actor_norm','shared_actor_norm'),('shared_weighted_critic_norm','shared_critic_norm'),
            ('shared_cosine','shared_cosine'),('shared_parameters','shared_parameters')):check(r[column],d.get(key))
        check(r['actor_preclip_norm_mean'],np.mean([x['preclip_norms'][0] for x in m]) if m else None)
        check(r['actor_clip_coefficient_mean'],np.mean([x['clip_coefficients'][0] for x in m]) if m else None)
        check(r['critic_preclip_norm_mean'],np.mean([x['preclip_norms'][1] for x in m]) if r['variant']=='D' else None)
    data=rows('actor_state_sensitivity');counts['actor_state_sensitivity']=len(data)
    a=read(str(TARGET/'protocol/A-prefix-policies.json'))['rows'];f=read(str(TARGET/'protocol/frozen-policies.json'))['rows']
    for r in data:
        u=int(r['update']);base=TARGET/'analysis/A-sensitivity'/f'{r["condition"]}-{r["training_seed"]}-{u:04d}.json' if r['variant']=='A' else folder(r)/f'sensitivity-{u:04d}.json'
        s=read(str(base));point=next(x for x in s['points'] if x['period']==int(r['period']) and x['scenario_seed']==int(r['scenario_seed']))
        for k in ('period','scenario_seed','pending','observation_hash','action','price_USD'):check(r[k],point[k])
        for k in ('D_pi','log_std'):check(r[k],s[k])
        choice=next(x for x in (a if r['variant']=='A' else f) if x['condition']==r['condition'] and x['seed']==int(r['training_seed']) and (r['variant']=='A' or x['variant']==r['variant']))
        check(r['selected'],choice['selected_update']==u)
    data=rows('action_probe_results');counts['action_probe_results']=len(data)
    for r in data:
        raw=read(str(TARGET/r['source_file']));entry=next(x for x in raw['rows'] if x['control']==r['control'])
        check(r['signal'],entry['signal']);check(r['probe_cash_USD'],entry['probe']['plan']['operating_profit'])
        check(r['canonical_changed_from_deterministic'],entry['canonical']!=raw['rows'][1]['canonical'])
    data=rows('business_metrics');counts['business_metrics']=len(data)
    for r in data:
        tail=f'{r["method"]}-{r["training_seed"]}' if r['training_seed'] else r['method']
        raw=read(str(TARGET/'evaluation'/f'{r["condition"]}-{tail}-s{r["scenario_seed"]}.json'))
        ledger=raw['ledger'];metrics=raw['metrics'];cfg=read(str(TARGET/'protocol'/(r['condition']+'.json')))['parameters']
        for col,key in (('profit_USD','realized_profit'),('revenue_USD','completed_fare'),('cost_USD','realized_cost'),
            ('completed_orders','completed_orders'),('backlog_penalty_USD','backlog_penalty'),('cancellation_penalty_USD','cancellation_penalty')):check(r[col],ledger[key])
        loaded=ledger['actual_loaded_distance']*cfg['loaded_cost'];empty=ledger['actual_empty_distance']*cfg['empty_cost'];waiting=ledger['actual_vehicle_waiting']*cfg['waiting_cost']
        for k,v in dict(loaded_cost_USD=loaded,empty_cost_USD=empty,waiting_cost_USD=waiting,solver_calls=metrics['solver_calls'],
            solver_seconds=metrics['solver_seconds'],policy_seconds=raw.get('policy_seconds')).items():check(r[k],v)
        if not np.isclose(loaded+empty+waiting+ledger['backlog_penalty']+ledger['cancellation_penalty'],ledger['realized_cost'],atol=1e-6,rtol=0):raise RuntimeError('independent cost decomposition')
        events=metrics['execution_events'];duration=sum(s['end']-s['start'] for v in events for s in v['segments'] if 'end' in s and 'start' in s)
        check(r['vehicle_utilization_travel_fraction_full_cooldown'],duration/(len(events)*ledger['cutoff_exclusive']))
    if counts!=dict(critic_diagnostics=21744,gradient_diagnostics=576,actor_state_sensitivity=864,action_probe_results=432,business_metrics=1008):raise RuntimeError('detailed export denominators')
    out=dict(status='PASSED',counts=counts,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        CSV_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in TARGET.glob('*.csv')},
        unchanged_statistical_family=True,no_source_records_modified=True)
    (TARGET/'qa/export-details.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('Detailed raw export checks PASSED',counts)


if __name__=='__main__':run()
