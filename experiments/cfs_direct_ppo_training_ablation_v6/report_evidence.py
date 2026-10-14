"""Non-numerical report summaries; no training, selection or new inference family."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v6'
PARENT=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_no_mc_v3'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(stem):
    with (TARGET/(stem+'.csv')).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def mean(values):return float(np.mean(values)) if values else None
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    critics=rows('critic_diagnostics');gradients=rows('gradient_diagnostics');actors=rows('actor_state_sensitivity')
    business=rows('business_metrics');probes=rows('action_probe_results');training=rows('training_diagnostics')
    protocol=read(TARGET/'protocol/preregistration.json');frozen=read(TARGET/'protocol/frozen-policies.json')
    out=dict(status='PASSED',scope='descriptive report evidence, unchanged official statistical family',
        source_sha256=digest(Path(__file__)),CSV_sha256={p.name:digest(p) for p in TARGET.glob('*.csv')},
        validation_selected_variant=frozen['validation_selected_variant'],conditions=[],models=[],costs=[])
    for name in protocol['conditions']:
        item=dict(condition=name,variants=[])
        for v in ('A','B','C','D'):
            ev={}
            if v!='C':
                for label,lo,hi in (('first8',1,8),('last8',9,16)):
                    group=[r for r in critics if r['condition']==name and r['variant']==v and r['period']=='-1'
                        and r['target']=='MC_RTG' and lo<=int(r['update'])<=hi]
                    ev[label]={k:mean([float(r[k]) for r in group]) for k in ('EV','RMSE_USD','predicted_mean_USD','target_mean_USD')}
                    ev[label]['signed_bias_USD']=ev[label]['predicted_mean_USD']-ev[label]['target_mean_USD']
            gr=[r for r in gradients if r['condition']==name and r['variant']==v]
            gradient=[]
            for label,lo,hi in (('first8',1,8),('last8',9,16)):
                group=[r for r in gr if lo<=int(r['update'])<=hi]
                gradient.append(dict(part=label,**{k:mean([float(r[k]) for r in group if r[k]!='']) for k in
                    ('shared_actor_norm','shared_weighted_critic_norm','shared_cosine','actor_clip_coefficient_mean')},
                    nonzero_actor_ratio_median=float(np.median([float(r['shared_weighted_critic_norm'])/float(r['shared_actor_norm'])
                        for r in group if r['shared_actor_norm'] and float(r['shared_actor_norm'])>0])) if v=='B' else None))
            selected=[r for r in actors if r['condition']==name and r['variant']==v and r['selected']=='True']
            sensitivity={}
            for t in (4,12,20):
                group=[r for r in selected if int(r['period'])==t]
                sensitivity[str(t)]=dict(mean_abs_signal=mean([float(np.abs(json.loads(r['action'])).mean()) for r in group]),
                    mean_price_USD=mean([float(np.mean(json.loads(r['price_USD']))) for r in group]))
            group=[r for r in probes if r['condition']==name and r['variant']==v]
            controls={c:dict(changed=sum(r['canonical_changed_from_deterministic']=='True' for r in group if r['control']==c),
                total=sum(r['control']==c for r in group)) for c in sorted(set(r['control'] for r in group))}
            visits=[r for r in training if r['condition']==name and r['variant']==v]
            trend={label:mean([float(r['training_mean_USD']) for r in visits if lo<=int(r['update'])<=hi])
                for label,lo,hi in (('first8',1,8),('last8',9,16))}
            item['variants'].append(dict(variant=v,critic=ev,gradient=gradient,reference_periods=sensitivity,
                selected_D_pi=mean([float(r['D_pi']) for r in selected]),probe_controls=controls,training_profit=trend))
            for seed in protocol['training_seeds']:
                folder=PARENT/'training'/name/str(seed) if v=='A' else TARGET/'training'/name/v/str(seed)
                logs=[read(folder/f'update-{u:04d}.json') for u in range(1,17)]
                row=dict(condition=name,variant=v,seed=seed)
                if v!='A':
                    row.update(read(folder/'initialization.json'))
                    row['phase_seconds']={k:sum(r[k] for r in logs) for k in ('rollout_with_replay_write_seconds',
                        'optimization_seconds','validation_with_replay_write_seconds')}
                    row['wall_seconds']=read(folder/'status.json')['wall_seconds']
                else:row['prefix_elapsed_seconds']=logs[-1].get('elapsed_seconds')
                out['models'].append(row)
        for v in ('A','B','C','D','MC','M0'):
            group=[r for r in business if r['condition']==name and r['method']==v]
            out['costs'].append(dict(condition=name,method=v,records=len(group),
                means={k:mean([float(r[k]) for r in group if r.get(k,'')!='']) for k in
                    ('completed_orders','revenue_USD','cost_USD','empty_cost_USD','backlog_penalty_USD','cancellation_penalty_USD',
                     'vehicle_utilization_travel_fraction_full_cooldown','solver_seconds','policy_seconds')}))
        out['conditions'].append(item)
    out['counterfactual_cases']=read(TARGET/'mechanism/cases.json')
    path=TARGET/'qa/report-evidence.json'
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Report evidence saved; descriptive only, no new statistical comparisons.')


if __name__=='__main__':run()
