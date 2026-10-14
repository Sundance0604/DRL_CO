"""Preregistered descriptive comparisons, no significance/selection search."""
import csv
import json
import subprocess
import numpy as np
from experiments.cfs_structured_ppo_q_mechanism.delivery import compute_kl
from .common import *


def rows():
    training=[];kl=[];beta=[];q=[];profit=[]
    for method,folder in (('old',OLD_TRAIN),('new',TARGET/'training')):
        for u in range(1,9):
            d=read_json(folder/f'update-{u:04d}.json')
            training.append(dict(method=method,update=u,planned_steps=8,effective_steps=d['minibatches'],skipped_steps=8-d['minibatches'],
                early_stop=d['kl_early_stop'],training_mean_USD=d['training_mean_USD'],policy_loss=d['policy_loss'],critic_EV=d['critic_EV'],
                critic_RMSE_USD=d['critic_RMSE_USD'],log_std_after=json.dumps(d['log_std']),approximate_KL_accepted_mean=d['approximate_kl']))
            p=MECHANISM/'raw/kl'/f'PPO20-{W40}-30001-u{u:02d}.json' if method=='old' else TARGET/'training/kl'/f'update-{u:04d}.json'
            checks=read_json(p)['checks']
            for c in checks:
                calc=compute_kl(c)
                if abs(calc['exact_KL']-c['exact_KL'])>1e-10 or abs(calc['independent_approximate_KL']-c['approximate_KL'])>2e-5:raise RuntimeError('independent KL differs')
                kl.append(dict(method=method,update=u,check=c['check'],epoch=c['epoch'],batch=c['batch_index'],
                    last_check=c==checks[-1],stopping_trigger=c['stopping_trigger'],accepted=c['accepted'],
                    approximate_KL_actual=c['approximate_KL'],**calc,alpha_mu_shift_RMS=c['alpha']['mu_shift_RMS'],beta_mu_shift_RMS=c['beta']['mu_shift_RMS'],
                    alpha_log_sigma_shift_RMS=c['alpha']['log_sigma_shift_RMS'],beta_log_sigma_shift_RMS=c['beta']['log_sigma_shift_RMS'],
                    log_std_before=json.dumps(c['log_std_before']),log_std_at_check=json.dumps(c['log_std_at_check']),
                    parameter_sha256=c['parameter_sha256'],source=str(p)))
    for p in sorted((TARGET/'probes').glob('*.json')):
        d=read_json(p);spec=d['spec'];new,zero=d['rows'];old=d['old'];old0=d['old_beta0']
        q.append(dict(scenario_seed=spec['scenario_seed'],period=spec['period'],q_active=spec['q_active'],max_feasible_q2_groups=spec['max_q2_groups'],
            old_beta_switch=old['canonical']!=old0['canonical'],new_beta_switch=new['canonical']!=zero['canonical'],
            new_vs_old_switch=new['canonical']!=old['canonical'],state_hash=spec['state_hash'],source=str(p)))
        for method,r in (('old',old),('new',new)):
            beta.append(dict(method=method,scenario_seed=spec['scenario_seed'],period=spec['period'],q_active=spec['q_active'],
                alpha=json.dumps(r['signal'][:10]),beta=json.dumps(r['signal'][10:]),beta_mean_absolute=float(np.mean(np.abs(r['signal'][10:]))),
                beta_max_absolute=max(abs(v) for v in r['signal'][10:]),raw_action=json.dumps(r['signal']),
                USD_prices=json.dumps(r['probe']['coefficients']),selected_slots=json.dumps(r['probe']['plan']['solver']['slot_occupancy']),
                chosen_q2_groups=sum(s['q']==2 and s['occupied'] for s in r['probe']['plan']['solver']['slot_occupancy']),
                status=r['probe']['plan']['solver']['status'],canonical=r['canonical'],source=str(p)))
    for s in read_json(TARGET/'data/development.json')['scenarios']:
        pair=dict(scenario_seed=s['seed'],scenario_hash=identity(s))
        for method in ('old','new'):
            r=read_json(TARGET/'evaluation'/f'{method}-s{s["seed"]}.json');m=r['metrics'];l=r['ledger']
            for k,v in dict(profit_USD=l['realized_profit'],revenue_USD=l['completed_fare'],total_cost_USD=l['realized_cost'],
                completed_orders=m['completed_orders'],empty_cost_USD=m['actual_empty_distance']*configuration()['parameters']['empty_cost'],
                solver_seconds=m['solver_seconds'],episode_seconds=m['episode_seconds'],cooldown_periods=m['cooldown_periods']).items():pair[method+'_'+k]=v
        pair['difference_USD']=pair['new_profit_USD']-pair['old_profit_USD'];profit.append(pair)
    return dict(training_comparison=training,kl_comparison=kl,beta_action_diagnostics=beta,q_active_comparison=q,paired_development_profit=profit)


def summary(data):
    result={}
    for method in ('old','new'):
        us=[r for r in data['training_comparison'] if r['method']==method]
        tr=[r for r in data['kl_comparison'] if r['method']==method and r['stopping_trigger']]
        bs=[json.loads(r['beta']) for r in data['beta_action_diagnostics'] if r['method']==method]
        b=np.array(bs)
        result[method]=dict(early_stops=sum(r['early_stop'] for r in us),steps=sum(r['effective_steps'] for r in us),planned_steps=64,
            trigger_beta_share_median=float(np.median([r['beta_exact_KL']/r['exact_KL'] for r in tr])) if tr else None,
            beta_mean_absolute=float(np.abs(b).mean()),beta_state_response_RMS=float(np.sqrt(np.mean((b-b.mean(0))**2))),
            active_switches=sum(r[method+'_beta_switch'] for r in data['q_active_comparison'] if r['q_active']),
            active_states=sum(r['q_active'] for r in data['q_active_comparison']),
            only_switches=sum(r[method+'_beta_switch'] for r in data['q_active_comparison'] if not r['q_active']))
    differences=[r['difference_USD'] for r in data['paired_development_profit']]
    result.update(profit_mean_difference_USD=float(np.mean(differences)),profit_directions=[int(np.sign(v)) for v in differences],
        new_profit_mean_USD=float(np.mean([r['new_profit_USD'] for r in data['paired_development_profit']])),
        old_profit_mean_USD=float(np.mean([r['old_profit_USD'] for r in data['paired_development_profit']])),
        H1_more_steps=result['new']['steps']>result['old']['steps'],
        H2_more_beta_switches=result['new']['active_switches']>result['old']['active_switches'],
        H3_positive_development_mean=float(np.mean(differences))>0,
        interpretation='descriptive single training seed/four development scenes; no p-values, no confirmation, no MC comparison')
    return result


def deliver():
    load()
    if (TARGET/'qa/analysis.json').exists():raise RuntimeError('do not overwrite analysis or reviewed report')
    data=rows();sentinel=TARGET/'qa/csv-marker.json'
    if not sentinel.exists():
        node='C:/Users/32599/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
        mark='C:/Users/32599/.codex/plugins/cache/openai-primary-runtime/spreadsheets/26.1011.11113/skills/spreadsheets/container_tools/mark_artifact_operation_started.mjs'
        r=subprocess.run([node,mark,'--operation-kind','create','--expected-output-count','5','--output-format','csv'],check=True,capture_output=True,text=True)
        write_json(sentinel,dict(at=now(),expected=5,stdout=r.stdout))
    for name,records in data.items():
        with (TARGET/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    s=summary(data);write_json(TARGET/'qa/summary.json',s)
    write_json(TARGET/'qa/analysis.json',dict(status='PASSED',rows={k:len(v) for k,v in data.items()},summary=s))
    (TARGET/'experiment_protocol.md').write_text('# β 初始探索尺度：实际协议\n\n```json\n'+json.dumps(read_json(TARGET/'protocol.json'),ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')


if __name__=='__main__':deliver()
