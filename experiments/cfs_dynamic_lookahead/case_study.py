"""Post-selected Western trajectory explanation, not a statistical sample."""
import json
from pathlib import Path
from experiment_core.storage import read_json,digest,atomic_json
from experiment_core.plugins import resolve
from model_families.single.data import matching_state
from model_families.single.planning import PeriodAction,VehicleAction,compile_action
from model_families.single.engine.mt_prototype import DecisionPlan
from model_families.single.runner import state_snapshot
from model_families.single.rolling.audit import feasible_now
from experiments.cfs_rho_endofday.run_experiment import write_csv


def action_of(plan):
    by_vehicle={int(r['vehicle_id']):dict(route={k:v for k,v in r.items() if k!='vehicle_id'},orders=[]) for r in plan['routes']}
    for a in plan['assignments']:by_vehicle[int(a['vehicle_id'])]['orders'].append(a['order_id'])
    actions=[VehicleAction(k,r['route'],tuple(r['orders'])) for k,r in by_vehicle.items()]
    actions += [VehicleAction(int(r['vehicle_id']),reposition_to=r['hub']) for r in plan['repositions']]
    return PeriodAction(plan['period'],tuple(actions))


def trace_action_key(row):
    p=row['plan']
    return digest({k:p[k] for k in ['assignments','routes','repositions']})


def replay_plan(sim,row):
    """Validate action/cost independently, then preserve recorded float summation.

    Original myopic objective extraction and compile_action can group sums in a
    different order. The state hash includes accumulated booked profit. Do not
    overwrite a physical state to force a hash match; retain the recorded monetary
    scalars only after an absolute 1e-6 agreement check.
    """
    compiled=compile_action(sim,action_of(row['plan']),{})
    recorded=(row['plan']['operating_profit'],row['plan']['augmented_solver_objective'])
    if any(abs(a-b)>1e-6 for a,b in zip(compiled[:2],recorded)):
        raise RuntimeError('case replay monetary value differs beyond floating tolerance')
    return DecisionPlan((*recorded,*compiled[2:]),sim.t,sim.revision)


def western_case(output,rows,reps):
    output=Path(output);failure=next(r for r in reps if 'failure' in r['roles'])
    condition=lambda r:r['region']==failure['region'] and r['orders']==failure['orders'] and r['fleet']==failure['fleet'] and r['rho_label']=='rho_max'
    selected=[r for r in rows if condition(r)]
    myopic={int(r['seed']):r for r in selected if r['policy']=='myopic'}
    candidates=[r for r in selected if r['policy']!='myopic']
    best=max(candidates,key=lambda r:r['realized_profit']-myopic[int(r['seed'])]['realized_profit'])
    baseline=myopic[int(best['seed'])];seed=int(best['seed'])
    if best['realized_profit']<=baseline['realized_profit']:
        selection_note='没有正向recovery seed；保留最大差值案例，不虚构改善。'
    else:selection_note='按最大wide-rho动态改善自动选取，仅作事后机制例，不代表平均收益。'
    target=output/'traces';target.mkdir(parents=True,exist_ok=True)
    paths=[output/'raw/runs'/r['run_id'] for r in [baseline,best]]
    traces=[read_json(path/'trace.json')[0]['steps'] for path in paths]
    cfg=resolve(read_json(paths[0]/'spec.resolved.json'))
    payload=read_json(output/'raw/datasets'/f'{failure["region"]}-n{failure["orders"]}-k{failure["fleet"]}-s{seed}.dataset.json')
    sim,orders=matching_state(payload['scenarios'][0],cfg.model.parameters,cfg.solver.parameters.model_dump())
    by_t={}
    for o in orders.values():by_t.setdefault(o.book_time,[]).append(o)
    alternatives={r['period']:r for r in traces[1] if not r.get('drain')}
    first=None;cross_period=None;local=None;common_state=False;table=[];all_candidates={'myopic':{},best['policy']:{}}
    for row in traces[0]:
        if row.get('drain'):continue
        t=row['period'];sim.begin_period(by_t.get(t,[]))
        if digest(state_snapshot(sim))!=row['plan']['state_hash']:raise RuntimeError('case replay does not match pre-action state')
        all_candidates['myopic'][str(t)]=feasible_now(sim)[1]
        other=alternatives[t]
        different=trace_action_key(row)!=trace_action_key(other)
        if different and first is None:first=t
        is_cross=any(r['new'] and row['before']['vehicles'][int(r['vehicle_id'])]['phase']=='idle' and
                     row['before']['vehicles'][int(r['vehicle_id'])]['hub']!=r['p'] for r in row['plan']['routes'])
        if different and is_cross and cross_period is None:
            cross_period=t;chosen=compile_action(sim,action_of(row['plan']),{})
            waiting=compile_action(sim,PeriodAction(t),{})
            local=dict(chosen_booked_profit=chosen[0],wait_booked_profit=waiting[0],immediate_advantage=chosen[0]-waiting[0])
            common_state=row['plan']['state_hash']==other['plan']['state_hash']
        sim.commit_plan(replay_plan(sim,row))
    alternate,_=matching_state(payload['scenarios'][0],cfg.model.parameters,cfg.solver.parameters.model_dump())
    for row in traces[1]:
        if row.get('drain'):continue
        t=row['period'];alternate.begin_period(by_t.get(t,[]))
        if digest(state_snapshot(alternate))!=row['plan']['state_hash']:raise RuntimeError('alternative case replay mismatch')
        all_candidates[best['policy']][str(t)]=feasible_now(alternate)[1]
        alternate.commit_plan(replay_plan(alternate,row))
    full_candidates=output/'raw/case-candidates';full_candidates.mkdir(parents=True,exist_ok=True)
    atomic_json(full_candidates/'western-all-periods.json',all_candidates)
    for policy,trace in zip(['myopic',best['policy']],traces):
        for row in trace:
            audit=row['plan']['solver'].get('deferral_audit',{})
            empty=sum(sim.net.d[row['before']['vehicles'][int(r['vehicle_id'])]['hub'],r['p']]
                      for r in row['plan']['routes'] if r['new'] and row['before']['vehicles'][int(r['vehicle_id'])]['phase']=='idle')
            table.append(dict(policy=policy,H=0 if policy=='myopic' else int(best['H']),seed=seed,t=row['period'],cooldown=bool(row.get('drain')),
                open_orders=json.dumps(row['before']['pool'],ensure_ascii=False),
                vehicle_states=json.dumps(row['before']['vehicles'],ensure_ascii=False),
                incoming_committed_supply=json.dumps(audit.get('incoming_committed_supply',[])),
                candidate_cross_hub_actions=json.dumps(audit.get('candidate_cross_hub_actions',[])),
                candidate_display_truncated=audit.get('candidate_display_truncated',False),
                full_candidates_ref='../raw/case-candidates/western-all-periods.json',
                chosen_action=json.dumps({k:row['plan'][k] for k in ['assignments','routes','repositions']}),
                empty_miles_committed=empty,orders_deferred=json.dumps(audit.get('orders',[])),
                cumulative_realized_profit=row['realized_ledger']['realized_profit'],
                realized_downstream_final_profit=baseline['realized_profit'] if policy=='myopic' else best['realized_profit']))
    write_csv(target/'western_case.csv',table)
    details=dict(seed=seed,regime=failure,myopic=baseline,alternative=best,first_divergence=first,
                 cross_hub_divergence=cross_period,local_comparison_to_wait=local,common_state_at_cross=common_state,
                 selection_note=selection_note)
    atomic_json(target/'western_case.json',details)
    text=f"### Western case：N={failure['orders']}, K={failure['fleet']}, seed={seed}\n\n{selection_note}\n\n"
    text+=f"Myopic wide最终profit={baseline['realized_profit']:.2f}；{best['policy']} H={int(best['H'])}为{best['realized_profit']:.2f}，差值={best['realized_profit']-baseline['realized_profit']:+.2f}。首个动作分歧t={first}。\n\n"
    if cross_period is not None:
        base=next(r for r in traces[0] if r['period']==cross_period)
        alt=alternatives[cross_period]
        chosen=base['plan'];audit=chosen['solver']['deferral_audit']
        text+=f"在t={cross_period}出现cross-hub动作分歧：Myopic执行{chosen['assignments']}，其路线为{[(r['vehicle_id'],r['p'],r['e']) for r in chosen['routes']]}；备选策略执行{alt['plan']['assignments']}。\n\n"
        text+=f"原myopic本期booked profit={local['chosen_booked_profit']:.2f}，固定承诺继续、其余等待的可行动作为{local['wait_booked_profit']:.2f}，单期优势={local['immediate_advantage']:+.2f}。这是局部booked比较，不是对最终realized目标的上界。\n\n"
        text+=f"Myopic当时已知incoming supply={audit['incoming_committed_supply']}；备选策略的暂缓审计={alt['plan']['solver']['deferral_audit']['orders']}。\n\n"
        text+=('该cross分歧时两个策略前状态hash相同，可直接比较当期选择，但后续都继续不同策略，仍不是单动作因果反事实。\n\n' if common_state else
               '该cross分歧前状态已不同，是此前控制的累积后果，不能将全天gain归因于该一期cross决策。\n\n')
    else:text+='此seed未出现可直接解释的cross动作分歧，改善主要出现在等待/本地路线或早期安排；不虚构“某一期错误cross”。\n\n'
    text+='| 策略 | delivered | empty miles | cross pickup miles | deliberate deferred orders | mean deferral periods |\n|---|---:|---:|---:|---:|---:|\n'
    for r in [baseline,best]:
        text+=f"| {r['policy']} H={int(r['H'])} | {int(r['delivered'])} | {r['actual_empty_distance']:.2f} | {r['empty_pickup_distance']:.2f} | {int(r['deliberate_deferral_count'])} | {r['mean_deliberate_deferral_periods']:.2f} |\n"
    if abs(best['actual_empty_distance']-baseline['actual_empty_distance'])<1e-6:
        text+=f"\n该案例空驶总量相同，但服务订单由{int(baseline['delivered'])}增至{int(best['delivered'])}，与更高完成收入一致；不支持将该seed的收益主要解释为减少empty deadhead。前期暂缓、后续配载与车辆时序共同变化，不能从一条事后选中的轨迹识别单一因果机制。\n"
    text+='\n完整逐期CSV含open orders、车辆、incoming pipeline、候选动作（最多12条显示，truncation flag显式）、选择、空驶、deferral和后续realized结果。每期全部当前cross候选已由两个策略的真实前状态重放枚举，保存在raw/case-candidates/western-all-periods.json，并通过CSV路径引用。重放先独立验证动作/账本金额误差不超过1e-6，再沿用记录的booked monetary scalars，以避免约1e-12累加顺序误差使严格state hash失配；不覆盖物理状态或原始trace。按seed最大改善选例具有选择偏差，正式CI不依赖该案例。\n'
    (target/'western_case.md').write_text(text,encoding='utf8')
    return text
