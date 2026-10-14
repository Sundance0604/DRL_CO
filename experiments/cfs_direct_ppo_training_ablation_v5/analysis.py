"""Preregistered crossed inference; pure NumPy, no policy training or solver."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v5'
PARENT=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_no_mc_v3'
NAMES=('california_corridor-n80-k3','western_multicluster-n40-k8','western_multicluster-n80-k3')
SEEDS=(30001,30002,30003)
METHODS=('A','B','C','D')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def crossed(matrix,family,replications=20000,seed=61000):
    x=np.asarray(matrix,float)
    if x.shape!=(3,24) or not np.isfinite(x).all():raise ValueError('crossed3x24 required')
    rng=np.random.default_rng(seed);out=[]
    for offset in range(0,replications,1000):
        n=min(1000,replications-offset)
        a=rng.integers(0,3,(n,3));b=rng.integers(0,24,(n,24))
        samples=x[a[:,:,None],b[:,None,:]].mean((1,2))
        ca=np.array([np.bincount(i,minlength=3) for i in a])/3
        cb=np.array([np.bincount(i,minlength=24) for i in b])/24
        check=np.einsum('bi,ij,bj->b',ca,x,cb)
        if not np.allclose(samples,check,atol=1e-8,rtol=1e-12):raise RuntimeError('bootstrap recalculation')
        out.extend(samples)
    return dict(mean_USD=float(x.mean()),seed_mean_USD=x.mean(1).tolist(),
        low95_USD=float(np.quantile(out,.025)),high95_USD=float(np.quantile(out,.975)),
        low_adjusted_USD=float(np.quantile(out,.05/(2*family))),
        high_adjusted_USD=float(np.quantile(out,1-.05/(2*family))),
        family=family,replications=replications,bootstrap_seed=seed,count_weighted_recalculation=True)


def csv_write(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (TARGET/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def error(p,y):
    p,y=np.asarray(p,float),np.asarray(y,float)
    return dict(predicted_mean_USD=float(p.mean()),target_mean_USD=float(y.mean()),
        RMSE_USD=float(np.sqrt(np.mean((p-y)**2))),MAE_USD=float(np.abs(p-y).mean()),
        EV=float(1-np.var(y-p)/np.var(y)) if np.var(y)>0 else None)


def choose(name,variant,seed):
    parent=read(TARGET/'protocol/A-prefix-policies.json') if variant=='A' else read(TARGET/'protocol/frozen-policies.json')
    return next(r for r in parent['rows'] if r['condition']==name and r['seed']==seed and
                (variant=='A' or r['variant']==variant))


def folder(name,variant,seed):
    return PARENT/'training'/name/str(seed) if variant=='A' else TARGET/'training'/name/variant/str(seed)


def collect():
    if read(TARGET/'manifest.json')['status']!='NUMERICAL_COMPLETE_QA_PENDING':raise RuntimeError('numeric complete required')
    paired=[];business=[];effects=[];training=[];gradients=[];critics=[];actor=[];probes=[];summaries=[]
    from experiments.cfs_direct_ppo_v3.analysis import business_row
    for name in NAMES:
        cfg=read(TARGET/'protocol'/(name+'.json'));tests=read(TARGET/'protocol/preregistration.json')['test_seeds']
        baselines={method:[read(TARGET/'evaluation'/f'{name}-{method}-s{s}.json') for s in tests]
                   for method in ('M0','MC')}
        matrices={v:np.empty((3,24)) for v in METHODS}
        for method,records in baselines.items():
            for scene,rec in zip(tests,records):business.append(business_row(name,None,scene,method,rec,cfg))
        for variant in METHODS:
            for rowidx,seed in enumerate(SEEDS):
                ch=choose(name,variant,seed);base=folder(name,variant,seed)
                checks=[(read(base/f'validation-{u:04d}.json')['mean_profit_USD'],u) for u in (8,16)]
                if ch['selected_update']!=max(checks,key=lambda x:(x[0],-x[1]))[1]:raise RuntimeError('checkpoint selection')
                for j,scene in enumerate(tests):
                    rec=read(TARGET/'evaluation'/f'{name}-{variant}-{seed}-s{scene}.json')
                    matrices[variant][rowidx,j]=rec['ledger']['realized_profit']
                    business.append(business_row(name,seed,scene,variant,rec,cfg))
                stoch=[read(TARGET/'validation'/f'{name}-{variant}-{seed}-s{s}.json')['ledger']['realized_profit']
                       for s in range(21008,21012)]
                det=[read(base/f'validation-{ch["selected_update"]:04d}-s{s}.json')['ledger']['realized_profit']
                     for s in range(21008,21012)]
                for u in range(1,17):
                    log=read(base/f'update-{u:04d}.json')
                    training.append(dict(condition=name,variant=variant,training_seed=seed,update=u,
                        training_mean_USD=log['training_mean_USD'],validation_mean_USD=log['validation_mean_USD'],
                        selected_update=ch['selected_update'],reward_divisor=10000 if variant=='B' else 1000,
                        policy_loss=log['policy_loss'],value_loss_native_units=log['value_loss'],
                        KL_accepted_mean=log['approximate_kl'],kl_early_stop=log['kl_early_stop'],
                        actor_steps=log.get('actor_steps',log['minibatches']),
                        critic_steps=0 if variant=='C' else log.get('critic_steps',log['minibatches']),
                        stochastic_validation_USD=float(np.mean(stoch)) if u==ch['selected_update'] else None,
                        paired_deterministic_validation4_USD=float(np.mean(det)) if u==ch['selected_update'] else None))
                    detailed=log.get('detailed_gradient') or {}
                    minib=log.get('minibatch_records',[])
                    gradients.append(dict(condition=name,variant=variant,training_seed=seed,update=u,
                        shared_actor_norm=detailed.get('shared_actor_norm'),shared_weighted_critic_norm=detailed.get('shared_critic_norm'),
                        shared_cosine=detailed.get('shared_cosine'),shared_parameters=detailed.get('shared_parameters'),
                        actor_preclip_norm_mean=float(np.mean([r['preclip_norms'][0] for r in minib])) if minib else None,
                        critic_preclip_norm_mean=float(np.mean([r['preclip_norms'][1] for r in minib])) if variant=='D' else None,
                        actor_clip_coefficient_mean=float(np.mean([r['clip_coefficients'][0] for r in minib])) if minib else None,
                        KL_stop_trigger=log.get('trigger'),missing_reason='historical A shared gradients unavailable' if variant=='A' else
                            ('no shared parameters' if variant in ('C','D') else None)))
                    if variant=='C':
                        critics.append(dict(condition=name,variant=variant,training_seed=seed,update=u,period=-1,
                            target='NA',missing_reason='no Critic by design',EV=None,RMSE_USD=None,MAE_USD=None));continue
                    recs=[read(base/f'training-episode-{i:06d}.json') for i in range(4*(u-1)+1,4*u+1)]
                    divisor=10000 if variant=='B' else 1000
                    preds=[];targets=[];lambdas=[];periods=[]
                    for rec in recs:
                        usd=[t['reward_USD'] for t in rec['transitions']]
                        rtg=np.cumsum(usd[::-1])[::-1]
                        for t,l in enumerate(rec['learning']):
                            preds.append(l['value']*divisor);targets.append(rtg[t]);lambdas.append(l['return']*divisor);periods.append(t)
                    for target_name,y in (('MC_RTG',targets),('GAE_lambda_return',lambdas)):
                        for period in (-1,*range(24)):
                            indices=[i for i,t in enumerate(periods) if period==-1 or t==period]
                            critics.append(dict(condition=name,variant=variant,training_seed=seed,update=u,period=period,
                                target=target_name,states=len(indices),**error(np.asarray(preds)[indices],np.asarray(y)[indices])))
                for u in (8,16):
                    sens=read(TARGET/'analysis/A-sensitivity'/f'{name}-{seed}-{u:04d}.json') if variant=='A' else read(base/f'sensitivity-{u:04d}.json')
                    for point in sens['points']:
                        actor.append(dict(condition=name,variant=variant,training_seed=seed,update=u,
                            selected=u==ch['selected_update'],D_pi=sens['D_pi'],log_std=sens['log_std'],**point))
        for i,seed in enumerate(SEEDS):
            for j,scene in enumerate(tests):
                paired.append(dict(condition=name,training_seed=seed,scenario_seed=scene,
                    M0_profit_USD=baselines['M0'][j]['ledger']['realized_profit'],
                    MC_profit_USD=baselines['MC'][j]['ledger']['realized_profit'],
                    **{v+'_profit_USD':float(matrices[v][i,j]) for v in METHODS},
                    **{v+'_minus_A_USD':float(matrices[v][i,j]-matrices['A'][i,j]) for v in ('B','C','D')}))
        mc=np.array([r['ledger']['realized_profit'] for r in baselines['MC']])
        summary=dict(condition=name,**{v+'_mean_USD':float(matrices[v].mean()) for v in METHODS},
            MC_mean_USD=float(mc.mean()),M0_mean_USD=float(np.mean([r['ledger']['realized_profit'] for r in baselines['M0']])))
        summaries.append(summary)
        for v in ('B','C','D'):effects.append(dict(condition=name,comparison=v+'-A',role='primary',**crossed(matrices[v]-matrices['A'],9)))
        for v in METHODS:effects.append(dict(condition=name,comparison=v+'-MC',role='secondary',**crossed(matrices[v]-mc[None,:],12)))
    for p in sorted((TARGET/'mechanism').glob('*-t*.json')):
        r=read(p);ref=r['rows'][1]['canonical']
        for row in r['rows']:
            probes.append(dict(condition=r['condition'],variant=r['variant'],training_seed=r['seed'],
                scenario_seed=r['spec']['scenario_seed'],period=r['spec']['period'],control=row['control'],
                canonical_changed_from_deterministic=row['canonical']!=ref,signal=row['signal'],
                probe_cash_USD=row['probe']['plan']['operating_profit'],source_file=p.relative_to(TARGET).as_posix()))
    if (len(paired),len(training),len(gradients),len(actor),len(probes))!=(216,576,576,864,432):
        # Actor: three conditions x4variants x3seeds x2checkpoints x12states=864.
        raise RuntimeError('analysis coverage mismatch '+str((len(paired),len(training),len(gradients),len(actor),len(probes))))
    return dict(paired_results=paired,training_diagnostics=training,gradient_diagnostics=gradients,
        critic_diagnostics=critics,actor_state_sensitivity=actor,action_probe_results=probes,
        business_metrics=business,statistical_comparisons=effects),summaries


def run():
    tables,summaries=collect()
    for name,rows in tables.items():csv_write(name,rows)
    write(TARGET/'analysis/statistics.json',dict(summaries=summaries,effects=tables['statistical_comparisons']))
    report=['# Direct PPO训练机制消融：从学习质量到实际控制收益','',
        '本轮保持10维战略价格、价格尺度、公开观测、MILP与现金账本不变，只干预训练机制。A复用冻结九模型的前16次训练前缀且仅8/16选模，B/C/D新训练27模型，每模型16update。用户授权训练规模减半，原32update协议不是本轮实际规模。以下结果来自全新确认性测试22044–22067，不是原测试重复诊断。','',
        '## 正式利润告诉我们','',
        '| 工况 | M0 | Pure MC | A | B | C | D |','|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:report.append('|'+r['condition']+'|'+'|'.join(f'{r[k+"_mean_USD"]:.2f}' for k in ('M0','MC','A','B','C','D'))+'|')
    report += ['', '单位USD/完整冷却运营日；每方案三个训练seed与同24场景，不把72差值当72独立运营日。','',
        '| 工况／主比较 | 差 USD | 普通95% | 九比较调整区间 |','|---|---:|---|---|']
    for r in tables['statistical_comparisons']:
        if r['role']=='primary':report.append(f'|{r["condition"]} {r["comparison"]}|{r["mean_USD"]:.2f}|[{r["low95_USD"]:.2f},{r["high95_USD"]:.2f}]|[{r["low_adjusted_USD"]:.2f},{r["high_adjusted_USD"]:.2f}]|')
    report += ['', '交叉bootstrap20000次、seed61000；对应训练seed与场景两轴各自重抽，主族九项Bonferroni，次族12项PPO−MC单独调整。仅三个模型seed使极端分位区间有限，不把未显著称为无效或等效。', '',
        '## 十个问题：证据与解释分开', '']
    questions=['奖励缩放是否改善Critic？','预测偏差与梯度量级是否随更新改善？','完整回报优势是否改善PPO？',
        'Actor–Critic解耦是否改善利润？','哪个方案学习到状态依赖？','随机策略改善能否转化为确定性部署？',
        '哪些方案超过A？','哪些方案缩小对MC的差距？','改善是否跨工况和seed稳定？','继续训练机制还是进入带q接口？']
    for i,q in enumerate(questions,1):
        report.extend([f'### 问题{i}：{q}',''])
        if i in (1,2):
            for name in NAMES:
                values=[]
                for v in ('A','B','D'):
                    records=[r for r in tables['critic_diagnostics'] if r['condition']==name and r['variant']==v and r['period']==-1 and r['target']=='MC_RTG']
                    first=np.mean([r['EV'] for r in records if r['update']<=8]);last=np.mean([r['EV'] for r in records if r['update']>=9])
                    rmse=np.mean([r['RMSE_USD'] for r in records if r['update']>=9])
                    values.append(f'{v} EV前8/末8={first:.4f}/{last:.4f}，末段RMSE={rmse:.2f}USD')
                report.append(name+'：'+'；'.join(values)+'。')
            report.append('这是各自on-policy访问分布上的pre-update质量，不是固定状态因果比较；真实RTG与GAE目标分列。A共享分项历史梯度缺失，不使用事后checkpoint冒充；B记录真实共享梯度，D两图独立。尺度改变影响整个优化动力学，不单归因于Critic范数。')
        elif i in (3,4,7):
            variant='C' if i==3 else 'D' if i==4 else None
            for r in tables['statistical_comparisons']:
                if r['role']=='primary' and (variant is None or r['comparison']==variant+'-A'):
                    status='调整后正向优势' if r['low_adjusted_USD']>0 else '调整后负向证据' if r['high_adjusted_USD']<0 else '调整后尚未建立方向性结论'
                    report.append(f'{r["condition"]} {r["comparison"]}：{r["mean_USD"]:+.2f}USD，{status}；seed均值={r["seed_mean_USD"]}。')
            report.append('C同时移除Critic训练并更换优势；D同时改变参数共享、优化器、裁剪及参数量。不能从组合效果识别其中某个单独因果机制。')
        elif i==5:
            for name in NAMES:
                pieces=[]
                for v in METHODS:
                    rows=[r for r in tables['actor_state_sensitivity'] if r['condition']==name and r['variant']==v and r['selected']]
                    pieces.append(f'{v} Dpi={np.mean([r["D_pi"] for r in rows]):.6f}')
                report.append(name+'：'+'；'.join(pieces)+'。')
            report.append('所有方案使用同12公开参考状态；响应变化不等于现金增益，固定价格校准与动态控制必须区分。见逐state价格、负载/早中晚结果及少量规范化MILP探针。')
        elif i==6:
            for name in NAMES:
                for v in METHODS:
                    rows=[r for r in tables['training_diagnostics'] if r['condition']==name and r['variant']==v and r['stochastic_validation_USD'] is not None]
                    report.append(f'{name} {v} 同4验证场景随机−确定性均值={np.mean([r["stochastic_validation_USD"]-r["paired_deterministic_validation4_USD"] for r in rows]):+.2f}USD。')
            report.append('每冻结策略/场景仅一次固定随机采样，属于辅助诊断，未用于选模；训练场景均值与验证均值分布不同，不能直接作部署差的因果估计。')
        elif i==8:
            for r in tables['statistical_comparisons']:
                if r['role']=='secondary':report.append(f'{r["condition"]} {r["comparison"]}={r["mean_USD"]:+.2f}USD，次族调整区间[{r["low_adjusted_USD"]:.2f},{r["high_adjusted_USD"]:.2f}]。')
            report.append('完整次族保留所有方案；方法赢家只由验证均值预选，不能测试后挑最高者。')
        elif i==9:
            report.append('九个主比较逐seed均值上表／统计CSV完整保留；条件之间与seed之间的方向/幅度不同，应逐项解释，不用总体平均掩盖保护工况。')
        else:report.append('应结合主族调整区间、跨seed方向、固定状态敏感性及Critic质量，决定后续资源配置。没有超过A不否定RL；有改善也不证明单一根因。q扩展本轮未实施，不能将未来假设写为已验证方案。最终人工审阅会基于实际图表和QA补充有依据的建议。')
        report.append('')
    budget=read(TARGET/'qa/numerical-budget.json')
    report += ['## 正确性、成本与范围','',f'实际数值预算记录：{budget["calls"]}次MILP，累计{budget["seconds"]/60:.2f}分钟；训练/验证及推理成本从原始计时分开统计，不重复累加update累计值。最终回放和交付另计。','',
        'v1只在smoke因基线核验计数写死4而停止，正式训练未开始；v2核验按配置读取，每轮完整重新执行，没有复用失败数值，保留历史与失败全部SHA。原动作、物理、价格、现金、随机场景规则和统计没有调参追结果。','',
        'A/B/C/D每t原始记录、实际shared梯度、独立网络、奖励/GAE/时间基线与完整冷却核验见qa；最终状态以final_manifest.json为准，NUMERICAL_COMPLETE不是完成。图表为SVG与600dpiPNG，数据来源为本目录八CSV，不生成PDF。']
    (TARGET/'direct_ppo_training_ablation_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    write(TARGET/'qa/analysis.json',dict(status='PASSED',CSV_files=8,paired_rows=len(tables['paired_results']),
        primary_comparisons=9,secondary_comparisons=12,count_weighted_bootstrap_checked=True))


if __name__=='__main__':run()
