"""Paired scientific analysis, no network or MILP execution."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from experiments.cfs_ppo_signal_action_diagnostic.analyse import read,write,sha

NAMES=('california_corridor-n80-k3','western_multicluster-n40-k8','western_multicluster-n80-k3')
LABELS=('CA80/3','W40/8','W80/3')
SEEDS=(30001,30002,30003)


def crossed(x,replications=20000,seed=42000):
    x=np.asarray(x,float)
    if x.shape!=(3,24) or not np.isfinite(x).all():raise ValueError('requires crossed 3 x24 matrix')
    rng=np.random.default_rng(seed);samples=[];independent=[]
    for offset in range(0,replications,1000):
        n=min(1000,replications-offset)
        a=rng.integers(0,3,(n,3));b=rng.integers(0,24,(n,24))
        draws=x[a[:,:,None],b[:,None,:]].mean(axis=(1,2))
        # A second, count-weighted implementation of the same realized resamples.
        counts_a=np.array([np.bincount(r,minlength=3) for r in a])/3
        counts_b=np.array([np.bincount(r,minlength=24) for r in b])/24
        independent.extend(np.einsum('bi,ij,bj->b',counts_a,x,counts_b))
        samples.extend(draws)
    if not np.allclose(samples,independent,atol=1e-8,rtol=1e-12):raise RuntimeError('independent crossed bootstrap differs')
    return dict(mean_USD=float(x.mean()),seed_mean_USD=x.mean(1).tolist(),
        marginal95=np.quantile(samples,[.025,.975]).tolist(),
        simultaneous98_333=np.quantile(samples,[.05/6,1-.05/6]).tolist(),
        replications=replications,bootstrap_seed=seed,count_weighted_recalculation=True)


def csv_rows(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def collect(target):
    m=read(target/'manifest.json')
    if m['status']!='NUMERICAL_COMPLETE_QA_PENDING':raise RuntimeError('numerical completion required')
    frozen=read(target/'protocol/frozen-policies.json');protocol=read(target/'protocol/preregistration.json')
    paired=[];business=[];curves=[];effects=[]
    for name in NAMES:
        baseline=[];matrix=[]
        for scene in protocol['splits']['test']:
            mc=read(target/'mc'/f'{name}-test-s{scene}.json');baseline.append(mc['ledger']['realized_profit'])
            business.append(business_row(name,None,scene,'Pure-MC',mc,read(target/'protocol'/(name+'.json'))))
        for seed in SEEDS:
            values=[];choice=next(r for r in frozen['rows'] if r['condition']==name and r['seed']==seed)
            vals=[]
            for update in (8,16,24,32):
                v=read(target/'training'/name/str(seed)/f'validation-{update:04d}.json')
                recomputed=sum(v['profits'])/16
                if abs(recomputed-v['mean_profit_USD'])>1e-8:raise RuntimeError('validation mean mismatch')
                vals.append((recomputed,update))
            expected=max(vals,key=lambda x:(x[0],-x[1]))
            if choice['selected_update']!=expected[1] or abs(choice['best_validation_USD']-expected[0])>1e-8:
                raise RuntimeError('checkpoint selection differs from validation')
            for update in range(1,33):
                r=read(target/'training'/name/str(seed)/f'update-{update:04d}.json')
                curves.append(dict(condition=name,training_seed=seed,update=update,training_mean_USD=r['training_mean_USD'],
                    validation_mean_USD=r['validation_mean_USD'],policy_loss=r['policy_loss'],critic_MSE=r['value_loss'],
                    KL=r['approximate_kl'],clip_fraction=r['clip_fraction'],gradient_norm=r['gradient_norm'],
                    actor_rollout_mean=json.dumps(r['action_mean']),actor_rollout_std=json.dumps(r['action_std']),
                    kl_early_stop=r['kl_early_stop']))
            for index,scene in enumerate(protocol['splits']['test']):
                r=read(target/'evaluation'/f'{name}-{seed}-s{scene}.json')
                value=r['ledger']['realized_profit'];values.append(value)
                paired.append(dict(condition=name,training_seed=seed,scenario_seed=scene,MC_profit_USD=baseline[index],
                    Direct_profit_USD=value,difference_USD=value-baseline[index],
                    selected_update=choice['selected_update'],record_file=f'evaluation/{name}-{seed}-s{scene}.json'))
                business.append(business_row(name,seed,scene,'Direct-PPO',r,read(target/'protocol'/(name+'.json'))))
            matrix.append(values)
        difference=np.array(matrix)-np.array(baseline)[None,:];statistics=crossed(difference)
        mean_mc=float(np.mean(baseline));mean_direct=float(np.mean(matrix))
        effect=dict(condition=name,MC_mean_USD=mean_mc,Direct_mean_USD=mean_direct,**statistics,
            MC_profit_CI95=crossed(np.tile(baseline,(3,1)))['marginal95'],
            Direct_profit_CI95=crossed(matrix)['marginal95'],
            difference_percent=100*statistics['mean_USD']/abs(mean_mc) if mean_mc else None)
        effects.append(effect)
    if (len(paired),len(business),len(curves))!=(216,288,288):raise RuntimeError('formal coverage mismatch')
    return paired,business,curves,effects


def business_row(name,seed,scene,method,r,cfg):
    ledger=r['ledger'];metrics=r['metrics'];events=metrics['execution_events']
    p=cfg['parameters']
    duration=sum(s['end']-s['start'] for v in events for s in v['segments'] if 'end' in s and 'start' in s)
    loaded=ledger['actual_loaded_distance']*p['loaded_cost'];empty=ledger['actual_empty_distance']*p['empty_cost']
    waiting=ledger['actual_vehicle_waiting']*p['waiting_cost']
    repos=sum(s['distance']*p['empty_cost'] for v in events for s in v['segments'] if s['kind']=='reposition')
    if abs(loaded+empty+waiting+ledger['backlog_penalty']+ledger['cancellation_penalty']-ledger['realized_cost'])>1e-6:
        raise RuntimeError('business cost decomposition differs from ledger')
    utilization=duration/(len(events)*ledger['cutoff_exclusive'])
    if not 0<=utilization<=1:raise RuntimeError('travel time utilization out of range')
    return dict(condition=name,training_seed=seed,scenario_seed=scene,method=method,
        profit_USD=ledger['realized_profit'],revenue_USD=ledger['completed_fare'],cost_USD=ledger['realized_cost'],
        completed_orders=ledger['completed_orders'],loaded_cost_USD=loaded,
        empty_cost_USD=empty,reposition_cost_USD=repos,waiting_cost_USD=waiting,
        backlog_penalty_USD=ledger['backlog_penalty'],cancellation_penalty_USD=ledger['cancellation_penalty'],
        unfinished=ledger['unfinished_committed'],solver_calls=metrics['solver_calls'],
        solver_seconds=metrics['solver_seconds'],episode_seconds=metrics['episode_seconds'],policy_seconds=r.get('policy_seconds'),
        MILP_failed_calls=0,vehicle_utilization_travel_fraction_full_cooldown=utilization,load_factor=ledger['load_factor'])


def data(target):
    if (target/'qa/analysis.json').exists() or not (target/'qa/csv-marker.json').exists():raise RuntimeError('no repeated analysis; marker required')
    paired,business,curves,effects=collect(target)
    rows=[dict(condition=e['condition'],MC_mean_USD=e['MC_mean_USD'],Direct_mean_USD=e['Direct_mean_USD'],
        difference_USD=e['mean_USD'],difference_percent=e['difference_percent'],CI95_low_USD=e['marginal95'][0],CI95_high_USD=e['marginal95'][1],
        family_CI_low_USD=e['simultaneous98_333'][0],family_CI_high_USD=e['simultaneous98_333'][1]) for e in effects]
    for name,rs in (('paired_results',paired),('business_metrics',business),('learning_curves',curves),('summary',rows)):csv_rows(target/(name+'.csv'),rs)
    write(target/'analysis/results.json',effects)
    frozen=read(target/'protocol/frozen-policies.json');eng=read(target/'qa/engineering.json');p=read(target/'protocol/preregistration.json')
    lines=['# 无 MC 的 Direct PPO 与纯 MC：正式配对数值比较','',
        '## 模型边界','',
        '本轮 PPO 独立替代 MC 资源表，不学习 MC 残差。Actor 输出五枢纽 near/far 共十维信号，训练收入分位数尺度换算直接价格，H1 MILP 完成原匹配、路线、容量与承诺。推理不加载 MC Table，不含 MC 特征、预训练、标签或奖励。Critic 保留用于 GAE，不是 MC 底座。',
        '','Direct Hub 特征16→12，主干输入301→281，其他集合编码、隐藏宽度64和 PPO 数学不变。价格不依赖q，是本轮预先指定的接口限制。MC 对照使用冻结 bank1 Table 和独立策略进程，没有 Actor 或残差。',
        '','## 数据与预算','',
        '每工况128训练场景、16验证、24新密封测试；训练20032–20159、验证21008–21023、测试22020–22043，避开历史实际使用 seed/content。原准备池已生成但未执行的区段不冒称已训练；CFS 衍生合成日不是真实日期重放。',
        '','每工况三个独立 seeds30001–30003，从随机初始化训练32×4=128日，验证8/16/24/32，按验证均值选择、并列取较早；测试之前全部九个模型已冻结，不择优报告seed。LR0.0003、clip0.2、gamma1、lambda0.95、epochs4、batch64、value0.5、entropy0、grad0.5、KL0.03、reward/1000。',
        '','价格尺度为 max(100 USD,5×训练收入Q75)：']
    for e in p['economics']:lines.append(f"- {e['condition']}：Q75={e['Q75_USD']:.2f} USD，五hub尺度均为 {e['scale_USD']:.2f} USD。系数5未按测试收益优化。")
    lines+=['','## 实现利润：主要结果','', '| 工况 | MC均值 USD | Direct均值 USD | 差 USD | 差% | 普通95%区间 USD | 三比较调整区间 USD |','|---|---:|---:|---:|---:|---|---|']
    for label,e in zip(LABELS,effects):lines.append(f"| {label} | {e['MC_mean_USD']:.2f} | {e['Direct_mean_USD']:.2f} | {e['mean_USD']:.2f} | {e['difference_percent']:.2f}% | [{e['marginal95'][0]:.2f}, {e['marginal95'][1]:.2f}] | [{e['simultaneous98_333'][0]:.2f}, {e['simultaneous98_333'][1]:.2f}] |")
    lines+=['','配对采用完全相同订单日与物理参数，24个主动时期及完整冷却。每工况3×24=72差值不是72个独立运营日；交叉重抽模型/场景20000次，固定seed42000，三主比较Bonferroni98.333% percentile区间，另列95%。仅三个模型，尾部区间精度有限。区间包含零不证明等效，不以loss或MILP objective替代利润。',
        '','## 所有训练seed与学习曲线','', '| 工况 | seed | 选中update | 验证均值 USD | 正式差 USD |','|---|---:|---:|---:|---:|']
    for e in effects:
        for i,seed in enumerate(SEEDS):
            r=next(r for r in frozen['rows'] if r['condition']==e['condition'] and r['seed']==seed)
            lines.append(f"| {e['condition']} | {seed} | {r['selected_update']} | {r['best_validation_USD']:.2f} | {e['seed_mean_USD'][i]:.2f} |")
    lines+=['','训练、验证均值随更新曲线与 policy loss、Critic MSE、KL、clip fraction、输出均值/std保存在learning_curves.csv。更新后的确定性验证收益与随机训练均值应分别解释；loss变小或选择较早checkpoint均不是收敛证明。',
        '','## 正确性与成本','',
        '零信号回归逐动作、有效战略系数和完整利润与M0一致；直接价格插值、终端、有限值和十维约束通过。训练/推理有实际 MC 读入/构造/query禁止 guard。每条训练、验证和测试轨迹进行现金/价格/状态/action/full-cooldown独立回放，GAE和奖励望远镜重算；非OPTIMAL不计成功样本。',
        '','工程尺度响应（仅训练日，不据正式利润校准）：']
    for r in eng['conditions']:lines.append(f"- {r['condition']}：{r['probe_count']}探针中{r['nonzero_price_changed_probes']}个非零价格改变整数动作。")
    totals={}
    for r in business:
        method=r['method'];totals.setdefault(method,{'solver_seconds':0.,'policy_seconds':0.,'solver_calls':0})
        for k in totals[method]:totals[method][k]+=r[k] or 0.
    traininfo=read(target/'qa/training.json')
    lines += ['',f"正式离线九模型训练/验证总墙钟 {sum(r['wall_seconds'] for r in traininfo['rows'])/60:.2f} 分钟；训练1152日+训练验证576日，共41472 MILP调用。模型每轮日志包含累计墙钟，不把重复累计日志相加。", 
        f"正式测试 Pure MC {totals['Pure-MC']['solver_calls']} 次 MILP，solver累计{totals['Pure-MC']['solver_seconds']:.2f}秒；Direct {totals['Direct-PPO']['solver_calls']}次，solver累计{totals['Direct-PPO']['solver_seconds']:.2f}秒，Actor推理累计{totals['Direct-PPO']['policy_seconds']:.2f}秒。Direct包含三模型，不能把总耗时当作单次延迟倍数。", 
        '','business_metrics.csv保留收益、完成订单、载货/空驶/车辆等待成本、积压与取消罚金，并独立复算成本总和。重定位是空驶的子项，不重复加总；本轮重定位关闭。车辆利用率定义为完整决策及冷却窗口内总旅行时长/(车数×结算cutoff)，与载荷利用率load_factor分别报告。不存在额外人工“延误罚金”：原模型的积压/取消罚金保持原定义。完整账本与车辆events均保留。',
        '','## 实证判断与七个问题','']
    positive=[label for label,e in zip(LABELS,effects) if e['simultaneous98_333'][0]>0]
    negative=[label for label,e in zip(LABELS,effects) if e['simultaneous98_333'][1]<0]
    lines += [f"1. 无MC的PPO已完成实际训练和运行：九模型、每模型128日，所有模型Actor/Critic参数发生更新，独立隔离与回放通过。",
        f"2. 经三主比较调整区间支持正向的工况：{positive or '无'}；支持负向的工况：{negative or '无'}。这不是普适算法优劣定理。",
        '3. 各工况改善/恶化的点估计及区间见主表，不合并不同工况收益，也不删除困难场景。',
        '4. 更新8/16/24/32的实际验证曲线全部保留；有限曲线不构成收敛证明，测试不决定增加训练。',
        '5. 三种子全部披露，种子差异是独立训练不确定性，不能用场景数掩盖。',
        '6. 离线训练成本与线上Actor/MILP耗时分开报告；MC表的历史构造成本为沉没成本，此处没有重新构表，不能据此宣称MC离线成本为零。',
        '7. 即使未超过MC，也只说明这个固定预算、尺度、观测与q无关直接价格接口的结果，不否定所有无MC RL。下一研究决策应依据验证趋势、动作响应与种子稳定性，不在看过测试后追加updates追结果。',
        '','本轮优先完成独立替代与正式利润比较，没有拓展联合动作网络、q动作或更复杂机制实验。没有修改历史MC/残差PPO、默认模型，没有提交或推送。',
        '','准备修订：direct_ppo_no_mc_v1 因旧机制数据读取路径误指sealed/而停止，尚未冻结源码、创建训练数据或运行新模型。失败证据保留；正式v2使用真实data/<condition>.json完成隔离检查。修改发生于数值冻结之前，不改模型、尺度或统计规则。预检曾发现小型测试配置省略GAE默认字段，已在训练入口标准化配置，正式全配置无此遗漏；预检失败不与正式样本混合。',
        '','## 证据入口','',
        '- experiment_protocol.md、protocol/preregistration.json：冻结边界及随机身份。',
        '- training/：九模型的完整日、更新日志、恢复包和选定推理模型。',
        '- mc/、evaluation/：Pure MC及Direct完整配对测试，原始美元账本与逐t记录。',
        '- paired_results.csv、summary.csv：逐场景差值及主比较。',
        '- figures/：利润、学习曲线、训练seed差异三组SVG/600dpiPNG与配置。',
        '- qa/、final_manifest.json：真实测试、独立统计/回放、历史SHA、视觉审阅及冻结状态。']
    (target/'direct_ppo_vs_mc_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(target/'qa/analysis.json',dict(status='DATA_PASSED_FIGURES_PENDING',paired_rows=216,
        independent_bootstrap=True,all_validation_selection_recomputed=True))
    print(json.dumps(effects,ensure_ascii=False),flush=True)


def figures(target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    qa=read(target/'qa/analysis.json')
    if qa['status']!='DATA_PASSED_FIGURES_PENDING':raise RuntimeError('no repeat plots')
    effects=read(target/'analysis/results.json')
    with (target/'learning_curves.csv').open(encoding='utf-8-sig',newline='') as f:curves=list(csv.DictReader(f))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name,source):
        for ext in ('png','svg'):fig.savefig(target/'figures'/(name+'.'+ext),dpi=600,bbox_inches='tight')
        write(target/'figures'/(name+'.config.json'),dict(source=source,dpi=600,code_sha256=sha(Path(__file__))))
        plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11,3.6))
    for ax,label,e in zip(axs,LABELS,effects):
        ax.bar([0,1],[e['MC_mean_USD'],e['Direct_mean_USD']],color=['#777777','#0072B2'],hatch='//')
        for x,mean,ci in ((0,e['MC_mean_USD'],e['MC_profit_CI95']),(1,e['Direct_mean_USD'],e['Direct_profit_CI95'])):
            ax.errorbar(x,mean,yerr=[[max(0,mean-ci[0])],[max(0,ci[1]-mean)]],color='black',capsize=3,fmt='none')
        ax.set_xticks([0,1],['Pure MC','Direct PPO']);ax.set_title(label);ax.set_ylabel('Mean full-cooldown profit (USD)')
        # The uncertainty being tested is paired difference, shown explicitly in panel text.
        ax.text(.5,.97,f"Difference: {e['mean_USD']:+.0f} USD\n95% CI [{e['marginal95'][0]:.0f}, {e['marginal95'][1]:.0f}]",ha='center',va='top',transform=ax.transAxes,fontsize=8)
        ax.set_ylim(0,max(e['MC_profit_CI95'][1],e['Direct_profit_CI95'][1])*1.3)
    fig.tight_layout();save(fig,'01-profit','summary.csv')
    fig,axs=plt.subplots(1,3,figsize=(11,3.6))
    for ax,name,label in zip(axs,NAMES,LABELS):
        for i,seed in enumerate(SEEDS):
            group=[r for r in curves if r['condition']==name and int(r['training_seed'])==seed and r['validation_mean_USD']!='']
            ax.plot([int(r['update']) for r in group],[float(r['validation_mean_USD']) for r in group],marker=('o','s','^')[i],ls=('-','--',':')[i],label=str(seed))
        mc=np.mean([read(target/'mc'/f'{name}-validation-s{s}.json')['ledger']['realized_profit'] for s in range(21008,21024)])
        ax.axhline(mc,color='black',ls='-.',label='Pure MC validation');ax.set_title(label);ax.set_xticks([8,16,24,32]);ax.set_xlabel('PPO update');ax.set_ylabel('Validation profit (USD)')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=8);fig.tight_layout(rect=(0,.12,1,1));save(fig,'02-learning','learning_curves.csv + mc validation records')
    fig,axs=plt.subplots(1,3,figsize=(11,3.6))
    for ax,label,e in zip(axs,LABELS,effects):
        ax.bar(range(3),e['seed_mean_USD'],color=['#0072B2','#D55E00','#009E73'],hatch='//');ax.axhline(0,color='black',lw=.8)
        ax.set_xticks(range(3),[str(s) for s in SEEDS]);ax.set_title(label);ax.set_xlabel('Independent training seed');ax.set_ylabel('Mean paired difference (USD)')
    fig.tight_layout();save(fig,'03-seeds','paired_results.csv')
    qa['status']='PASSED';qa['figure_runtime']='repository Matplotlib; bundled Python used for data analysis'
    write(target/'qa/analysis.json',qa)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('data','figures'));p.add_argument('--attempt',required=True)
    a=p.parse_args();globals()[a.stage](Path(a.attempt).resolve())
