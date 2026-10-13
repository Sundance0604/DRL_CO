"""Scientific exports in bundled numpy; drawing only uses existing Matplotlib."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_no_mc_v3'
TARGET=ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v1'
NAMES=('california_corridor-n80-k3','western_multicluster-n40-k8','western_multicluster-n80-k3')
SEEDS=(30001,30002,30003);LABELS=('CA80/3','W40/8','W80/3')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_write(stem,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (TARGET/(stem+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def bootstrap(x):
    x=np.asarray(x,dtype=float);rng=np.random.default_rng(53000);a=rng.integers(0,x.shape[0],(20000,x.shape[0]));b=rng.integers(0,24,(20000,24))
    s=x[a[:,:,None],b[:,None,:]].mean((1,2))
    ca=np.array([np.bincount(v,minlength=x.shape[0]) for v in a])/x.shape[0]
    cb=np.array([np.bincount(v,minlength=24) for v in b])/24
    other=np.einsum('bi,ij,bj->b',ca,x,cb)
    if not np.allclose(s,other,rtol=1e-12,atol=1e-8):raise RuntimeError('bootstrap recalculation mismatch')
    return dict(mean_USD=float(x.mean()),CI95=np.quantile(s,[.025,.975]).tolist(),CI_adjusted=np.quantile(s,[.05/18,1-.05/18]).tolist(),seed_means=x.mean(1).tolist(),crossed_models=x.shape[0],scenarios=24)

def data():
    if (TARGET/'qa/analysis.json').exists() or not (TARGET/'qa/csv-marker.json').exists():raise RuntimeError('marker/repetition gate')
    if read(TARGET/'manifest.json')['status']!='NUMERICAL_COMPLETE_QA_PENDING':raise RuntimeError('numerical gate')
    write(TARGET/'qa/analysis-freeze.json',dict(status='FROZEN_BEFORE_ANALYSIS',source_SHA=sha(Path(__file__))))
    paired=[];summary=[];business=[];diagnostics=[];sensitivity=[]
    for name in NAMES:
        m0=[];mc=[];direct=[]
        cfg=read(PARENT/'protocol'/(name+'.json'))
        for s in range(22020,22044):
            z=read(TARGET/'m0'/f'{name}-s{s}.json');m=read(PARENT/'mc'/f'{name}-test-s{s}.json');m0.append(z['ledger']['realized_profit']);mc.append(m['ledger']['realized_profit'])
            for method,r in [('M0',z),('Pure-MC',m)]:business.append(business_row(name,None,s,method,r,cfg))
        for seed in SEEDS:
            vals=[]
            for i,s in enumerate(range(22020,22044)):
                r=read(PARENT/'evaluation'/f'{name}-{seed}-s{s}.json');v=r['ledger']['realized_profit'];vals.append(v)
                paired.append(dict(condition=name,training_seed=seed,scenario_seed=s,M0_USD=m0[i],MC_USD=mc[i],Direct_USD=v,
                    MC_minus_M0_USD=mc[i]-m0[i],Direct_minus_M0_USD=v-m0[i],Direct_minus_MC_USD=v-mc[i],post_hoc=True))
                business.append(business_row(name,seed,s,'Direct-PPO',r,cfg))
            direct.append(vals)
            d=read(TARGET/'diagnostics'/f'{name}-{seed}.json')
            actors={r['update']:r for r in d['actor']}
            for r in d['critic']:
                row=dict(condition=name,training_seed=seed,**r);a=actors.get(r['update'])
                if a:
                    row.update(D_pi=a['D_pi'],selected=a['selected'],near_far_difference_mean=a['near_far_difference_mean'])
                    for dim in range(10):
                        for metric in ('mean','std','positive_fraction','negative_fraction','log_std','clipped_std'):row[f'action{dim}_{metric}']=a[metric][dim]
                if r['update']==32:row.update({k:v for k,v in d['gradient_diagnostic'].items() if isinstance(v,(float,int,str))})
                diagnostics.append(row)
        x=np.array(direct);m0=np.array(m0);mc=np.array(mc)
        summary.append(dict(condition=name,M0=float(m0.mean()),MC=float(mc.mean()),Direct=float(x.mean()),
            MC_M0=bootstrap((mc-m0)[None,:]),Direct_M0=bootstrap(x-m0[None,:]),Direct_MC=bootstrap(x-mc[None,:])))
    with (PARENT/'paired_results.csv').open(encoding='utf-8-sig',newline='') as f:old=list(csv.DictReader(f))
    for new,prior in zip(paired,old,strict=True):
        if (new['condition'],new['training_seed'],new['scenario_seed'])!=(prior['condition'],int(prior['training_seed']),int(prior['scenario_seed'])) or abs(new['Direct_minus_MC_USD']-float(prior['difference_USD']))>1e-8:raise RuntimeError('parent contrast changed')
    state_summaries=[]
    for path in sorted((TARGET/'sensitivity').glob('state*.json')):
        r=read(path);spec=r['spec'];ref=r['rows'][1];ref_coeff={(c['time'],c['hub'],c['supply_slot']):c['value'] for c in ref['probe']['coefficients']}
        for j,a in enumerate(r['rows']):
            coefficients=a['probe']['coefficients'];diff=[c['value']-ref_coeff[(c['time'],c['hub'],c['supply_slot'])] for c in coefficients]
            row=dict(state_id=path.stem,condition=spec['condition'],training_seed=spec['seed'],scenario_seed=spec['scenario_seed'],
                period=spec['period'],pending=spec['pending'],idle_fraction=spec['idle_fraction'],kind=a['kind'],control_index=j,
                direction=a.get('direction'),alpha=a.get('alpha'),replicate=a.get('replicate'),changed=a['canonical']!=ref['canonical'],
                raw_hash_changed=a['probe']['action_hash']!=ref['probe']['action_hash'],
                price_mean_USD=float(np.mean([c['value'] for c in coefficients])) if coefficients else None,
                price_max_abs_USD=max((abs(c['value']) for c in coefficients),default=0),
                price_change_mean_abs_USD=float(np.mean(np.abs(diff))) if diff else 0,
                objective=a['probe']['plan']['augmented_solver_objective'],booked_operating_USD=a['probe']['plan']['operating_profit'],
                solver_status=a['probe']['plan']['solver']['status'],seconds=a['seconds'],
                accepted_orders=len(a['probe']['plan']['assignments']),routes=len(a['probe']['plan']['routes']))
            row.update({f'action{i}':v for i,v in enumerate(a['signal'])});sensitivity.append(row)
        state_summaries.append(dict(condition=spec['condition'],state_id=path.stem,**{kind:float(np.mean([s['changed'] for s in sensitivity if s['state_id']==path.stem and s['kind']==kind])) for kind in ('zero','stochastic','perturbation')}))
    cases=read(TARGET/'counterfactual/results.json')
    for c in cases:c['future_realized_difference_USD']=c['difference_USD']-c['current_realized_difference_USD']
    csv_write('m0_comparison',paired);csv_write('actor_critic_diagnostics',diagnostics);csv_write('action_sensitivity',sensitivity)
    csv_write('business_metrics',business);csv_write('counterfactual_results',cases if cases else [dict(status='NO_CHANGED_CASE',reason='no canonical switch in bounded states')])
    flat=[dict(condition=r['condition'],M0_USD=r['M0'],MC_USD=r['MC'],Direct_USD=r['Direct'],**{k+'_'+field:value for k in ('MC_M0','Direct_M0','Direct_MC') for field,value in [('mean',r[k]['mean_USD']),('CI95_low',r[k]['CI95'][0]),('CI95_high',r[k]['CI95'][1]),('adjusted_low',r[k]['CI_adjusted'][0]),('adjusted_high',r[k]['CI_adjusted'][1])]}) for r in summary]
    csv_write('comparison_summary',flat)
    write(TARGET/'analysis/results.json',dict(profit=summary,switch=state_summaries,cases=cases))
    report(summary,diagnostics,sensitivity,cases,business)
    write(TARGET/'qa/analysis.json',dict(status='DATA_PASSED',rows=[len(paired),len(diagnostics),len(sensitivity),len(business),len(cases),len(flat)],
        independent_bootstrap=True,parent_contrast_exact=True,source_SHA=sha(Path(__file__))))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

def business_row(name,seed,s,method,r,cfg):
    l=r['ledger'];p=cfg['parameters']
    return dict(condition=name,training_seed=seed,scenario_seed=s,method=method,profit_USD=l['realized_profit'],
        completed_orders=l['completed_orders'],revenue_USD=l['completed_fare'],cost_USD=l['realized_cost'],
        loaded_cost_USD=l['actual_loaded_distance']*p['loaded_cost'],empty_cost_USD=l['actual_empty_distance']*p['empty_cost'],
        backlog_penalty_USD=l['backlog_penalty'],cancellation_penalty_USD=l['cancellation_penalty'],unfinished=l['unfinished_committed'])

def report(summary,diagnostics,sensitivity,cases,business):
    lines=['# Direct PPO 失败机制诊断与下一阶段架构评估','',
        '2026-10-13。本轮不重新训练、不修改Direct/MC/物理模型。复用已评价测试集补M0属于事后诊断；机制使用验证侧公开状态，不能冒称新独立确认性实验。',
        '','## M0告诉我们：战略信号究竟是否有益','',
        '| 工况 | M0 USD | Pure MC USD | Direct USD | Direct−M0 USD（95%区间） | Direct−MC USD |','|---|---:|---:|---:|---|---:|']
    for label,r in zip(LABELS,summary):lines.append(f"| {label} | {r['M0']:.2f} | {r['MC']:.2f} | {r['Direct']:.2f} | {r['Direct_M0']['mean_USD']:.2f} [{r['Direct_M0']['CI95'][0]:.2f},{r['Direct_M0']['CI95'][1]:.2f}] | {r['Direct_MC']['mean_USD']:.2f} |")
    lines+=['','每工况24个相同测试场景、三个Direct训练seed，MC/M0只有一个确定性策略实例。20,000次重抽样，seed53000；Direct对训练seed与场景交叉重抽，MC−M0只按场景重抽。九个事后比较同时列95%和Bonferroni99.444%描述性区间，详见comparison_summary.csv。没有把72个模型—场景差值当72个独立日。旧Direct−MC逐值与原冻结CSV完全一致。',
        '','### 全部训练seed','', '| 工况 | seed | Direct−M0 USD | Direct−MC USD |','|---|---:|---:|---:|']
    for r in summary:
        for i,seed in enumerate(SEEDS):lines.append(f"| {r['condition']} | {seed} | {r['Direct_M0']['seed_means'][i]:.2f} | {r['Direct_MC']['seed_means'][i]:.2f} |")
    lines+=['','### 业务分解','', '| 工况／方法 | 完成订单 | 收入 USD | 成本 USD | 空驶 USD |','|---|---:|---:|---:|---:|']
    for label,name in zip(LABELS,NAMES):
        for method in ('M0','Pure-MC','Direct-PPO'):
            g=[b for b in business if b['condition']==name and b['method']==method];v={k:np.mean([b[k] for b in g]) for k in ('completed_orders','revenue_USD','cost_USD','empty_cost_USD')}
            lines.append(f"| {label}/{method} | {v['completed_orders']:.2f} | {v['revenue_USD']:.2f} | {v['cost_USD']:.2f} | {v['empty_cost_USD']:.2f} |")
    m0text='\n'.join(lines)+'\n';(TARGET/'m0_comparison_report.md').write_text(m0text,encoding='utf-8')
    lines+=['','## Actor告诉我们：变化存在，但不等于适应有效','',
        '每个模型用固定96个验证公开状态（四场景×24t，来自该模型best轨迹）评价全部8/16/24/32checkpoint。参考均值只用这些验证状态；Dpi为十维平均绝对偏离。时间和订单负载影响混合在自然状态分布中，不能将相关性自动解释为有效状态适应。原始分时段／负载点保留在diagnostics/*.json。',
        '','| 工况 | seed | best update | Dpi | 平均绝对near USD | 平均绝对far USD | pooled Critic MC EV | MC RMSE USD |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        for seed in SEEDS:
            d=read(TARGET/'diagnostics'/f'{name}-{seed}.json');a=next(x for x in d['actor'] if x['selected']);v=d['pooled_critic']
            near=np.mean([np.abs(p['near_USD']) for p in a['points']]);far=np.mean([np.abs(p['far_USD']) for p in a['points']])
            lines.append(f"| {name} | {seed} | {a['update']} | {a['D_pi']:.6f} | {near:.2f} | {far:.2f} | {v['critic_MC_EV']:.6f} | {1000*v['critic_MC_RMSE_reward_units']:.2f} |")
    lines+=['','旧MC+PPO机制报告的平均信号偏离约0.000319–0.000964，仅作为描述性参照。旧指标来自MC轨迹/旧测试分布，本轮来自Direct自身验证轨迹；MC残差/直接价格、输入、训练预算与状态分布均不同，不能据比值宣布控制更好。接近零的归一化信号也可能是数百至数千USD，actual查询价格另外保存于探针原始记录和action_sensitivity.csv。',
        '','## Critic告诉我们：收益归因的量纲与证据','',
        '原训练pre-update Critic与完整实际现金return-to-go/1000比较，另与GAE lambda-return比较，不能将两种target混称真实回报。每模型128episode、3072期，EV按原始序列池化仅描述质量，不把时段当独立样本。每update GAE递推独立检查。历史只有混合loss裁剪前总梯度，不能恢复每次Actor/Critic分项或全部中间Critic权重。',
        '','新增冻结最终recovery梯度诊断使用最后4个pre-update32训练日、post-update32权重，分别求policy loss和0.5×value loss梯度，不更新optimizer。不是历史训练梯度；其范数比与cosine只能支持后续训练机制检查，不单独证明梯度干扰导致政策损失。',
        '','| 工况 | seed | 事后Actor梯度范数 | 事后加权Critic梯度范数 | 比值 | cosine |','|---|---:|---:|---:|---:|---:|']
    for name in NAMES:
        for seed in SEEDS:
            g=read(TARGET/'diagnostics'/f'{name}-{seed}.json')['gradient_diagnostic'];lines.append(f"| {name} | {seed} | {g['actor_norm']:.4f} | {g['weighted_critic_norm']:.4f} | {g['weighted_critic_actor_ratio']:.2f} | {g['full_parameter_cosine']:.4f} |")
    lines+=['','## MILP告诉我们：连续信号如何变成整数动作','',
        '54个预选验证状态，覆盖九模型的t4/12/20及公开负载最低/最高，完整保留无空闲等受限状态。每状态zero、det、4个随机样本、3固定方向×0.05/0.2/0.6扰动，共15探针。固定随机seed51000/52000加状态索引，不按利润选方向。所有probe在相同完整状态求解。',
        '','规范化只互换当前物理状态/承诺完全等价的同质车辆，保留每条内部路线、具体订单和分配组合。非等价车辆不可任意去ID。raw hash与规范化业务变化分列；动作变化不等于收益改善。',
        '','| 工况 | zero−det不同 | 随机−det不同 | 扰动−det不同 |','|---|---:|---:|---:|']
    for name in NAMES:
        fields=[]
        for kind in ('zero','stochastic','perturbation'):
            g=[r for r in sensitivity if r['condition']==name and r['kind']==kind];fields.append(f"{sum(r['changed'] for r in g)}/{len(g)} ({100*np.mean([r['changed'] for r in g]):.1f}%)")
        lines.append('| '+name+' | '+' | '.join(fields)+' |')
    lines+=['','比率描述的是预选状态及其有限扰动分布，不是总体探索概率的独立样本置信区间；同一状态内15次求解相关，54状态也嵌套于少数模型和场景。不可比较不同价格下的augmented objective作为真实利润改善。',
        '','## 反事实告诉我们：当期切换与日终结果并不相同','',
        '每模型按冻结顺序选择首个发生规范化业务变化的控制（zero优先，然后随机，再扰动），不是按收益找最优动作。至多9例，共同后续策略是该模型冻结Direct，从t+1恢复，环境隐藏未来完全相同、不进入Actor输入。确定性参考分支必须完整复现旧原始账本，变体全日独立回放。',
        '','| 工况 | seed／scenario／t | 控制 | 当期实际现金差 USD | 后续现金差 USD | 最终利润差 USD |','|---|---|---|---:|---:|---:|']
    for c in cases:lines.append(f"| {c['condition']} | {c['seed']}/{c['scenario_seed']}/{c['period']} | {c['variant']} | {c['current_realized_difference_USD']:.2f} | {c['future_realized_difference_USD']:.2f} | {c['difference_USD']:.2f} |")
    lines+=['','即时booked operating profit另列CSV，不能当已经完成运输的实际现金。精选改变案例及单一已生成未来只有条件性机制意义，不推断所有未来上的平均因果收益。',
        '','## 七个问题与识别边界','']
    for label,r in zip(LABELS,summary):lines.append(f"1. {label}：Direct−M0={r['Direct_M0']['mean_USD']:.2f}USD，95%={r['Direct_M0']['CI95']}，九比较调整={r['Direct_M0']['CI_adjusted']}。方向／区间据此解释，不靠三均值排序证明根因。")
    lines+=['2. Actor确实输出随这些公开状态变化的信号；Dpi只是数值敏感性，未完成固定信号长期对照，不宣布有效适应。',
        '3. 随机与扰动的规范化切换率见上表，同时保留受限状态和所有失败检查；不能仅凭continuous动作维数判断探索有效。',
        '4. 当前动作变化具有或不具有日终经济影响由上述配对反事实逐例说明；案例不能代替完整政策平均优势。',
        '5. H1可用Direct−M0检验当前价格是否比零战略项更差；H3有实际switch证据但无总体阈值分布；H4有logged Critic与独立梯度诊断。H2的q无关是代码事实，表达上限及因果损失仍需公平新模型对照，不能从MC胜出单独识别。',
        '6. 先依据此诊断做训练侧最小检查，再在独立协议比较结构化动作；不得利用22020–22043反复调参，也不默认九模型再次大规模训练。',
        '7. 带q结构有可接入性，但不是保证收益的方案。scheme_b_architecture_proposal.md给最小潜变量PPO与decoder设计、本轮未正式训练。',
        '','## QA与成本边界','',
        '原Direct v3及其全部数值／源／历史输入SHA只读核验；72M0+72zero、pilot与正式反事实完整回放；probe状态不变、OPTIMAL、整数行动现金编译检查；12项预检仅工程阶段，不替代最终相关pytest XML。图表实际查看与全部最终SHA成功才COMPLETED。详细真实gate与耗时见qa/，不虚构未保存历史梯度。',
        '','本轮CSV/SVG/600dpiPNG配套冻结数据、原始账本和config。相关统计使用bundled numpy；绘图仅用仓库现有Matplotlib；未安装库、未提交／推送、未改平台默认、未启动方案B训练。']
    (TARGET/'direct_ppo_failure_diagnosis_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def figures():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    qa=read(TARGET/'qa/analysis.json')
    if qa['status']!='DATA_PASSED':raise RuntimeError('plot order/repeat')
    results=read(TARGET/'analysis/results.json');profit=results['profit']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name,source):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(TARGET/'figures'/(name+'.'+ext),dpi=600,bbox_inches='tight')
        write(TARGET/'figures'/(name+'.config.json'),dict(source=source,analysis_SHA=sha(Path(__file__)),dpi=600));plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11,3.3))
    for ax,label,r in zip(axs,LABELS,profit):
        ax.bar(range(3),[r['M0'],r['MC'],r['Direct']],color=['#999999','#009E73','#0072B2'],hatch='//');ax.set_xticks(range(3),['M0','MC','Direct']);ax.set_title(label);ax.set_ylabel('Mean full-cooldown profit (USD)')
        ax.text(.5,.98,f"Direct - M0: {r['Direct_M0']['mean_USD']:+.0f} USD",transform=ax.transAxes,ha='center',va='top',fontsize=8);ax.set_ylim(0,max(r['M0'],r['MC'],r['Direct'])*1.2)
    save(fig,'01-m0','comparison_summary.csv; CI in comparison_summary.csv')
    fig,axs=plt.subplots(1,3,figsize=(11,3.3))
    for ax,name,label in zip(axs,NAMES,LABELS):
        for i,seed in enumerate(SEEDS):
            d=read(TARGET/'diagnostics'/f'{name}-{seed}.json');ax.plot([x['update'] for x in d['critic']],[x['critic_MC_EV'] for x in d['critic']],label=str(seed),ls=('-','--',':')[i])
        ax.axhline(0,color='black',lw=.7);ax.set_title(label);ax.set_xlabel('PPO update');ax.set_ylabel('Logged Critic EV vs realized RTG');ax.legend(fontsize=7)
    save(fig,'02-critic','actor_critic_diagnostics.csv')
    fig,axs=plt.subplots(1,3,figsize=(11,3.3))
    for ax,name,label in zip(axs,NAMES,LABELS):
        g=[r for r in results['switch'] if r['condition']==name];rates=[np.mean([r[k] for r in g]) for k in ('zero','stochastic','perturbation')]
        ax.bar(range(3),rates,color=['#999999','#0072B2','#D55E00'],hatch='//');ax.set_xticks(range(3),['Zero','Random','Perturbed']);ax.set_ylim(0,1);ax.set_title(label);ax.set_ylabel('Canonical decision switch fraction')
    save(fig,'03-switch','action_sensitivity.csv')
    fig,axs=plt.subplots(1,3,figsize=(11,3.3))
    for ax,name,label in zip(axs,NAMES,LABELS):
        g=[c for c in results['cases'] if c['condition']==name]
        if g:ax.bar(range(len(g)),[c['difference_USD'] for c in g],color='#0072B2',hatch='//');ax.set_xticks(range(len(g)),[str(c['seed']) for c in g])
        else:ax.text(.5,.5,'No changed cases',transform=ax.transAxes,ha='center')
        ax.axhline(0,color='black',lw=.7);ax.set_title(label);ax.set_ylabel('Variant - reference final profit (USD)');ax.set_xlabel('Training seed (one selected case)')
    save(fig,'04-counterfactual','counterfactual_results.csv')
    qa['status']='PASSED';write(TARGET/'qa/analysis.json',qa)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('data','figures'));a=p.parse_args();globals()[a.stage]()
