"""Paired inference, seven publication figures and reproducible research report."""
import argparse,csv,math,zipfile,logging,os,platform
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t,skew
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
logging.getLogger('fontTools').setLevel(logging.WARNING)
from experiment_core.storage import atomic_json,read_json,digest
from experiments.cfs_rho_endofday.run_experiment import write_csv,file_hash,ROOT
from .common import DEFAULT_OUTPUT

ALIASES={'california_corridor':'California','northeast_cluster':'Northeast','western_multicluster':'Western','la_sf':'LA-SF'}
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
MARKERS=['o','s','^','D']
METRICS=['realized_profit','service_rate','actual_empty_distance','empty_pickup_distance','completed_fare',
         'deliberate_deferral_rate','mean_deliberate_deferral_periods','deferral_success_rate',
         'mean_RH_solve_time','p95_RH_solve_time','runtime_seconds','solver_time','actual_order_waiting',
         'actual_vehicle_waiting','average_occupancy','cross_hub_pickup_rate']
METRICS += ['actual_loaded_distance','cancellation_rate','pending_unassigned','deliberate_deferral_count',
            'orders_eventually_served_after_deferral','orders_lost_after_deferral','max_RH_solve_time',
            'RH_solve_count','max_variable_count','max_constraint_count','max_window_gap',
            'orders_per_vehicle','requested_load_per_fleet_capacity']


def read_rows(path):
    with Path(path).open(encoding='utf8',newline='') as f:raw=list(csv.DictReader(f))
    categorical={'run_id','stage','region','rho_label','policy','info_mode','status','data_hash','scenario_hash',
                 'spec_hash','code_hash','accounting_version','solver_version','error','error_code'}
    result=[]
    for row in raw:
        result.append({k:(v if k in categorical else None if v=='' else float(v)) for k,v in row.items()})
    return result


def statistics(values,seed=20261005,bootstrap=10000):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('missing/nonfinite statistical sample')
    n=len(a);mean=float(a.mean());sd=float(a.std(ddof=1)) if n>1 else None
    half=float(student_t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
    asymmetry=float(skew(a,bias=False)) if n>2 and sd and sd>1e-12 else 0.
    row=dict(n=n,mean=mean,SD=sd,median=float(np.median(a)),CI_low=mean-half if half is not None else None,
             CI_high=mean+half if half is not None else None,skew=asymmetry,bootstrap_low=None,bootstrap_high=None)
    if abs(asymmetry)>1:
        rng=np.random.default_rng(seed);means=np.mean(a[rng.integers(0,n,size=(bootstrap,n))],axis=1)
        row.update(bootstrap_low=float(np.quantile(means,.025)),bootstrap_high=float(np.quantile(means,.975)))
    return row


def condition_key(r):return (r['region'],int(r['orders']),int(r['fleet']),r['rho_label'],r['policy'],int(r['H']))
def regime(r):return (r['region'],int(r['orders']),int(r['fleet']))
def tag(key):return f'{ALIASES[key[0]]} N={key[1]}, K={key[2]}'


def summarize(rows,config):
    groups=defaultdict(list)
    for r in rows:
        if r['status']!='COMPLETED':raise RuntimeError('failed runs retained: do not silently infer from survivors')
        groups[condition_key(r)].append(r)
    summaries=[]
    for key,group in sorted(groups.items()):
        if len({r['seed'] for r in group})!=len(group):raise RuntimeError('seed duplication')
        for metric in METRICS:
            measured=[r[metric] for r in group if r.get(metric) is not None]
            if not measured:continue
            s=statistics(measured,config['bootstrap_seed'],config['bootstrap_replications'])
            summaries.append(dict(region=key[0],orders=key[1],fleet=key[2],rho_label=key[3],policy=key[4],H=key[5],
                                  metric=metric,missing_sample_count=len(group)-len(measured),**s))
    return summaries,groups


def paired(groups,config,kind,reg,rho,a,H,b,bH,bRho=None):
    one=groups[(*reg,rho,a,H)];two=groups[(*reg,bRho or rho,b,bH)]
    A={int(r['seed']):r for r in one};B={int(r['seed']):r for r in two}
    if set(A)!=set(B):raise RuntimeError('unpaired seed sets')
    for seed in A:
        if A[seed]['scenario_hash']!=B[seed]['scenario_hash']:raise RuntimeError('pair uses different frozen scenario')
    result=[]
    for metric in ['realized_profit','service_rate','actual_empty_distance','empty_pickup_distance','deliberate_deferral_rate','actual_order_waiting']:
        diffs=[A[s][metric]-B[s][metric] for s in sorted(A)]
        result.append(dict(comparison=kind,region=reg[0],orders=reg[1],fleet=reg[2],rho_label=rho,H=H,
                           policy_a=a,policy_b=b,rho_b=bRho or rho,metric=metric,
                           **statistics(diffs,config['bootstrap_seed'],config['bootstrap_replications'])))
    return result


def compute_pairs(groups,config,pilot=False):
    result=[]
    regimes=sorted({k[:3] for k in groups})
    for reg in regimes:
        result+=paired(groups,config,'cross_hub_myopic_gain',reg,'rho_max','myopic',0,'myopic',0,'rho0')
        if pilot:continue
        for rho in config['formal_rho_labels']:
            for H in config['horizons']:
                result+=paired(groups,config,'known_supply_value',reg,rho,'known_supply',H,'myopic',0)
                result+=paired(groups,config,'oracle_demand_preview_value',reg,rho,'oracle_preview',H,'known_supply',H)
                result+=paired(groups,config,'oracle_preview_total_lookahead_gain',reg,rho,'oracle_preview',H,'myopic',0)
                if rho=='rho_max':
                    for policy in ['known_supply','oracle_preview']:
                        result+=paired(groups,config,'wide_rho_recovery_vs_own_myopic',reg,rho,policy,H,'myopic',0,'rho0')
    return result


def validate_raw(output,rows,manifest):
    """Independent read-only trace audit, not inferred from completion status."""
    datasets={};steps=0;decisions=0;files={}
    for index,r in enumerate(rows,1):
        dataset_name=f'{r["region"]}-n{int(r["orders"])}-k{int(r["fleet"])}-s{int(r["seed"])}.dataset.json'
        if dataset_name not in datasets:datasets[dataset_name]=read_json(output/'raw/datasets'/dataset_name)
        payload=datasets[dataset_name];scenario=payload['scenarios'][0]
        if digest(payload)!=r['data_hash'] or digest(scenario)!=r['scenario_hash']:
            raise RuntimeError('frozen dataset hash mismatch')
        by_id={o['id']:o for o in scenario['orders']}
        path=output/'raw/runs'/r['run_id']
        if digest(read_json(path/'spec.resolved.json'))!=r['spec_hash']:
            raise RuntimeError('run spec hash mismatch')
        if r['code_hash']!=manifest['source_code_digest']:raise RuntimeError('mixed execution source')
        trace_path=path/'trace.json';trace=read_json(trace_path)[0]['steps'];assigned=set()
        for s in trace:
            steps+=1;t=s['period']
            if not all(s['invariants'].values()):raise RuntimeError('physical trace invariant')
            for order in s['before']['pool']:
                oid=order['id'] if isinstance(order,dict) else order
                if by_id[oid]['book_time']>t:raise RuntimeError('future order leaked into physical pool')
            for a in s['plan']['assignments']:
                oid=a['order_id']
                if by_id[oid]['book_time']>t or oid in assigned:raise RuntimeError('early or duplicate assignment')
                assigned.add(oid)
            if s.get('drain'):
                if s['plan']['assignments']:raise RuntimeError('new assignment after decision horizon')
                continue
            decisions+=1;w=s['plan']['solver'].get('window')
            if r['policy']=='myopic':continue
            if w['commit_end']!=t+1 or w['window_start']!=t:raise RuntimeError('not one-period receding control')
            if w['information_mode']!=r['policy']:raise RuntimeError('mislabelled information mode')
            if any(by_id[o]['book_time']>t for o in w['visible_orders']):raise RuntimeError('future base visibility')
            preview=w['preview_order_ids']
            if r['policy']=='known_supply' and preview:raise RuntimeError('future demand in supply planner')
            expected={o['id'] for o in scenario['orders'] if t<o['book_time']<=t+int(r['H'])}
            if r['policy']=='oracle_preview' and set(preview)!=expected:raise RuntimeError('preview interval mismatch')
        if len(assigned)!=r['assigned']:raise RuntimeError('assignment trace count mismatch')
        if abs(r['realized_profit']-(r['completed_fare']-r['realized_cost']))>1e-6:
            raise RuntimeError('realized profit identity mismatch')
        files[r['run_id']]=file_hash(trace_path)
        if index%250==0:print(f'Trace audit: {index}/{len(rows)}',flush=True)
    audit=dict(status='PASSED',runs=len(rows),steps=steps,decision_steps=decisions,
               assertions=['frozen data/spec/source hashes','physical invariants','booking visibility',
                           'no early or duplicate assignment','supply has no preview','exact oracle interval',
                           'one-period commit','no cooldown new assignment','realized profit identity'],trace_sha256=files)
    atomic_json(output/'processed/trace_audit.json',audit)
    return {k:v for k,v in audit.items() if k!='trace_sha256'}


def diagnostics(output,rows,config):
    """Small-sample diagnostics, explicitly excluded from formal inference."""
    from experiment_core.plugins import resolve
    from model_families.single.data import matching_state
    from model_families.single.runner import state_snapshot
    from model_families.single.planning import compile_action
    from model_families.single.rolling.controller import observed_state
    from .case_study import action_of
    lookup={(regime(r),r['rho_label'],r['policy'],int(r['H']),int(r['seed'])):r for r in rows}
    result=[];checks=[]
    for stage,reference,reference_H in [('H1_diagnostic','myopic',0),('full_preview','oracle_preview',8)]:
        sample=read_rows(output/(stage+'_results.csv'))
        expected=len({regime(r) for r in rows})*len(config['formal_rho_labels'])*len(config['full_preview_seeds'])
        if len(sample)!=expected or any(r['status']!='COMPLETED' for r in sample):
            raise RuntimeError('incomplete diagnostic matrix')
        for r in sample:
            b=lookup[(regime(r),r['rho_label'],reference,reference_H,int(r['seed']))]
            if r['scenario_hash']!=b['scenario_hash']:raise RuntimeError('diagnostic scenario mismatch')
            result.append(dict(stage=stage,region=r['region'],orders=r['orders'],fleet=r['fleet'],
                rho_label=r['rho_label'],seed=r['seed'],policy=r['policy'],H=r['H'],
                reference_policy=reference,reference_H=reference_H,realized_profit=r['realized_profit'],
                reference_profit=b['realized_profit'],paired_profit_difference=r['realized_profit']-b['realized_profit'],
                runtime_seconds=r['runtime_seconds'],solver_time=r['solver_time'],
                limited_window_count=r['limited_window_count'],max_window_gap=r['max_window_gap']))
            if stage!='H1_diagnostic':continue
            path=output/'raw/runs'/r['run_id'];spec=resolve(read_json(path/'spec.resolved.json'))
            payload=read_json(output/'raw/datasets'/f'{r["region"]}-n{int(r["orders"])}-k{int(r["fleet"])}-s{int(r["seed"])}.dataset.json')
            sim,orders=matching_state(payload['scenarios'][0],spec.model.parameters,spec.solver.parameters.model_dump())
            by_t=defaultdict(list)
            for o in orders.values():by_t[o.book_time].append(o)
            for step in read_json(path/'trace.json')[0]['steps']:
                if step.get('drain'):continue
                sim.begin_period(by_t[step['period']])
                if digest(state_snapshot(sim))!=step['plan']['state_hash']:raise RuntimeError('H1 replay state mismatch')
                observed=observed_state(sim)
                p=sim.parameters
                one=observed.solve_plan(p['cross_hub'],p['empty_cost'],p['reposition'],None,{})
                actual=compile_action(sim,action_of(step['plan']),{})
                checks.append(dict(run_id=r['run_id'],period=step['period'],
                    RH1_current_objective=actual[1],myopic_same_state_objective=one[1],
                    objective_difference=actual[1]-one[1],
                    same_assignment_set=sorted(actual[2]['x'])==sorted(one[2]['x'])))
                sim.commit_plan(actual)
    write_csv(output/'processed/diagnostics.csv',result)
    write_csv(output/'processed/H1_same_state_objectives.csv',checks)
    text=['### H=1 与全日预览：独立诊断，不计入正式CI','',
          '每工况/半径只使用两个既定seed 1010、1011。以下是小样本描述，不用作总体显著性结论。',
          '', '| 诊断 | 工况 | rho | mean配对profit差 | 两seed差值范围 |','|---|---|---|---:|---:|']
    for stage in ['H1_diagnostic','full_preview']:
        for reg in sorted({regime(r) for r in rows}):
            for rho in config['formal_rho_labels']:
                d=[r['paired_profit_difference'] for r in result if r['stage']==stage and regime(r)==reg and r['rho_label']==rho]
                label='Supply H1 − Myopic' if stage=='H1_diagnostic' else 'oracle_preview full − H8'
                text.append(f'| {label} | {tag(reg)} | {rho} | {np.mean(d):+.2f} | [{min(d):+.2f}, {max(d):+.2f}] |')
    gap=max(abs(r['objective_difference']) for r in checks)
    text += ['',f"两类诊断合计{len(result)}次run，非OPTIMAL可行窗口数={int(sum(r['limited_window_count'] for r in result))}，最大window gap={max(r['max_window_gap'] for r in result):.6g}。",'',
             f'H1逐期真实前状态重放：{len(checks)}期；相同状态下H1与原myopic augmented objective的最大绝对差={gap:.8g}。',
             '单期最优解可能非唯一；H1采用不同变量/约束顺序，最优目标一致不保证选同一车辆、延迟或路线。此类tie会使后续状态分叉，故不能从H1全天差值直接断言信息泄漏。逐期目标检查详见processed/H1_same_state_objectives.csv。']
    if gap>1e-4:text+=['**注意：本矩阵存在超过数值容差的同状态目标差异；H1不能被宣称与原myopic完全等价，需要按逐期记录进一步定位。**']
    return '\n'.join(text)


def figures(output,report,summaries,pilot_pairs,pairs,representatives):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
                         'legend.fontsize':6.5,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600})
    target=report/'figures';target.mkdir(parents=True,exist_ok=True)
    S={(r['region'],r['orders'],r['fleet'],r['rho_label'],r['policy'],r['H'],r['metric']):r for r in summaries}
    P={(r['comparison'],r['region'],r['orders'],r['fleet'],r['rho_label'],r['policy_a'],r['H'],r['metric']):r for r in pairs}
    regs=[regime(r) for r in representatives]
    config=dict(renderer='matplotlib',font='DejaVu Serif',width_inches=7.2,PNG_dpi=600,formats=['png','pdf','svg'],
                colors=COLORS,markers=MARKERS,uncertainty='95% Student-t CI; paired differences are computed per seed',
                inputs={n:file_hash(output/n) for n in ['regime_map_results.csv','rolling_results.csv','paired_differences.csv','summary.csv']})
    def save(fig,name):
        for ext in config['formats']:fig.savefig(target/(name+'.'+ext),dpi=600)
        plt.close(fig)
    def curve(ax,x,stats,color,marker,label,ls='-'):
        ax.plot(x,[s['mean'] for s in stats],color=color,marker=marker,linestyle=ls,label=label,markersize=3)
        ax.fill_between(x,[s['CI_low'] for s in stats],[s['CI_high'] for s in stats],color=color,alpha=.12)
        if list(x)==[2,4,8]:ax.set_xticks([2,4,8])
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.0),constrained_layout=True)
    for ax,region in zip(axes,['california_corridor','northeast_cluster','western_multicluster']):
        for i,K in enumerate([3,5,8]):
            r=sorted([r for r in pilot_pairs if r['region']==region and r['fleet']==K and r['metric']=='realized_profit'],key=lambda r:r['orders'])
            curve(ax,[s['orders']/K for s in r],r,COLORS[i],MARKERS[i],f'K={K}')
        ax.axhline(0,color='.5',lw=.7);ax.set_title(ALIASES[region]);ax.set_xlabel('Orders / vehicle');ax.legend()
    axes[0].set_ylabel('Wide rho - own myopic profit');save(fig,'01_myopic_regime_map')
    def panels(title):
        fig,axes=plt.subplots(math.ceil(len(regs)/2),2,figsize=(7.2,2.4*math.ceil(len(regs)/2)),squeeze=False,constrained_layout=True)
        for ax in axes.flat[len(regs):]:ax.set_visible(False)
        return fig,list(axes.flat)[:len(regs)]
    fig,axes=panels('profit')
    for ax,reg in zip(axes,regs):
        base=S[(*reg,'rho_max','myopic',0,'realized_profit')]
        ax.axhline(base['mean'],color='.4',ls=':',label='Myopic')
        for policy,color,marker in [('known_supply',COLORS[0],'o'),('oracle_preview',COLORS[1],'^')]:
            curve(ax,[2,4,8],[S[(*reg,'rho_max',policy,h,'realized_profit')] for h in [2,4,8]],color,marker,
                  'RH-Supply' if policy=='known_supply' else 'oracle_preview',ls='-' if policy=='known_supply' else '--')
        ax.set_title(tag(reg));ax.set_xlabel('H (decision periods)');ax.set_ylabel('Realized profit');ax.legend()
    save(fig,'02_horizon_realized_profit')
    fig,axes=panels('decomposition')
    for ax,reg in zip(axes,regs):
        for kind,policy,color,marker,label in [('known_supply_value','known_supply',COLORS[0],'o','Known-supply gain'),
                                               ('oracle_demand_preview_value','oracle_preview',COLORS[1],'^','oracle_preview increment')]:
            curve(ax,[2,4,8],[P[(kind,*reg,'rho_max',policy,h,'realized_profit')] for h in [2,4,8]],color,marker,label)
        ax.axhline(0,color='.5',lw=.7);ax.set_title(tag(reg));ax.set_xlabel('H');ax.set_ylabel('Paired profit difference');ax.legend()
    save(fig,'03_future_information_decomposition')
    failure=next(regime(r) for r in representatives if 'failure' in r['roles'])
    comparisons=[('rho0','myopic',0),('rho_max','myopic',0)]+[('rho_max',p,h) for p in ['known_supply','oracle_preview'] for h in [2,4,8]]
    values=[S[(*failure,r,p,h,'realized_profit')] for r,p,h in comparisons]
    fig,ax=plt.subplots(figsize=(7.2,3.0),constrained_layout=True)
    ax.errorbar(range(8),[r['mean'] for r in values],yerr=np.array([[r['mean']-r['CI_low'] for r in values],[r['CI_high']-r['mean'] for r in values]]),fmt='o',color=COLORS[0],capsize=3)
    ax.set_xticks(range(8),['M own','M wide','Supply 2','Supply 4','Supply 8','oracle_preview 2','oracle_preview 4','oracle_preview 8'],rotation=25)
    ax.set_ylabel('Realized profit (95% CI)');ax.set_title(tag(failure));save(fig,'04_freedom_recovery')
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.8),constrained_layout=True)
    for row,policy in enumerate(['known_supply','oracle_preview']):
        for ax,metric,label in zip(axes[row],['deliberate_deferral_rate','mean_deliberate_deferral_periods','deferral_success_rate'],['Deferral rate','Mean deferred periods','Deferral success']):
            for i,reg in enumerate(regs):
                vals=[S.get((*reg,'rho_max',policy,h,metric)) for h in [2,4,8]]
                xs=[h for h,s in zip([2,4,8],vals) if s is not None]
                ax.plot(xs,[s['mean'] for s in vals if s is not None],color=COLORS[i%4],marker=MARKERS[i%4],markersize=3,label=tag(reg))
                ci=[(h,s) for h,s in zip([2,4,8],vals) if s is not None and s['CI_low'] is not None]
                if ci:ax.fill_between([h for h,s in ci],[s['CI_low'] for h,s in ci],[s['CI_high'] for h,s in ci],color=COLORS[i%4],alpha=.08)
            ax.set_xlabel('H');ax.set_xticks([2,4,8]);ax.set_title(('RH-Supply' if row==0 else 'oracle_preview')+'\n'+label)
        axes[row,0].legend(fontsize=5.5)
    save(fig,'05_deferral_behavior')
    fig,ax=plt.subplots(figsize=(7.2,3.4),constrained_layout=True)
    for i,reg in enumerate(regs):
        xy=[(S[(*reg,'rho_max',p,h,'actual_empty_distance')]['mean'],S[(*reg,'rho_max',p,h,'service_rate')]['mean'])
            for p,h in [('myopic',0),('known_supply',8),('oracle_preview',8)]]
        ax.plot([x for x,y in xy],[y for x,y in xy],color=COLORS[i%4],lw=1,ls=['-','--','-.',':'][i%4],label=tag(reg))
        for (x,y),marker,label in zip(xy,['s','o','^'],['M','S8','P8']):
            ax.scatter(x,y,color=COLORS[i%4],marker=marker,s=20)
            ax.annotate(label,(x,y),xytext=(3,-9 if label=='M' else 5),textcoords='offset points',fontsize=6)
    ax.set_xlabel('Mean empty vehicle miles');ax.set_ylabel('Mean service rate')
    ax.set_title('M = Myopic; S8 = RH-Supply H8; P8 = oracle_preview H8',fontsize=8)
    ax.legend();save(fig,'06_empty_miles_service_rate')
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.1),constrained_layout=True)
    for ax,metric,label,scale in zip(axes,['mean_RH_solve_time','p95_RH_solve_time','solver_time'],['Mean window solve (ms)','Mean of run p95 (ms)','Total daily solver time (s)'],[1000,1000,1]):
        for i,reg in enumerate(regs):
            for policy,ls in [('known_supply','-'),('oracle_preview','--')]:
                ax.plot([2,4,8],[scale*S[(*reg,'rho_max',policy,h,metric)]['mean'] for h in [2,4,8]],
                        color=COLORS[i%4],marker=MARKERS[i%4],markersize=3,ls=ls,label=tag(reg)+(' S' if ls=='-' else ' oracle'))
        ax.set_xlabel('H');ax.set_xticks([2,4,8]);ax.set_ylabel(label)
    handles=[plt.Line2D([],[],color=COLORS[i],marker=MARKERS[i],label=tag(reg)) for i,reg in enumerate(regs)]
    handles += [plt.Line2D([],[],color='.2',ls='-',label='RH-Supply'),plt.Line2D([],[],color='.2',ls='--',label='oracle_preview')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=6.5)
    save(fig,'07_runtime_scaling')
    atomic_json(output/'processed/plotting_config.json',config)


def report_text(manifest,reps,pilot_pairs,pairs,rows,case,increments,diagnostic_text):
    lines=['# 第二次数值仿真：压力分区与动态前视基准','',
           '所有收益均为 full-cooldown realized profit。RH-Supply 是合法在线信息；**oracle_preview 仅用于未来订单信息放松诊断，不能称可部署策略**。',
           '',f"压力扫描：{manifest['regime_map']['run_count']} 次运行，10 个 paired seeds。正式动态矩阵：{manifest['rolling']['run_count']} 次，30 个独立于筛选 pilot 的连续 seeds 1010–1039。失败不删除。",
           '', '## 1. 基线、数据与压力设计','',
           'Phase 0：40次关键基线精确复现旧归档利润，California wide-own +2118.83、Western −275.07，rho0仅own hub；31项前置测试通过。源快照与hash在 phase0/。旧实验输出不覆盖。',
           '', '同一 CFS adapter、原 WGT_FACTOR 加权无放回抽样、双向OD配平与原坐标/时间/运费假设。CFS货值不是运费，时间不是观测时间，network不是实际公路。不同N是不同实验条件，不是增加replication。',
           '', 'N∈{20,40,80}，K∈{3,5,8}，rho0 / rho_mid / rho_max 取旧报告距离levels，mid取正levels下中位（旧50%分位代表值）。初始车位按完整eligible CFS的outbound WGT_FACTOR总和排序；K3取前三、K5每hub一车、K8再补前三，tie按hub ID，无额外随机源。与旧fleet ID seed-shuffle区分，Phase0复现仍用原冻结fleet。',
           '', 'LA–SF sanity另运行30次（10个paired seeds、3个rho标签），两个正rho的结构重复不当独立样本。与rho0的利润差只在约1e-11数值误差量级，符合两hub sanity预期。',
           '', '## 2. 算法与信息边界','',
           '复用独立 single rolling 模块：联合车辆时空流、路线宏动作、订单和容量；传播当前commitment的future handoff/release，允许等待后匹配；k=1逐期重规划。V=0，无ValueProvider、fluid、学习、RL或multi-backbone。原myopic solver目标不改。',
           '', '正式进程固定PYTHONHASHSEED=0、Gurobi Seed=0、Threads=1，避免集合约束顺序随worker变化。Pilot默认hash seed，仅作预算验证，不并入正式CI；manifest记录这一区别。',
           '', 'H沿用平台的决策期数：[t,min(t+H,T))。这与任务文档闭区间记号有一个边界差别：H=1是当前单期；preview仍揭示(t,t+H]，但恰在t+H预订的订单没有本窗口决策slot，不得提前分配/接货。未来未booking订单只传给专用窗口接口，绝不提前塞给当前M_t或真实pool。未来batch ready不偷看，仍用当前可见上界。',
           '', '模型限制：窗口内部把已选路线作为宏动作执行到端点，中间额外接单/延长在下一次真实replan重考虑。故它不是所有逐期M_t动作序列的精确展开，oracle_preview也不是全局最优上界。长路线全段费用和真实deadline计入；T后仅冷却执行承诺，不优化。',
           '', 'known_supply−myopic是“相同在线信息下改用多期规划”的净执行增益，包含当前backlog延期、当前路线联合安排与由这些决策产生的未来车辆复用，并非把外生incoming supply单独作因果识别。若要只识别pipeline信息本身，需要额外的信息屏蔽消融，本轮不虚构这个分解。',
           '', 'Deferral：当前至少存在一个满足pickup/endpoint锚点与逐弧容量的车辆/订单coalition，但本期未分配且未取消，标deliberate_deferral；另分no_vehicle、infeasible_unassignment、cancellation。计数=唯一订单、rate分母=arrivals，periods为累计故意暂缓期数。future_supply_available保守地指原承诺最终idle release可服务该订单；所有中间hub arrivals另完整记录，不能将中途受约束车辆误当自由idle。',
           '', '每run的mean deferral periods先按该run暂缓订单平均，无暂缓订单的run定义为0，再对run等权汇总。Deferral success rate没有分母时为不适用，统计时保留missing_sample_count，不以0补值，也不与跨seed合并分子的pooled ratio混称。',
           '', '## 3. 统计口径与筛选','',
           '每条件分别报告mean、sample SD、median、Student-t95% CI。差值先按相同seed、相同scenario hash配对；|skew|>1追加固定seed的10000次paired bootstrap percentile CI。Pilot选择最大正向gain、Western最负gain、tight(80/3)、loose(20/8)，重复合并；formal使用新30 seeds，避免筛选样本重复当验证样本。选择规则、角色和pilot gain在representative_regimes.json。',
           '', '| 角色 | 工况 | pilot wide-own gain |','|---|---|---:|']
    for r in reps:lines.append(f"| {', '.join(r['roles'])} | {tag(regime(r))} | {r['pilot_gain']:.2f} |")
    lines+=['','## 4. 正式配对结果（wide rho）','',
            '| 工况 | H | 已知供给增益 [t95% CI] | Oracle需求信息增量 [t95% CI] | Supply wide − Myopic own [t95% CI] |','|---|---:|---:|---:|---:|']
    def fmt(r):return f"{r['mean']:+.2f} [{r['CI_low']:+.2f}, {r['CI_high']:+.2f}]"
    lookup={(r['comparison'],regime(r),r['rho_label'],r['policy_a'],r['H']):r for r in pairs if r['metric']=='realized_profit'}
    baseline_gaps=[lookup[('cross_hub_myopic_gain',regime(rep),'rho_max','myopic',0)] for rep in reps]
    for rep in reps:
        reg=regime(rep)
        for H in [2,4,8]:
            a=lookup[('known_supply_value',reg,'rho_max','known_supply',H)]
            b=lookup[('oracle_demand_preview_value',reg,'rho_max','oracle_preview',H)]
            c=lookup[('wide_rho_recovery_vs_own_myopic',reg,'rho_max','known_supply',H)]
            lines.append(f'| {tag(reg)} | {H} | {fmt(a)} | {fmt(b)} | {fmt(c)} |')
    lines+=['','独立formal seeds的myopic wide−own差值（角色按pilot定义，不强迫formal重现同方向）：',
            '', '| 工况 | formal Myopic wide−own [t95% CI] |','|---|---:|']
    for r in baseline_gaps:lines.append(f'| {tag(regime(r))} | {fmt(r)} |')
    lines+=['','H8直接配对的freedom recovery（wide RH−wide Myopic），两部分区间不能直接相加：',
            '', '| 工况 | Supply recovery [t95% CI] | oracle_preview recovery [t95% CI] |','|---|---:|---:|']
    for rep in reps:
        reg=regime(rep)
        a=lookup[('known_supply_value',reg,'rho_max','known_supply',8)]
        b=lookup[('oracle_preview_total_lookahead_gain',reg,'rho_max','oracle_preview',8)]
        lines.append(f'| {tag(reg)} | {fmt(a)} | {fmt(b)} |')
    lines+=['','H8主要差值的完整统计口径（不是混合参数条件的样本）：',
            '', '| 工况 | 差值 | n | mean | sample SD | median | Student-t95% CI | bootstrap95% CI |',
            '|---|---|---:|---:|---:|---:|---|---|']
    for rep in reps:
        reg=regime(rep)
        for kind,policy,label in [('known_supply_value','known_supply','Supply−Myopic'),
                                  ('oracle_demand_preview_value','oracle_preview','oracle_preview−Supply')]:
            r=lookup[(kind,reg,'rho_max',policy,8)]
            boot=f"[{r['bootstrap_low']:+.2f}, {r['bootstrap_high']:+.2f}]" if r['bootstrap_low'] is not None else '未触发偏态规则'
            lines.append(f"| {tag(reg)} | {label} | {r['n']} | {r['mean']:+.2f} | {r['SD']:.2f} | {r['median']:+.2f} | [{r['CI_low']:+.2f}, {r['CI_high']:+.2f}] | {boot} |")
    lines+=['','## 5. 研究问题逐项回答','',
            'Q1（压力是否主因）：逐区域压力map见图1与regime_map_paired.csv；以下仅描述各条件的均值，不把27个不同条件当独立场景混算CI。','']
    for region in ['california_corridor','northeast_cluster','western_multicluster']:
        g=[r for r in pilot_pairs if r['region']==region and r['metric']=='realized_profit']
        tight=next(r for r in g if r['orders']==80 and r['fleet']==3);loose=next(r for r in g if r['orders']==20 and r['fleet']==8)
        lines.append(f"- {ALIASES[region]}：9个压力条件中{sum(r['mean']<0 for r in g)}个mean gain负；tight {fmt(tight)}，loose {fmt(loose)}。供需压力与空间配置同时变化，不能仅凭这个map断言因果。")
    lines+=['','本轮不支持“failure主要集中于tight supply”的概括：三个区域80/3的pilot平均gain均非负；所选Western最负pilot为40/8（5 orders/vehicle），不是80/3（26.67 orders/vehicle）。且formal未稳定重现该负向差值，空间配置/参数/场景交互不能用单一N/K解释。',
            '', 'Q2/Q3（信息分解）：上表分别给known_supply−myopic、oracle_preview−known_supply。有限H、零terminal value和宏动作限制下，后者可能为负；它是同一规划器不同信息的执行差值，不是信息理论非负价值定理。',
            '', 'H8下，Western20/8和40/8的在线Supply增益CI下界>0，80/3则跨0；三个Western工况的oracle额外增益都显著高于0（逐项marginal区间）。因此仅known-supply结构不能解释本矩阵全部动态改善，但oracle潜力不能当作可预测、可学习或可部署的保证。',
            '', 'Q4（Western是否修复）：']
    failure=next(r for r in reps if 'failure' in r['roles']);reg=regime(failure)
    base_gap=lookup[('cross_hub_myopic_gain',reg,'rho_max','myopic',0)]
    lines.append(f"该pilot-selected failure工况在新formal seeds下Myopic wide−own={fmt(base_gap)}。"+
                 ('formal平均负向重现，但CI是否完全负向需以区间判断。' if base_gap['mean']<0 else
                  'formal平均不再负向；本轮不能声称已在独立样本可靠复现负向failure。'))
    lines.append('')
    for H in [2,4,8]:
        c=lookup[('wide_rho_recovery_vs_own_myopic',reg,'rho_max','known_supply',H)]
        conclusion=('CI下界大于0，支持超过own对照；'+('且修复该工况的平均负向差值。' if base_gap['mean']<0 else '但formal myopic均值本身未退化，不称修复负向failure。')) if c['CI_low']>0 else 'CI包含或低于0，不能声称已可靠超过own对照。'
        lines.append(f"- H={H}：Supply wide相对Myopic own为{fmt(c)}；"+conclusion)
    lines+=['','Q5（是否H2/4饱和）：不预设单调或饱和。H4−H2、H8−H4的配对统计保存于horizon_increments.csv。H4/H8增益比例仅在H8平均增益>0时有描述意义；超过100%可能说明非单调，不表示精确等价。',
            '', '| 工况 | Supply H4 / H8 mean gain | Supply H8−H4 [t95% CI] | Oracle H8−H4 [t95% CI] |','|---|---:|---:|---:|']
    for rep in reps:
        reg=regime(rep)
        gain4=lookup[('known_supply_value',reg,'rho_max','known_supply',4)]['mean']
        gain8=lookup[('known_supply_value',reg,'rho_max','known_supply',8)]['mean']
        ratio=f'{gain4/gain8:.1%}' if gain8>0 else 'n.a.'
        inc=[next(r for r in increments if regime(r)==reg and r['rho_label']=='rho_max' and r['policy_a']==policy and r['H']==8 and r['metric']=='realized_profit') for policy in ['known_supply','oracle_preview']]
        lines.append(f'| {tag(reg)} | {ratio} | {fmt(inc[0])} | {fmt(inc[1])} |')
    lines+=['','有限H、终端V=0和后续重规划下，增大H并不保证全天利润单调；CI跨0不能证明饱和或等价。',
            '', 'Q6（机制）：下面是Supply H8相对myopic wide的配对变化。服务率和空驶应联合解释；暂缓是按可行性记录的operational行为，不证明优化器有意将每个订单安排到未来，低价值拒绝也可能在取消前被计入该口径。',
            '', '| 工况 | Δservice rate [CI] | Δempty miles [CI] | Δcross pickup miles [CI] |','|---|---:|---:|---:|']
    for rep in reps:
        reg=regime(rep)
        m=[next(r for r in pairs if r['comparison']=='known_supply_value' and regime(r)==reg and r['rho_label']=='rho_max' and r['policy_a']=='known_supply' and r['H']==8 and r['metric']==metric) for metric in ['service_rate','actual_empty_distance','empty_pickup_distance']]
        lines.append(f"| {tag(reg)} | {m[0]['mean']:+.4f} [{m[0]['CI_low']:+.4f}, {m[0]['CI_high']:+.4f}] | {fmt(m[1])} | {fmt(m[2])} |")
    lines+=['','service、wait、occupancy、deferral和pipeline详见图5/6与完整机制CSV。空间分布的期末车辆逐t状态可审计，但本轮不声称已把空间平衡效应因果分解。case只作轨迹解释，不能把事后挑选seed当总体证据。',
            '', 'Q7（是否值得继续RL）：']
    evidence=[r for r in pairs if r['comparison']=='oracle_demand_preview_value' and r['rho_label']=='rho_max' and r['metric']=='realized_profit' and r['H']==8]
    lines.append(f"4类代表工况（合并后{len(evidence)}类）H8中，{sum(r['CI_low']>0 for r in evidence)}类的oracle额外增益CI下界>0，{sum(r['CI_high']<0 for r in evidence)}类CI上界<0。")
    if any(r['CI_low']>0 for r in evidence):
        lines.append('Recommendation：存在需进一步研究需求预测/价值逼近的工况，但先校准未来需求与terminal effects，验证宏动作限制和计算预算。Oracle gap不是可学习性证据，暂不直接启动复杂RL。必须要求未来方法在同信息、同预算下超过RH-Supply，而不是与oracle混称可部署收益。')
    else:
        lines.append('Recommendation：本矩阵尚无稳健的额外未来需求信息价值证据，优先完善结构性known-supply lookahead、终端效应和真实业务标定；暂不投入复杂RL。CI跨0不证明收益为零，需要更大独立检验才能宣称等价。')
    lines+=['','上述CI都是各个预先规定比较的marginal区间，未作多重比较控制；整组显著性不能由单个CI拼接宣称。正式新seed降低工况筛选的重复使用偏差，但研究仍为机制探索，不是全国经营效果的confirmatory trial。',
            '', '## 6. Seven figures','']
    captions=['Myopic regime map','Horizon vs realized profit','Future-information decomposition','Western freedom recovery','Deferral behavior','Empty mileage vs service rate','Runtime scaling']
    names=['01_myopic_regime_map','02_horizon_realized_profit','03_future_information_decomposition','04_freedom_recovery','05_deferral_behavior','06_empty_miles_service_rate','07_runtime_scaling']
    for caption,name in zip(captions,names):lines+=['### '+caption,'',f'![{caption}](figures/{name}.png)','']
    limited=sum(r.get('limited_window_count',0) or 0 for r in rows)
    lines+=['## 7. Runtime、失败与验收','',f"正式运行{len(rows)}次；失败{sum(r['status']!='COMPLETED' for r in rows)}；有非OPTIMAL可行解的窗口数={int(limited)}。全部承诺完成。窗口gap不是全天gap；timeout、status、变量/约束规模逐run保存。运行时间含构建、执行、审计和输出开销，求解器runtime单独记录。",'',
            '图7的p95先对每run的window runtime取95%分位，再对30个run等权平均；并非把所有windows混在一起算global p95。每日solver time是求解器runtime之和，daily wall time另存CSV。3个仿真worker并行、各Gurobi一线程；耗时受本机调度影响，不作为独立于硬件的速度保证。', '',
            'Run wall time包含本次执行、审计和输出，不包含父进程最初的CFS读取/场景冻结；manifest各stage elapsed也从冻结后开始计时，不等于脚本从启动到退出的全部耗时。', '',
            f"独立读取原始trace审计：{manifest['trace_audit']['runs']}次run、{manifest['trace_audit']['steps']}个时点，结果{manifest['trace_audit']['status']}。核对冻结数据/spec/source hash、物理invariants、booking边界、无提前/重复assignment、Supply无preview、oracle区间、k=1与realized利润恒等式；逐trace SHA在processed/trace_audit.json。",'',
            'Canonical tests A–E、车位确定性、booking边界、ETA更新测试通过。Full-preview与H1小样本诊断单独保存，不并入30-seed正式均值。所有先验规则写config，未挑formal seed，未删除失败样本。',
            '', f"整套回归在正式固定环境PYTHONHASHSEED=0下{manifest['qa_validation']['full_suite']['passed']}项通过；另有一条Starlette/AnyIO弃用warning。默认hash环境的一次全套运行104通过/1失败：Test C在零等待/积压成本下强制‘立即服务’，而‘稍后服务’也是相同window最优目标，断言对tie敏感。为保持冻结源码身份，本轮没有中途改写该fixture；未来应检查window目标与可行性而非强制一个非唯一动作。首次失败与重测都保存在processed/qa_validation.json，不以固定环境通过掩盖这一测试局限。",
            '', '另在独立临时目录重跑California N80/K3、seed1010、wide rho、oracle_preview H8，利润差0，逐期状态/动作轨迹完全一致；原正式run未覆盖。',
            '', diagnostic_text,
            '', '## 8. Western轨迹','',case,
            '', '## 9. 可复现文件','',
            '- raw/datasets：完整冻结数据与observed/generated记录；raw/runs：spec、hash、metrics、逐t trace、dual ledger、状态。',
            '- summary.csv：每条件/指标完整统计；paired_differences.csv：信息分解、恢复与机制统计；horizon_increments.csv：H差值；processed/diagnostics.csv：H1/full-preview小样本。',
            '- manifest.json：config、baseline config/hash、code SHA逐文件/快照、CFS SHA、Python/solver版本、运行数量、seed、protected files。',
            '- processed/plotting_config.json：图尺寸、字体、颜色/符号、统计定义及输入hash；traces/western_case.csv与western_case.md：可读案例。',
            '- analysis-source.snapshot.zip：后处理脚本独立快照；不因调整报告文本而将旧数值输出假称新运行。',
            '', '复现：先运行run_regime_map，再run_rolling_horizon --pilot，再run_rolling_horizon，最后analysis。同运行源码/config/CFS身份可续跑；改引擎必须新output。旧基线source snapshot在phase0。',
            '', '限制：货运shipment不是passenger，每条被当作不可分割load；不同seed可重抽同一PUMS记录，CI为生成场景replication不包含调查误差。坐标、几何网络、时间、运费和成本都是原实验假设。Formal按pilot选择工况，不能外推全部27条件。所有oracle图明确标识信息放松，不称可部署或full-horizon最优。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(DEFAULT_OUTPUT));args=p.parse_args()
    output=Path(args.output);manifest=read_json(output/'manifest.json');config=manifest['config']
    manifest['qa_validation']=read_json(output/'processed/qa_validation.json')
    pilot=read_rows(output/'regime_map_results.csv');rows=read_rows(output/'rolling_results.csv')
    if manifest['rolling']['failed_count'] or len(rows)!=manifest['rolling']['run_count']:
        raise RuntimeError('formal matrix incomplete or failed')
    summaries,groups=summarize(rows,config);_,pilot_groups=summarize(pilot,config)
    if len(groups)!=len(read_json(output/'representative_regimes.json'))*len(config['formal_rho_labels'])*7:
        raise RuntimeError('missing formal experimental condition')
    if any({int(r['seed']) for r in group}!=set(config['formal_seeds']) for group in groups.values()):
        raise RuntimeError('formal replication set does not match preregistered seeds')
    if any(r['unfinished_committed'] or r['assigned']!=r['delivered'] for r in rows):
        raise RuntimeError('formal commitment accounting invariant')
    pilot_pairs=compute_pairs(pilot_groups,config,True);pairs=compute_pairs(groups,config)
    increments=[]
    for reg in sorted({regime(r) for r in rows}):
        for rho in config['formal_rho_labels']:
            for policy in ['known_supply','oracle_preview']:
                for H,bH in [(4,2),(8,4)]:increments+=paired(groups,config,'horizon_increment',reg,rho,policy,H,policy,bH)
    write_csv(output/'summary.csv',summaries);write_csv(output/'paired_differences.csv',pairs)
    write_csv(output/'regime_map_paired.csv',pilot_pairs);write_csv(output/'horizon_increments.csv',increments)
    report=ROOT/'reports/cfs_dynamic_lookahead';report.mkdir(parents=True,exist_ok=True)
    reps=read_json(output/'representative_regimes.json')
    audit_rows=rows+pilot
    for stage in ['sanity','rolling_pilot','full_preview','H1_diagnostic']:
        audit_rows+=read_rows(output/(stage+'_results.csv'))
    manifest['trace_audit']=validate_raw(output,audit_rows,manifest)
    diagnostic_text=diagnostics(output,rows,config)
    figures(output,report,summaries,pilot_pairs,pairs,reps)
    from .case_study import western_case
    case=western_case(output,rows,reps)
    (report/'dynamic_lookahead_report.md').write_text(report_text(manifest,reps,pilot_pairs,pairs,rows,case,increments,diagnostic_text),encoding='utf8')
    with zipfile.ZipFile(output/'analysis-source.snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['analysis.py','case_study.py']:
            f=Path(__file__).parent/name;z.write(f,f.relative_to(ROOT).as_posix())
        z.write(ROOT/'tests/test_dynamic_analysis.py','tests/test_dynamic_analysis.py')
    manifest.update(status='COMPLETED',analysis_source_hash={name:file_hash(Path(__file__).parent/name) for name in ['analysis.py','case_study.py']},
                    analysis_input_hashes={name:file_hash(output/name) for name in ['regime_map_results.csv','rolling_results.csv']},
                    figures=7,formal_samples_per_condition=len(config['formal_seeds']),
                    postprocessing_tests_sha256=file_hash(ROOT/'tests/test_dynamic_analysis.py'),
                    qa_validation=read_json(output/'processed/qa_validation.json'),
                    host=dict(OS=platform.platform(),processor_identifier=os.environ.get('PROCESSOR_IDENTIFIER'),
                              logical_processors=os.cpu_count(),simulation_workers=config['max_parallel'],solver_threads=1))
    from .common import fingerprint
    if digest(fingerprint())!=manifest['source_code_digest']:raise RuntimeError('execution source changed during numerical study')
    original=read_json(output/'phase0/regression.json')
    if file_hash(ROOT/'results/cfs_rho_endofday/manifest.json')!=original['archived_manifest_hash']:
        raise RuntimeError('baseline archive manifest changed')
    source=Path(manifest['source']['path']);stat=source.stat()
    if stat.st_size!=manifest['source']['size_bytes'] or stat.st_mtime_ns!=manifest['source']['mtime_ns']:
        raise RuntimeError('CFS source metadata changed during study')
    print('Reverifying final CFS SHA256 (read only)...',flush=True)
    if file_hash(source)!=manifest['source']['sha256']:raise RuntimeError('CFS source content changed during study')
    manifest['source_sha256_reverified']=True
    for name,sha in manifest['protected_hashes'].items():
        if file_hash(ROOT/name)!=sha:raise RuntimeError('protected network/accounting/adapter modified')
    manifest['analysis_output_hashes']={str(f.relative_to(ROOT)).replace('\\','/'):file_hash(f)
        for f in sorted(list(report.rglob('*'))+list((output/'processed').glob('*'))+
                        list((output/'traces').glob('*'))+[output/'summary.csv',output/'paired_differences.csv',
                                                         output/'regime_map_paired.csv',output/'horizon_increments.csv'])
        if f.is_file()}
    atomic_json(output/'manifest.json',manifest)
    print('Analysis complete:',report/'dynamic_lookahead_report.md',flush=True)


if __name__=='__main__':main()
