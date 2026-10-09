"""Scripted statistics, seven publication figures and evidence-based report."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t, skew
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_core.storage import read_json, atomic_json
from experiment_core.plugins import resolve
from model_families.single.runner import matching
from .run_experiment import ROOT, write_csv, file_hash

COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
METRICS=['realized_profit','booked_profit','booked_minus_realized','service_rate','cross_pickup_rate',
         'actual_empty_distance','empty_pickup_distance','actual_loaded_distance','completed_fare',
         'cancellation_rate','load_factor','solver_time','candidate_route_count',
         'assigned_undelivered_at_T','credited_unrealized_fare_at_T','remaining_committed_vehicle_time',
         'cooldown_periods','mean_eligible_hubs','pickup_components','largest_pickup_component',
         'pending_unassigned','unfinished_orders','unfinished_committed']


def statistics(values):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('missing/nonfinite statistical observations')
    n=len(a);std=float(a.std(ddof=1)) if n>1 else 0.0
    half=float(t.ppf(.975,n-1))*std/math.sqrt(n) if n>1 else 0
    return dict(n=n,mean=float(a.mean()),sd=std,median=float(np.median(a)),
                ci_low=float(a.mean()-half),ci_high=float(a.mean()+half))


def interval(values):
    s=statistics(values)
    return s['mean'],s['mean']-s['ci_low']


def save_figure(fig, directory, name):
    fig.savefig(directory/(name+'.png'),dpi=400,bbox_inches='tight')
    fig.savefig(directory/(name+'.svg'),bbox_inches='tight')
    fig.savefig(directory/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def horizon_probe(output,manifest):
    config=manifest['config']
    region=config['regions'][0]
    s=read_json(output/'raw'/(region['id']+'-'+str(config['seeds'][0])+'.dataset.json'))['scenarios'][0]
    original=next(o for o in s['orders'] if o['departure']=='06-348' and o['destination']=='06-488')
    rows=[]
    for booking in range(config['horizon']):
        scenario=json.loads(json.dumps(s));scenario['hubs']=[original['departure']]
        o=dict(original,book_time=booking,start_time=booking,end_time=booking+100,penalty=0)
        scenario['orders']=[o]
        spec=resolve({'name':'controlled-time-translation','dataset':{'dataset_id':'probe','revision':'probe'},
                      'model':{'id':'single_level_matching','parameters':{k:config[k] for k in
                      ['speed','capacity','loaded_cost','empty_cost']}|{'cross_hub':False,'max_delay':0,'waiting_cost':0}},
                      'solver':{'backend':'gurobi','parameters':config['solver']},
                      'evaluation':{'accounting_version':'single-realized-v1','terminal_policy':'cooldown_realized'}})
        m,trace=matching(spec,scenario,config['policy_seed'],lambda:None)
        rows.append(dict(booking_time=booking,source_shipment_id=original['id'],
                         booked_profit=m['booked_profit'],realized_full=m['realized_profit'],
                         realized_at_T=m['ledger_at_T']['realized_profit'],
                         delivery_credit_at_T=m['ledger_at_T']['credit_on_delivery_diagnostic'],
                         cooldown_periods=m['cooldown_periods'],unfinished_full=m['unfinished_orders']))
    atomic_json(output/'processed/horizon_probe.json',rows)
    return rows


def analyze(output,report_dir):
    output=Path(output).resolve();report_dir=Path(report_dir).resolve()
    manifest=read_json(output/'manifest.json')
    if manifest['status']!='COMPLETED':
        raise RuntimeError('cannot report an incomplete study')
    with (output/'processed/runs.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    string_fields={'run_id','region','variant','pickup_rule','scenario_hash','training_seed'}
    rows=[{k:v if k in string_fields or v=='' else float(v) for k,v in r.items()} for r in rows]
    regions=[r['id'] for r in manifest['config']['regions']]
    groups={}
    for r in rows:
        groups.setdefault((r['region'],r['variant']),[]).append(r)
    summary=[];paired=[]
    for (region,variant),group in groups.items():
        for metric in METRICS:
            summary.append(dict(region=region,variant=variant,rho=group[0]['rho'],metric=metric,
                                **statistics([r[metric] for r in group])))
        baseline={r['seed']:r for r in groups[region,'rho-0']}
        for metric in ['realized_profit','service_rate','actual_empty_distance','booked_minus_realized']:
            differences=[]
            for row in group:
                base=baseline[row['seed']]
                if row['scenario_hash']!=base['scenario_hash']:
                    raise RuntimeError('paired physical scenario mismatch')
                differences.append(row[metric]-base[metric])
            stat=statistics(differences)
            bs_low=bs_high=''
            if len(differences)>2 and np.std(differences)>0 and abs(skew(differences,bias=False))>1:
                rng=np.random.default_rng(20261004)
                means=rng.choice(differences,(10000,len(differences)),replace=True).mean(axis=1)
                bs_low,bs_high=np.quantile(means,[.025,.975]).tolist()
            paired.append(dict(region=region,variant=variant,metric=metric,baseline='rho-0',**stat,
                               bootstrap_ci_low=bs_low,bootstrap_ci_high=bs_high))
    write_csv(output/'summary.csv',summary);write_csv(output/'processed/paired_differences.csv',paired)
    details={r['run_id']:r for r in read_json(output/'processed/details.json')}
    figures=report_dir/'figures';figures.mkdir(parents=True,exist_ok=True)
    plotconfig={'backend':'matplotlib','width_double_column_inches':7.2,'dpi':400,
                'font':'DejaVu Serif','colors':COLORS,'markers':['o','s','^','D'],
                'uncertainty':'two-sided Student t 95% CI of independent scenario-seed means; paired contrasts stored separately',
                'formats':['PNG','SVG','PDF'],'input':str(output/'processed/runs.csv'),
                'data_sha256':file_hash(output/'processed/runs.csv')}
    atomic_json(output/'processed/plotting_config.json',plotconfig)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':9,
                         'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,
                         'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    def radius_groups(region):
        return sorted([g for (r,v),g in groups.items() if r==region and v.startswith('rho-')],key=lambda g:g[0]['rho'])
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.6))
    for i,(region,ax) in enumerate(zip(manifest['config']['regions'],axs.flat)):
        data=read_json(output/'raw'/(region['id']+'-'+str(manifest['config']['seeds'][0])+'.dataset.json'))['scenarios'][0]
        xy=data['coordinates']
        for a,b,w in data['edges']:
            ax.plot([xy[a][1],xy[b][1]],[xy[a][0],xy[b][0]],color='#777777',lw=1)
        for a,(lat,lon) in xy.items():
            ax.scatter(lon,lat,color=COLORS[i],s=18,zorder=3)
            ax.annotate(a,(lon,lat),xytext=(3,3),textcoords='offset points',fontsize=6)
        ax.set(title=region['id'].replace('_',' '),xlabel='Longitude (assumed)',ylabel='Latitude (assumed)')
        ax.margins(.22)
    fig.suptitle('Geometry-informed metro networks — not observed highways',fontsize=10)
    fig.tight_layout();save_figure(fig,figures,'network')
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.1))
    for i,(region,ax) in enumerate(zip(regions,axs.flat)):
        gg=radius_groups(region);x=[g[0]['rho'] for g in gg]
        ax.plot(x,[g[0]['mean_eligible_hubs'] for g in gg],'o-',color=COLORS[i],label='Eligible other hubs')
        ax.plot(x,[g[0]['largest_pickup_component'] for g in gg],'s--',color='#333333',label='Largest component')
        ax.set(title=region.replace('_',' '),xlabel=r'$\rho$ (network miles)',ylabel='Hubs')
        ax.legend()
    fig.tight_layout();save_figure(fig,figures,'rho_eligibility')
    fig,axs=plt.subplots(4,4,figsize=(7.2,9.4))
    performance=[('realized_profit','Profit (assumed $)'),('service_rate','Completed / orders'),
                 ('actual_empty_distance','Empty vehicle miles'),('cross_pickup_rate','Cross / assignments')]
    for i,region in enumerate(regions):
        gg=radius_groups(region);x=[g[0]['rho'] for g in gg]
        for j,(metric,label) in enumerate(performance):
            means,errors=zip(*(interval([r[metric] for r in g]) for g in gg))
            ax=axs[i,j];ax.errorbar(x,means,yerr=errors,color=COLORS[i],marker=['o','s','^','D'][i],capsize=2,lw=1)
            ax.set(xlabel=r'$\rho$ (miles)',ylabel=label)
            if j==0:ax.set_title(region.replace('_',' '),fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle('Paired CFS scenarios; means and 95% scenario-seed confidence intervals',fontsize=9)
    fig.tight_layout();save_figure(fig,figures,'rho_profit')
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.2))
    for i,region in enumerate(regions):
        selected=[r for r in rows if r['region']==region and r['pickup_rule']=='network_radius']
        axs[0].scatter([r['booked_profit'] for r in selected],[r['realized_profit'] for r in selected],
                       s=12,alpha=.65,c=COLORS[i],marker=['o','s','^','D'][i],label=region.replace('_',' '))
        gg=radius_groups(region);means,errors=zip(*(interval([r['booked_minus_realized'] for r in g]) for g in gg))
        axs[1].errorbar([g[0]['rho'] for g in gg],means,yerr=errors,color=COLORS[i],marker=['o','s','^','D'][i],capsize=2)
    limits=axs[0].get_xlim();axs[0].plot(limits,limits,'k--',lw=.8,label='Equality')
    axs[0].set(xlabel='Booked profit (assumed $)',ylabel='Full realized profit (assumed $)');axs[0].legend(fontsize=5.5)
    axs[1].axhline(0,color='black',lw=.7);axs[1].set(xlabel=r'$\rho$ (miles)',ylabel='Booked minus full realized ($)')
    fig.tight_layout();save_figure(fig,figures,'booked_vs_realized')
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.0))
    for i,(region,ax) in enumerate(zip(regions,axs.flat)):
        g=radius_groups(region)[-1];means=[];errors=[]
        for length in ['0','5','10','20','until_completion']:
            m,e=interval([details[r['run_id']]['cooldown_scan'][length]['realized_profit'] for r in g]);means.append(m);errors.append(e)
        ax.errorbar([0,5,10,20,30],means,yerr=errors,color=COLORS[i],marker=['o','s','^','D'][i],capsize=2)
        ax.set_xticks([0,5,10,20,30],['0','5','10','20','Full'])
        ax.set(title=region.replace('_',' '),xlabel='Cool-down diagnostic length',ylabel='Realized profit ($)')
    fig.tight_layout();save_figure(fig,figures,'cooldown_convergence')
    probes=horizon_probe(output,manifest)
    fig,ax=plt.subplots(figsize=(7.2,3))
    x=[p['booking_time'] for p in probes]
    ax.plot(x,[p['booked_profit']-p['realized_at_T'] for p in probes],'o-',color=COLORS[0],label='Booked minus cutoff realized')
    ax.plot(x,[p['booked_profit']-p['delivery_credit_at_T'] for p in probes],'s--',color=COLORS[1],label='Booked minus delivery-credit diagnostic')
    ax.plot(x,[p['booked_profit']-p['realized_full'] for p in probes],'^:',color=COLORS[2],label='Booked minus full realized')
    ax.set(xlabel='Booking time (same CFS-sourced physical order shifted in time)',ylabel='Accounting difference (assumed $)',
           title='Controlled horizon diagnostic — one order, no scenario-level CI')
    ax.legend();fig.tight_layout();save_figure(fig,figures,'horizon_bias')
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.2))
    for i,region in enumerate(regions):
        gg=radius_groups(region)
        for ax,key,label in zip(axs,['candidate_route_count','solver_time'],['Candidate routes (sum over decisions)','MILP runtime (seconds / scenario)']):
            m,e=zip(*(interval([r[key] for r in g]) for g in gg))
            ax.errorbar([g[0]['rho'] for g in gg],m,yerr=e,color=COLORS[i],marker=['o','s','^','D'][i],capsize=2,
                        label=region.replace('_',' '))
            ax.set(xlabel=r'$\rho$ (network miles)',ylabel=label)
    axs[0].legend(fontsize=6);fig.tight_layout();save_figure(fig,figures,'runtime')
    build_report(output,report_dir,manifest,rows,groups,summary,paired,probes)
    print('Analysis/report complete:',report_dir,flush=True)


def build_report(output,report_dir,manifest,rows,groups,summary,paired,probes):
    config=manifest['config'];regions=[r['id'] for r in config['regions']]
    def group_summary(region,variant,metric):
        return next(r for r in summary if r['region']==region and r['variant']==variant and r['metric']==metric)
    def fmt(s):return f"{s['mean']:.2f} [{s['ci_low']:.2f}, {s['ci_high']:.2f}]"
    def line(s):return s+'\n'
    text='# CFS network-rho 与日终真实核算仿真实验报告\n\n'
    text+='日期：2026-10-04。统计及图表全部由 `experiments/cfs_rho_endofday/analysis.py` 从原始运行结果生成。正式实验状态：'+manifest['status']+'。\n\n'
    text+='## 1. Executive Summary\n\n'
    text+=f"完成 {manifest['scenario_count']} 个配对 CFS 场景、{len(rows)} 次正式运行，每个条件 {len(config['seeds'])} 个连续种子，每场景 {config['orders']} 个订单。网络距离/rho 规则通过多跳/长边/零半径/单调/固定路径测试；纯执行冷却期完成全部承诺，没有日终再次优化。\n\n"
    cutoff_unfinished=statistics([r['assigned_undelivered_at_T'] for r in rows]);fullbias=statistics([r['booked_minus_realized'] for r in rows])
    text+=f"在 T 截止时，每次运行平均仍有 {cutoff_unfinished['mean']:.2f} 个已分配未交付订单；正式结果全部 unfinished committed=0。booked−full-realized 的条件混合平均为 {fullbias['mean']:.2f}（仅描述性，不作为独立样本 CI）。**提前收入的截止偏差被消除，但 booked−full-realized 不一定为正**：计划 backbone 全段按 loaded_cost 计费，而实际按占载/空载区分，两种成本口径本身也可能不同。\n\n"
    text+='Go：以 d_G≤rho 作为正式资格，adjacency 只作显式诊断。Go：以后论文比较采用 full-cooldown realized_profit；旧指标保留供复现，不能把 booked objective 当实际利润。收益提升不是预设结论，见逐区域配对 CI。\n\n'
    text+='## 2. Data：观测值与生成值\n\n'
    text+=f"唯一源：`{manifest['source']['path']}`；CFS 2022 PUMS indexed SQLite，{manifest['source']['size_bytes']:,} bytes，mtime_ns={manifest['source']['mtime_ns']}；完整 SHA256 `{manifest['source']['sha256']}`。SQLite metadata 声明 37,576,546 shipments；此次只读查询局部 OD，未下载/替换源文件。\n\n"
    text+='参考本地 Intercity 分支 `agent/paper-solution-approaches`，commit `'+manifest['reference']['commit']+'`。适配 catalog area code、SQLite schema/index、双向筛选、log(U)/WGT_FACTOR 加权 reservoir；reference 四个文件 hash 见 manifest。未复制其业务模型或截断时间窗。\n\n'
    text+='过滤：MODE∈{111 for-hire truck,112 company-owned truck}；domestic N；不同明确 metro；weight∈(0,40000] lbs；大圆距离≥50 miles；WGT_FACTOR>0；每一方向至少 10 条。超过容量的整单排除，不裁剪货重。每对方向均衡分配实验订单，再在每个 OD stratum 内按 WGT_FACTOR 加权无放回抽样。**跨 OD 均衡设计不是全国自然 OD 分布，也不是无偏 survey total 估计。**\n\n'
    text+='| 分类 | 字段/解释 |\n|---|---|\n| CFS observed | shipment ID, origin/destination CFS area, weight/lbs, goods value/$, great-circle distance/miles, mode, SCTG, TEMP_CNTL, EXPORT, HAZMAT, WGT_FACTOR |\n| generated | booking uniform 0..23，ready=booking+0..2，deadline=ready+network travel+4..8；不截到 T |\n| model scale | load=ceil(weight/1000 lbs)，容量40单位；无拆单/裁剪 |\n| assumed tariff | fare=4×network OD miles+0.03×weight lbs；**不是 CFS goods value 或观测运费** |\n| assumed costs | occupied $2/mile，empty $2.2/mile，backlog $5/order-period，expiry $100/order，vehicle wait $0/period |\n| generated network/fleet | 近似代表城市坐标、kNN/MST、stretch1.2、每 hub 一辆；不是观测场站、道路、车辆 |\n\n'
    text+='以上 temporal fields 是基于 CFS 抽样需求构造的模型时间假设，不是 CFS observed timestamps，也未以实际货运时间标定。用户指南第2/3/5/8/9页：货值不是运费；2022只给大圆距离、不提供 routed mileage；survey weight含采样/噪声调整。\n\n'
    text+='| 区域 | CFS areas / 代表位置 | 支持双向 OD 数 | rho levels (miles) |\n|---|---|---:|---|\n'
    for region in config['regions']:
        levels=sorted({r['rho'] for r in rows if r['region']==region['id'] and r['pickup_rule']=='network_radius'})
        text+=f"| {region['id']} | "+'; '.join(k+': '+v for k,v in region['names'].items())+f" | {len(manifest['data_regions'][region['id']]['supported_pairs'])} | "+', '.join(f'{r:.2f}' for r in levels)+' |\n'
    text+='\n完整方向计数在 manifest.data_regions；不满足最低双向数量的 pair 排除（例如 DC–NY），没有虚构 OD。每个场景 `raw/*.source.json` 保存全部抽中观测字段及对应生成订单。\n\n'
    text+=f"冻结场景总抽样记录数（重复抽中计数）={manifest['sampled_record_draws']}；去重源shipment ID={manifest['unique_sampled_shipments']}。不同seed可抽中同一PUMS记录，不能声称全部观测样本互不重叠。\n\n"
    text+='## 3. Network Construction\n\n'
    text+='CFS areas → 配置中近似代表城市 lat/lon（非 area centroid）→ 区域局部 equirectangular 投影/英里 → deterministic kNN(k=1)+MST 保证连通 → edge length=1.2×投影欧氏距离 → Dijkstra d_G → own hub + {v≠u:d_G≤rho}。stretch 是 geometry-based 假设。无真实 highway 数据，全部称 **geometry-informed metro network**。东北区是相对稠密的 metro 子集，不声称20英里城市设施密集网络；20英里多跳逻辑另由明确 toy test 验证。\n\n'
    text+='路径 primary 权重采用边长十进制的精确有理数之和；距离完全相等时，secondary 全局唯一二进制边 rank 加和确定唯一 simple path。secondary 不扰动 primary physical distance，与 seed/插边顺序无关，保证反向路径与连续子路径一致。travel time=Σ_edge max(1,ceil(edge miles/50mph))，期间1小时；这是逐边上取整，不是整个 OD 一次上取整。\n\n'
    text+='![Geometry-informed network](figures/network.png)\n\n![Eligibility](figures/rho_eligibility.png)\n\n'
    text+='## 4. Accounting Correction\n\n'
    text+='优化器未改：assignment fare−booked route/reposition−stage wait/backlog−expiry +既有value−delay tie-break。正式实验 myopic/zero V；没有新学习组件或政策。\n\n'
    text+='Decision ledger 保存 assignment fare、计划费用、stage penalties、stage objective。Realized ledger 记录段起讫时刻/载重、已行驶里程（未完段按时间线性分摊）、pickup/delivery事件；fare只在delivery计入。真实成本=occupied miles×loaded_cost+empty miles×empty_cost+实际车辆非移动等待×waiting_cost+已发生stage backlog/expiry penalties。订单等待时间独立记录（booking→pickup或cancel/T），不人为新增 assigned waiting 费用。empty pickup / reposition / empty service 均可通过段 kind 审计；loaded 为实际 onboard>0，不等同计划 backbone。\n\n'
    text+='决策 t=0..T−1；cutoff [0,T) 尚未处理恰在 T 的到达。冷却期从 T 处理这些事件，无新单/分配/战略求解/调位，只完成固定承诺。全车辆决策期非行驶时间计等待；T后只计仍未完成承诺的停等，不对提前结束车辆虚计等待。自动 cap 由每辆剩余 arrival/hold+剩余边旅行时间的最大值+事件处理1期推导，未完成报错，禁止静默截断。legacy drain_committed 也改为纯执行。\n\n'
    text+=f"订单穷尽分类：completed + cancelled + unfinished_orders = arrivals；unfinished_orders = unfinished_committed + pending_unassigned。冷却期仅保证unfinished_committed=0，不把T时未分配订单静默转为完成或取消。正式每run平均仍有 {np.mean([r['pending_unassigned'] for r in rows]):.2f} 个pending unassigned（最大 {max(r['pending_unassigned'] for r in rows):.0f}），属于未服务需求，保留账簿并计入服务率分母。\n\n"
    text+='兼容性：原 report_pending / legacy-assignment-v1 仍保留 booked operating_profit；新 single-realized-v1 / cooldown_realized 的 operating_profit 为 realized_profit。不能混合两个版本计算均值。新 rho 默认0，即 own-hub；复现旧邻接必须明确 pickup_rule=legacy_adjacency。\n\n'
    text+='## 5. Experimental Design / statistics\n\n'
    text+=f"种子：{config['seeds']}；policy_seed={config['policy_seed']}；无 training seed。4区×10场景×各区rho levels加邻接baseline；共{len(rows)}次。各 setting 使用同一冻结 CFS抽样订单/时间窗/fleet/hash、cost/fare/solver，仅改变pickup rule/rho；两个核算器从同一执行路径计算，**不分别随机重跑**。T={config['horizon']}，{config['orders']}orders，每hub一车；solver={config['solver']}；reposition=False，max_delay=2（既有规则，不是新增delayed matching）。rho=0加d_G正距离分布25/50/75/100百分位nearest实际距离，去重。\n\n"
    text+='样本=独立场景种子（不同抽样/时间窗）；参数不是replication，同一场景不同半径是paired observations。summary.csv保存每条件每指标mean、sample SD、median、Student t 95% CI。paired_differences.csv保存同seed对rho0差值的同套统计；明显偏态（|skew|>1）额外固定bootstrap seed、10,000重采样CI。小样本CI不能证明正态、全国代表性或真实商业利润。\n\n'
    text+='## 6. Results\n\n| 区域 / setting | realized mean [95% CI] | SD | median | served rate | empty miles | cross rate |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for region in regions:
        variants=sorted({r['variant'] for r in rows if r['region']==region})
        for variant in variants:
            s=group_summary(region,variant,'realized_profit')
            text+=f"| {region}/{variant} | {fmt(s)} | {s['sd']:.2f} | {s['median']:.2f} | {group_summary(region,variant,'service_rate')['mean']:.3f} | {group_summary(region,variant,'actual_empty_distance')['mean']:.2f} | {group_summary(region,variant,'cross_pickup_rate')['mean']:.3f} |\n"
    text+='\n![Performance](figures/rho_profit.png)\n\n| 区域 / setting vs rho0 | paired Δrealized mean [95% CI] | paired SD | median | n |\n|---|---:|---:|---:|---:|\n'
    for p in paired:
        if p['metric']=='realized_profit' and p['variant']!='rho-0':
            text+=f"| {p['region']}/{p['variant']} | {fmt(p)} | {p['sd']:.2f} | {p['median']:.2f} | {p['n']} |\n"
    text+='\n![Booked vs realized](figures/booked_vs_realized.png)\n\n![Cooldown](figures/cooldown_convergence.png)\n\n![Horizon](figures/horizon_bias.png)\n\n![Computational cost](figures/runtime.png)\n\n'
    text+='## 7. Mechanism and policy ranking\n\n'
    for region in regions:
        variants=sorted({r['variant'] for r in rows if r['region']==region and r['pickup_rule']=='network_radius'},key=lambda v:groups[region,v][0]['rho'])
        lo,hi=variants[0],variants[-1]
        peak=group_summary(region,hi,'realized_profit');base=group_summary(region,lo,'realized_profit')
        by_booked=sorted(variants,key=lambda v:group_summary(region,v,'booked_profit')['mean'],reverse=True)
        by_realized=sorted(variants,key=lambda v:group_summary(region,v,'realized_profit')['mean'],reverse=True)
        text+=f"- {region}：最大rho相对0，Δprofit={peak['mean']-base['mean']:+.2f}；eligible other hubs {group_summary(region,lo,'mean_eligible_hubs')['mean']:.1f}→{group_summary(region,hi,'mean_eligible_hubs')['mean']:.1f}，service {group_summary(region,lo,'service_rate')['mean']:.3f}→{group_summary(region,hi,'service_rate')['mean']:.3f}，empty miles {group_summary(region,lo,'actual_empty_distance')['mean']:.1f}→{group_summary(region,hi,'actual_empty_distance')['mean']:.1f}，cross {group_summary(region,hi,'cross_pickup_rate')['mean']:.3f}。mean booked排序 {by_booked}，realized排序 {by_realized}；排序{'发生改变' if by_booked!=by_realized else '未改变'}（只描述均值，不证明显著）。\n"
    text+='\n必须联合检查eligible、cross、empty、service：扩大资格不保证收益单调，额外空驶与myopic局部决策可抵消服务收益。LA–SF仅两hub，所有正rho相同资格，不能通过此区声称多跳pooling；多跳语义由5hub区域及toy回归验证。dense-cluster是否贡献更多不预设，逐区域绝对/配对结果已给出，不能跨不同数据分布直接作因果比较。邻接baseline即使直接连接很长也允许，因此物理含义与rho不同，不当正式距离定义。\n\n'
    text+='| 区域 | 平均 assigned-undelivered at T | 平均 credited-unrealized fare at T | booked−full-realized mean [95% CI]（最大rho） |\n|---|---:|---:|---:|\n'
    for region in regions:
        v=max((v for r,v in groups if r==region and v.startswith('rho-')),key=lambda v:groups[region,v][0]['rho'])
        text+=f"| {region} | {group_summary(region,v,'assigned_undelivered_at_T')['mean']:.2f} | {group_summary(region,v,'credited_unrealized_fare_at_T')['mean']:.2f} | {fmt(group_summary(region,v,'booked_minus_realized'))} |\n"
    text+='\nbooked−full realized包含费用定义差异，不应全部称horizon“虚增”。horizon_probe控制同一个LA→SF货运订单，固定物理任务，只移动booking；full realized范围='+f"[{min(p['realized_full'] for p in probes):.2f}, {max(p['realized_full'] for p in probes):.2f}]"+'，说明无虚假临近日终额外利润。credit-on-delivery diagnostic只移动收入口径、仍用booked costs，因而只是诊断，不是正式实支核算。\n\n'
    text+='## 8. Robustness / computation\n\n'
    text+=f"完成rho=0及分位扫描、连续10种子、4个真实支持区域、cooldown0/5/10/20/full诊断；20不强行视为足够，full按自动cap完成。最大实际cooldown={max(r['cooldown_periods'] for r in rows):.0f}期；总MILP runtime={sum(r['solver_time'] for r in rows):.2f}秒，study wall={manifest['elapsed_seconds']:.2f}秒。runtime为此本机测量，不保证跨机器一致。汇总所有candidate数在runs.csv，图7提供按rho CI。\n\n"
    text+='未校准/扫描tariff、成本、坐标、stretch、容量、demand intensity；本结论只对配置的限定实验有效。Bootstrap只是补充，小样本/部分相同PUMS记录跨seed重复导致统计推论有限。\n\n'
    text+='## 9. Limitations\n\n'
    text+='CFS是货运，不是passenger数据；每条shipment可能包含多车/多件，实验排除超过单车40,000lbs订单，剩下视作不可分割负载。模型字段passenger仅沿用load变量命名。区域metro/state-part不等于具体地址，近似位置非centroid、非真实道路，投影有误差。几何stretch及所有时间/车辆/运费/费用是假设，未验证实际经济有效性。CFS PUMS有survey/noise/topcoding，实验CI是仿真replication CI而非CFS调查误差。只有myopic/zero V policy和radius alternatives，没有验证RL/rollout优劣。只实施本次第1与第4项。\n\n'
    text+='## 10. Go / no-go and scope\n\n'
    text+='Go：采用network-distance rho资格（单位英里），不再把adjacency当物理pickup半径。Go：所有后续single executed-policy论文比较显式采用 single-realized-v1 + cooldown_realized，保留booked账供debug。No-go：把几何网络称为真实highway，把货值称为运费，把生成时间称观测时间，或把本小规模结果当实际商业收益验证。其他模型族尚未接入此realized evaluator，不能直接混合其费用语义。\n\n'
    text+='既有监督/rollout代码未扩展；未实现 delayed matching、future supply、ValueProvider、RL/neural、multi-backbone、dynamic direction。Future work仅可讨论真实路网/真实时间费用标定，更广budget验证，不在本任务实施。\n\n'
    text+='## 11. Reproducibility and acceptance\n\n'
    text+=f"Python {manifest['python']}；packages `{json.dumps(manifest['packages'])}`。基线git commit `{manifest['git_commit']}`；本任务未自动commit/push，运行使用工作树源码hash，不假称已提交版本。所有核心/runner/adapter/分析源码逐文件 SHA256 保存在 manifest.source_code_sha256；config完整保存在manifest及configs/main.json。\n\n"
    text+='在仓库根目录 PowerShell：\n\n```powershell\n.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=workspace/cfs-task-qa/reproduce-unique\n.venv/Scripts/python.exe -B -m experiments.cfs_rho_endofday.run_experiment\n.venv/Scripts/python.exe -B -m experiments.cfs_rho_endofday.analysis\n```\n\n'
    text+='已有输出只允许相同source/config/code安全续跑；源码或配置改变必须用新的 `--output`，避免覆盖旧结果。原始CFS无需复制进仓库；只保存抽中observed rows。测试临时目录放workspace/（ignored），避免Windows coordinator lock被打包为源码。\n\n'
    text+='- `implementation_audit.md`：修改前审计。\n- `results/cfs_rho_endofday/raw/*.source.json` / `*.dataset.json`：观测/生成/冻结场景。\n- `raw/runs/<id>/`：spec、manifest、dataset、metrics、trace每t、decision-ledger、realized-ledger。\n- `processed/runs.csv`：每run全指标；`summary.csv`：每condition所有统计；`paired_differences.csv`：配对差值。\n- `processed/details.json`：T截断/冷却扫描；`horizon_probe.json`：控制时间平移。\n- `processed/plotting_config.json`：输入hash、论文尺寸、字体、色/符号、CI定义；figures共7张，每张PNG400dpi、PDF/SVG矢量。\n\n'
    text+='验收：范围限定✓；local CFS source✓；Intercity adaptation✓；rho语义与network tests✓；双账✓；T后纯执行✓；formal paired runs✓；全部commitment完成✓；观测/生成区分✓；原始CSV/JSON✓；scripted7figures✓；报告与reproduction✓。全量tests最终结果另保存 `test-results.json`，以该文件为准，不隐瞒失败。\n'
    report_dir.mkdir(parents=True,exist_ok=True)
    (report_dir/'simulation_report.md').write_text(text,encoding='utf-8')
    # Generated report is derived output, not a manually edited statistical artifact.


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/cfs_rho_endofday'))
    p.add_argument('--report-dir',default=str(ROOT/'reports/cfs_rho_endofday'))
    a=p.parse_args();analyze(a.output,a.report_dir)
