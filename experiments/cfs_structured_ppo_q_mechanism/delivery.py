"""Bundled NumPy scientific exports and independent non-training verification."""
import argparse
import csv
import json
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from .common import *


def csv_write(name,rows):
    if not rows:raise RuntimeError('do not fabricate an empty result '+name)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (TARGET/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def compute_kl(c):
    old,new=np.array(c['old_mu']),np.array(c['new_mu'])
    s,t=np.array(c['old_sigma'])[None,:],np.array(c['new_sigma'])[None,:]
    z=np.array(c['latent']);mean=(old-new)**2/(2*t*t);scale=np.log(t/s)+s*s/(2*t*t)-.5
    dim=old.shape[1]
    # Jacobian cancels at the same saved latent. Approximate components are not additive.
    lr=-.5*((z-new)/t)**2+.5*((z-old)/s)**2+np.log(s/t)
    full=lr.sum(1);alpha=lr[:,:10].sum(1)
    return dict(exact_KL=float((mean+scale).sum(1).mean()),alpha_exact_KL=float((mean[:,:10]+scale[:,:10]).sum(1).mean()),
        alpha_mean_KL=float(mean[:,:10].sum(1).mean()),alpha_std_KL=float(scale[:,:10].sum(1).mean()),
        beta_exact_KL=float((mean[:,10:]+scale[:,10:]).sum(1).mean()) if dim==20 else None,
        beta_mean_KL=float(mean[:,10:].sum(1).mean()) if dim==20 else None,
        beta_std_KL=float(scale[:,10:].sum(1).mean()) if dim==20 else None,
        independent_approximate_KL=float(np.mean(np.expm1(full)-full)),
        alpha_only_approximate_KL=float(np.mean(np.expm1(alpha)-alpha)),
        beta_only_approximate_KL=float(np.mean(np.expm1(lr[:,10:].sum(1))-lr[:,10:].sum(1))) if dim==20 else None)


def collect():
    kl=[];updates=[];probes=[]
    for p in sorted((TARGET/'raw/kl').glob('*.json')):
        d=read_json(p);s=d['summary'];updates.append(s)
        for c in d['checks']:
            v=compute_kl(c)
            if abs(v['exact_KL']-c['exact_KL'])>1e-10 or abs(v['independent_approximate_KL']-c['approximate_KL'])>2e-5:
                raise RuntimeError('independent KL calculation differs')
            row={k:s[k] for k in ('method','condition','training_seed','update','dimension','planned_steps','effective_steps','skipped_steps','early_stop','completed_epochs','policy_loss')}
            row.update(check=c['check'],epoch=c['epoch'],batch_index=c['batch_index'],batch_size=len(c['indices']),
                actual_approximate_KL=c['approximate_KL'],**v,stopping_trigger=c['stopping_trigger'],accepted=c['accepted'],
                alpha_mu_shift_RMS=c['alpha']['mu_shift_RMS'],alpha_log_sigma_shift_RMS=c['alpha']['log_sigma_shift_RMS'],
                beta_mu_shift_RMS=c['beta']['mu_shift_RMS'] if c['beta'] else None,
                beta_log_sigma_shift_RMS=c['beta']['log_sigma_shift_RMS'] if c['beta'] else None,
                beta_mean_denominator_sensitivity=c.get('beta_mean_denominator_sensitivity'),
                beta_mean_KL_with_paired_alpha_sigma=c.get('beta_mean_KL_with_paired_alpha_sigma'),
                log_std_before=json.dumps(c['log_std_before']),log_std_at_check=json.dumps(c['log_std_at_check']),
                alpha_old_mu=json.dumps([row[:10] for row in c['old_mu']]),
                alpha_new_mu=json.dumps([row[:10] for row in c['new_mu']]),
                parameter_sha256=c['parameter_sha256'],raw_file=str(p))
            kl.append(row)
    for p in sorted((TARGET/'raw/probes').glob('*.json')):
        d=read_json(p);spec=d['spec']
        for r in d['rows']:
            info=r['probe']['plan']['solver'];occ=info['slot_occupancy'];cs=r['probe']['coefficients']
            probes.append(dict(scenario_seed=spec['scenario_seed'],period=spec['period'],training_seed=d['training_seed'],
                state_hash=spec['state_hash'],observation_hash=spec['observation_hash'],q_active_certified=spec['q_active'],
                max_feasible_q2_groups=spec['max_q2_groups'],idle_vehicles=spec['idle_vehicles'],pending_orders=spec['pending_orders'],
                control=r['control'],action20=json.dumps(r['signal']),canonical=r['canonical'],changed_vs_beta0=r['changed_vs_beta0'],
                queried_q_slots=len(cs),queried_q_ge2=sum(c['supply_slot']>=2 for c in cs),
                chosen_q2_groups=sum(x['q']==2 and x['occupied'] for x in occ),
                constructed_price_outside_old_envelope=r['any_constructed_price_outside_old_envelope'],
                MILP_objective=info['window_objective'],solver_status=info['status'],solver_seconds=info['runtime'],
                assigned_orders=len(r['probe']['plan']['assignments']),routes=len(r['probe']['plan']['routes']),raw_file=str(p)))
            probes[-1].update(actual_USD_coefficients=json.dumps(cs),
                selected_release_slots=json.dumps(occ),queried_q=json.dumps(sorted({c['supply_slot'] for c in cs})),
                canonical_business_plan=json.dumps({k:r['probe']['plan'][k] for k in ('assignments','routes','repositions','expired')}))
    return kl,updates,probes,read_json(TARGET/'qa/probes.json')['cases']


def marker(count):
    p=TARGET/'qa/csv-marker.json'
    if p.exists():return
    node='C:/Users/32599/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
    script='C:/Users/32599/.codex/plugins/cache/openai-primary-runtime/spreadsheets/26.1011.11113/skills/spreadsheets/container_tools/mark_artifact_operation_started.mjs'
    r=subprocess.run([node,script,'--operation-kind','create','--expected-output-count',str(count),'--output-format','csv'],capture_output=True,text=True,check=True)
    write_json(p,dict(at=now(),expected=count,stdout=r.stdout))


def exports():
    """Schema-only enrichment/recalculation; never reselect or overwrite reports."""
    load();kl,updates,probes,cases=collect()
    if not (TARGET/'qa/csv-marker.json').exists():raise RuntimeError('scientific CSV marker missing')
    csv_write('kl_components.csv',kl);csv_write('q_active_probes.csv',probes)
    if cases:csv_write('counterfactual_results.csv',cases)


def analyze():
    load()
    if (TARGET/'qa/analysis.json').exists():raise RuntimeError('analysis already delivered')
    kl,updates,probes,cases=collect();marker(3 if cases else 2)
    csv_write('kl_components.csv',kl);csv_write('q_active_probes.csv',probes)
    if cases:csv_write('counterfactual_results.csv',cases)
    aggregate=[]
    for name in NAMES:
        for method in ('PPO10','PPO20'):
            us=[u for u in updates if u['condition']==name and u['method']==method]
            triggers=[r for r in kl if r['condition']==name and r['method']==method and r['stopping_trigger']]
            aggregate.append(dict(condition=name,method=method,updates=len(us),early_stops=sum(u['early_stop'] for u in us),
                planned_steps=sum(u['planned_steps'] for u in us),effective_steps=sum(u['effective_steps'] for u in us),
                effective_mean=float(np.mean([u['effective_steps'] for u in us])),
                trigger_exact_mean=float(np.mean([r['exact_KL'] for r in triggers])) if triggers else None,
                trigger_approximate_mean=float(np.mean([r['actual_approximate_KL'] for r in triggers])) if triggers else None,
                beta_share_median=float(np.median([r['beta_exact_KL']/r['exact_KL'] for r in triggers])) if method=='PPO20' and triggers else None,
                beta_mean_share=float(np.sum([r['beta_mean_KL'] for r in triggers])/np.sum([r['beta_exact_KL'] for r in triggers])) if method=='PPO20' and triggers else None,
                trigger_full_but_alpha_only_below=sum(r['alpha_only_approximate_KL']<=.03 for r in triggers),
                mean_sensitivity_factor=float(np.mean([r['beta_mean_denominator_sensitivity'] for r in triggers])) if method=='PPO20' and triggers else None,
                exact_trigger_below_actual_threshold=sum(r['exact_KL']<=.03 for r in triggers)))
    q=[]
    for active in (True,False):
        pairs=[r for r in probes if r['q_active_certified']==active and r['control']=='A_learned']
        fixed=[r for r in probes if r['q_active_certified']==active and r['control'].startswith('C_')]
        inside=[r for r in fixed if not r['constructed_price_outside_old_envelope']]
        q.append(dict(q_active=active,learned_pairs=len(pairs),learned_switches=sum(r['changed_vs_beta0'] for r in pairs),
            fixed_probes=len(fixed),fixed_switches=sum(r['changed_vs_beta0'] for r in fixed),
            inside_envelope_probes=len(inside),inside_envelope_switches=sum(r['changed_vs_beta0'] for r in inside)))
    summary=dict(aggregate=aggregate,q=q,cases=cases)
    write_json(TARGET/'qa/summary.json',summary);reports(summary)
    write_json(TARGET/'qa/analysis.json',dict(status='PASSED',kl_rows=len(kl),update_rows=len(updates),probe_rows=len(probes),CF_cases=len(cases)))
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-m','experiments.cfs_structured_ppo_q_mechanism.delivery','plot'],check=True)


def reports(s):
    def save(name,text):(TARGET/name).write_text(text,encoding='utf-8')
    protocol=read_json(TARGET/'protocol.json')
    save('experiment_protocol.md','# 实际执行协议\n\n```json\n'+json.dumps(protocol,ensure_ascii=False,indent=2)+'\n```\n\n原始库/模型只读；数值源码与本协议在执行前SHA冻结。PPO优化仅重建保存的历史轨迹，不产生新训练数据或新模型；正式测试未访问。状态冻结发生于q-feasibility检查后、扰动前，未使用隐藏未来利润。\n')
    table='|工况|模型|早停/24|有效优化/192|每轮平均有效步|触发时β精确KL占比中位数|\n|---|---|---:|---:|---:|---:|\n'
    for r in s['aggregate']:
        share=f'{100*r["beta_share_median"]:.2f}%' if r['beta_share_median'] is not None else '不适用'
        table+=f'|{r["condition"]}|{r["method"]}|{r["early_stops"]}|{r["effective_steps"]}|{r["effective_mean"]:.2f}|{share}|\n'
    kl='# KL 训练几何：实际触发时刻的重建\n\n'+table+'\n'
    kl+='18个冻结模型×8次update，从保存的观测/latent/旧logp和初始随机流重建。原backend的检查行被只读记录；原始日志、最终全部网络tensor及PPO20最终随机状态一致。没有用最终checkpoint推造中间策略，没有生成新的rollout。\n\n'
    kl+='每个实际检查保存old/new mu、sigma、latent、参数hash；触发时保存完整网络/Adam/随机流快照。逐dim精确高斯KL=均值项+标准差项，以Torch分布KL及bundled NumPy独立核对。old→new方向固定，条件状态集就是实际检查minibatch。tanh数学变换KL对应latent高斯，但原早停用样本近似 `(exp(logratio)-1)-logratio`，有限样本并不要求它等于解析KL。α/β的解析KL可加，分量近似KL不能直接相加。\n\n'
    for r in s['aggregate']:
        if r['method']!='PPO20':continue
        kl+=f'{r["condition"]}：β精确KL的均值移动项占比 {100*r["beta_mean_share"]:.2f}%；在固定轨迹上，{r["trigger_full_but_alpha_only_below"]}/{r["early_stops"]} 次真实触发的α单独样本近似KL仍≤0.03。β均值位移的分母敏感性约 {r["mean_sensitivity_factor"]:.2f} 倍（只替换解析项的配对α方差作算术对照，没有改变任何实际策略）。{r["exact_trigger_below_actual_threshold"]}/{r["early_stops"]} 次触发的解析KL≤0.03，说明必须区分真实近似触发与解析诊断。\n\n'
    kl+='分母敏感性不等于“方差造成早停”的独立因果估计，扩维、均值更新与不同on-policy数据共同改变轨迹。也没有拿每维KL平均来替代原联合概率或阈值。\n'
    save('kl_geometry_report.md',kl)
    qt='|状态类别|模型—状态对|已学β切换|固定β探针|固定β切换|旧包络内探针|旧包络内切换|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in s['q']:
        qt+=f'|{"q-active" if r["q_active"] else "q-only"}|{r["learned_pairs"]}|{r["learned_switches"]}|{r["fixed_probes"]}|{r["fixed_switches"]}|{r["inside_envelope_probes"]}|{r["inside_envelope_switches"]}|\n'
    selection=read_json(TARGET/'states-frozen.json')
    q='# W40 q-active：合法候选与已学控制\n\n'+qt+'\n'
    q+='先在20个开发状态对原MILP副本最大化sum q2，只换诊断目标，约束/变量界/整数类型SHA相同。OPTIMAL且目标0证明当前H1 macro-action模型没有任何q≥2方案；正值有完整合法证人，区别于仅查询了q系数。不是全系统所有未来控制的可行域证明。\n\n'
    q+=f'请求8个q-active、2个q-only，实际取得 {selection["actual"]["q_active"]} 和 {selection["actual"]["q_only"]}。不足类别没有按结果补搜索。先seed/time排序冻结，再运行3个训练seed的A/B/D与预设single/all方向四斜率；同状态不变，规范化仅合并真实同质车辆。\n\n'
    q+='“旧包络内”采用保守标准：该探针**全部构建/查询的非终端价格**均在各hub±Sc内。这保证所有真正有效项也在内；超出标记可能包含不可达项，不能据此断言超出可行价格包络。经验比例仅属于预选有限状态/扰动集合。\n\n'
    if s['cases']:
        for r in s['cases']:
            q+=f'案例{r["case"]}：seed{r["scenario_seed"]}/t{r["period"]}/model{r["training_seed"]}/{r["control"]}，完整冷却ΔP={r["profit_difference_USD"]:.2f} USD，当期现金差={r["current_cash_difference_USD"]:.2f}，后续差={r["later_cash_difference_USD"]:.2f}。按首个切换规则，不按收益挑选。\n\n'
    else:q+='没有符合预设条件的切换案例，未运行长期反事实，不将未运行填0。\n\n'
    q+='反事实两个分支保留相同完整状态/隐藏订单日，下期共同冻结PPO10/B8/30001，完整冷却；隐藏未来仅进入环境，不进Actor。窗口MILP objective与实现现金不是同一指标。\n'
    save('q_active_report.md',q)
    b=read_json(TARGET/'qa/budget.json')
    main='# PPO20 为什么尚未有效利用 q：轻量机制诊断\n\n'
    main+='## 实验范围\n\n本轮不重新比较正式测试利润，不重训九个模型。只重建历史KL检查，并在W40开发状态检查合法q-active控制与少量反事实。原模型、MILP约束、现金、联合logp、学习率、KL阈值和探索方差不改。\n\n'
    main+='## 训练优化告诉我们什么\n\n'+table+'\n'+kl.split('每个实际检查')[1].split('分母敏感性不等于')[0]+'\n'
    main+='## 有效槽位告诉我们什么\n\n'+qt+'\n'+q.split('请求8个')[1]+'\n'
    main+='## 十问与下一步\n\n'
    questions=[
        '为什么20维更容易早停？实际重建显示新增β的KL累加、较小方差对均值位移的敏感性与策略均值更新共同相关；这些是固定轨迹机制证据，不是不同训练设定间的因果分解。',
        'β贡献多少？见各工况触发时β/总精确KL比例；数值来自实际检查参数，不是update结束值。',
        '均值还是方差变化？使用解析均值/方差项分开检查，不能将非零log_std变化自动认定为主因。',
        '减少多少优化？计划每模型每update8个minibatch，表中有效步数与原日志一致；拒绝的检查不是一次优化，未完成epoch不冒称完成。',
        '已学β能否影响q-active决策？按上表有限状态/模型配对结果判断，不推广成总体概率。',
        '人工β能否影响同状态？固定α，预先设定single/all方向与±.2/±.6；全部候选保留，不寻找收益赢家。',
        '是否仅因为价格包络扩大？看所有查询价格仍在±Sc内的切换子集；若该子集出现改变，就不能把全部变化归于包络扩大，但也未证明所有新动作是10维任意α都不可达。',
        '是否具有长期经济价值？只使用首2不同状态的共同未来案例；正负效应都保留，小量目的性案例不是总体收益检验。',
        '优先改善哪里？结合KL与q-active对照后做有条件判断，不能仅由β存在交互或某例收益推导必须转联合动作。',
        '是否值得下一轮训练？仅提出单工况单seed、≤8-update独立方案，不在本轮启动；旧正式测试不再作为新确认集。']
    main+='\n\n'.join(f'{i+1}. {v}' for i,v in enumerate(questions))
    main+=f'\n\n## 成本与局限\n\n新增数值/离线重建累计 {b["seconds"]:.2f} 秒，{b["calls"]} 次新MILP。没有新增正式测试日或训练rollout；历史训练成本不消失。开发状态有目的筛选，3个模型共享状态，不视为独立场景。跨PPO10/20使用各自on-policy轨迹，KL对照不是相同状态/同动作分布的随机干预。全部实际QA、原始数据和SHA见qa与final_manifest。\n'
    save('structured_ppo_q_mechanism_report.md',main)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=read_json(TARGET/'qa/summary.json');plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'svg.fonttype':'none'})
    def save(fig,name,source):
        fig.tight_layout();fig.savefig(TARGET/'figures'/f'{name}.svg');fig.savefig(TARGET/'figures'/f'{name}.png',dpi=600)
        write_json(TARGET/'figures'/f'{name}.json',dict(source_csv=source,source_module_sha256=sha(__file__),dpi=600,
            note='descriptive only; not independent replications',size_inches=list(fig.get_size_inches())));plt.close(fig)
    labels=('CA','W40','W80');x=np.arange(3)
    fig,ax=plt.subplots(figsize=(5.5,3))
    rs=[r for r in s['aggregate'] if r['method']=='PPO20']
    beta=[r['beta_share_median'] for r in rs]
    ax.bar(x,1-np.array(beta),label='alpha',color='#0072B2');ax.bar(x,beta,bottom=1-np.array(beta),label='beta',color='#E69F00',hatch='//')
    ax.set_xticks(x,labels);ax.set_ylabel('Median exact KL share at true trigger');ax.legend(loc='upper center',ncol=2,bbox_to_anchor=(.5,1.2))
    save(fig,'01_KL_components','kl_components.csv')
    fig,ax=plt.subplots(figsize=(5.5,3))
    for i,method in enumerate(('PPO10','PPO20')):
        rs=[r for r in s['aggregate'] if r['method']==method]
        ax.bar(x+(i-.5)*.3,[r['effective_mean'] for r in rs],.3,label=method,hatch='' if i==0 else '//')
    ax.axhline(8,color='.4',linestyle='--',label='planned 8');ax.set_xticks(x,labels);ax.set_ylabel('Optimized minibatches per update');ax.legend(loc='upper center',ncol=3,bbox_to_anchor=(.5,1.2))
    save(fig,'02_effective_steps','kl_components.csv')
    fig,ax=plt.subplots(figsize=(5.5,3))
    for i,kind in enumerate(('learned','fixed')):
        ratios=[r[kind+'_switches']/r[kind+('_pairs' if kind=='learned' else '_probes')] if r[kind+('_pairs' if kind=='learned' else '_probes')] else np.nan for r in s['q']]
        ax.bar(np.arange(2)+(i-.5)*.3,ratios,.3,label=kind,hatch='' if i==0 else '//')
        for j,r in enumerate(s['q']):
            denom=r[kind+('_pairs' if kind=='learned' else '_probes')]
            ax.text(j+(i-.5)*.3,(ratios[j] if denom else 0.)+.002,
                f'{r[kind+"_switches"]}/{denom}' if denom else 'not sampled',ha='center',fontsize=8)
    ax.set_ylim(0,.16);ax.set_xticks([0,1],['q-active','q-only']);ax.set_ylabel('Canonical switch fraction vs beta0');ax.legend(loc='upper center',ncol=2,bbox_to_anchor=(.5,1.2))
    save(fig,'03_q_active_switches','q_active_probes.csv')
    write_json(TARGET/'qa/figures.json',{str(p):sha(p) for p in (TARGET/'figures').iterdir() if p.is_file()})


def verify():
    load();kl,updates,probes,cases=collect()
    files={str(p):sha(p) for p in TARGET.rglob('*') if p.is_file() and p.is_relative_to(TARGET/'raw')}
    # Verify CSV serialization against each owning raw row, including NA vs0.
    for name,expected in (('kl_components.csv',kl),('q_active_probes.csv',probes),('counterfactual_results.csv',cases)):
        if not expected:continue
        actual=list(csv.DictReader((TARGET/name).open(encoding='utf-8-sig')))
        if len(actual)!=len(expected):raise RuntimeError('CSV row count')
        for a,e in zip(actual,expected):
            for k,v in e.items():
                if v is None:
                    if a[k]!='':raise RuntimeError('missing not blank')
                elif isinstance(v,(float,int)) and not isinstance(v,bool):
                    if float(a[k])!=v:raise RuntimeError('CSV number differs '+k)
                elif a[k]!=str(v):raise RuntimeError('CSV category differs '+k)
    m=load();verify_hashes(Path('/'),m['history_hashes'])
    verify_hashes(Path('/'),read_json(TARGET/'qa/figures.json'))
    for p in (TARGET/'figures').glob('*.svg'):ET.parse(p)
    write_json(TARGET/'qa/exports.json',dict(status='PASSED',kl_rows=len(kl),updates=len(updates),probes=len(probes),CF=len(cases),
        independent_Gaussian_and_sampled_KL=True,all_CSV_cells_recalculated=True,raw_hashes=files))


def freeze(reviewed):
    m=load()
    if not reviewed:raise RuntimeError('latest3PNG must actually be viewed')
    for name in ('kl','selection','probes','analysis','exports','physical'):
        if read_json(TARGET/'qa'/(name+'.json'))['status']!='PASSED':raise RuntimeError('mandatory QA incomplete '+name)
    verify_hashes(Path('/'),read_json(TARGET/'qa/exports.json')['raw_hashes'])
    if read_json(TARGET/'qa/physical.json')['source_sha256']!=sha(Path(__file__).with_name('physical_qa.py')):raise RuntimeError('physical QA source changed')
    suites=list(ET.parse(TARGET/'qa/pytest-final.xml').getroot().iter('testsuite'))
    if any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped')):raise RuntimeError('tests failed')
    n=sum(int(s.get('tests',0)) for s in suites)
    if n<20:raise RuntimeError('test coverage incomplete')
    tested_files=[ROOT/'tests'/name for name in ('test_structured_ppo_q.py','test_structured_q_mechanism.py','test_structured_q_mechanism_delivery.py')]
    write_json(TARGET/'qa/tests.json',dict(status='PASSED',tests=n,xml_sha256=sha(TARGET/'qa/pytest-final.xml'),
        scope='related numerical and delivery regression, not full-platform matrix',test_sources={str(p):sha(p) for p in tested_files}))
    write_json(TARGET/'qa/visual.json',dict(status='PASSED',actual_latest_png_count=3,at=now(),SVG_same_source_XML_checked=True))
    b=read_json(TARGET/'qa/budget.json')
    if b['calls']>500 or b['seconds']>600:raise RuntimeError('budget exceeded')
    if sum(p.stat().st_size for p in TARGET.rglob('*') if p.is_file())>512*2**20:raise RuntimeError('output budget exceeded')
    verify_hashes(Path('/'),m['history_hashes']);verify_hashes(Path('/'),read_json(TARGET/'qa/figures.json'))
    hashes={p.relative_to(TARGET).as_posix():sha(p) for p in TARGET.rglob('*') if p.is_file() and p not in (TARGET/'manifest.json',TARGET/'final_manifest.json')}
    m.update(status='COMPLETED',completed_at=now(),artifact_hashes=hashes,delivery_sha256=sha(__file__),
        delivery_sources={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('physical_qa.py'),ROOT/'tests/test_structured_q_mechanism_delivery.py')})
    write_json(TARGET/'manifest.json',m);write_json(TARGET/'final_manifest.json',m);verify_hashes(TARGET,hashes)
    print('COMPLETED',len(hashes),'hashes',sha(TARGET/'final_manifest.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('analyze','exports','plot','verify','freeze'));p.add_argument('--reviewed',action='store_true');a=p.parse_args()
    if a.stage=='freeze':freeze(a.reviewed)
    else:globals()[a.stage]()


if __name__=='__main__':main()
