"""Delivery and predeclared same-state q ablations; no edits to frozen numerics."""
import argparse
import copy
import csv
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from .common import *


def mechanism():
    from experiments.cfs_direct_ppo_diagnosis.common import restore,canonical,finish
    from model_families.single.runner import state_snapshot
    from model_families.single.learning.ppo.networks import sample_action
    from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
    from .control import upgrade,replay
    from .training import Policy
    import torch
    load()
    if (TARGET/'qa/mechanism.json').exists():return
    # Freeze supplemental mechanism code before its first numerical execution.
    if not (TARGET/'protocol/mechanism-freeze.json').exists():
        write_json(TARGET/'protocol/mechanism-freeze.json',dict(sha256=sha(__file__),at=now(),
            controls=['old10det','q20det','q20beta0','q20stochastic'],states=18,seeds=SEEDS,
            counterfactual='first beta0/det switch per condition, order reference then training seed; no profit choice'))
    elif sha(__file__)!=read_json(TARGET/'protocol/mechanism-freeze.json')['sha256']:
        raise RuntimeError('frozen supplemental mechanism changed')
    refs=read_json(TARGET/'protocol/references.json');budget=Budget();cases=[];total=0
    try:
        for name in NAMES:
            first=None;c=cfg(name)
            for spec in [r for r in refs if r['condition']==name]:
                scene=next(s for s in scenes(name,'validation') if s['seed']==spec['scenario_seed'])
                record=read_json(spec['source'])
                for seed in SEEDS:
                    with no_mc_guard():
                        old=old_policy(name,seed);new=Policy(TARGET/'training'/name/str(seed)/'inference-0008')
                        ep,o=restore(scene,c,record,spec['period'],budget);ep=upgrade(ep);before=state_snapshot(ep._sim)
                        det=new.act(o);zero=det[:10]+[0.]*10
                        gen=torch.Generator().manual_seed(72000+seed+scene['seed']+spec['period'])
                        controls=[('old10',list(old.act(o))+[0.]*10),('q20',det),('beta0',zero),
                            ('stochastic',sample_action(new.network,o,gen,False)[0].tolist())]
                        rows=[]
                        for label,signal in controls:
                            probe=ep.probe(signal);rows.append(dict(control=label,signal=signal,probe=probe,
                                canonical=canonical(probe['plan'],before)));total+=1
                        if state_snapshot(ep._sim)!=before:raise RuntimeError('mechanism probe mutated state')
                        if first is None and rows[1]['canonical']!=rows[2]['canonical']:
                            first=(spec,scene,record,rows[1],rows[2],seed)
                    write_json(TARGET/'validation/probes'/f'{name}-{seed}-s{scene["seed"]}-t{spec["period"]}.json',
                        dict(condition=name,training_seed=seed,spec=spec,state=before,rows=rows))
            if first:
                spec,scene,record,det,zero,seed=first;profits=[];continuation=old_policy(name,30001)
                class Common:
                    def act(self,o):return list(continuation.act(o))+[0.]*10
                with no_mc_guard():
                    for label,row in (('q20',det),('beta0',zero)):
                        ep,o=restore(scene,c,record,spec['period'],budget);ep=upgrade(ep)
                        r=finish(ep,row['signal'],Common());r['replay']=replay(scene,c,r)
                        write_json(TARGET/'validation'/f'{name}-cf-{label}.json',r);profits.append(r['ledger']['realized_profit'])
                cases.append(dict(condition=name,training_seed=seed,scenario_seed=scene['seed'],period=spec['period'],
                    difference_USD=profits[0]-profits[1],selection='first canonical switch; NOT profit selected'))
            else:cases.append(dict(condition=name,reason='no learned beta canonical switch in finite probes'))
        write_json(TARGET/'qa/mechanism.json',dict(status='PASSED',probes=total,expected=216,cases=cases))
    finally:budget.save()


def rows_csv(name,rows):
    if not rows:raise RuntimeError('no fabricated empty CSV: '+name)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (TARGET/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def csv_marker(count):
    p=TARGET/'qa/csv-marker.json'
    if p.exists():return
    node='C:/Users/32599/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
    marker='C:/Users/32599/.codex/plugins/cache/openai-primary-runtime/spreadsheets/26.1011.11113/skills/spreadsheets/container_tools/mark_artifact_operation_started.mjs'
    r=subprocess.run([node,marker,'--operation-kind','create','--expected-output-count',str(count),'--output-format','csv'],check=True,capture_output=True,text=True)
    write_json(p,dict(at=now(),expected_count=count,stdout=r.stdout))


def analyze():
    from .analysis import crossed
    load();pilot=read_json(TARGET/'qa/pilot.json');promote=pilot['promote']
    if (TARGET/'qa/analysis.json').exists():raise RuntimeError('do not overwrite analyzed report')
    if promote and not (TARGET/'qa/mechanism.json').exists():raise RuntimeError('mechanism first')
    csv_marker(8 if promote else 3)
    coverage=copy.deepcopy(pilot['coverage']);probes=[];cases=[]
    for p in sorted((TARGET/'pilot').glob('*-t*.json')):
        d=read_json(p);base=d['rows'][0]
        occ=base['probe']['plan']['solver']['slot_occupancy']
        cov=next(r for r in coverage if (r['condition'],r['seed'],r['period'])==
                 (d['spec']['condition'],d['spec']['scenario_seed'],d['spec']['period']))
        cov.update(constructed_groups=sum(r['q']==1 for r in occ),
                   occupied_groups=sum(r['q']==1 and r['occupied'] for r in occ),
                   occupied_groups_ge2=sum(r['q']==2 and r['occupied'] for r in occ))
        for r in d['rows']:
            occ=r['probe']['plan']['solver']['slot_occupancy']
            probes.append(dict(condition=d['spec']['condition'],scenario_seed=d['spec']['scenario_seed'],period=d['spec']['period'],
                direction=r['direction'],slope=r['slope'],canonical_changed=r['canonical']!=base['canonical'],
                canonical=r['canonical'],occupied_q_ge2=sum(v['occupied'] for v in occ if v['q']>=2),
                served=len(r['probe']['plan']['assignments']),routes=len(r['probe']['plan']['routes']),
                solver_seconds=r['probe']['plan']['solver']['runtime'],record_file=str(p)))
    for r in pilot['cases']:cases.append(dict(stage='pilot',**r))
    if promote:
        for r in read_json(TARGET/'qa/mechanism.json')['cases']:cases.append(dict(stage='learned',**r))
    rows_csv('q_slot_coverage.csv',coverage);rows_csv('expressivity_probe.csv',probes);rows_csv('counterfactual_cases.csv',cases)
    paired=[];curves=[];sensitivity=[];diagnostics=[];validation=[];effects=[];means=[]
    if promote:
        for name in NAMES:
            matrices={method:np.zeros((3,8)) for method in ('PPO10','PPO20','MC','M0')}
            for i,seed in enumerate(SEEDS):
                for j,s in enumerate(TESTS):
                    row=dict(condition=name,training_seed=seed,scenario_seed=s)
                    for method in matrices:
                        suffix=f'-{seed}' if method.startswith('PPO') else ''
                        r=read_json(TARGET/'evaluation'/f'{name}-{method}{suffix}-s{s}.json')
                        matrices[method][i,j]=r['ledger']['realized_profit'];row[method+'_profit_USD']=matrices[method][i,j]
                    row['difference20_10_USD']=row['PPO20_profit_USD']-row['PPO10_profit_USD']
                    row['difference20_MC_USD']=row['PPO20_profit_USD']-row['MC_profit_USD'];paired.append(row)
                folder=TARGET/'training'/name/str(seed)
                for u in range(1,9):
                    d=read_json(folder/f'update-{u:04d}.json')
                    diagnostics.append(dict(condition=name,training_seed=seed,update=u,training_mean_USD=d['training_mean_USD'],
                        KL=d['approximate_kl'],KL_early_stop=d['kl_early_stop'],clip_fraction=d['clip_fraction'],
                        policy_loss=d['policy_loss'],value_loss=d['value_loss'],critic_RMSE_USD=d['critic_RMSE_USD'],
                        critic_EV=d['critic_EV'],gradient_norm=d['gradient_norm'],
                        alpha_mean=json.dumps(d['action_mean'][:10]),beta_mean=json.dumps(d['action_mean'][10:]),
                        beta_mean_abs=float(np.mean(np.abs(d['action_mean'][10:]))),log_std=json.dumps(d['log_std'])))
                oldfolder=OLD/'training'/name/'B'/str(seed)
                for s in range(21008,21012):
                    a=read_json(oldfolder/f'validation-0008-s{s}.json');b=read_json(folder/f'validation-s{s}.json')
                    validation.append(dict(condition=name,training_seed=seed,scenario_seed=s,update=8,
                        PPO10_profit_USD=a['ledger']['realized_profit'],PPO20_profit_USD=b['ledger']['realized_profit']))
            means.append(dict(condition=name,**{k:float(v.mean()) for k,v in matrices.items()}))
            for other in ('PPO10','MC','M0'):
                effects.append(dict(condition=name,comparison='PPO20-'+other,**crossed(matrices['PPO20']-matrices[other])))
        for p in sorted((TARGET/'validation/probes').glob('*.json')):
            d=read_json(p);det=d['rows'][1]
            for r in d['rows']:
                sensitivity.append(dict(condition=d['condition'],training_seed=d['training_seed'],scenario_seed=d['spec']['scenario_seed'],
                    period=d['spec']['period'],control=r['control'],changed_vs_q20=r['canonical']!=det['canonical'],canonical=r['canonical']))
                for c in r['probe']['coefficients']:
                    curves.append(dict(condition=d['condition'],training_seed=d['training_seed'],scenario_seed=d['spec']['scenario_seed'],
                        period=d['spec']['period'],control=r['control'],hub=c['hub'],release_time=c['time'],q=c['supply_slot'],
                        price_USD=c['value'],alpha=c['alpha'],beta=c['beta'],psi=c['psi']))
        rows_csv('paired_results.csv',paired);rows_csv('q_value_curves.csv',curves);rows_csv('action_sensitivity.csv',sensitivity)
        rows_csv('training_diagnostics.csv',diagnostics);rows_csv('validation_results.csv',validation)
        write_json(TARGET/'qa/statistics.json',dict(status='PASSED',effects=effects,means=means,unit='3 crossed model seeds x8 paired scenarios'))
    write_json(TARGET/'qa/analysis.json',dict(status='PASSED',csv_count=8 if promote else 3,paired=len(paired),curves=len(curves),
        diagnostics=len(diagnostics),sensitivity=len(sensitivity),validation=len(validation)))
    report(pilot,means,effects,sensitivity,diagnostics)
    write_json(TARGET/'qa/plot-input.json',dict(pilot=pilot,means=means,curves=curves,
        sensitivity=sensitivity,diagnostics=diagnostics,validation=validation))
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-m','experiments.cfs_structured_ppo_q.delivery','plot','--directory',str(TARGET)],check=True)


def report(pilot,means,effects,sensitivity,diagnostics):
    p=pilot['promote'];budget=read_json(TARGET/'qa/budget.json')
    def text(name,content):(TARGET/name).write_text(content,encoding='utf-8')
    audit='''# 在线 q 的语义与实际覆盖

q 是相同 (释放时刻,枢纽) sink-flow 的数量前缀，不是车辆编号。原 add_release_slots 对每组构建 K 个 binary z，sum z=sum path-flow，z_q<=z_(q-1)，目标加入 sum V(q)z_q。route/wait/locked-arrival 的流变量通过等式连接 z，订单assignment x 与 route y 原约束不变。

异质车辆的当前承诺/位置/路线决定其可行流，slot本身不绑定车辆身份。车辆不能任意置换；canonical只合并完整当前状态相同车辆。releases按(time,hub)排序，q从1开始。候选由observed_state及当前订单生成，不读未来订单；承诺车辆只投影锁定移动。tau>=T在建模前排除战略项。

所有组都会查询q=1..K，即便某组不可达。因此query coverage不是occupied coverage，更不是q有效控制频率。本轮分别保存查询、选中z、所有固定扰动中选中q>=2及规范化动作差。也没有把构建的 route候选数当合法联合动作数。

结构秩扣除真实MILP等式及固定变量行空间，容差1e-8。它只描述仿射松弛，未完全扣除不等式导致的不可达方向，不能称精确integer hull表达维数。真实canonical switch且occupiedq>=2是更直接的有限控制证据。MC的q来自资源注入，不等于这个group prefix；新PPO不读取MC标签，但可加性仍存在。
'''
    text('q_semantics_audit.md',audit)
    proto=read_json(TARGET/'protocol/preregistration.json')
    text('experiment_protocol.md','# 轻量实验协议\n\n'+json.dumps(proto,ensure_ascii=False,indent=2)+'\n\n用户的轻量要求优先于附件16-update/24-test建议，改动在数值结果前登记。保留三工况三seed，训练缩为8update，验证仅update8，正式新测试缩为8场景。旧10D只用B8原权重，不用事后B16 best。旧训练成本没有消失，推理复用不宣称总成本对等。历史完整QA只读引用，当前实际核验所用输入SHA和所有新增轨迹。\n')
    table='|工况|预选状态|beta扰动切换/144|基准占用q>=2槽位|仿射秩扩展状态|\n|---|---:|---:|---:|---:|\n'
    for name,label in zip(NAMES,LABELS):
        c=[r for r in pilot['coverage'] if r['condition']==name];r=[r for r in pilot['ranks'] if r['condition']==name]
        table+=f'|{label}|{len(c)}|{sum(x["changed"] for x in c)}|{sum(x["occupied_q_ge2"] for x in c)}|{sum(x["expanded"]>x["alpha"] for x in r)}|\n'
    text('expressivity_pilot_report.md','# q 方向控制能力探针\n\n'+table+'\n六个完整日β=0逐coeff/目标/整数动作/最终账本一致。18状态、每状态25求解，其中24非零固定扰动；所有退化状态保留。\n\n晋级：'+str(p)+'。人工扰动能改变动作不等于PPO能学会，也不等于该动作更有经济价值。只报告首个切换案例，不按利润筛选。完整系数/slot occupancy/状态/路线见pilot原始JSON。\n')
    main='# 结构化 q 价格 PPO 与纯 MC：轻量实验报告\n\n'
    main+='## 本次回答的研究问题\n\n仅扩大战略价格表示，原single MILP、公开观测、现金核算及完整冷却不变。用户要求优先轻量，因此本轮不是附件九模型16-update/24测试确认实验，而是8-update、8测试日探索。reward-scaled B仅作为工程共同基准，不是已证明胜者。\n\n'
    main+='采用任务书 $V=Sc(\\alpha+\\beta\\psi)$，alpha10在前、beta10在后，near/far按hub交错，psi=(q-1)/(K-1)，K=1为0，terminal归0。已有方案用(1-|alpha|)压缩斜率，本轮未采用；价格包络从±Sc扩至±2Sc，形状与包络不能完全分离归因。\n\n'
    main+='## 先检查表示是否真的参与决策\n\n'+table+'\n'
    if p:
        main+='## 同预算完整冷却利润（USD/日）\n\n|工况|M0|MC|PPO10/B8|PPO20/8|20−10|20−MC|\n|---|---:|---:|---:|---:|---:|---:|\n'
        for r,label in zip(means,LABELS):
            main+=f'|{label}|{r["M0"]:.2f}|{r["MC"]:.2f}|{r["PPO10"]:.2f}|{r["PPO20"]:.2f}|{r["PPO20"]-r["PPO10"]:.2f}|{r["PPO20"]-r["MC"]:.2f}|\n'
        main+='\n配对单位为三训练seed×八完整场景的交叉结构，不是24独立场景。两万次bootstrap不会增加真实样本信息；每个比较族三个工况Bonferroni调整。验证只看update8，未用新测试选架构/超参数/训练seed。测试来自此前生成但未执行的22068–22075密封CFS池，不声称新增真实日期。\n\n'
        main+='|比较|工况|均差|95%区间|Bonferroni3区间|三个seed均差|\n|---|---|---:|---|---|---|\n'
        for r in effects:
            main+=f'|{r["comparison"]}|{LABELS[NAMES.index(r["condition"])]}|{r["mean_USD"]:.2f}|{r["CI95"][0]:.2f}, {r["CI95"][1]:.2f}|{r["Bonferroni3"][0]:.2f}, {r["Bonferroni3"][1]:.2f}|'+', '.join(f'{v:.2f}' for v in r['seed_means_USD'])+'|\n'
        main+='\n## 表达、学习和业务动作的区别\n\n'
        for name,label in zip(NAMES,LABELS):
            q=[r for r in sensitivity if r['condition']==name and r['control']=='beta0'];ds=[r for r in diagnostics if r['condition']==name]
            main+=f'{label}：已学beta归零后同状态canonical变化 {sum(r["changed_vs_q20"] for r in q)}/{len(q)}；训练记录KL早停 {sum(r["KL_early_stop"] for r in ds)}/{len(ds)}。完整alpha/beta曲线、逐update Critic RMSE/EV、KL/clip/梯度与各seed见CSV。\n\n'
        main+='首个已学切换反事实采用相同隐藏完整订单日、下期共同B8/30001；未来只用于环境评价，不传Actor。案例只是机制线索，不是总体政策增益。\n\n'
    else:main+='预设可控性门未满足，停止训练；后续模型/测试/利润优势均未执行，不能制造空表或称PPO20普遍无效。\n\n'
    main+='## 任务书十问的答案\n\n'
    answers=[
        '不同q都被构建查询，但占用/有效变化必须另看；覆盖表区分三种分母。',
        '在本批存在等式商空间中的新增方向；不是精确整数可行域维数证明。',
        f'固定alpha人工beta共发生 {sum(r["changed"] for r in pilot["coverage"])} 次canonical改变；不把只换车辆ID算业务变化。',
        '若训练已执行，价格CSV保留每个状态各hub的alpha/beta；参数非零和实际动作作用分别统计。',
        '按20−10各工况效应和调整区间判断，不以某seed最高利润选赢家。' if p else '未训练，未评价。',
        '按20−MC的独立次比较族判断；是否缩差不等于稳定优于MC。' if p else '未训练，未评价。',
        '本次是小样本探索，所有seed/工况保留；区间包含0不等于无效。',
        '人工可控、已学斜率、整数切换和实际利润是四层证据，不能因利润较低直接判代码错误。',
        'beta0同状态消融可定位直接接入，仍不能排除随机训练路径和20维探索几何差异。',
        '先据轻量结果决定是否值得16update确认；不自动扩训练、换算法或转联合动作。']
    main+='\n'.join(f'{i+1}. {a}' for i,a in enumerate(answers))+'\n\n'
    main+=f'## 成本与证据边界\n\n数值阶段累计 {budget["seconds"]/60:.2f} 分钟，{budget["calls"]} 次求解预约；硬上限30分钟/20000次/2GiB。编辑、最终复核另计，不能混入已测数值时间。旧PPO10训练只读复用不是免费训练。原历史最终测试与现金回放只引用其冻结QA，不说本次重跑整个历史。当前QA独立重算全部新增现金/冷却/价格/GAE/Actor与训练更新、归一化整数动作及所有实际输入/产物SHA；最终通过清单见qa。\n'
    text('structured_ppo_q_vs_mc_report.md',main)


def figures(pilot,means,curves,sensitivity,diagnostics,validation):
    # Bundled Python lacks Matplotlib; existing repo Matplotlib is used only here.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'svg.fonttype':'none'})
    def save(fig,name,sources):
        fig.tight_layout();fig.savefig(TARGET/'figures'/f'{name}.svg');fig.savefig(TARGET/'figures'/f'{name}.png',dpi=600)
        write_json(TARGET/'figures'/f'{name}.json',dict(source_csv=sources,dpi=600,units='USD unless fraction/count',
            size_inches=list(fig.get_size_inches()),style='colorblind-friendly; markers/linestyles',exploratory=True));plt.close(fig)
    if not means:
        fig,ax=plt.subplots(figsize=(5,3));ax.bar(LABELS,[sum(r['changed'] for r in pilot['coverage'] if r['condition']==n) for n in NAMES])
        ax.set_ylabel('Canonical switches /144 probes');save(fig,'01_control_probe',['q_slot_coverage.csv']);return
    fig,axs=plt.subplots(1,3,figsize=(9,3))
    for ax,r,label in zip(axs,means,LABELS):
        methods=('M0','MC','PPO10','PPO20');ax.bar(methods,[r[k] for k in methods],color=['.6','#E69F00','#0072B2','#009E73'])
        ax.set_title(label);ax.tick_params(axis='x',rotation=30);ax.set_ylabel('Full-cooldown profit (USD/day)')
    save(fig,'01_profit',['paired_results.csv'])
    fig,axs=plt.subplots(1,3,figsize=(9,3))
    for ax,name,label in zip(axs,NAMES,LABELS):
        for method,mark in (('PPO10','o'),('PPO20','s')):
            ys=[np.mean([r[method+'_profit_USD'] for r in validation if r['condition']==name and r['training_seed']==seed]) for seed in SEEDS]
            ax.plot([1,2,3],ys,marker=mark,label=method)
        ax.set_title(label);ax.set_xticks([1,2,3],['30001','30002','30003']);ax.set_ylabel('Validation profit at update8 (USD)');ax.legend()
    save(fig,'02_validation',['validation_results.csv'])
    fig,axs=plt.subplots(1,3,figsize=(9,3))
    for ax,name,label in zip(axs,NAMES,LABELS):
        first=next(r for r in curves if r['condition']==name and r['control']=='q20')
        for control,style in (('q20','-'),('beta0','--')):
            rs=[r for r in curves if r['condition']==name and r['control']==control and
                all(r[k]==first[k] for k in ('training_seed','scenario_seed','period','hub','release_time'))]
            ax.plot([r['q'] for r in rs],[r['price_USD'] for r in rs],style,marker='o',label=control)
        ax.set_title(label+' fixed first query group');ax.set_xlabel('Release-group q');ax.set_ylabel('Strategic price (USD)');ax.legend()
    save(fig,'03_q_curves',['q_value_curves.csv'])
    fig,ax=plt.subplots(figsize=(5,3));x=np.arange(3)
    for k,control in enumerate(('old10','beta0','stochastic')):
        y=[np.mean([r['changed_vs_q20'] for r in sensitivity if r['condition']==n and r['control']==control]) for n in NAMES]
        ax.bar(x+(k-1)*.25,y,.25,label=control)
    ax.set_xticks(x,LABELS);ax.set_ylabel('Same-state canonical switch fraction');ax.legend();save(fig,'04_switches',['action_sensitivity.csv'])
    fig,axs=plt.subplots(1,3,figsize=(9,3))
    for ax,name,label in zip(axs,NAMES,LABELS):
        for seed,mark in zip(SEEDS,('o','s','^')):
            rs=[r for r in diagnostics if r['condition']==name and r['training_seed']==seed]
            ax.plot([r['update'] for r in rs],[r['beta_mean_abs'] for r in rs],marker=mark,label=str(seed))
        ax.set_title(label);ax.set_xlabel('PPO update');ax.set_ylabel('Mean absolute batch beta');ax.legend()
    save(fig,'05_beta_learning',['training_diagnostics.csv'])
    hashes={str(p):sha(p) for p in (TARGET/'figures').iterdir() if p.is_file()}
    write_json(TARGET/'qa/figure-hashes.json',hashes)


def plot():
    d=read_json(TARGET/'qa/plot-input.json')
    figures(**d)


def verify_exports():
    """All eight flat exports traced to raw owners, not just paired profit."""
    counts={}
    def rows(name):
        out=list(csv.DictReader((TARGET/name).open(encoding='utf-8-sig')));counts[name]=len(out);return out
    def equal(actual,expected):
        if expected is None:
            if actual!='':raise RuntimeError('NA export differs')
        elif isinstance(expected,bool):
            if actual!=str(expected):raise RuntimeError('boolean export differs')
        elif isinstance(expected,(float,int)):
            if float(actual)!=expected:raise RuntimeError('numeric export differs')
        elif actual!=str(expected):raise RuntimeError('text export differs')
    pilot=read_json(TARGET/'qa/pilot.json')
    for r in rows('q_slot_coverage.csv'):
        key=(r['condition'],int(r['seed']),int(r['period']))
        raw=next(v for v in pilot['coverage'] if (v['condition'],v['seed'],v['period'])==key)
        for k,v in raw.items():equal(r[k],v)
        d=read_json(TARGET/'pilot'/f'{key[0]}-s{key[1]}-t{key[2]}.json')
        occ=d['rows'][0]['probe']['plan']['solver']['slot_occupancy']
        for k,v in dict(constructed_groups=sum(x['q']==1 for x in occ),occupied_groups=sum(x['q']==1 and x['occupied'] for x in occ),
            occupied_groups_ge2=sum(x['q']==2 and x['occupied'] for x in occ)).items():equal(r[k],v)
    cache={}
    for r in rows('expressivity_probe.csv'):
        file=r['record_file']
        if file not in cache:cache[file]=read_json(file)
        d=cache[file]
        raw=next(v for v in d['rows'] if str(v['direction'])==r['direction'] and v['slope']==float(r['slope']))
        for k,v in dict(canonical=raw['canonical'],canonical_changed=raw['canonical']!=d['rows'][0]['canonical'],
            served=len(raw['probe']['plan']['assignments']),routes=len(raw['probe']['plan']['routes']),
            occupied_q_ge2=sum(x['occupied'] for x in raw['probe']['plan']['solver']['slot_occupancy'] if x['q']>=2),
            solver_seconds=raw['probe']['plan']['solver']['runtime']).items():equal(r[k],v)
    for r in rows('counterfactual_cases.csv'):
        source=pilot if r['stage']=='pilot' else read_json(TARGET/'qa/mechanism.json')
        raw=next(v for v in source['cases'] if v['condition']==r['condition'])
        for k,v in raw.items():equal(r[k],v)
    if pilot['promote']:
        for r in rows('training_diagnostics.csv'):
            d=read_json(TARGET/'training'/r['condition']/r['training_seed']/f'update-{int(r["update"]):04d}.json')
            aliases=dict(KL='approximate_kl',KL_early_stop='kl_early_stop',critic_RMSE_USD='critic_RMSE_USD',critic_EV='critic_EV')
            for k in ('training_mean_USD','KL','KL_early_stop','clip_fraction','policy_loss','value_loss','critic_RMSE_USD','critic_EV','gradient_norm'):
                equal(r[k],d[aliases.get(k,k)])
            if json.loads(r['alpha_mean'])!=d['action_mean'][:10] or json.loads(r['beta_mean'])!=d['action_mean'][10:] or json.loads(r['log_std'])!=d['log_std']:
                raise RuntimeError('actor export differs')
            equal(r['beta_mean_abs'],float(np.mean(np.abs(d['action_mean'][10:]))))
        for r in rows('validation_results.csv'):
            a=read_json(OLD/'training'/r['condition']/'B'/r['training_seed']/f'validation-0008-s{r["scenario_seed"]}.json')
            b=read_json(TARGET/'training'/r['condition']/r['training_seed']/f'validation-s{r["scenario_seed"]}.json')
            equal(r['PPO10_profit_USD'],a['ledger']['realized_profit']);equal(r['PPO20_profit_USD'],b['ledger']['realized_profit'])
        owners={}
        for p in (TARGET/'validation/probes').glob('*.json'):
            d=read_json(p);owners[d['condition'],d['training_seed'],d['spec']['scenario_seed'],d['spec']['period']]=d
        for r in rows('action_sensitivity.csv'):
            d=owners[r['condition'],int(r['training_seed']),int(r['scenario_seed']),int(r['period'])]
            raw=next(v for v in d['rows'] if v['control']==r['control'])
            equal(r['canonical'],raw['canonical']);equal(r['changed_vs_q20'],raw['canonical']!=d['rows'][1]['canonical'])
        price_index={}
        for key,d in owners.items():
            for r in d['rows']:
                for c in r['probe']['coefficients']:price_index[(*key,r['control'],c['hub'],c['time'],c['supply_slot'])]=c
        for r in rows('q_value_curves.csv'):
            c=price_index[r['condition'],int(r['training_seed']),int(r['scenario_seed']),int(r['period']),r['control'],r['hub'],int(r['release_time']),int(r['q'])]
            for k,other in dict(price_USD='value',alpha='alpha',beta='beta',psi='psi').items():equal(r[k],c[other])
        # Paired profit/statistics are independently checked in verify().
        counts['paired_results.csv']=len(list(csv.DictReader((TARGET/'paired_results.csv').open(encoding='utf-8-sig'))))
    write_json(TARGET/'qa/exports.json',dict(status='PASSED',row_counts=counts,all_columns_owners_checked=True))


def verify():
    import torch
    from .control import replay
    from .training import build
    from .analysis import crossed
    from experiments.cfs_direct_ppo_diagnosis.common import observation,canonical,restore
    from experiments.cfs_direct_ppo_v3.qa import no_mc_guard,replay as old_replay
    from experiments.cfs_direct_ppo_training_ablation_v6.backend import update
    from model_families.single.learning.ppo.networks import sample_action
    from model_families.single.learning.ppo.config import PPOConfig
    from experiments.cfs_ppo_strategic_control.study_execution import replay as base_replay
    from experiments.cfs_direct_ppo_v3.run import PARENT as mc
    from experiments.cfs_ppo_strategic_control.study_execution import action_from_plan,physical_hash
    from model_families.single.planning import compile_action
    from model_families.single.runner import state_snapshot,serialize_plan
    load();torch.set_num_threads(1);start=time.monotonic();days=0;learning_count=0;probes=0
    pilot=read_json(TARGET/'qa/pilot.json')
    def check_probe(d,row):
        spec=d['spec'];c=cfg(spec['condition']);scene=next(s for s in scenes(spec['condition'],'validation') if s['seed']==spec['scenario_seed'])
        ep,o=restore(scene,c,read_json(spec['source']),spec['period'])
        if state_snapshot(ep._sim)!=d['state']:raise RuntimeError('independent probe initial state')
        signal=row['signal'];p=row['probe']['plan'];info=copy.deepcopy(p['solver'])
        plan=compile_action(ep._sim,action_from_plan(p),info);serial=serialize_plan(plan,ep._sim.t,o.hash,info)
        if physical_hash(serial)!=row['probe']['action_hash'] or plan[0]!=p['operating_profit'] or info['accounting']!=p['accounting']:
            raise RuntimeError('independent probe integer/cash')
        groups={};fleet=len(ep._sim.vehs)
        for x in info['slot_occupancy']:groups.setdefault((x['time'],x['hub']),[]).append(x)
        for xs in groups.values():
            if [x['q'] for x in xs]!=list(range(1,fleet+1)) or any(a['occupied']<b['occupied'] for a,b in zip(xs,xs[1:])):
                raise RuntimeError('slot index/prefix')
        for x in row['probe']['coefficients']:
            i=c['action']['hubs'].index(x['hub']);ell=(x['time']-spec['period'])/(24-spec['period'])
            psi=(x['supply_slot']-1)/(fleet-1) if fleet>1 else 0
            v=c['action']['scales'][i]*((1-ell)*(signal[2*i]+psi*signal[10+2*i])+ell*(signal[2*i+1]+psi*signal[11+2*i]))
            if abs(v-x['value'])>1e-7:raise RuntimeError('independent probe price')
    for name in NAMES:
        val={s['seed']:s for s in scenes(name,'validation')}
        for p in (TARGET/'pilot').glob(f'{name}-nesting-*.json'):
            r=read_json(p);s=val[r['old']['scenario_seed']]
            old_replay(s,cfg(name),r['old']);replay(s,cfg(name),r['new']);days+=2
        for p in (TARGET/'pilot').glob(f'{name}-cf-*.json'):
            r=read_json(p);replay(val[r['scenario_seed']],cfg(name),r);days+=1
        for p in (TARGET/'pilot').glob(f'{name}-s*-t*.json'):
            d=read_json(p)
            for r in d['rows']:
                if canonical(r['probe']['plan'],d['state'])!=r['canonical']:raise RuntimeError('pilot canonical mismatch')
                check_probe(d,r)
                probes+=1
        if not pilot['promote']:continue
        train={s['seed']:s for s in scenes(name,'train')};test={s['seed']:s for s in scenes(name,'test')}
        for seed in SEEDS:
            c=cfg(name,seed);folder=TARGET/'training'/name/str(seed);net=build(seed)
            optim=(torch.optim.Adam(net.parameters(),lr=.0003),)
            sampling=torch.Generator().manual_seed(seed+1001);batches=torch.Generator().manual_seed(seed+1002)
            for u in range(1,9):
                batch=[]
                for i in range((u-1)*4+1,u*4+1):
                    r=read_json(folder/f'training-{i:04d}.json');replay(train[r['scenario_seed']],c,r);days+=1
                    for tr,l in zip(r['transitions'],r['learning']):
                        obs=observation(tr['observation']);a,z,lp,v=sample_action(net,obs,sampling)
                        if a.tolist()!=l['action'] or z.tolist()!=l['latent'] or lp.item()!=l['logp'] or v.item()!=l['value']:
                            raise RuntimeError('actor/latent RNG/logp/value independent reconstruction')
                        batch.append(dict(observation=obs,latent=z,logp=lp.item(),advantage=l['advantage'],**{'return':l['return']}));learning_count+=1
                diag=update(net,optim,batch,PPOConfig.model_validate(c['ppo']),batches,'B')
                original=read_json(folder/f'update-{u:04d}.json')
                for key in ('policy_loss','value_loss','approximate_kl','gradient_norm','clip_fraction'):
                    if diag[key]!=original[key]:raise RuntimeError('PPO optimization reconstruction '+key)
            saved=torch.load(folder/'recovery.pt',weights_only=False,map_location='cpu')
            if any(not torch.equal(v,saved['network'][k]) for k,v in net.state_dict().items()):raise RuntimeError('final training tensors')
            if not torch.equal(saved['sampling_rng'],sampling.get_state()) or not torch.equal(saved['minibatch_rng'],batches.get_state()):
                raise RuntimeError('final training RNG')
            for p in folder.glob('validation-s*.json'):
                r=read_json(p);replay(val[r['scenario_seed']],c,r);days+=1
            for method in ('PPO10','PPO20'):
                for s in TESTS:
                    r=read_json(TARGET/'evaluation'/f'{name}-{method}-{seed}-s{s}.json')
                    (old_replay if method=='PPO10' else replay)(test[s],c,r);days+=1
        baseline=read_json(mc/'protocol'/(name+'.json'));baseline['baseline']=str(mc/baseline['baseline'])
        for method in ('MC','M0'):
            c=copy.deepcopy(baseline)
            if method=='M0':c['baseline']=dict(method='zero')
            for s in TESTS:
                r=read_json(TARGET/'evaluation'/f'{name}-{method}-s{s}.json');base_replay(test[s],c,r);days+=1
        for p in (TARGET/'validation').glob(f'{name}-cf-*.json'):
            r=read_json(p);replay(val[r['scenario_seed']],cfg(name),r);days+=1
    for p in (TARGET/'validation/probes').glob('*.json'):
        d=read_json(p)
        for r in d['rows']:
            if canonical(r['probe']['plan'],d['state'])!=r['canonical']:raise RuntimeError('learned canonical mismatch')
            check_probe(d,r)
            probes+=1
    if pilot['promote']:
        raw=list(csv.DictReader((TARGET/'paired_results.csv').open(encoding='utf-8-sig')))
        for r in raw:
            for method in ('PPO10','PPO20','MC','M0'):
                suff='-'+r['training_seed'] if method.startswith('PPO') else ''
                d=read_json(TARGET/'evaluation'/f'{r["condition"]}-{method}{suff}-s{r["scenario_seed"]}.json')
                if float(r[method+'_profit_USD'])!=d['ledger']['realized_profit']:raise RuntimeError('paired export mismatch')
        stats=read_json(TARGET/'qa/statistics.json')
        for e in stats['effects']:
            other=e['comparison'].split('-')[1]
            x=np.array([[float(next(r for r in raw if r['condition']==e['condition'] and int(r['training_seed'])==seed and int(r['scenario_seed'])==s)['PPO20_profit_USD'])-
                         float(next(r for r in raw if r['condition']==e['condition'] and int(r['training_seed'])==seed and int(r['scenario_seed'])==s)[other+'_profit_USD']) for s in TESTS] for seed in SEEDS])
            if any(crossed(x)[k]!=e[k] for k in ('mean_USD','CI95','Bonferroni3','seed_means_USD')):raise RuntimeError('export statistics mismatch')
    verify_hashes(Path('/'),read_json(TARGET/'qa/figure-hashes.json'))
    if pilot['promote']:verify_hashes(Path('/'),read_json(TARGET/'protocol/models-frozen.json')['hashes'])
    m=read_json(TARGET/'manifest.json')
    if sha(TARGET/'protocol/preregistration.json')!=m['preregistration_sha256']:raise RuntimeError('protocol changed')
    verify_exports()
    write_json(TARGET/'qa/replay.json',dict(status='PASSED',days=days,learning_steps=learning_count,probes=probes,
        cash_price_cooldown=True,Actor_GAE_PPO_RNG_reconstructed=True,statistics_from_raw_recomputed=True,
        seconds=time.monotonic()-start,history_scope='used historical inputs only; not whole historical experiment replay'))


def freeze(reviewed):
    m=load()
    if not reviewed:raise RuntimeError('actual latest PNG visual review required')
    for name in ('pilot','analysis','replay','exports'):
        if read_json(TARGET/'qa'/(name+'.json'))['status']!='PASSED':raise RuntimeError('QA incomplete '+name)
    if read_json(TARGET/'qa/pilot.json')['promote']:
        for name in ('evaluation','mechanism','test-isolation','statistics'):
            if read_json(TARGET/'qa'/(name+'.json'))['status']!='PASSED':raise RuntimeError('QA incomplete '+name)
    xml=TARGET/'qa/pytest-final.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
    tests=sum(int(s.get('tests',0)) for s in suites)
    if tests<21 or any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped')):raise RuntimeError('real final pytest failed')
    write_json(TARGET/'qa/tests.json',dict(status='PASSED',tests=tests,xml_sha256=sha(xml)))
    write_json(TARGET/'qa/visual.json',dict(status='PASSED',reviewed_at=now(),png_count=len(list((TARGET/'figures').glob('*.png'))),
        note='latest PNGs actually viewed; SVG checked same-source XML, not independent render'))
    verify_hashes(Path('/'),m['history_hashes'])
    verify_hashes(Path('/'),read_json(TARGET/'qa/figure-hashes.json'))
    for p in (TARGET/'figures').glob('*.svg'):ET.parse(p)
    files={p.relative_to(TARGET).as_posix():sha(p) for p in TARGET.rglob('*') if p.is_file() and p.name not in ('manifest.json','final_manifest.json')}
    m.update(status='COMPLETED',completed_at=now(),artifact_hashes=files,
        delivery_source_sha256=sha(__file__),numeric_source_unchanged=sources()==m['source'])
    write_json(TARGET/'manifest.json',m);write_json(TARGET/'final_manifest.json',m)
    verify_hashes(TARGET,files)
    print('COMPLETED',len(files),'artifacts',sha(TARGET/'final_manifest.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('mechanism','analyze','plot','verify','freeze'));p.add_argument('--reviewed',action='store_true');p.add_argument('--directory');a=p.parse_args()
    if a.directory:
        if a.stage not in ('plot','analyze'):raise RuntimeError('directory override is nonnumerical-only')
        globals()['TARGET']=Path(a.directory)
        from . import common as localcommon
        localcommon.TARGET=Path(a.directory)
    if a.stage=='freeze':freeze(a.reviewed)
    else:globals()[a.stage]()


if __name__=='__main__':main()
