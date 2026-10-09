"""Evidence-gated report, whole-world lineage audit, and immutable delivery."""
import argparse
from pathlib import Path
import numpy as np
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.value.providers import MCTableValueProvider
from experiments.cfs_resource_policy_iteration.protocol import read_compressed
from .common import OUTPUT,HISTORY,ROOT,ALPHAS,key,file_hash,now,code_hashes
from .evaluation import blend
from .analysis import finalize_analysis

def world_audit(cfg):
    qp=OUTPUT/'qa/world_lineage_audit.json'
    if qp.exists():return read_json(qp)
    count=steps=calls=coefficients=0;hash_index={}
    for a in ALPHAS:
        for n in [2,3]:
            root=OUTPUT/'raw'/f'alpha_{a}';manifest=read_json(OUTPUT/'manifests'/f'alpha_{a}_raw_V{n}.json')
            assert file_hash(OUTPUT/manifest['targets_ref'])==manifest['target_sha256']
            tables={cid:read_json(OUTPUT/'tables'/f'alpha_{a}'/f'deployed_V{n-1}'/(cid+'.json')) for cid in [key(c) for c in cfg['conditions']]}
            providers={cid:MCTableValueProvider(model) for cid,model in tables.items()}
            prices={}
            for itempath in sorted((root/'targets'/f'iter{n}'/'raw').glob('*.json')):
                item=read_json(itempath)
                for wi in item['worlds']:
                    path=root/wi['path'];assert file_hash(path)==wi['sha256']
                    w=read_compressed(path);cid=w['condition'];line=manifest['lineage'][cid]
                    assert w['identity']==wi['identity'] and w['policy_hash']==line['parent_policy_hash']
                    assert w['table_hash']==line['parent_table_hash'] and w['replay']['status']=='PASSED'
                    assert abs(w['replay']['total_difference'])<=1e-6
                    assert w['metrics']['limited_window_count']==0 and w['metrics']['unfinished_committed']==0
                    for row in w['trace']:
                        if row.get('drain'):
                            assert not row['plan']['assignments'] and not row['plan']['routes'];continue
                        solver=row['plan']['solver'];assert solver['status']=='OPTIMAL'
                        assert solver.get('formulation')=='single-route-macro-action-slot-value/v1'
                        assert solver.get('information_mode')=='known_supply'
                        for v in solver.get('release_coefficients',[]):
                            pricekey=(cid,v['time'],v['hub'],v['supply_slot'])
                            if pricekey not in prices:
                                prices[pricekey]=providers[cid].marginal_release_value(v['time'],v['hub'],v['supply_slot'],dict(T=24))
                            assert v['value']==prices[pricekey], 'Stored coefficient not from deployed parent'
                            coefficients+=1
                    count+=1;steps+=w['replay']['periods'];calls+=w['value_calls']
                    hash_index[path.relative_to(OUTPUT).as_posix()]=wi['sha256']
            print(f'World audit alpha {a}, E(D{n}): {count} worlds cumulative.',flush=True)
    assert count==93600
    atomic_json(OUTPUT/'manifests/raw_world_sha256.json',dict(files=hash_index))
    report=dict(status='PASSED',worlds=count,replay_steps=steps,value_calls=calls,
        coefficients_independently_recomputed=coefficients,checked_at=now(),all_parent_lineage_correct=True,
        all_OPTIMAL=True,no_preview=True,full_cooldown=True,raw_world_index_sha256=file_hash(OUTPUT/'manifests/raw_world_sha256.json'))
    atomic_json(qp,report);return report

def blend_audit(cfg):
    rows=[]
    for a in ALPHAS:
        for n in range(4):
            f=read_json(OUTPUT/'manifests'/f'alpha_{a}_deployed_V{n}.json')
            for p in f['proof']:
                raw=read_json(p['raw_ref']);assert file_hash(p['raw_ref'])==p['raw_sha256']
                if n==0:expected=raw
                else:
                    previous=read_json(OUTPUT/p['previous_ref']);assert file_hash(OUTPUT/p['previous_ref'])==p['previous_sha256']
                    expected=blend(previous,raw,ALPHAS[a])
                assert read_json(OUTPUT/p['deployed_ref'])==expected
                if n<2:assert p['raw_sha256']==file_hash(HISTORY/'tables'/f'V{n}'/(p['condition']+'.json'))
                rows.append(p)
    result=dict(status='PASSED',tables=len(rows),checked_at=now(),exact_scalar_recalculation=True,
        common_V0_V1_identical_to_A=True,later_raw_evaluates_actual_damped_parent=True)
    atomic_json(OUTPUT/'qa/damping_recalculation.json',result);return result

def evidence(cfg):
    freeze=read_json(OUTPUT/'manifests/alpha_freeze.json');a=freeze['selected_alpha']
    formal=read_json(OUTPUT/'processed/formal_statistics.json');diag=read_json(OUTPUT/'processed/diagnostic_statistics.json')
    acts=read_json(OUTPUT/'processed/formal_action_summary.json')
    tables=read_json(OUTPUT/'processed/table_metrics.json');e=cfg['effectiveness']
    rows=[]
    w80=key(next(c for c in cfg['conditions'] if c['orders']==80 and c['region']=='western_multicluster'))
    safety=next(r for r in formal['paired'] if r['condition']==w80 and r['comparison']=='D4-U4')
    margin=e['W80_noninferiority_margin_fraction']*abs(safety['reference_mean'])
    safe=safety['CI95_lower']>-margin
    for c in cfg['conditions']:
        cid=key(c);profit=next(r for r in formal['paired'] if r['condition']==cid and r['comparison']=='D4-U4')
        dprofit=next(r for r in diag['paired'] if r['condition']==cid and r['comparison']=='D4-U4')
        def ac(path,n):return next(r for r in acts if r['condition']==cid and r['alpha']==path and r['kind']=='adjacent' and r['iteration']==n)['mean']
        def tr(path,n):return next(r for r in tables if r['condition']==cid and r['alpha']==path and r['kind']=='deployed' and r['iteration']==n)['relative_L2']
        d_action=float(np.mean([ac(a,3),ac(a,4)]));u_action=float(np.mean([ac('undamped',3),ac('undamped',4)]))
        action_ok=d_action<=e['action_improvement_ratio']*u_action and ac(a,4)<=ac(a,3)+e['action_trend_tolerance']
        d_table=float(np.mean([tr(a,3),tr(a,4)]));u_table=float(np.mean([tr('undamped',3),tr('undamped',4)]))
        table_ok=d_table<=e['table_worsening_ratio']*u_table and tr(a,4)<=e['table_worsening_ratio']*tr(a,2)
        initial=next(r for r in formal['summary'] if r['condition']==cid and r['policy']=='U1')['mean']
        profit_ok=profit['mean']>=e['profit_margin_fraction']*abs(initial) and profit['Holm_p']<=.05
        direction=dprofit['mean']>0 and profit['mean']>0
        joint=profit_ok and action_ok and table_ok and direction and safe and not freeze['exploratory_selection']
        rows.append(dict(condition=cid,alpha=a,profit_difference=profit['mean'],paired_t95=[profit['CI95_lower'],profit['CI95_upper']],
            paired_bootstrap95=[profit['bootstrap95_lower'],profit['bootstrap95_upper']],Holm_p=profit['Holm_p'],
            diagnostic_profit_difference=dprofit['mean'],late_damped_action=d_action,late_undamped_action=u_action,
            late_damped_relativeL2=d_table,late_undamped_relativeL2=u_table,
            profit_evidence=profit_ok,action_evidence=action_ok,table_evidence=table_ok,
            direction_agrees=direction,W80_noninferiority_supported=safe,
            joint_stabilization_supported=joint,
            interpretation='stabilization supported under preregistered criteria' if joint else
                'profit improvement only / stabilization not established' if profit['mean']>0 else 'stabilization not established'))
    primary=[r for r in rows if r['condition'].startswith('california') or '-n40-' in r['condition']]
    result=dict(alpha=a,W80_margin_USD=margin,W80_t95_lower=safety['CI95_lower'],
        W80_noninferiority_supported=safe,regimes=rows,
        damping_effective=any(r['joint_stabilization_supported'] for r in primary),
        failure_does_not_prove_ineffectiveness=True,no_fixed_point_residual=True)
    atomic_json(OUTPUT/'processed/evidence.json',result);return result

def report(cfg):
    e=evidence(cfg);selection=read_json(OUTPUT/'manifests/alpha_freeze.json')
    stats=read_json(OUTPUT/'processed/formal_statistics.json');a=e['alpha']
    text=['# A2：真正的自一致阻尼资源价值策略迭代','',
        '数值与分析阶段已完成。完整测试、视觉审阅、历史只读校验与最终冻结状态，以 `manifests/final_manifest.json` 为准。','',
        '## 1. 已确认的研究口径','',
        '用户于2026-10-07明确确认真正的自一致更新。共同 raw V0=E(M0)、raw V1=E(D1)=E(U1)保留历史一致性；各 alpha 后续 raw V2、V3重新评价其实际 D2、D3，不能要求它们继续等于 A 的 undamped表。',
        'Vtilde0=V0；Vraw_n=E(Dn)；Vtilde_n=(1−alpha)Vtilde_(n−1)+alpha Vraw_n；D(n+1)=G(Vtilde_n)。E是完整冷却J(q)−J(q−1)，没有J−g或ACTION correction。',
        '唯一变量alpha=.25/.50/.75。物理、MILP、参数、construction场景、time/hub/q网格、线性插值、负值及终端规则与A相同。不同release(time,hub)可分计价仍是架构限制，不是完整联合状态Bellman价值。','',
        '## 2. 历史复现、样本与冻结顺序','',
        'A的223个交付文件校验、16张表从全保存标签精确重建、20次旧政策独立求解、16组旧资源配对标签独立重算通过。前六轮只读索引记录全部原文件内容hash，未调用历史writer。',
        'Construction8000–8059；diagnostic8200–8219；pilot8950–8951；formal9100–9139。每工况正式n=40，独立单位scenario；policy seed=0，training seed=null。相同场景跨算法配对，t/hub/q/alpha不是额外独立replication。',
        '共同E0/E1在来源一致性证明后复用；3 alpha×2次新E×15,600=93,600反事实世界，均保存完整逐t状态、动作、ledger、冷却及回放。D1=U1精确别名：formal1440标签、1280独立仿真，不增加场景统计自由度。','',
        '## 3. 诊断alpha选择','',
        f'全局冻结alpha={ALPHAS[a]:g}，exploratory_selection={selection["exploratory_selection"]}。主排序只使用CA/W40诊断数据，W80利润安全筛查。formal和pilot在选择冻结后才运行。',
        '利润崩溃定义：D4−D1均值损失超过D1绝对均值5%，且配对t95上限<0；表末步relativeL2比首步上升>10%；动作step4比step2上升>2个百分点。最后两步动作差优先，1个百分点内近似；表L2其次，.02内近似；利润最后，1%U0尺度内近似；最终优先.5。全部筛查不合格时按预注册最少失败项继续固定矩阵，但不宣称稳定化。','',
        '| alpha | 筛查失败数 | 最后两步动作差 | 最后两步relativeL2 | 诊断D4均值USD |','|---|---:|---:|---:|---:|']
    for r in selection['records']:text.append(f'|{r["alpha_value"]:g}|{r["failed_screens"]}|{r["last_two_action"]:.4f}|{r["last_two_relative_L2"]:.4f}|{r["diagnostic_mean_profit"]:.2f}|')
    text+=['','## 4. 新正式集利润与配对比较','',
        '| 策略 | '+' | '.join(key(c) for c in cfg['conditions'])+' |','|---|'+'---:|'*4]
    for label in [f'U{i}' for i in range(5)]+[f'D{i}' for i in range(1,5)]:
        text.append('| '+label+' | '+' | '.join(f'{next(r for r in stats["summary"] if r["condition"]==key(c) and r["policy"]==label)["mean"]:.2f}' for c in cfg['conditions'])+' |')
    text+=['','主family跨4工况的D4−U4统一两侧Holm；20项路径比较和8项中间跨路径差分别校正。t95和10,000次paired bootstrap未作多重区间校正，与Holm p分列。SD、median、skewness和逐scenario差全部保存。','',
        '| 工况 | D4−U4 USD | t95 | bootstrap95 | Holm p | 诊断D4−U4 |','|---|---:|---|---|---:|---:|']
    for r in e['regimes']:text.append(f'|{r["condition"]}|{r["profit_difference"]:+.2f}|{r["paired_t95"]}|{r["paired_bootstrap95"]}|{r["Holm_p"]:.6g}|{r["diagnostic_profit_difference"]:+.2f}|')
    text+=['','## 5. 稳定化联合证据','',
        f'预注册damping effectiveness证据：{e["damping_effective"]}。不满足证据阈值并不证明阻尼无效，更不证明结构极限或DRL必要。',
        f'W80非劣容差={e["W80_margin_USD"]:.2f} USD；D4−U4 t95下限={e["W80_t95_lower"]:.2f}；支持非劣={e["W80_noninferiority_supported"]}。不显著不等于无伤害。','',
        '| 工况 | profit证据 | action证据 | table证据 | 方向一致 | 联合稳定化 |','|---|---|---|---|---|---|']
    for r in e['regimes']:text.append('| '+r['condition']+' | '+' | '.join(str(r[k]) for k in ['profit_evidence','action_evidence','table_evidence','direction_agrees','joint_stabilization_supported'])+' |')
    text+=['','动作使用固定U0完整状态库、双方独立求解、全部24期，top-level排序不改变路线序列和车辆ID；正式40场景只是冻结后泛化佐证。自己的闭环状态分布单独报告SMD、Wasserstein和支持范围外比例。SMD零方差且均值不同不填零。',
        '振荡全部原始/部署单格路径、符号变化、方向反转、range和自然idle-q占据率保留；图中展示选定alpha各工况range最大的5格。这是有限四步描述，不证明极限环。',
        '负值保留，q非单调不修正；表差同时包含parent政策和自然状态分布变化，不能解读为固定完整状态上的纯operator因果差。MC_SE条件于有限经验parent，不包括整条PI链重新估计不确定性。','',
        '阻尼公式机械地缩小单步部署表更新，因此不能仅凭deployed relativeL2/RMSE下降声称稳定化；raw/deployed变化同时导出，动作、正式利润与W80安全证据不可省略。','',
        '状态分布中的release_delay_mean使用只读rolling.audit最终idle_release ETA，历史last-delivery摘要offset另存以审计。此处仅后处理，不改冻结的E/G或仿真物理，不参与alpha选择；diagnostic开始前已固定分析源码。','',
        '## 6. 运行时、QA、交付与限制','',
        'offline wall与worker累计分列，共同E0/E1复用不伪造新的计算时间；在线context/lookup/build/solver/transition/ledger与完整pipeline分开，嵌套计时不能重复相加。Gurobi均为受限MILP OPTIMAL，不是全日全局最优证明。',
        '运行时是本地共享机器的流程实测；alpha=.25的部分construction曾与实现预检并行，不是隔离硬件的微基准。小幅耗时差不能直接归因alpha；QA保存实际测试起止与elapsed供审计。',
        '源码、查询、parent table/policy、共同future与配对资源底座身份逐世界检查；保存系数从实际parent表独立复算；T后无新dispatch、完整承诺冷却、全部现金与物理回放通过。阻尼表逐格精确复算，D1/U1别名明确。',
        '九组论文图提供600dpi PNG、字体嵌入矢量PDF、SVG、source CSV和plot config，并提供九页本地图册。测试和渲染检查完成后再冻结全部A2，之后才启动B数值实验。',
        '没有E(D4)、没有固定点残差或理论收敛证明。没有bucket、ACTION correction、神经网络、DRL、SAC/PPO、平台默认算法更新或Git提交/推送。剩余机制需独立B实验及支持充分性审查，不能仅凭本轮失败宣称Strategic RL必要。','',
        '详细文件导航、字段及恢复入口见README。processed CSV可由Python直接读取；所有统计要按scenario配对，不按文件行数扩大样本。']
    (OUTPUT/'damping_policy_iteration_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return e

def verify_history():
    historical=read_json(OUTPUT/'manifests/historical_readonly.json')
    for i,(name,sha) in enumerate(historical['files'].items()):
        assert file_hash(ROOT/name)==sha,name
        if (i+1)%20000==0:print(f'Final historical byte check {i+1}/{len(historical["files"])}',flush=True)
    assert file_hash(Path(read_json(OUTPUT/'manifest.json')['baseline_config']['data_source']))==historical['CFS_sha256']
    result=dict(status='PASSED',files=len(historical['files']),checked_at=now(),CFS_verified=True)
    atomic_json(OUTPUT/'qa/historical_integrity.json',result);return result

def main(stage):
    cfg=read_json(OUTPUT/'config/main.json')
    if stage=='analysis':finalize_analysis(cfg);report(cfg);return
    if stage=='worlds':blend_audit(cfg);world_audit(cfg);return
    if stage=='history':verify_history();return
    assert stage=='freeze'
    m=read_json(OUTPUT/'manifest.json');assert code_hashes()==m['source_code_sha256']
    assert file_hash(OUTPUT/'analysis.py')==read_json(OUTPUT/'manifests/selection_source_freeze.json')['sha256']
    for name in ['baseline_regression','world_lineage_audit','damping_recalculation','historical_integrity','pdf_visual_audit','tests']:
        assert read_json(OUTPUT/'qa'/(name+'.json'))['status']=='PASSED',name
    assert (OUTPUT/'damping_policy_iteration_report.md').exists()
    fm=read_json(OUTPUT/'figures/figure_manifest.json');assert len(fm['names'])==9
    selection=read_json(OUTPUT/'manifests/alpha_freeze.json')
    formal=read_json(OUTPUT/'manifests/formal_matrix.json')
    assert selection['created_at']<formal['completed_at'] and formal['runs']==1440
    files={}
    for folder in ['config','policies','tables','processed','figures','qa']:
        for p in (OUTPUT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:files[p.relative_to(OUTPUT).as_posix()]=file_hash(p)
    for p in [OUTPUT/'README.md',OUTPUT/'damping_policy_iteration_report.md']:
        files[p.relative_to(OUTPUT).as_posix()]=file_hash(p)
    m.update(status='COMPLETED',completed_at=now(),selected_alpha=selection['selected_alpha'],deliverable_sha256=files,
        A2_frozen_before_B=True)
    atomic_json(OUTPUT/'manifest.json',m);atomic_json(OUTPUT/'manifests/final_manifest.json',m)
    print('A2 completed and immutable. B may now start.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['analysis','worlds','history','freeze'],required=True)
    main(p.parse_args().stage)
