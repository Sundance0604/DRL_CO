"""Evidence-backed QA and report; completion requires tests and visual QA."""
import time,zipfile
from collections import defaultdict
from xml.etree import ElementTree
import numpy as np
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.value.providers import MCTableValueProvider
from .common import OUTPUT,ROOT,key,file_hash,now,verify_history,numerical_hashes,INTERPOLATION,TERMINAL,write_csv
from .policies import load_policies
from .protocol import read_compressed,restore,continuation,action_hash

def value_audit(trace,policy):
    calls=0;solves=0
    provider=None if policy['table'] is None else MCTableValueProvider(policy['table'])
    for r in trace:
        if r.get('drain'):continue
        info=r['plan']['solver'];assert info['status']=='OPTIMAL';solves+=1
        if provider:
            assert info.get('formulation')=='single-route-macro-action-slot-value/v1','Value-policy trace used another formulation or hidden M0'
            assert info.get('information_mode')=='known_supply'
            for c in info['release_coefficients']:
                assert c['time']<24
                expected=provider.marginal_release_value(c['time'],c['hub'],c['supply_slot'],dict(T=24))
                assert abs(c['value']-expected)<=1e-9;calls+=1
    return calls,solves

def audit():
    cfg=read_json(OUTPUT/'config/main.json');m=read_json(OUTPUT/'manifest.json')
    assert numerical_hashes()==m['source_code_sha256']
    policies={n:load_policies(n) for n in range(5)};index=[];worlds=pairs=steps=solves=calls=0
    comparisons=[];source_policy_checks=[]
    for n in range(4):
        iteration=read_json(OUTPUT/'manifests'/f'iteration_{n}_manifest.json')
        for c in cfg['conditions']:
            cid=key(c);lineage=iteration['lineage'][cid];p=policies[n][cid]
            assert lineage['continuation_policy_hash']==p['policy_hash']
            assert lineage['continuation_table_hash']==p.get('table_hash')
            assert lineage['interpolation_rule_hash']==digest(INTERPOLATION) and lineage['terminal_rule_hash']==digest(TERMINAL)
            assert lineage['table_hash']==policies[n+1][cid]['table_hash']
            for seed in cfg['construction_seeds']:
                j=read_json(OUTPUT/'targets'/f'iter{n}'/'raw'/f'{cid}-s{seed}.json')
                parent=read_compressed(OUTPUT/j['parent_trace'])
                assert parent['policy_hash']==p['policy_hash'] and parent['table_hash']==p.get('table_hash')
                pc,ps=value_audit(parent['trace'],p)
                assert pc==parent['value_calls'];solves+=ps;calls+=pc
                snapshots={};profits={};futurehashes=set()
                for record in j['worlds']:
                    path=OUTPUT/record['path'];w=read_compressed(path)
                    assert w['identity']==record['identity'] and w['policy_hash']==p['policy_hash'] and w['table_hash']==p.get('table_hash')
                    assert w['replay']['status']=='PASSED' and abs(w['replay']['total_difference'])<1e-6
                    assert w['metrics']['unfinished_committed']==0
                    assert w['replay']['periods']==len(w['trace']);worlds+=1;steps+=len(w['trace'])
                    wc,ws=value_audit(w['trace'],p);assert wc==w['value_calls'];calls+=wc;solves+=ws
                    for row in w['trace']:
                        if row.get('drain'):continue
                        assert all(all(row['invariants'].values()) for _ in [0])
                        visible={o['id'] for o in row['before']['pool']}
                        assert all(a['order_id'] in visible for a in row['plan']['assignments'])
                    snapshots[w['time'],w['hub'],w['q']]=w['initial']
                    profits[w['time'],w['hub'],w['q']]=w['metrics']['realized_profit'];futurehashes.add(w['future_hash'])
                    index.append(dict(path=record['path'],sha256=file_hash(path),iteration=n,condition=cid,seed=seed,
                        policy_hash=p['policy_hash'],table_hash=p.get('table_hash'),replay_steps=len(w['trace']),provider_calls=wc))
                assert len(futurehashes)==1
                for row in j['rows']:
                    if row['terminal_anchor']:continue
                    t,h,q=row['time'],row['hub'],row['q'];a,b=snapshots[t,h,q-1],snapshots[t,h,q]
                    assert {k:v for k,v in a.items() if k!='vehicles'}=={k:v for k,v in b.items() if k!='vehicles'}
                    assert a['vehicles']==b['vehicles'][:-1] and digest(a)==row['paired_common_hash']
                    assert abs(profits[t,h,q]-profits[t,h,q-1]-row['target'])<1e-6;pairs+=1
                if seed==cfg['construction_seeds'][0]:
                    record=j['worlds'][1];saved=read_compressed(OUTPUT/record['path'])
                    scenario=read_json(OUTPUT/saved['scenario_ref'])['scenarios'][0]
                    from .common import parameters,rho_levels
                    s,o=restore(scenario,parameters(m['baseline_config'],rho_levels(scenario,m['baseline_config'])['rho_max']),
                        m['baseline_config']['solver'],saved['initial'])
                    new=continuation(s,o,p,read_json(OUTPUT/'tables/priors.json')[cid])
                    assert abs(new['metrics']['realized_profit']-saved['metrics']['realized_profit'])<1e-6
                    assert new['final_hash']==saved['final_hash']
                    assert [action_hash(r['plan']) for r in new['trace'] if not r.get('drain')]==[action_hash(r['plan']) for r in saved['trace'] if not r.get('drain')]
                    comparisons.append(dict(iteration=n,condition=cid,profit_difference=new['metrics']['realized_profit']-saved['metrics']['realized_profit'],
                        physical_and_actions_identical=True,parent_policy_hash=p['policy_hash'],table_hash=p.get('table_hash')))
            source_policy_checks.append(dict(iteration=n,condition=cid,parent=p['policy_id'],parent_hash=p['policy_hash'],
                parent_table=p.get('table_ref'),new_policy=policies[n+1][cid]['policy_id'],new_policy_table_hash=lineage['table_hash']))
        print(f'QA iter{n}: cumulative worlds={worlds}, paired marginals={pairs}, steps={steps}',flush=True)
    write_csv(OUTPUT/'processed/world_integrity_index.csv',index)
    lineage=dict(status='PASSED',rows=source_policy_checks,worlds=worlds,provider_calls=calls,optimizer_periods=solves,
        every_value_period_parent_coefficients_recomputed=True,no_hidden_zero_fallback=True)
    atomic_json(OUTPUT/'qa/policy_lineage_audit.json',lineage)
    atomic_json(OUTPUT/'qa/paired_world_audit.json',dict(status='PASSED',worlds=worlds,pairs=pairs,
        initial_common_prefix_exact=True,only_extra_idle=True,future_path_common=True,targets_recomputed=True))
    atomic_json(OUTPUT/'qa/independent_reproduction.json',dict(status='PASSED',runs=len(comparisons),rows=comparisons))
    atomic_json(OUTPUT/'qa/information_boundary_audit.json',dict(status='PASSED',worlds=worlds,
        pure_slot_coefficients_verified=calls,nonoptimal_solves=0,assigned_only_observed_pool=True,oracle_in_E=False,
        provider_actual_future_inputs=False,action_correction=False,formal_for_tables=False))
    runs=[]
    for stage in ['diagnostic','pilot','formal']:runs+=read_json(OUTPUT/'processed'/f'{stage}_runs.json')
    for r in runs:
        assert r['status']=='COMPLETED' and abs(r['replay_profit_difference'])<1e-6 and r['unfinished_committed']==0
        assert file_hash(OUTPUT/r['trace_ref'])==r['trace_sha256']
        data=read_compressed(OUTPUT/r['trace_ref'])
        if r['policy'].startswith('P'):
            vc,vs=value_audit(data['trace'],policies[int(r['policy'][1:])][r['condition']]);assert vc==r['value_calls']
        for row in data['trace']:
            if not row.get('drain'):assert row['plan']['solver']['status']=='OPTIMAL'
    replay=dict(status='PASSED',resource_worlds=worlds,resource_world_steps=steps,
        policy_runs=len(runs),policy_steps=sum(r['replay_steps'] for r in runs),all_physical_exact=True,cash_tolerance_USD=1e-6)
    atomic_json(OUTPUT/'qa/replay_audit.json',replay)
    return dict(lineage=lineage,replay=replay,independent_reproduction=len(comparisons))

def convergence(cfg,stats,tab,states,actions):
    output=[];threshold=cfg['convergence']
    for c in cfg['conditions']:
        cid=key(c);last=[]
        for k in [3,4]:
            gain=next(r for r in stats['comparisons'] if r['condition']==cid and r['comparison']==f'P{k}-P{k-1}')
            table=next(r for r in tab['changes'] if r['condition']==cid and r['to_V']==k-1 and r['scope']=='observed_shared' and r['segment']=='all')
            action=next(r for r in actions if r['condition']==cid and r['iteration']==k)
            shift=[r for r in states['shifts'] if r['condition']==cid and r['from_pi']==k-1 and r['absolute_SMD'] is not None]
            undefined=sum(r['absolute_SMD'] is None for r in states['shifts'] if r['condition']==cid and r['from_pi']==k-1)
            smd=float(np.mean([r['absolute_SMD'] for r in shift]));margin=threshold['paired_gain_fraction']*abs(gain['reference_mean'])
            last.append(dict(step=k,paired_gain_USD=gain['mean'],paired_gain_CI=[gain['CI95_lower'],gain['CI95_upper']],
                mean_gain_small=abs(gain['mean'])<=margin,CI_within_equivalence_band=gain['CI95_lower']>=-margin and gain['CI95_upper']<=margin,
                table_relative_L2=table['relative_L2'],action_disagreement=action['mean'],state_mean_abs_SMD=smd,undefined_SMD_cells=undefined,
                jointly_small=abs(gain['mean'])<=margin and table['relative_L2']<=threshold['relative_table_l2'] and
                    action['mean']<=threshold['action_disagreement'] and smd<=threshold['mean_abs_SMD'] and undefined==0))
        # Small thresholds alone insufficient: last two table/action/state changes must shrink.
        nonincreasing=all(last[1][f]<=last[0][f]+1e-12 for f in ['table_relative_L2','action_disagreement','state_mean_abs_SMD'])
        stable=all(r['jointly_small'] for r in last) and nonincreasing
        pi1=next(r for r in stats['comparisons'] if r['condition']==cid and r['comparison']=='P1-P0')
        pi4=next(r for r in stats['comparisons'] if r['condition']==cid and r['comparison']=='P4-P0')
        oracle=next(r for r in stats['comparisons'] if r['condition']==cid and r['comparison']=='D-P4')
        adjacent=[next(r for r in stats['comparisons'] if r['condition']==cid and r['comparison']==f'P{k}-P{k-1}') for k in range(1,5)]
        mean_alternations=sum(a['mean']*b['mean']<0 for a,b in zip(adjacent,adjacent[1:]))
        corrected_regressions=[r['comparison'] for r in adjacent if r['mean']<0 and r['Holm_p']<=.05]
        improved=pi4['mean']>0 and pi4['Holm_p']<=.05
        category=('empirically stable with positive descriptive preview gap; H-C-compatible, gap magnitude reported separately' if stable and oracle['CI95_lower']>0 else
            'H-A-compatible empirical stability and multiplicity-corrected improvement' if stable and improved else
            'empirically stable without corrected cumulative improvement' if stable else
            'H-B-compatible corrected adjacent deterioration; not a proof of a cycle' if corrected_regressions else
            'not empirically stabilized in four fixed improvements')
        output.append(dict(condition=cid,last_two_steps=last,nonincreasing_joint=nonincreasing,empirically_stabilized=stable,
            descriptive_CI_band_containment=all(r['CI_within_equivalence_band'] for r in last),
            pi1_gain=pi1['mean'],pi4_gain=pi4['mean'],pi4_Holm_p=pi4['Holm_p'],preview_gap=oracle['mean'],
            adjacent_mean_direction_alternations=mean_alternations,corrected_adjacent_regressions=corrected_regressions,
            corrected_cumulative_improvement=improved,category=category,
            caveat='restricted class/protocol only; no exact convergence or global optimality'))
    return output

def markdown_table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'

def report(qa):
    cfg=read_json(OUTPUT/'config/main.json');m=read_json(OUTPUT/'manifest.json');stats=read_json(OUTPUT/'processed/policy_statistics.json')
    tab=read_json(OUTPUT/'processed/table_diagnostics.json');states=read_json(OUTPUT/'processed/state_diagnostics.json');actions=read_json(OUTPUT/'processed/action_summary.json')
    conclusions=convergence(cfg,stats,tab,states,actions);atomic_json(OUTPUT/'processed/stabilization_assessment.json',conclusions)
    text=['# 第六次数值实验 A：Self-Consistent MC Resource-Value Policy Iteration','',
        '研究日期：2026-10-07。模型族：single。独立目录：`experiments/cfs_resource_policy_iteration/`。',
        '数值矩阵与研究QA已完成；最终测试/视觉审查/交付状态以 `manifests/final_manifest.json` 为准。',
        '', '## 1. 研究设计与数学含义','',
        '本轮固定四次改善：pi0=M0/Zero；V0=E(pi0)、pi1=G(V0)；V1=E(pi1)、pi2=G(V1)；V2=E(pi2)、pi3=G(V2)；V3=E(pi3)、pi4=G(V3)。',
        '唯一目标是 $V^{pi}_{res}(t,u,q)=E[J^{pi}(S_t^{u,q})-J^{pi}(S_t^{u,q-1})]$。J 使用完整 realized ledger，包括 T 后承诺冷却；没有扣 g，没有 ACTION correction、其他拟合器、Fluid 更新、神经网络或 DRL。',
        'G 保留原 H=1 资源 slot-prefix MILP。pi0 保留原 M0；pi1..pi4 只用对应前一张 V 表，每期重规划。所有价值系数为求解前标量，没有新增非线性 joint post-state 或 route-slot 联合变量。',
        '接口细节：每辆车的路径流只进入一个sink，按其release(time,hub)分组；每组sum(prefix slots)=sum(release indicators)，且z_q≤z_(q−1)。因此负价不能通过少计车辆或跳选后槽被暗中截断。不同释放时刻的组分别计价，G并未重建跨时刻完整联合库存或重叠需求机会；这是保持不变的可分接入近似，不是完整状态Bellman价值。',
        '按任务书只执行四次E/improvement，未额外计算E(pi4)或第五次G。因此即使最后两步达到经验稳定标准，也没有直接测量V3−E(pi4)的固定点残差，更不能宣称已验证精确自一致或Bellman固定点。',
        'q 仍是查询 hub 的第 q 辆空闲车：共同底座移除该 hub 空闲车，其他 hub 车和全部承诺保留，再加入共同资源前缀。配对世界仅多一辆空闲车。引擎按数组索引寻址，因此共同车使用同样的机械编号映射，承诺内容/物理字段不变。',
        '正式车队 K 不变；反事实世界可大于 K。固定 q 端点外推扩到2K，q>K 显式标记 unsupported_endpoint，不借缺槽回退到M0。不把这些格算进主表变化指标。',
        '', '## 2. 数据、split、信息与局限','',
        '四工况 CA80/3、W40/8、W80/3、W20/8；沿用 CFS 2022 PUMS、kNN1+MST、stretch1.2、50mph、capacity40、loaded/empty cost2/2.2、fare4*miles+.03*pounds、backlog5、expiry100、waiting0、maxdelay2、T24、reposition=false。Network-rho_max 与前五轮不变。CFS货物价值不是fare，实验 USD 利润不是实测经营预测。',
        f"CFS SHA-256：`{m['source']['sha256']}`。数值源码身份：`{m['code_hash']}`。", 
        'Construction8000–8059（60/工况）；diagnostic8100–8119（20）；pilot8900/8901（2）；formal9000–9039（40）。Scenario seed与policy seed0分开；training seed=null。所有正式政策运行完全相同场景，以40个独立场景的配对差统计，不能把参数或t/hub/q行当样本。',
        '为避免第五轮共享 donor 的跨父场景依赖，本轮每场景只使用自身未来路径；不同迭代共用相同生成场景实现 CRN。状态在每轮用当前 pi 重新采集，未来仅给 scheduler，online policy 不预览。此 MC 是有限 sampled-trajectory 估计，不是精确条件期望，也不是强信息状态的 Bellman evaluation。',
        '七个时间节点、查询hub旋转和q网格沿用旧MC；线性time/q插值、负值保留、T后free resource为0，所有四轮一致。正式集未用于 table、iteration、damping或alpha选择。四轮固定完成；可选 damping 未执行，第六轮B未启动。',
        '表格跨轮同格比较虽然支持域相同，但父策略自然状态不同；差异同时反映 evaluation operator 和 state distribution shift。没有将其解读成固定完整状态上的纯策略因果差。动作比较另用固定pi0诊断状态库、每个t的同一个完整状态，双方独立求解，而不是比较两条不同闭环轨迹的相同时间。',
        'Construction scenario原始生成相互独立，但后续父table共享并由同一个60场景bank构造。各格MC_SE仅是给定经验父policy的描述性离散度，不包括重估整条PI链的不确定性，也不提供全链无条件置信保证。重复有限construction bank可能过拟合；正式40场景独立于此bank，用来评价冻结政策的泛化表现。',
        '本轮没有额外固定状态资源-evaluation bank，因此表值变化来源的进一步分解仍待单独研究。固定pi0动作库也不能代表所有后期策略可能到达的状态；各政策自己的state distribution单独记录。',
        '', '## 3. Phase 0与来源关系','',
        '第五轮16次M0/MC Table/RH4/RH8独立重新求解，利润和物理/动作轨迹完全一致；四工况旧MC资源标签重建一致。不是读取旧摘要后自称复现。',
        'A–B人工例子：M0拿当前fare6，V0在t1的A资源价值10使pi1等待并赚10；重新评估t0的q1/q2边际从[6,10]变成[10,6]，pi2动作稳定。它证明当前代码确实重新评估改变的政策，也展示q非单调未被强行修正。不能由toy推导一般精确PI定理。',
        'T后新增空闲车与共同在途承诺的配对toy：无新dispatch、完整冷却、利润差0；原已有承诺仍赚fare/付成本并在两边抵消。',
        '另一个可手算estimand toy（仅历史语义诊断，不进入E/G）：T=2，当前短路A→B赚20，t1的B→A赚100；强制当前长路A→C赚10且release=2。baseline J=120、g=20；长路J=10、g=10，因此terminal candidate的Yroll=(10−10)−(120−20)=−100。与此同时，t=T只增加free resource的配对利润差仍为0。完整独立物理回放通过，见qa/terminal_estimand_distinction.json。',
        f"逐世界 lineage 审查 {qa['lineage']['worlds']:,} 个反事实世界；{qa['lineage']['provider_calls']:,} 次价值系数按父table重新计算一致。每期明确纯slot formulation，没有隐藏M0 fallback、Oracle训练或ACTION correction。每工况每轮还做一次独立重新求解，共{qa['independent_reproduction']}次，利润/动作/最终物理一致。",
        '', '## 4. 正式政策结果','']
    names=['P0','P1','P2','P3','P4','R0','R1','D'];conds=[key(c) for c in cfg['conditions']]
    lookup={(r['condition'],r['policy']):r for r in stats['summary']}
    text.append(markdown_table(['策略','CA80/3','W40/8','W80/3','W20/8'],[[p]+[f"{lookup[cid,p]['mean']:.2f}" for cid in conds] for p in names]))
    text+=['每格40个独立场景 mean realized profit (USD)，不等于显著最优。RH4/RH8是known-supply参考；PreviewH8非可部署诊断，不是全日认证上界，也不能把gap宣称为RL可回收潜力。',
        '预注册主family共36项：每工况相邻4组、P2/P3/P4相对P0的3组、P4−RH4/RH8的2组。统一两侧Holm；t95/10000次paired bootstrap未作多重区间校正。4个Preview−P4描述性另列。完整SD、median、skewness和逐scenario差均保存。','']
    text.append(markdown_table(['工况','比较','mean差USD','paired t95','Holm p'],[
        [r['condition'].replace('california_corridor','CA').replace('western_multicluster','W'),r['comparison'],f"{r['mean']:+.2f}",
         f"[{r['CI95_lower']:.2f},{r['CI95_upper']:.2f}]",'描述性' if r['Holm_p'] is None else f"{r['Holm_p']:.6g}"] for r in stats['comparisons']]))
    text+=['','## 5. 表、动作、状态、q与振荡','']
    text.append(markdown_table(['工况','表差','RMSE USD','relativeL2','Spearman','sign change'],[
        [r['condition'],f"V{r['to_V']}-V{r['from_V']}",f"{r['RMSE']:.2f}",f"{r['relative_L2']:.4f}",
         'n.a.' if r['Spearman'] is None else f"{r['Spearman']:.4f}",f"{r['sign_flip_rate']:.2%}"]
        for r in tab['changes'] if r['scope']=='observed_shared' and r['segment']=='all']))
    text.append(markdown_table(['工况','动作改善步','same-state日均变化比例','20场景t95'],[
        [r['condition'],r['iteration'],f"{r['mean']:.2%}",f"[{r['CI95_lower']:.2%},{r['CI95_upper']:.2%}]"] for r in actions]))
    formalactions=read_json(OUTPUT/'processed/formal_action_summary.json')
    text.append(markdown_table(['工况','动作改善步','正式集same-state变化比例','40场景t95'],[
        [r['condition'],r['iteration'],f"{r['mean']:.2%}",f"[{r['CI95_lower']:.2%},{r['CI95_upper']:.2%}]"] for r in formalactions]))
    text.append('两组动作诊断分开统计，主稳定性判据使用预声明的pi0诊断状态库，正式状态库只作泛化佐证。正式pi0状态通过冻结源码独立重算采集，并与保存的正式运行核对利润、逐期动作和最终物理hash精确复现。排序规范忽略数组序列化顺序但保留vehicle ID；正式集没有参与table、iteration或模型选择。')
    qsum=[]
    for cid in conds:
        for n in range(4):
            rr=[r for r in tab['monotonicity'] if r['condition']==cid and r['iteration']==n];neg=[r for r in tab['negative'] if r['condition']==cid and r['iteration']==n]
            qsum.append([cid,n,f"{sum(r['violation'] for r in rr)}/{len(rr)}",f"{max((r['magnitude_USD'] for r in rr),default=0):.2f}",len(neg)])
    text.append(markdown_table(['工况','V轮次','q边际非递增违例/比较格','最大违例USD','负值格'],qsum))
    text+=['q违例是诊断，不做isotonic；负值可能来自政策动作改变、机会竞争、成本、MC噪声或稀疏支持，不自动等于代码错。每格保留支持状态、MC_SE、持久性和原始配对利润；q之间依赖，不能把两个SE当独立作显著性检验。`negative_value_components.csv`对直接观察的负表价格逐对重建fare/loaded/empty/backlog/expiry增量及首次业务利润/动作改变，核对利润identity；插值产生的负q明确标记没有该q直接标签，不伪造归因。成分重建是核算分解，不是独立识别唯一行为机制。',
        '振荡全部格保存V0..V3、符号变化、更新方向反转、range和idle-q占据频率。图中仅按range显示各工况最大的5格，选择规则透明，不冒充显著性。高q在自然轨迹很少占据时，其变化不能与常用资源格同等解释。',
        'State shift逐hub/time提供idle、incoming、backlog、committed、utilization、已知release delay的SMD/Wasserstein及旧范围外占比。零方差且均值不同的SMD保持null并计数，绝不写0表示“无变化”。',
        '', '## 6. 稳定性与任务书10个问题','']
    for r in conclusions:
        text+=[f"### {r['condition']}",'',f"pi1−pi0={r['pi1_gain']:+.2f} USD；pi4−pi0={r['pi4_gain']:+.2f} USD，Holm p={r['pi4_Holm_p']:.6g}。最后两步联合小变化标准：{r['empirically_stabilized']}；描述性类别：{r['category']}。收益t95全部落入容差带（描述性，未作多重等价检验）：{r['descriptive_CI_band_containment']}。Preview−pi4={r['preview_gap']:.2f} USD，仅信息参考。"]
        for a in r['last_two_steps']:
            text.append(f"- step{a['step']}: gain={a['paired_gain_USD']:+.2f}, relativeL2={a['table_relative_L2']:.4f}, same-state action={a['action_disagreement']:.2%}, meanabsSMD={a['state_mean_abs_SMD']:.4f}, undefinedSMD={a['undefined_SMD_cells']}。")
        text.append('')
        cid=r['condition'];comparisons={x['comparison']:x for x in stats['comparisons'] if x['condition']==cid}
        tt=[x for x in tab['changes'] if x['condition']==cid and x['scope']=='observed_shared' and x['segment']=='all'];tt.sort(key=lambda x:x['to_V'])
        aa=sorted([x for x in actions if x['condition']==cid],key=lambda x:x['iteration'])
        shifts=[]
        for n in range(4):
            ss=[x['absolute_SMD'] for x in states['shifts'] if x['condition']==cid and x['from_pi']==n and x['absolute_SMD'] is not None]
            shifts.append(float(np.mean(ss)))
        oo=[x for x in tab['oscillations'] if x['condition']==cid]
        worst=sorted(oo,key=lambda x:(-x['range_USD'],x['hub'],x['time'],x['q']))[:3]
        oscillating=sum(x['update_direction_reversal_count']>0 for x in oo)
        negative=[x for x in tab['negative'] if x['condition']==cid]
        qviol=[x for x in tab['monotonicity'] if x['condition']==cid]
        def effect(name):
            x=comparisons[name]
            return f"{x['mean']:+.2f} USD，t95[{x['CI95_lower']:.2f},{x['CI95_upper']:.2f}]，Holm={x['Holm_p']:.6g}"
        text += [
            '1. **第一次MC是否有增益？** P1−P0：'+effect('P1-P0')+'。方向和多重校正证据分开解释。',
            f"2. **V1−V0是否改变？** RMSE={tt[0]['RMSE']:.2f} USD，relativeL2={tt[0]['relative_L2']:.4f}，Spearman={tt[0]['Spearman']}，三类sign change={tt[0]['sign_flip_rate']:.2%}。",
            '3. **pi2是否进一步增益？** P2−P1：'+effect('P2-P1')+'；不以累计差替代增量差。',
            f"4. **表变化是否逐渐减小？** relativeL2序列={[round(x['relative_L2'],4) for x in tt]}；连续非增={all(b['relative_L2']<=a['relative_L2'] for a,b in zip(tt,tt[1:]))}。RMSE序列={[round(x['RMSE'],2) for x in tt]}。",
            f"5. **同状态动作差异是否减小？** 四步日均变化比例={[round(x['mean'],4) for x in aa]}；连续非增={all(b['mean']<=a['mean'] for a,b in zip(aa,aa[1:]))}，固定pi0状态库、20场景。",
            f"6. **状态分布是否稳定？** 各步meanabsSMD={[round(x,4) for x in shifts]}；最后两步small判据和undefined分母计数见上。Wasserstein/support另存，不把时间格当独立样本。",
            f"7. **振荡在哪些格？** {oscillating}/{len(oo)}个部署支持格有更新方向反转。最大range三格："+
                '；'.join(f"hub={x['hub']},t={x['time']},q={x['q']},range={x['range_USD']:.2f},sign_flips={x['sign_flip_count']},idle_q_frequency={x['occupancy_frequency']:.3f}" for x in worst)+'。仅有限步诊断，非周期极限证明。',
            f"8. **q违例和负值？** 四张表合计q违例{sum(x['violation'] for x in qviol)}/{len(qviol)}次，负值{len(negative)}格，其中mean+2SE仍负的条件描述格{sum(x['negative_two_SE_support'] for x in negative)}。没有裁剪或isotonic；SE不含整条PI重估不确定性。",
            f"9. **pi4是否empirically stabilized？** 固定协议的联合small+nonincreasing判据={r['empirically_stabilized']}；没有exact PI/convergence/最优V证明。小mean变化与宽CI不等价于收益等价。",
            '10. **还有什么限制？** P4−RH4：'+effect('P4-R0')+'；P4−RH8：'+effect('P4-R1')+
                '。V仍忽略完整backlog/other-hub/incoming条件，slot可分与有限MC支持仍在；本轮不能判定主要误差来源，也不能从Preview gap推断DRL必要。','']
    text+=['', '## 7. 运行时与核算','']
    text.append(markdown_table(['E轮次','wall秒','worker秒','paired worlds','provider calls'],[
        [n,f"{m[f'iteration_{n}']['wall_seconds']:.2f}",f"{m[f'iteration_{n}']['worker_seconds']:.2f}",m[f'iteration_{n}']['worlds'],m[f'iteration_{n}']['value_calls']] for n in range(4)]))
    efficiency=read_json(OUTPUT/'processed/offline_evaluation_gain_efficiency.json')
    text.append(markdown_table(['工况','额外E轮次','worker秒','相邻gain USD','gain USD/离线worker小时'],[
        [r['condition'],r['evaluation_iteration'],f"{r['evaluation_worker_seconds']:.2f}",f"{r['paired_mean_increment_USD']:+.2f}",
            f"{r['mean_daily_increment_USD_per_offline_worker_hour']:+.2f}"] for r in efficiency]))
    text+=['目标世界含独立物理回放和完整逐期存储，因此offline时间不是只解MILP；三个worker的wall与summed-worker不同，不能相加。offline parent/simulation/serialization/aggregation/interpolation/QA分别记录，worker为包含这些部分的总量。纯provider lookup、context、build、solver、transition、postprocess、trace、ledger都有独立计时。lookup是build子集，release-slot assembly含lookup，cooldown是transition子集；不能重复相加。原M0内部build/extraction不改源代码，只在planner overhead合并标识，不冒充纯build。USD/离线worker小时仅摊销描述，不是现金ROI。',
        '', '## 8. QA、文件导航和未做内容','',
        f"全反事实独立回放{qa['replay']['resource_worlds']:,}世界、{qa['replay']['resource_world_steps']:,}步；全部policy运行{qa['replay']['policy_runs']:,}次、{qa['replay']['policy_steps']:,}步，物理精确，cash容差1e-6 USD，承诺完整冷却。",
        '`README.md`逐模块/目录/字段/统计含义导航；`targets`为资源标签，`tables`为四轮价值面，`policies`为五个政策的反向来源，`raw/worlds`与`raw/traces`为逐t完整记录，`processed`为分析表，`qa`为审查，`figures`含16组PNG/PDF/SVG与source/config。',
        '初次尝试为补齐独立计时而主动中止，完整归档于raw/failed_attempts/timing-v1，不混入最终样本；仅观测计时与等价聚合分段，物理/候选/控制/目标/种子不变。重做Phase0并全量重跑，四张新旧V0文件SHA必须精确相同，结果见qa/timing_repair_parity.json。修订前没有本轮formal outcomes可选。',
        '前五轮与CFS最终内容校验、pytest、字体矢量渲染/视觉检查以及交付SHA在final_manifest。未做：可选damping、第六轮B、ACTION correction、joint post-state、neural/DRL/SAC/PPO、分支切换/提交/推送。MC资源评估和近似policy iteration属于经典ADP思想，不能因没有DRL就说没有数据驱动价值估计。不能把本实验新增代码解释为已经更新平台默认框架。','']
    (OUTPUT/'resource_policy_iteration_report.md').write_text('\n'.join(text),encoding='utf-8')

def main():
    p=OUTPUT/'qa/research_audit.json'
    if p.exists():qa=read_json(p)
    else:
        qa=audit();atomic_json(p,qa)
    report(qa)
    # Artifact authoring / full tests should be finished before this final gate.
    required=['vector_pdf_audit.json','pytest.xml']
    if not all((OUTPUT/'qa'/n).exists() for n in required):
        print('Research QA/report ready; final tests and visual QA remain. Historical integrity is verified at the final gate.',flush=True);return
    visual=read_json(OUTPUT/'qa/vector_pdf_audit.json');assert visual['status']=='PASSED'
    actionqa=read_json(OUTPUT/'qa/canonical_same_state_action_audit.json')
    assert actionqa['status']=='PASSED' and actionqa['formal_states']==160*24
    tests=ElementTree.parse(OUTPUT/'qa/pytest.xml').getroot()
    suites=[tests] if tests.tag=='testsuite' else list(tests.iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert counts['failures']==counts['errors']==0
    history=verify_history()
    analysisfiles={p.relative_to(ROOT).as_posix():file_hash(p) for name in ['analysis.py','action_analysis.py','figures.py','delivery.py','pdf_qa.py'] if (p:=OUTPUT/name).exists()}
    analysistest=ROOT/'tests/test_resource_policy_iteration_analysis.py'
    analysisfiles[analysistest.relative_to(ROOT).as_posix()]=file_hash(analysistest)
    snapshot=OUTPUT/'raw/analysis.snapshot.zip'
    with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as z:
        for name in analysisfiles:z.write(ROOT/name,name)
    files={p.relative_to(OUTPUT).as_posix():file_hash(p) for p in OUTPUT.rglob('*') if p.is_file() and
        not any(x in p.parts for x in ['raw','__pycache__','pdf_render']) and p.name not in ['manifest.json','final_manifest.json']}
    final=dict(status='COMPLETED',created_at=now(),formal_runs=1280,pilot_runs=64,diagnostic_runs=400,
        construction_scenarios_per_condition=60,iterations=4,policies=5,research_qa=qa,pytest=counts,
        historical=history,vector_pdf_qa=visual,analysis_source_sha256=analysisfiles,analysis_snapshot_sha256=file_hash(snapshot),
        deliverable_sha256=files,git_action='none requested for this sixth study',not_run=['damping','sixth B','DRL','action correction'])
    atomic_json(OUTPUT/'manifests/final_manifest.json',final)
    m=read_json(OUTPUT/'manifest.json');m.update(status='COMPLETED',completed_at=final['created_at']);atomic_json(OUTPUT/'manifest.json',m)
    print(dict(status='COMPLETED',formal_runs=1280,files=len(files),pytest=counts),flush=True)

if __name__=='__main__':main()
