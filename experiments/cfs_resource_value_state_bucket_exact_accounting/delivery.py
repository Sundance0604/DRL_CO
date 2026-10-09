"""B evidence/report, exhaustive provenance audits, readonly checks and freeze."""
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.value.providers import MCTableValueProvider
from experiments.cfs_resource_policy_iteration.protocol import resource_world,pair_audit,state_identity
from experiments.cfs_resource_policy_iteration.targets import profiled_table_model
from experiments.cfs_rollout_target_alignment.queries import pack_state
from .common import OUTPUT,A2,ROOT,key,file_hash,now,code_hashes
from .protocol import read_compressed,restore
from .pressure import pressure,bucket,expected_prior,thresholds
from .provider import BucketValueProvider
from .analysis import finalize_analysis

def check_integer_cash(plan):
    diagnostic=plan['solver']['integer_action_accounting'];a=plan['accounting']
    assert diagnostic['integer_action_preserved']
    assert diagnostic['original_integer_action_hash']==diagnostic['independently_compiled_action_hash']
    cost=sum(v for k,v in a.items() if k not in ['assignment_revenue','business_cost','profit_identity_residual'])
    assert abs(cost-a['business_cost'])<=1e-6
    assert abs(a['assignment_revenue']-cost-plan['operating_profit'])<=1e-6
    assert abs(plan['operating_profit']-diagnostic['integer_action_profit_USD'])<=1e-6
    assert abs(diagnostic['integer_action_profit_USD']-diagnostic['raw_solver_profit_USD']-
        diagnostic['integer_minus_raw_profit_USD'])<=1e-6

def target_audit(cfg):
    path=OUTPUT/'qa/world_lineage_audit.json'
    if path.exists():return read_json(path)
    m=read_json(OUTPUT/'manifest.json');priors=read_json(OUTPUT/'tables/priors.json')
    index={};count=steps=pairs=0
    for stage in ['construction','diagnostic']:
        root=OUTPUT/'raw'/stage;mf=read_json(OUTPUT/'manifests'/f'{stage}_target_manifest.json')
        assert file_hash(OUTPUT/mf['targets_ref'])==mf['targets_sha256']
        queries={r['id']:r for r in read_json(OUTPUT/'manifests'/f'{stage}_query_freeze.json')['queries']}
        for ip in sorted((root/'targets/iter0/raw').glob('*.json')):
            item=read_json(ip);job=queries[ip.stem];cid=job['condition']
            scenario=read_json(job['scenario_ref'])['scenarios'][0]
            assert digest(scenario)==job['scenario_hash']
            parent=read_compressed(root/item['parent_trace']);bank={}
            future=digest([vars(o) for _,o in sorted(restore(scenario,job['parameters'],job['solver'],next(iter(parent['state_packets'].values())))[1].items())])
            expected_policy=digest(dict(condition=cid,parent='M0',code=m['code_hash']))
            assert expected_policy==job['parent_policy_hash']==parent['policy_hash'] and parent['table_hash'] is None
            for t,packet in parent['state_packets'].items():
                state,_=restore(scenario,job['parameters'],job['solver'],packet)
                bank[int(t)]=(state,pressure(state,priors[cid]['pressure_prior']))
            worlds={}
            for wi in item['worlds']:
                p=root/wi['path'];assert file_hash(p)==wi['sha256'];w=read_compressed(p)
                assert w['identity']==wi['identity'] and w['future_hash']==future
                assert w['policy_hash']==expected_policy and w['table_hash'] is None and w['value_calls']==0
                assert w['replay']['status']=='PASSED' and abs(w['replay']['total_difference'])<=1e-6
                assert w['metrics']['limited_window_count']==0 and w['metrics']['unfinished_committed']==0
                state,pp=bank[w['time']];h=w['hub'];pr=pp[h];b=bucket(pr['R'],priors[cid]['thresholds'])
                expected=resource_world(state,h,w['q'])
                assert w['initial']==pack_state(expected)
                assert w['pre_injection_bucket']==b and w['pre_injection_pressure']==pr
                assert w['pre_injection_state_hash']==state_identity(state)
                for row in w['trace']:
                    if row.get('drain'):assert not row['plan']['routes'] and not row['plan']['assignments']
                    else:
                        assert row['plan']['solver']['status']=='OPTIMAL'
                        check_integer_cash(row['plan'])
                worlds[w['time'],h,w['q']]=w
                index[p.relative_to(OUTPUT).as_posix()]=wi['sha256'];count+=1;steps+=w['replay']['periods']
            for r in item['rows']:
                if r['terminal_anchor']:assert r['time']==24 and r['target']==0;continue
                t,h,q=r['time'],r['hub'],r['q'];lo,hi=worlds[t,h,q-1],worlds[t,h,q]
                assert lo['future_hash']==hi['future_hash']==r['future_hash']
                assert lo['pre_injection_bucket']==hi['pre_injection_bucket']==r['bucket']
                state,pp=bank[t];audit=pair_audit(resource_world(state,h,q-1),resource_world(state,h,q),h)
                assert audit['common_hash']==r['paired_common_hash'] and r['bucket_computed_before_resource_replacement']
                assert hi['metrics']['realized_profit']-lo['metrics']['realized_profit']==r['target'];pairs+=1
        print(f'B {stage} audit: {count} worlds cumulative.',flush=True)
    assert count==72800
    atomic_json(OUTPUT/'manifests/raw_world_sha256.json',dict(files=index))
    result=dict(status='PASSED',worlds=count,replay_steps=steps,paired_targets=pairs,checked_at=now(),
        same_M0_parent=True,pre_injection_bucket_recomputed=True,same_future=True,all_OPTIMAL=True,
        full_cooldown=True,no_bucket_world_relabel=True,all_world_integer_cash_checked=True,
        raw_world_index_sha256=file_hash(OUTPUT/'manifests/raw_world_sha256.json'))
    atomic_json(path,result);return result

def table_and_pressure_audit(cfg):
    priors=read_json(OUTPUT/'tables/priors.json');base=read_json(OUTPUT/'manifest.json')['baseline_config']
    rows=read_json(OUTPUT/'targets/construction/targets.json');survey=[];cells=0
    for c in cfg['conditions']:
        cid=key(c);scenarios=[read_json(OUTPUT/'raw/datasets'/f'{cid}-s{s}.dataset.json')['scenarios'][0] for s in cfg['construction_seeds']]
        assert expected_prior(scenarios,base['capacity'])==priors[cid]['pressure_prior']
        for s in cfg['construction_seeds']:
            survey.extend(read_compressed(OUTPUT/'raw/survey'/f'{cid}-s{s}.json.gz')['pressure_rows'])
        assert thresholds([r for r in survey if r['condition']==cid])==priors[cid]['thresholds']
        rr=[r for r in rows if r['condition']==cid];original,_,_=profiled_table_model(rr,2*c['fleet'])
        actual=read_json(OUTPUT/'tables/base_tuq'/(cid+'.json'));bm=read_json(OUTPUT/'tables/bucket_tuqb'/(cid+'.json'))
        assert actual['values']==original['values'] and bm['base']==actual
        assert bm['base_and_bucket_same_targets_sha256']==file_hash(OUTPUT/'targets/construction/targets.json')
        groups=defaultdict(list)
        for r in rr:
            if not r['terminal_anchor']:groups[r['hub'],r['time'],r['q'],r['bucket']].append(r['target'])
        for h in actual['values']:
            for t in actual['values'][h]['1']['times']:
                if t>=24:continue
                for q in bm['measured_q']:assert sum(len(groups[h,t,q,b]) for b in range(3))==len(cfg['construction_seeds'])
        for h,qs in actual['values'].items():
            for b in range(3):
                for q,rec in qs.items():
                    for i,t in enumerate(rec['times']):
                        means=[]
                        for qq in bm['measured_q']:
                            yy=groups[h,t,qq,b]
                            assert bm['support'][h][str(b)][str(qq)][str(t)]==len(yy)
                            means.append(float(np.mean(yy)) if len(yy)>=10 else qs[str(qq)]['values'][i])
                        assert bm['buckets'][str(b)]['values'][h][q]['values'][i]==float(np.interp(int(q),bm['measured_q'],means));cells+=1
    threshold=read_json(OUTPUT/'manifests/threshold_freeze.json');table=read_json(OUTPUT/'manifests/table_freeze.json')
    assert threshold['created_at']<table['created_at'] and threshold['formal_seen'] is False and threshold['targets_seen'] is False
    assert table['formal_seen'] is False and table['diagnostic_seen'] is False
    r=dict(status='PASSED',checked_at=now(),cells_recomputed=cells,thresholds_construction_only=True,
        expected_prior_construction_only=True,base_bucket_targets_identical=True,no_clipping=True,no_monotonicity_constraint=True)
    atomic_json(OUTPUT/'qa/table_pressure_recalculation.json',r);return r

def deployment_audit(cfg):
    priors=read_json(OUTPUT/'tables/priors.json');count=steps=queries=coefficients=0
    for stage in ['diagnostic','pilot','formal']:
        for r in read_json(OUTPUT/'processed'/f'{stage}_runs.json'):
            spec=read_json(OUTPUT/'raw/runs'/r['id']/'spec.json');tracepath=OUTPUT/r['trace_ref']
            assert file_hash(tracepath)==r['trace_sha256'];tr=read_compressed(tracepath)
            assert tr['replay']['status']=='PASSED' and abs(tr['replay']['total_difference'])<=1e-6
            assert r['limited_window_count']==0 and r['unfinished_committed']==0
            context={}
            for t,packet in tr['state_packets'].items():
                state,_=restore(spec['scenario'],spec['parameters'],spec['solver'],packet)
                pp=pressure(state,priors[r['condition']]['pressure_prior'])
                context[int(t)]=dict(T=24,period=int(t),pressure=pp,
                    bucket={h:bucket(v['R'],priors[r['condition']]['thresholds']) for h,v in pp.items()})
            p=BucketValueProvider(spec['policy']['table']) if r['policy']=='B2' else MCTableValueProvider(spec['policy']['table']) if r['policy']=='B1' else None
            for q in tr.get('lookup_queries',[]):
                s=context[q['observation_period']];v=p.marginal_release_value(q['release_time'],q['hub'],q['q'],s)
                assert q==p.query_log[-1] and q['value_USD']==v and q['current_bucket_not_future'];queries+=1
            if r['policy']=='B2':assert p.calls==r['value_calls'] and p.fallback_calls==r['fallback_calls']
            for row in tr['trace']:
                if row.get('drain'):
                    assert not row['plan']['routes'] and not row['plan']['assignments'];continue
                solver=row['plan']['solver'];assert solver['status']=='OPTIMAL'
                if r['policy']=='B0':check_integer_cash(row['plan'])
                if r['policy'] in ['B1','B2']:
                    assert solver['information_mode']=='known_supply' and solver['formulation']=='single-route-macro-action-slot-value/v1'
                    for v in solver.get('release_coefficients',[]):
                        actual=p.marginal_release_value(v['time'],v['hub'],v['supply_slot'],context[row['period']])
                        assert actual==v['value'];coefficients+=1
            count+=1;steps+=r['replay_steps']
        print(f'B {stage} deployment audit: {count} runs cumulative.',flush=True)
    assert count==1488
    result=dict(status='PASSED',checked_at=now(),runs=count,replay_steps=steps,
        bucket_queries_independently_recomputed=queries,slot_coefficients_independently_recomputed=coefficients,
        explicit_fallback=True,online_legal_information=True,full_cooldown=True,all_OPTIMAL=True)
    atomic_json(OUTPUT/'qa/deployment_audit.json',result);return result

def evidence(cfg):
    formal=read_json(OUTPUT/'processed/formal_statistics.json');diagnostic=read_json(OUTPUT/'processed/diagnostic_statistics.json')
    variance=read_json(OUTPUT/'processed/variance_corroboration.json');fallback=read_json(OUTPUT/'processed/formal_fallback_summary.json')
    records=[];e=cfg['effectiveness']
    for c in cfg['conditions']:
        cid=key(c);r=next(r for r in formal['paired'] if r['condition']==cid and r['comparison']=='B2-B1')
        dr=next(r for r in diagnostic['paired'] if r['condition']==cid and r['comparison']=='B2-B1')
        vr=next(r for r in variance if r['condition']==cid);fr=next(r for r in fallback if r['condition']==cid)
        gain=r['mean']>=e['problematic_gain_fraction']*abs(r['reference_mean']) and r['Holm_p']<=.05
        records.append(dict(condition=cid,formal_gain_USD=r['mean'],t95=[r['CI95_lower'],r['CI95_upper']],
            bootstrap95=[r['bootstrap95_lower'],r['bootstrap95_upper']],Holm_p=r['Holm_p'],diagnostic_gain_USD=dr['mean'],
            formal_positive=r['mean']>0,stable_gain_supported=gain and dr['mean']>0,
            heldout_MSE_improvement_USD2=vr['mean'],variance_corroborated=vr['corroborated'],
            query_fallback_daily_mean=fr['mean'],query_fallback_pooled=fr['pooled_rate'],
            nonterminal_query_fallback_pooled=fr['nonterminal_pooled_rate'],fallback_supported=fr['mean']<=e['maximum_formal_fallback_fraction']))
    keyrows=[r for r in records if '-n20-' not in r['condition']]
    problematic=[r for r in records if r['condition'].startswith('california') or '-n40-' in r['condition']]
    w80=next(r for r in formal['paired'] if r['comparison']=='B2-B1' and r['condition'].startswith('western') and '-n80-' in r['condition'])
    margin=e['W80_noninferiority_margin_fraction']*abs(w80['reference_mean']);safe=w80['CI95_lower']>-margin
    majority=sum(r['formal_positive'] for r in keyrows)>=e['majority_positive']
    # The same problematic regime must support the paired gain AND independent
    # predictive heterogeneity reduction, with adequate conditional support.
    mechanism=any(r['stable_gain_supported'] and r['variance_corroborated'] and r['fallback_supported'] for r in problematic)
    supported=majority and safe and mechanism and all(r['fallback_supported'] for r in keyrows)
    result=dict(regimes=records,majority_key_positive=majority,W80_noninferiority_supported=safe,
        W80_noninferiority_margin_USD=margin,problematic_regime_mechanism_supported=mechanism,
        bucket_effective=supported,failure_does_not_prove_ineffectiveness=True,
        no_B_PI=True,no_RL=True,current_bucket_future_release_transfer_is_approximation=True)
    atomic_json(OUTPUT/'processed/evidence.json',result);return result

def report(cfg):
    e=evidence(cfg);stats=read_json(OUTPUT/'processed/formal_statistics.json')
    lines=['# B：最小供需状态扩展的独立数值实验','',
        '数值与分析已完成；全部测试、图表视觉审阅、历史只读校验及最终冻结，以 manifests/final_manifest.json 为准。','',
        '## 1. 实验边界与数学定义','',
        'A2完成并冻结后才启动B。B不使用阻尼，不做bucket PI，只在共同M0父策略下构造一次base与bucket MC value；MILP约束、收入成本、资源替换与完整冷却均不变。',
        'R=(Bload+Dnear)/(Midle+Inear+1)，Hb=4、eta=1。订单load按capacity=40折成车辆当量，资源是车辆数。Bload只含可见ready backlog；Dnear=已知且近4期ready bookings+construction历史均值的尚未预订近期待到load，互不重复。Inear只含已锁定承诺的近4期最终release，不是优化预测供给。',
        'Inear复用现有rolling.audit的最终idle_release事件，不把历史last-delivery时间当释放，也不重复处理已到达的当前期事件。实际优化/推进产生的边界反例测试通过，历史物理与摘要代码保持只读。',
        '每工况construction的M0原始状态上固定c33/c67：R<=c33 surplus；c33<R<=c67 balanced；R>c67 scarce。阈值相等时不人为抖动或重分桶，balanced可为空。',
        'Y=J^M0(q)−J^M0(q−1)是完整冷却后的finite-resource边际值，不是route advantage、ACTION bonus或Q(S,a)。在原始共同状态计算一次桶，再移除查询hub的idle并分别注入q/q−1，两世界共用原始桶、原始可见状态和同一未来订单。',
        '在线当前观测hub桶作为未来release(time,hub)的查表上下文；没有candidate-dependent post-state或未来桶预测。因此它仍是可分slot价格近似，当前桶向未来释放时点的转移假设是明确限制，不能当作完整条件Bellman价值。','',
        '## 2. 数据、共享标签与样本单位','',
        'Construction8300–8359；diagnostic8400–8419；pilot8960–8961；formal9200–9239。四工况正式各40个独立scenario，policy seed=0，training seed=null。B0/B1/B2/R0/R1/D全部用完全相同场景配对，共960次formal仿真。',
        'B0=M0；B1=共同标签的base V(t,u,q)；B2=相同标签按pre-injection桶分组的V(t,u,q,b)；R0/R1=RH4/RH8；D=OraclePreviewH8，不是经过认证的全日upper bound。',
        '为使三桶有可用支持，base与bucket都采用同一7个历史time节点、全部5hub和原q节点；每原始tuq先有60construction场景再分桶。这是B内部共同采样设计，相对A的hub轮换加密，不能把B1−旧A1的差单独归因于状态扩展。唯一直接扩展对照是同标签B2−B1。',
        '54,600个construction和18,200个独立diagnostic资源世界全部保存逐t动作、状态、ledger、冷却及回放。诊断标签仅用于独立预测误差佐证，不拟合表、不选阈值、eta或算法。','',
        '## 3. Support与回退','',
        '原始cell记录count、sample_count、conditional MC_SE、bucket、terminal anchor。查询需要的任何非终端线性插值源角count<10，或q>K，则整个查询明确回退到同一base V(t,u,q)。终端t>=24恒为0，不要求抽样支持。负值保留；不施加q或bucket单调性。',
        '部署整数t/q全网格另存source corners、最小源count、interpolated/unsupported/fallback标识。插值后的联合MC_SE未估计，不将源格SE伪装成插值SE。',
        'count>=10是最低样本门，不是完整状态或桶内R范围充分覆盖的证明。原targets兼容legacy feature_row不进入两种MC均值拟合；权威桶输入是pre_injection_pressure。经济收入成本与网络坐标/载荷转换仍是固定实验假设，利润不是CFS直接观测财务结果。',
        'fallback rate主口径是每场景全部pre-solve coefficient调用中的回退比例，再跨40场景取均值。另报汇总池化率、非终端池化率、各桶/原因计数。它包含最终未选用的slot及终端零查询，不能解释为已执行资源的回退占比。','',
        '## 4. 正式利润与配对统计','',
        '| 策略 | '+' | '.join(key(c) for c in cfg['conditions'])+' |','|---|'+'---:|'*4]
    for label in ['B0','B1','B2','R0','R1','D']:
        lines.append('| '+label+' | '+' | '.join(f'{next(r for r in stats["summary"] if r["condition"]==key(c) and r["policy"]==label)["mean"]:.2f}' for c in cfg['conditions'])+' |')
    lines+=['','主family是4工况B2−B1，统一两侧Holm；16项其他算法比较单独Holm；Preview差只作描述。t95与10,000次paired bootstrap是未校正区间，Holm p单列。SD、median、skewness与逐scenario差保留。t/hub/q/bucket/query不是独立replication。','',
        '| 工况 | B2−B1 USD | t95 | bootstrap95 | Holm p | diagnostic方向 | fallback均值 |','|---|---:|---|---|---:|---:|---:|']
    for r in e['regimes']:lines.append(f'|{r["condition"]}|{r["formal_gain_USD"]:+.2f}|{r["t95"]}|{r["bootstrap95"]}|{r["Holm_p"]:.6g}|{r["diagnostic_gain_USD"]:+.2f}|{r["query_fallback_daily_mean"]:.4f}|')
    lines+=['','## 5. 异质性、动作与机制证据','',
        'Construction采用同tuq总体方差=加权桶内方差+桶间方差的精确分解。分组后加权桶内方差必然不增，不能单靠这项下降证明状态有用。',
        '独立diagnostic用冻结construction表预测Y，计算base/bucket squared residual；先将一个scenario的全部cell误差差取平均，再做20独立scenario的paired t95/bootstrap及4工况Holm。要求base−bucket MSE均值>0、t95下限>0、Holm<=.05，才视为异质性减少的独立佐证。不用diagnostic分组均值重新中心化或拟合。',
        '同状态动作分析以B0完整24期状态为固定基准，两方分别求解。输出精确完整hash、route、assignment、wait/reposition及release-hub差；不舍入，不把不同车辆ID合并。桶占比按每场景24期×5hub计算后跨scenario统计。','',
        f'预注册bucket effectiveness证据={e["bucket_effective"]}；关键工况多数正均值={e["majority_key_positive"]}；问题工况联合利润/独立MSE/支持证据={e["problematic_regime_mechanism_supported"]}；W80非劣证据={e["W80_noninferiority_supported"]}，容差={e["W80_noninferiority_margin_USD"]:.2f} USD。',
        '问题工况需至少CA或W40达到>=1%base均值的正式配对改善及Holm<=.05、diagnostic方向一致、独立MSE佐证和fallback均值<=.5；关键工况fallback也须<=.5。W80要求配对t95下限>-1%base均值，不以不显著代替非劣。未达联合阈值只能说证据不足，不能证明bucket无效或不存在state aliasing。','',
        '| 工况 | 独立MSE改善USD² | 方差佐证 | 稳定利润佐证 | 非终端回退率 |','|---|---:|---|---|---:|']
    for r in e['regimes']:lines.append(f'|{r["condition"]}|{r["heldout_MSE_improvement_USD2"]:.3f}|{r["variance_corroborated"]}|{r["stable_gain_supported"]}|{r["nonterminal_query_fallback_pooled"]:.4f}|')
    lines+=['','## 6. QA、运行时与交付','',
        '阈值与期望prior从construction独立复算；所有世界的原始pre-injection压力、桶、资源初始状态、同future及边际标签校验。Base/bucket从相同标签逐格重算，unsupported占位值必须经查询回退门。在线所有桶query和保存MILP系数由逐t合法状态与冻结表独立重算。',
        '所有派发期求解OPTIMAL、无超时静默接受，T后只完整冷却无新dispatch，cash与physical replay一致。传统策略与base adapter先在历史场景独立回归；历史A、A2与更早研究全部内容hash最终再检查。',
        '十一组600dpi PNG/vector PDF/SVG、source CSV和plot config，并有十一页本地图册。Offline wall/worker与在线context/lookup/build/solver/transition/ledger分开，嵌套计时不相加。',
        '单次parent M0改进不意味着自一致或收敛；不使用神经网络、DRL、ACTION correction或阻尼，没有更新平台默认算法。仍有有限样本、稀疏桶、当前桶向未来释放时间转移、可分resource prices及有限PreviewH8的限制。失败不自动支持Strategic RL：需独立确认充分支持、剩余全日政策gap、joint interaction/nonseparability。','',
        '具体文件导航、字段和恢复方式见README。A2与B分别冻结，不共享formal数据做参数选择，不自动启动B-PI或RL。']
    (OUTPUT/'resource_value_state_bucket_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    joint_report(cfg,e);return e

def joint_report(cfg,b):
    """Answer all twelve questions without converting a failed test into no-effect."""
    a=read_json(A2/'processed/evidence.json');alpha=read_json(A2/'manifests/alpha_freeze.json')
    av={r['condition']:r for r in a['regimes']};bv={r['condition']:r for r in b['regimes']}
    ca=key(next(c for c in cfg['conditions'] if c['region'].startswith('california')))
    w40=key(next(c for c in cfg['conditions'] if c['orders']==40))
    atables=read_json(A2/'processed/table_metrics.json');osc=read_json(A2/'processed/oscillations.json')
    cuts=read_json(OUTPUT/'tables/priors.json');variance=read_json(OUTPUT/'processed/variance_corroboration.json')
    formal=read_json(OUTPUT/'processed/formal_statistics.json')
    contrasts=read_json(OUTPUT/'processed/bucket_contrast.json')
    w80=key(next(c for c in cfg['conditions'] if c['orders']==80 and c['region'].startswith('western')))
    w20=key(next(c for c in cfg['conditions'] if c['orders']==20))
    case='I' if a['damping_effective'] and b['bucket_effective'] else 'II' if b['bucket_effective'] else 'III' if a['damping_effective'] else 'IV'
    lines=['# A2 + B 独立机制诊断联合汇总','',
        'A2先完整执行并冻结，B随后独立执行；不同正式seed银行不相互配对，不共享正式结果调参。两项实验分别只有阻尼更新和三桶状态扩展，未同时改变两个因素。','',
        '此处“有效”仅表示达到预注册联合证据阈值；“未建立”不是证明无效。任务书2×2矩阵作为组织证据的框架，不是足以排除所有竞争解释的因果识别。','',
        '## 任务书第35节：12问逐项回答','']
    local_profit_leaders={cid:max(alpha['records'],key=lambda r:next(x['D4_mean'] for x in r['conditions'] if x['condition']==cid))['alpha_value'] for cid in [ca,w40,w80]}
    w20_base=next(r for r in formal['paired'] if r['condition']==w20 and r['comparison']=='B1-B0')
    mean_abs_base={key(c):float(np.mean([abs(r['base_value_USD']) for r in contrasts if r['condition']==key(c)])) for c in cfg['conditions']}
    answers=[
        ('CA的后续policy deterioration能否被damping明显缓解？',
         f'正式D4−U4={av[ca]["profit_difference"]:+.2f} USD，t95={av[ca]["paired_t95"]}，Holm={av[ca]["Holm_p"]:.6g}；利润证据={av[ca]["profit_evidence"]}，动作证据={av[ca]["action_evidence"]}，表证据={av[ca]["table_evidence"]}。联合稳定化={av[ca]["joint_stabilization_supported"]}，不能只以利润增加替代全部判据。'),
        ('W40的table/action instability能否被damping缓解？',
         f'最后两步动作差damped/undamped={av[w40]["late_damped_action"]:.4f}/{av[w40]["late_undamped_action"]:.4f}；relativeL2={av[w40]["late_damped_relativeL2"]:.4f}/{av[w40]["late_undamped_relativeL2"]:.4f}；动作/表证据={av[w40]["action_evidence"]}/{av[w40]["table_evidence"]}。部署表变化变小部分由阻尼公式机械产生，仍需利润和独立动作证据。'),
        ('Damping是否破坏W80/3原本较好的表现？',
         f'D4−U4={av[w80]["profit_difference"]:+.2f} USD，t95={av[w80]["paired_t95"]}。预注册非劣容差{a["W80_margin_USD"]:.2f} USD，非劣支持={a["W80_noninferiority_supported"]}；未通过意味着不能排除超过容差的损失，不等于证明发生损害。'),
        ('最优alpha是否跨工况一致？',
         f'只冻结了一个全局alpha={alpha["alpha_value"]:g}，没有各工况独立alpha选择或不同alpha的formal评估。诊断D4利润均值最大者（仅描述，不是预注册稳定化最优）：CA={local_profit_leaders[ca]:g}、W40={local_profit_leaders[w40]:g}、W80={local_profit_leaders[w80]:g}。这不足以认定存在跨工况共同最优alpha；全局选择遵循动作/表优先的lexicographic规则，不能在formal后重选。'),
        ('Coarse imbalance bucket是否降低同一(t,u,q)内的resource-value heterogeneity？',
         'Construction加权桶内方差下降是分组机械结果。独立20诊断scenario的冻结表预测残差MSE佐证：'+ '；'.join(f'{r["condition"]}: base−bucket MSE={r["mean"]:+.3f} USD², t95={[r["CI95_lower"],r["CI95_upper"]]}, Holm={r["Holm_p"]:.6g}, supported={r["corroborated"]}' for r in variance)+'。不对diagnostic标签重新拟合或中心化。'),
        ('Bucket policy是否稳定优于base MC Table？',
         '同标签B2−B1正式配对证据：'+'；'.join(f'{r["condition"]}: {r["formal_gain_USD"]:+.2f} USD, t95={r["t95"]}, Holm={r["Holm_p"]:.6g}, diagnostic={r["diagnostic_gain_USD"]:+.2f}, stable profit={r["stable_gain_supported"]}' for r in b['regimes'])+f'。整体联合有效性={b["bucket_effective"]}；未达到阈值不等于证明无效。'),
        ('Bucket是否主要帮助CA/W40，而对W80影响小？',
         f'CA/W40增量分别{bv[ca]["formal_gain_USD"]:+.2f}/{bv[w40]["formal_gain_USD"]:+.2f} USD；W80={bv[w80]["formal_gain_USD"]:+.2f} USD，t95={bv[w80]["t95"]}，非劣支持={b["W80_noninferiority_supported"]}。需要结合各自区间、Holm与诊断方向，不能凭均值大小或不显著判断影响小；也未预注册跨工况差异的因果交互检验。'),
        ('W20是否仍表现为资源宽松、value signal弱？',
         f'W20的B1−B0={w20_base["mean"]:+.2f} USD，t95={[w20_base["CI95_lower"],w20_base["CI95_upper"]]}，Holm={w20_base["Holm_p"]:.6g}；B2−B1={bv[w20]["formal_gain_USD"]:+.2f} USD。Construction绝对base边际价格均值（原始非终端tuq等权）为'+ '；'.join(f'{cid}={v:.2f} USD' for cid,v in mean_abs_base.items())+f'；W20压力分位阈值c33/c67={cuts[w20]["thresholds"]["c33"]:.6g}/{cuts[w20]["thresholds"]["c67"]:.6g}。这些是资源压力/价格与收益的描述，结合原始count/MC_SE审查；condition-specific桶相对占比不能单独证明绝对资源宽松，也未对“信号弱”预注册二元判据。'),
        ('A2与B哪个解释更强？',
         f'A2联合证据={a["damping_effective"]}，B联合证据={b["bucket_effective"]}，组织性编码Case {case}。'+('B的最小状态扩展机制证据更符合本轮联合阈值。' if b['bucket_effective'] and not a['damping_effective'] else 'A2的更新稳定性机制证据更符合本轮联合阈值。' if a['damping_effective'] and not b['bucket_effective'] else '两者均达阈值，但本轮没有直接比较其解释强度。' if a['damping_effective'] else '两者均未建立完整联合证据，不能据此选出更强机制。')+'A2/B使用不同formal场景和不同干预；不直接配对其利润增量，更不能仅比较p值大小来给机制强度排序。'),
        ('剩余问题更像update instability、state aliasing、separability还是有限MC噪声？',
         '阻尼改善动作/表与利润支持更新机制，独立桶MSE/利润改善支持状态混叠机制，但它们不排除其他解释。当前查询回退daily mean：'+'；'.join(f'{r["condition"]}={r["query_fallback_daily_mean"]:.4f}' for r in b['regimes'])+'。高回退、有限条件MC_SE与利润区间均可能限制识别；没有独立MC-bank重复、E(D4)固定点残差、完整联合资源交互或未来桶预测实验，不能确定剩余问题主要是哪一个，也不能把Case IV证明为可分架构结构上限。'),
        ('是否已有证据进入bucketed self-consistent PI？',
         ('达到预注册bucket联合证据，可作为另行预注册bucket PI的研究动机，但一次改进不证明多轮稳定，本轮不自动启动。' if b['bucket_effective'] else '尚未达到预注册bucket联合证据；先审查支持、回退、独立预测误差和利润区间，不凭某一正均值自动进入B-PI。')),
        ('是否已有足够证据进入Strategic RL？',
         '本轮不自动满足。没有完整固定点、经过认证的全日最优gap或joint interaction/nonseparability独立证据；PreviewH8差不是全日上界或RL可回收利润。即使Case IV也先审查MC支持与精度，并另行预注册机制实验，不能由未达判据推导RL必要。')]
    assert len(answers)==12
    for number,(question,answer) in enumerate(answers,1):
        lines.extend([f'### {number}. {question}','',answer,''])
    lines+=['## 补充证据与复现细节','',
        '### CA 后续路径比较','',
        f'同新formal场景D4−U4={av[ca]["profit_difference"]:+.2f} USD，t95={av[ca]["paired_t95"]}，Holm={av[ca]["Holm_p"]:.6g}；联合稳定化证据={av[ca]["joint_stabilization_supported"]}。',
        'D1=U1，因此D4−U4也等于两条路径从共同第一步到第四步的利润变化差；不能用新formal与A旧formal直接作paired差。正利润差但动作/表不稳定只能称profit improvement。','',
        '## 2. W40 的 table/action 不稳定是否减少？','',
        f'最后两步same-state动作差damped={av[w40]["late_damped_action"]:.4f}，undamped={av[w40]["late_undamped_action"]:.4f}；relativeL2分别={av[w40]["late_damped_relativeL2"]:.4f}/{av[w40]["late_undamped_relativeL2"]:.4f}；动作证据={av[w40]["action_evidence"]}，表证据={av[w40]["table_evidence"]}。',
        '动作在共同U0完整状态库上独立solve，自己的闭环状态分布另报；未计算E(D4)或固定点残差，不把四步趋缓叫理论收敛。','',
        '## 3. 最终选择哪个 alpha，是否受 formal 影响？','',
        f'全局alpha={alpha["alpha_value"]:g}；exploratory_selection={alpha["exploratory_selection"]}；选择只用8200–8219诊断集，lexicographic筛查/近似容差和.5优先在结果前固定，pilot/formal均在alpha冻结后启动。',
        '共同raw V0/V1与A一致；raw V2/V3评估实际alpha-specific D2/D3，允许与A不同，是用户明确确认的真正自一致协议。','',
        '## 4. 是否破坏原先表现较好的 W80？','',
        f'A2非劣证据={a["W80_noninferiority_supported"]}，容差={a["W80_margin_USD"]:.2f} USD，paired t95下限={a["W80_t95_lower"]:.2f}；B非劣证据={b["W80_noninferiority_supported"]}，容差={b["W80_noninferiority_margin_USD"]:.2f} USD。',
        '未通过非劣门槛只能说不能排除超过容差的损失，不能把不显著解释为安全。','',
        '## 5. 阻尼是否减少符号变化、方向反转和格振荡？','',
        '| 工况 | undamped平均range USD | damped平均range USD | undamped方向反转均值 | damped方向反转均值 |','|---|---:|---:|---:|---:|']
    for c in cfg['conditions']:
        cid=key(c);u=[r for r in osc if r['condition']==cid and r['alpha']=='undamped'];d=[r for r in osc if r['condition']==cid and r['alpha']==alpha['selected_alpha']]
        lines.append(f'|{cid}|{np.mean([r["range_USD"] for r in u]):.3f}|{np.mean([r["range_USD"] for r in d]):.3f}|{np.mean([r["update_direction_reversal_count"] for r in u]):.3f}|{np.mean([r["update_direction_reversal_count"] for r in d]):.3f}|')
    lines+=['','这些是原始native非终端格的四步描述，不是独立formal样本或极限环证据。完整sign_flip_count、raw/deployed路径和自然idle-q占据率均保存。','',
        '## 6. Bucket 是否提供了不同的资源价值信号？','',
        '| 工况 | c33 | c67 | tied |','|---|---:|---:|---|']
    for c in cfg['conditions']:
        cid=key(c);cc=cuts[cid]['thresholds'];lines.append(f'|{cid}|{cc["c33"]:.6g}|{cc["c67"]:.6g}|{cc["tied_thresholds"]}|')
    lines+=['','同tuq三桶raw mean、conditional MC_SE及实际supported/fallback查表价格逐格保存；有raw均值差不等于有部署价格差。仅supported桶对照才支持regime-dependent marginal value，不强制单调。','',
        '## 7. Bucket 对利润是否有稳定增量？','',
        '| 工况 | B2−B1 USD | t95 | Holm | diagnostic USD | 稳定利润证据 |','|---|---:|---|---:|---:|---|']
    for c in cfg['conditions']:
        r=bv[key(c)];lines.append(f'|{r["condition"]}|{r["formal_gain_USD"]:+.2f}|{r["t95"]}|{r["Holm_p"]:.6g}|{r["diagnostic_gain_USD"]:+.2f}|{r["stable_gain_supported"]}|')
    lines+=['','B1/B2同construction标签、同M0父策略和同formal未来；B1相对旧A1还改变了共同hub采样密度，不能将旧A1到B2全部收益归因于bucket。','',
        '## 8. Within-cell heterogeneity 是否减少？','',
        '| 工况 | 独立MSE改善 USD² | t95 | Holm | corroborated |','|---|---:|---|---:|---|']
    for r in variance:lines.append(f'|{r["condition"]}|{r["mean"]:.3f}|{[r["CI95_lower"],r["CI95_upper"]]}|{r["Holm_p"]:.6g}|{r["corroborated"]}|')
    lines+=['','Construction加权桶内方差下降是分组的机械结果，仅描述。上表用独立20诊断场景及冻结construction表的预测残差，先scenario内平均，再paired t95/bootstrap/Holm；不从diagnostic重新拟合均值。','',
        '## 9. Support 是否充分，fallback 是否过高？','',
        '| 工况 | query每日均值率 | pooled率 | 非终端pooled率 | 支持回退门槛 |','|---|---:|---:|---:|---|']
    for c in cfg['conditions']:
        r=bv[key(c)];lines.append(f'|{r["condition"]}|{r["query_fallback_daily_mean"]:.4f}|{r["query_fallback_pooled"]:.4f}|{r["nonterminal_query_fallback_pooled"]:.4f}|{r["fallback_supported"]}|')
    lines+=['','分母是所有pre-solve coefficient查询，不是已选择资源；终端查询不回退会影响总体分母，非终端率单列。阈值ties、空桶、任何角点count<10及q>K都显式可见。高回退时阴性结果可能是支持不足，不是状态信息无用。','',
        '## 10. 对应2×2诊断矩阵哪一格？','',
        f'按“达到证据阈值/未达到证据阈值”的编码，A2={a["damping_effective"]}，B={b["bucket_effective"]}，组织性编码Case {case}。',
        'Case I支持两种机制都值得继续；II优先继续状态扩展；III优先稳定更新；IV仅表明两种有限诊断均未建立正面联合证据。特别是IV不能直接证明低维资源价值达到结构极限，仍需检查支持、效应精度和未测joint interactions。','',
        '## 11. 下一步是否需要 bucket PI、damped ADP 或联合方法？','',
        ('可以优先设计独立bucket PI，并在新的预注册实验中检验是否还需要阻尼；本轮不自动运行。' if b['bucket_effective'] else
         '可以优先设计独立damped ADP/trust-region/mixture-policy对照，先保留固定状态；本轮不自动运行。' if a['damping_effective'] else
         '先审查稀疏支持、误差区间、资源价格对未来regime的转移和联合资源交互，再决定新的状态或更新实验；本轮不自动加复杂度。'),
        '本轮不将两变量同时改动以追求更高收益，也不复用formal结果选择新的阈值/alpha。','',
        '## 12. 是否已经满足进入 Strategic RL 的门槛？','',
        '没有自动满足。MC table与有限self-consistent迭代、阻尼和最小状态扩展已按各自协议检验，但没有完整固定点、理论结构极限或joint interaction/nonseparability的独立因果证据。',
        'PreviewH8−B2只能描述有限提前信息基准差，不能视为全日最优gap或RL可回收利润。即使编码IV，也需充分支持及精度、稳健剩余政策差、联合交互证据与新的预注册实验，才讨论Strategic RL。',
        '如果未来进入RL，MILP仍负责可行性、matching/routing/capacity/commitments，学习层仅提供状态依赖价格或粗粒度目标；本轮没有实现或启动任何RL训练。','',
        '## 交付与复现','',
        'A2/B各有独立报告、config、源码/数据/policy/table/query冻结、QA和最终manifest。正式统计以每工况40独立scenario配对，图表为本地600dpiPNG/嵌入字体矢量PDF/SVG，使用数据与config一并保存。',
        '是否完整完成最终交付，须检查两目录final_manifest状态和内容hash。没有Git提交/推送、平台默认模型替换或对旧实验写入。']
    (OUTPUT/'A2_B_joint_diagnostic_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def verify_history():
    history=read_json(OUTPUT/'manifests/historical_readonly.json')
    for i,(name,sha) in enumerate(history['files'].items()):
        assert file_hash(ROOT/name)==sha,name
        if (i+1)%20000==0:print(f'B final readonly check {i+1}/{len(history["files"])}',flush=True)
    assert file_hash(Path(read_json(OUTPUT/'manifest.json')['baseline_config']['data_source']))==history['CFS_sha256']
    result=dict(status='PASSED',files=len(history['files']),checked_at=now(),A2_readonly=True,
        original_failed_B_readonly=True,CFS_verified=True)
    atomic_json(OUTPUT/'qa/historical_integrity.json',result);return result

def main(stage):
    cfg=read_json(OUTPUT/'config/main.json')
    if stage=='analysis':
        finalize_analysis(cfg);report(cfg)
        disclosure=('本报告来自用户授权的新独立 exact-accounting B attempt。原 B 在 seed8405 触发金额一致性停止，完整保留且只读。'
            '唯一修正是从原求解器提取的整数动作独立重算执行金额，保留原浮点profit/objective诊断；'
            '不改变动作、求解器容差、物理、种子、分桶和统计规则，全部B数值重新运行，未混入原attempt缓存。'
            'A2仍为原始不可变冻结结果，跨A2/B解释须保留这一数值核算修正的边界。\n\n')
        for name in ['resource_value_state_bucket_report.md','A2_B_joint_diagnostic_report.md']:
            p=OUTPUT/name;p.write_text(disclosure+p.read_text(encoding='utf-8'),encoding='utf-8')
        return
    if stage=='worlds':table_and_pressure_audit(cfg);target_audit(cfg);deployment_audit(cfg);return
    if stage=='history':verify_history();return
    assert stage=='freeze';m=read_json(OUTPUT/'manifest.json');assert code_hashes()==m['source_code_sha256']
    assert file_hash(OUTPUT/'analysis.py')==read_json(OUTPUT/'manifests/analysis_source_freeze.json')['sha256']
    for name in ['baseline_regression','accounting_amendment','table_pressure_recalculation','world_lineage_audit','deployment_audit','historical_integrity','pdf_visual_audit','tests']:
        assert read_json(OUTPUT/'qa'/(name+'.json'))['status']=='PASSED',name
    assert read_json(OUTPUT/'manifests/formal_run_manifest.json')['runs']==960
    freeze=read_json(OUTPUT/'manifests/policy_freeze.json')
    assert freeze['created_at']<read_json(OUTPUT/'manifests/formal_run_manifest.json')['created_at'] and freeze['formal_seen'] is False
    assert len(read_json(OUTPUT/'figures/figure_manifest.json')['names'])==11
    files={}
    for folder in ['config','tables','policies','processed','figures','qa']:
        for p in (OUTPUT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:files[p.relative_to(OUTPUT).as_posix()]=file_hash(p)
    for name in ['README.md','ACCOUNTING_AMENDMENT.md','resource_value_state_bucket_report.md','A2_B_joint_diagnostic_report.md']:
        assert (OUTPUT/name).exists();files[name]=file_hash(OUTPUT/name)
    m.update(status='COMPLETED',completed_at=now(),deliverable_sha256=files,A2_was_frozen_before_B=True)
    atomic_json(OUTPUT/'manifest.json',m);atomic_json(OUTPUT/'manifests/final_manifest.json',m)
    print('B completed and immutable. No bucket PI or RL launched.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['analysis','worlds','history','freeze'],required=True)
    main(p.parse_args().stage)
