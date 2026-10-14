"""Read-only paired-label and public-prefix audit; no MC generation."""
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import numpy as np
from .common import *


def historical_seeds():
    used=set();evidence=[]
    def visit(obj,key=''):
        if isinstance(obj,dict):
            for k,v in obj.items():visit(v,k)
        elif isinstance(obj,list):
            if ('seed' in key.lower() or key in ['train','validation','test','pilot_train','pilot_validation','formal','diagnostic','development','construction','engineering']) and not any(s in key.lower() for s in ['training','policy','bootstrap']):
                used.update(int(x) for x in obj if isinstance(x,int))
            for v in obj:
                if isinstance(v,(dict,list)):visit(v,key)
        elif isinstance(obj,int) and ('scenario_seed' in key or key=='seed'):
            used.add(obj)
    for root in [ROOT/'experiments',ROOT/'reports/cfs_ppo_strategic_control']:
        for directory,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if d not in ['raw','data','datasets','models','tables','targets','training',
                'evaluation','sealed','figures','processed','__pycache__','workspace','qa']]
            for name in files:
                p=Path(directory)/name
                if p.suffix=='.json' and p.stat().st_size<4*2**20:
                    try:visit(read(p));evidence.append(str(p))
                    except (ValueError,KeyError):pass
    # Also reserve all historical frozen scenario filenames, even unused preparation.
    for root in [ROOT/'experiments',ROOT/'results']:
        for base in root.glob('cfs_*'):
            for folder in [base/'raw/datasets']:
                if folder.exists():
                    import re
                    for p in folder.glob('*.dataset.json'):
                        m=re.search(r'-s(\d+)\.dataset',p.name)
                        if m:used.add(int(m[1]))
    return used,evidence


def prepare(target=TARGET):
    from experiments.cfs_resource_policy_iteration.protocol import restore, state_identity, pair_audit
    from experiments.cfs_classical_value.common import parameters, rho_levels
    from model_families.single.data import matching_state
    from model_families.single.planning import compile_action
    from experiments.cfs_ppo_strategic_control.study_execution import action_from_plan
    from model_families.single.value.providers import MCTableValueProvider
    target=Path(target)
    if target.exists():raise FileExistsError('new attempt required')
    target.mkdir(parents=True)
    for name in ['data','models','training','qa','raw','figures']:(target/name).mkdir()
    shutil.copyfile('D:/download/writing-block (7).md',target/'task.md')
    base=read(ROOT/'experiments/cfs_rho_endofday/configs/main.json')
    construction=read(OLD/'targets/construction/targets.json')
    diagnostic=read(OLD/'targets/diagnostic/targets.json')
    raw=[r for r in construction if r['condition']==NAME]
    evaluate=[r for r in diagnostic if r['condition']==NAME]
    origin_manifest=read(OLD/'manifests/final_manifest.json')
    assert origin_manifest['status']=='COMPLETED'
    source_hashes={}
    def protect(p,expected=None):
        value=sha(p)
        if expected is not None and value!=expected:raise RuntimeError('historical hash mismatch '+str(p))
        source_hashes[str(p)]=value
    for stage in ['construction','diagnostic']:
        m=read(OLD/f'manifests/{stage}_target_manifest.json')
        protect(OLD/m['targets_ref'],m['targets_sha256']);protect(OLD/f'manifests/{stage}_target_manifest.json')
    protect(OLD/'manifests/final_manifest.json')
    for rel,digest in origin_manifest['source_code_sha256'].items():protect(ROOT/rel,digest)
    table=OLD/'tables/base_tuq'/f'{NAME}.json'
    protect(table,origin_manifest['deliverable_sha256'][table.relative_to(OLD).as_posix()])
    seeds=sorted({r['scenario_seed'] for r in raw});rng=random.Random(41001);rng.shuffle(seeds)
    train=sorted(seeds[:48]);val=sorted(seeds[48:]);test=sorted({r['scenario_seed'] for r in evaluate})
    assert len(seeds)==60 and len(test)==20 and not(set(seeds)&set(test))
    scenarios={}
    for seed in seeds+test:
        p=OLD/'raw/datasets'/f'{NAME}-s{seed}.dataset.json';protect(p)
        scenarios[seed]=read(p)['scenarios'][0]
    physical=[{k:v for k,v in s.items() if k not in ['seed','id','split']} for s in scenarios.values()]
    from experiment_core.storage import digest
    assert len({digest(s) for s in physical})==len(physical)
    hubs=sorted(scenarios[train[0]]['nodes'])
    prior={h:sum(sum(o['departure']==h for o in scenarios[s]['orders']) for s in train)/(len(train)*24) for h in hubs}
    write(target/'data/train_demand_prior.json',prior)
    feature_rows=[];pair_checks=[];prefix_checks=[];history_cost=[]
    for seed in sorted(scenarios):
        stage='construction' if seed in seeds else 'diagnostic'
        scenario=scenarios[seed];p=parameters(base,rho_levels(scenario,base)['rho_max'])
        out=OLD/'raw'/stage
        jobpath=out/'targets/iter0/raw'/f'{NAME}-s{seed}.json'
        job=read(jobpath);protect(jobpath)
        parentpath=out/job['parent_trace'];protect(parentpath);parent=read(parentpath)
        history_cost.append(dict(seed=seed,stage=stage,total_job_seconds=job['seconds'],
            solver_periods=job['solver_periods'],simulation_seconds=job['simulation_seconds'],
            serialization_seconds=job['serialization_seconds'],qa_seconds=job['qa_seconds']))
        assert parent['policy_hash']==raw[0]['continuation_policy_hash'] and parent['table_hash'] is None
        packets=parent['state_packets'];states={}
        sim,orders=matching_state(copy.deepcopy(scenario),p,base['solver'])
        for tr in parent['trace']:
            if tr.get('drain'):continue
            t=sim.t;sim.begin_period([o for o in orders.values() if o.book_time==t])
            if str(t) in packets:
                from experiments.cfs_rollout_target_alignment.queries import pack_state
                check_packet(pack_state(sim),packets[str(t)])
                # The original saved packet, including its diagnostic J, is the
                # source of features; J is never a predictor input or cash reward.
                states[t],_=restore(scenario,p,base['solver'],packets[str(t)])
            plan=compile_action(sim,action_from_plan(tr['plan']),{})
            assert math.isclose(plan[0],tr['plan']['operating_profit'],abs_tol=1e-6,rel_tol=1e-10)
            sim.commit_plan(plan)
        prefix_checks.append(dict(seed=seed,public_nodes=len(states),prefix_actions=24))
        worlds={}
        for w in job['worlds']:
            wp=out/w['path']
            # Inspect the q=1/2 worlds at three preselected nodes in every scenario.
            if '-h'+hubs[0]+'-' in wp.name and any('-t'+str(t)+'-' in wp.name for t in [0,12,23]) and any('-q'+str(q)+'.' in wp.name for q in [0,1,2]):
                protect(wp,w['sha256']);worlds[wp.name]=read(wp)
        group=[r for r in (raw if stage=='construction' else evaluate) if r['scenario_seed']==seed]
        for r in group:
            if r['terminal_anchor']:continue
            assert r['paired_state_check'] and r['continuation_policy_id']=='M0' and r['continuation_table_hash'] is None
            assert r['continuation_policy_hash']==parent['policy_hash']
            assert math.isclose(r['profit_q']-r['profit_previous'],r['target'],abs_tol=1e-7)
            tau=r['time'];t=decision_time(tau)
            assert state_identity(states[tau])==r['state_hash']
            known={k:o for k,o in orders.items() if o.book_time<=t}
            row=features(states[t],known,r['hub'],tau,r['q'],prior)
            row.update(target=r['target'],scenario_seed=seed,split='train' if seed in train else 'validation' if seed in val else 'evaluation',
                source_release_state_hash=r['state_hash'],public_state_hash=state_identity(states[t]),
                future_hash=r['future_hash'],mc_repetitions=1)
            feature_rows.append(row)
            if r['hub']==hubs[0] and tau in [0,12,23] and r['q'] in [1,2]:
                a=worlds[f'{NAME}-s{seed}-t{tau}-h{r["hub"]}-q{r["q"]-1}.json.gz']
                b=worlds[f'{NAME}-s{seed}-t{tau}-h{r["hub"]}-q{r["q"]}.json.gz']
                assert a['future_hash']==b['future_hash']==r['future_hash']
                lo,_=restore(scenario,p,base['solver'],a['initial']);hi,_=restore(scenario,p,base['solver'],b['initial'])
                pair_audit(lo,hi,r['hub'])
                assert a['replay']['status']==b['replay']['status']=='PASSED'
                assert math.isclose(b['metrics']['realized_profit']-a['metrics']['realized_profit'],r['target'],abs_tol=1e-6)
                pair_checks.append(dict(seed=seed,release_time=tau,q=r['q'],future_equal=True,only_one_free_vehicle=True))
    csv_write(target/'data/supervised_samples.csv',feature_rows)
    for stage in ['train','validation','evaluation']:
        rows=[r for r in feature_rows if r['split']==stage]
        csv_write(target/f'data/{stage}.csv',rows)
    # Freeze all six new days before labels are fitted or any policy profit is seen.
    used,usage_evidence=historical_seeds();fresh=[];candidate=21028
    while len(fresh)<6:
        if candidate not in used:fresh.append(candidate)
        candidate+=1
    from model_families.single.cfs_adapter import read_pools,make_scenario
    region=next(r for r in base['regions'] if r['id']=='western_multicluster')
    pools,_,_=read_pools(base['data_source'],region['coordinates'])
    scenes=[];provenance=[]
    for seed in fresh:
        cfg={**base,'orders':40}
        scene,pr=make_scenario(region,pools,seed,cfg);scene['hubs']=scenarios[train[0]]['hubs']
        scene['id']=f'autogluon-W40-development-s{seed}';scene['split']='test'
        assert digest({k:v for k,v in scene.items() if k not in ['id','seed','split']}) not in {digest(x) for x in physical}
        scenes.append(scene);provenance.append(pr)
    write(target/'data/development_scenarios.json',scenes);write(target/'qa/development_provenance.json',provenance)
    cfg=dict(parameters=parameters(base,rho_levels(scenes[0],base)['rho_max']),solver={**base['solver'],'threads':1},baseline=str(table))
    write(target/'data/runtime_config.json',cfg)
    csv_write(target/'qa/historical_label_cost.csv',history_cost)
    ys=np.array([r['target'] for r in feature_rows]);mc=MCTableValueProvider(read(table))
    summary=dict(raw_W40_records=len(raw)+len(evaluate),nonterminal_supervised_records=len(feature_rows),
        train_groups=train,validation_groups=val,evaluation_groups=test,development_seeds=fresh,
        rows_by_split={s:sum(r['split']==s for r in feature_rows) for s in ['train','validation','evaluation']},
        q_counts={str(q):sum(r['q']==q for r in feature_rows) for q in [1,2,3,4,6,8]},
        label_mean=float(ys.mean()),label_variance=float(ys.var()),label_min=float(ys.min()),label_max=float(ys.max()),
        repetitions_per_state_query=1,within_state_MC_standard_error=None,
        public_prefixes_checked=len(prefix_checks),paired_world_checks=len(pair_checks),
        label_continuation='frozen legacy M0, original exact integer cash, rolling until full cooldown',
        q_semantics='replace idle queried-hub fleet at release with q versus q-1, NOT additional q over original idle',
        feature_time_rule='previous saved public node; tau=0 uses t=0; each raw pair label used ONCE',
        repeated_future_within_day=True,synthetic_CFS=True,MC_uses_all_60_construction_groups=True,
        overlap_groups=0,feature_columns=dict(Basic=BASIC,State=STATE),new_MC_labels=0,
        excluded_data=['round3 older cash','round4 freeze/action-conditioned','round5 action advantage','round6 pi1/pi2/pi3','failed bucket attempt'],
        historical_usage_metadata_files=len(usage_evidence),status='PASSED')
    write(target/'dataset_manifest.json',summary)
    write(target/'qa/data_preflight.json',dict(status='PASSED',prefix_checks=prefix_checks,pair_checks=pair_checks))
    write(target/'qa/readonly_inputs.json',source_hashes)
    write(target/'qa/model_source_hashes.json',{str(p):sha(p) for p in (ROOT/'model_families/single').rglob('*.py')})
    train_abs=float(max(abs(r['target']) for r in feature_rows if r['split']=='train'))
    protocol=dict(status='PREPARED',fit_time_limit=150,hard_fit_wall_seconds=180,fit_models=2,
        total_numerical_seconds=900,maximum_new_solves=1500,expected_main_solves=576,
        num_cpus=1,num_gpus=0,num_bag_folds=0,num_stack_levels=0,presets='medium_quality',
        training_seed=41002,split_seed=41001,max_output_bytes=2*2**30,price_clipping=None,
        extreme_flag_USD=train_abs,official_fit_docs='https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html')
    write(target/'protocol.json',protocol)
    audit=['# 监督标签数据审计','',
        '结论：存在可复用的、单一 frozen M0 后续策略、精确整数现金账本的 W40 配对资源标签。仅选第六轮精确核算版本，不混入其他 continuation 或 action advantage。',
        '',f'原始 W40 共 {len(raw)+len(evaluate):,} 条（含解析终端锚点）；使用 {len(feature_rows):,} 条非终端独立配对记录，80个完整场景组。训练48组、验证12组、监督评价20组，按场景划分，不按行随机拆分。',
        '', '## 当前时刻与释放时刻的对齐', '',
        '旧标签在 τ 时刻用自然 M0 状态替换被查询枢纽的空闲资源。新输入使用同一天较早已存的公开快照 t：τ=0 时 t=0；其余采用前一个保存节点。两反事实世界在 [t,τ) 共同执行已冻结 M0，前缀完全相同、现金在差值中抵消；τ 时刻才执行旧资源替换，然后继续相同 M0 到完整冷却。因此旧原始配对差值可作为较早公开状态条件下的未来释放边际样本，不重新求解或复制汇总表值。全部80天的24个原整数决策已独立编译并核验7个快照；记录来源标签 hash 与输入 public hash 分开保存。',
        '', '这定义的是共同 M0 前缀下的延迟资源替换价值，不能称为候选动作条件价值或最优 Bellman 价值。q是替换后的资源数量；在线q是释放组前缀槽位序号，沿用原资源表近似映射，二者并未被证明严格等价。',
        '',f'标签均值 {ys.mean():.4f} USD，方差 {ys.var():.4f}，范围 [{ys.min():.4f},{ys.max():.4f}]。q≥2：{sum(r["q"]>=2 for r in feature_rows):,} 条。每个完整状态/查询仅一个配对未来，不能估计该状态的MC重复标准误；跨场景方差不是这个标准误。',
        '', '公开特征20项，禁止 scenario ID、future hash、利润、未来订单和释放后的状态。需求强度仅由48个训练场景估计；旧全60组构建prior和bucket不作为模型特征。保留旧公开快照中的完整承诺，只用可见pool与已承诺路线。',
        '', 'Pure MC 使用已冻结第六轮 base_tuq（全60个构建场景）；AG仅48组训练、12组选择。这是保留强基准的比较，不是三模型完全等训练样本比较；独立预测评价的20组不参与任何模型构建。',
        '',f'独立配对反事实世界抽查 {len(pair_checks)} 组，核对原始文件SHA、共同未来、仅差一辆自由车辆及冷却差值。所有16,800标签重新核算profit_q−profit_previous；不把这些行当16,800个独立场景。',
        '', '原标签构建并非零成本。各W40原作业累计时间、模拟时间、序列化、QA和求解期数保存于qa/historical_label_cost.csv；本次新增MC标签为零。',
        '',f'闭环场景按21028起递增、跳过所有已登记或已冻结历史seed，预先选为 {fresh}。此规则采用保守排除，不声称配置登记等于实际已使用。新场景不是新的正式测试集，只是开发评价。']
    (target/'supervised_data_audit.md').write_text('\n'.join(audit)+'\n',encoding='utf-8')
    text='''# 监督资源价值 AutoGluon 轻量实验协议

仅W40/8；两个模型共享48训练/12验证场景组与同一配对标签。Basic使用真实表接口的τ/h/q三项；State新增17项决策前公开摘要。20个独立历史诊断场景仅用于冻结后监督评价，6个新开发场景仅用于闭环评价。终端锚点不参与拟合和主预测误差，τ≥24价格严格为0。

标签使用之前已完成的精确核算M0原始资源替换对。较早快照与释放标签的共同M0前缀均经原动作编译核对；不把未来状态输入模型，不复制MC表值制造标签。q的资源替换数量与在线slot语义差异保留并披露。

AutoGluon medium_quality，显式独立tuning_data，无bagging/stacking，无额外HPO，CPU1/GPU0。每模型fit软time_limit=150秒，保留30秒余量，实际fit不得超过180秒；最多两次fit。原任务示例为180，本次主动预留开销，两方法相同设置。禁止根据评价利润重选模型、尺度或范围；负价与非单调q保留，不裁剪价格，只按训练最大绝对值标记极端预测。

所有4方法使用同一个原H1 ValueRouteMILPPlanner、同样物理、约束、solver、完整冷却账本；只更换provider。预测为求解前批量固定美元系数；缓存键隐含当前状态，明确覆盖整数τ与所有h/q；terminal gate保持不变。在线实际占用由已提交路线投影释放组重建，并与原MILP目标的slot奖金核对。

最多24完整运营日/576主求解，总新增求解不超过1500，新增数值阶段不超过900秒，产物上限2GiB，单进程solver串行。开发日配对结果只描述，不作六日总体显著性论证。监督排序按同场景/同公开时刻/同释放时刻分组，误差另报告q≥2与符号错误。原始一对未来/状态不能提供MC重复SE。

原数据/价格表/源码只读hash核验；独立原动作重放、容量/ready/deadline/完整结算检查。不追加RL、第二轮或新MC标签。若预检、求解、预算或账本失败，保留失败并停止，不删除不利结果。
'''
    (target/'experiment_protocol.md').write_text(text,encoding='utf-8')
    print('DATA_PREFLIGHT_PASSED',summary['rows_by_split'],'development',fresh,flush=True)


if __name__=='__main__':prepare()
