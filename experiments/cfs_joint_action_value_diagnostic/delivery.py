"""Post-numerical verification and reporting. No source/selection/statistic changes.

Every replay uses persisted futures and actions, not a newly sampled world or
another optimization. This module is explicitly outside numerical source freeze.
"""
import argparse
import itertools
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from experiment_core.storage import read_json, atomic_json, digest
from experiments.cfs_resource_policy_iteration.protocol import restore, replay_world
from experiments.cfs_rollout_target_alignment.protocol import read_compressed
from .common import OUTPUT, ROOT, B, A2, config, sha, now, numerical_hashes, gate, csv_output

QA=['historical_regression','joint_lineage','world_lineage','bank_recalculation',
    'historical_integrity','paired_statistics','tests','pdf_visual_audit','complete_artifact_integrity']

def writable():
    m=read_json(OUTPUT/'manifest.json')
    assert m['status']!='COMPLETED','COMPLETED is immutable'
    assert not (OUTPUT/'qa/stop_condition_failure.json').exists(),'Failed attempt is immutable'
    assert m['status']=='NUMERICAL_COMPLETE_ANALYSIS_QA_PENDING',m['status']
    assert numerical_hashes()==m['source_code_sha256'],'Frozen numerical source changed'
    assert sha(OUTPUT/'analysis.py')==read_json(OUTPUT/'manifests/analysis_preregistration.json')['analysis_sha256']
    return m

def passed(name,**fields):
    value=dict(status='PASSED',checked_at=now(),**fields)
    atomic_json(OUTPUT/'qa'/(name+'.json'),value);return value

def marker(kind,count):
    p=OUTPUT/'qa'/f'{kind}_create_marker.json'
    if p.exists():return
    runtime=Path('C:/Users/32599/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
    package='spreadsheets/26.1005.12141/skills/spreadsheets' if kind=='csv' else 'pdf/26.1005.12141/skills/pdf'
    script=Path('C:/Users/32599/.codex/plugins/cache/openai-primary-runtime')/package/'container_tools/mark_artifact_operation_started.mjs'
    subprocess.run([str(runtime),str(script),'--operation-kind','create','--expected-output-count',str(count),'--output-format',kind],check=True)
    atomic_json(p,dict(status='EXECUTED_ONCE',at=now(),expected_output_count=count,format=kind))

def supplement(rows):
    """Additional economic denominators; never changes the preregistered analysis."""
    from .actions import historical
    from .states import restore_row
    from experiments.cfs_resource_value_state_bucket_exact_accounting.protocol import provider
    from experiments.cfs_resource_value_state_bucket_exact_accounting.pressure import summary
    fallback=[];sensitivity=[];group_stats=[]
    for r in rows:
        actions=read_json(r['action_ref']['path'])['actions'];state,_=restore_row(read_json(r['state_ref']['path']))
        policies,prior=historical(r['condition']);online=summary(state,prior)
        generating=next(a for a in actions if 'B2' in a['labels'])['generating_solver']['B2']
        bp=provider(policies['B2']);potential=generating.get('release_coefficients',[])
        for c in potential:
            v=bp.marginal_release_value(c['time'],c['hub'],c['supply_slot'],online)
            assert abs(v-c['value'])<=1e-6
        potential_queries=bp.query_log
        for selector,choice in r['selections'].items():
            action=next(a for a in actions if a['action_hash']==choice['action_hash'])
            price=next(a for a in actions if a['action_hash']==r['selections']['price']['action_hash'])
            occupied=action['scores']['B2']['lookup_queries']
            def coefs(a):
                return {(c['release_time'],c['hub'],c['q']):c for c in a['scores']['B2']['lookup_queries']}
            a,b=coefs(action),coefs(price)
            changed=[a.get(k,b.get(k)) for k in a.keys()^b.keys()]
            def metrics(query,prefix):
                absolute=sum(abs(x['value_USD']) for x in query)
                fallback_value=sum(abs(x['value_USD']) for x in query if x['fallback'])
                nonterminal=[x for x in query if x['release_time']<state.T]
                return {prefix+'_count':len(query),prefix+'_fallback_count':sum(x['fallback'] for x in query),
                    prefix+'_nonterminal_count':len(nonterminal),prefix+'_nonterminal_fallback_count':sum(x['fallback'] for x in nonterminal),
                    prefix+'_absolute_value_USD':absolute,prefix+'_fallback_absolute_value_USD':fallback_value,
                    prefix+'_fallback_value_fraction':fallback_value/absolute if absolute else None}
            fallback.append({k:r[k] for k in ['stage','condition','source','scenario_seed','state_id']}|
                dict(selector=selector,action_hash=choice['action_hash'],changed_action_from_price=choice['action_hash']!=price['action_hash'],
                     generating_coefficients_source='B2 full original H1 solve, including physically merged candidates',
                     economic_denominator='sum absolute coefficients, NOT sum with cancellation; terminal zeros explicit',
                     changed_definition='occupied (time,hub,q) symmetric difference from price action',
                     **metrics(potential_queries,'all'),**metrics(occupied,'selected'),**metrics(changed,'changed')))
        if r['sensitivity']:
            for left,right in [('select32','price'),('select32','M0'),('select32','select8')]:
                a=r['selections'][left]['action_hash'];b=r['selections'][right]['action_hash']
                diffs=[x['returns'][a]-x['returns'][b] for x in r['sensitivity']]
                se=float(np.std(diffs,ddof=1)/np.sqrt(len(diffs)))
                sensitivity.append({k:r[k] for k in ['condition','source','scenario_seed','state_id']}|
                    dict(stage='diagnostic',comparison=left+'-'+right,continuation='pi1',paths=len(diffs),
                         mean_difference_USD=float(np.mean(diffs)),paired_MC_SE_USD=se,path_differences_USD=diffs,
                         independent_scenario_count_not_paths=True))
    from .analysis import statistics
    for cid,src,comp in sorted({(r['condition'],r['source'],r['comparison']) for r in sensitivity}):
        vals=[r['mean_difference_USD'] for r in sensitivity if (r['condition'],r['source'],r['comparison'])==(cid,src,comp)]
        group_stats.append(dict(condition=cid,source=src,comparison=comp,continuation='pi1',**statistics(vals),
            secondary_exploratory=True,only_four_diagnostic_scenarios=True))
    csv_output('fallback_economic_supplement',fallback)
    if sensitivity:csv_output('continuation_sensitivity_supplement',sensitivity)
    atomic_json(OUTPUT/'processed/supplement.json',dict(continuation_statistics=group_stats,
        primary_analysis_unchanged=True,created_at=now(),delivery_sha256=sha(OUTPUT/'delivery.py')))

def replay_saved(path):
    w=read_compressed(path)
    if 'state_ref' in w:
        row=read_json(w['state_ref']['path']);future=read_json(w['future_ref'])
        scene=row['scenario']|dict(orders=row['scenario']['orders']+future['future_orders'])
        state,orders=restore(scene,row['parameters'],row['solver'],w['initial'])
        assert w['numeric_code_hash']==row['numeric_code_hash']
        assert w['future_hash']==digest([vars(orders[k]) for k in sorted(orders)])
        expected=digest(dict(state=w['initial'],future=future['stream_hash'],action=w['action_hash'],
            continuation=w['continuation_hash'],numeric_code_hash=row['numeric_code_hash']))
        assert expected==w['identity'],path
        assert future['visible_order_hash']==digest({k:vars(v) for k,v in sorted(restore(row['scenario'],row['parameters'],row['solver'],row['packet'])[1].items())})
        assert future['hidden_source_orders_used'] is False
        if w['action_hash'] is not None:
            packet=read_json(OUTPUT/'raw/actions'/f'{row["state_id"]}.json')
            a=next(a for a in packet['actions'] if a['action_hash']==w['action_hash'])
            assert w['current_physical_action']==a['physical']
    elif 'scenario' in w:state,orders=restore(w['scenario'],w['parameters'],w['solver'],w['initial'])
    else:
        job=read_json(Path(path).parent.parent/(Path(path).name.split('-t')[0]+'.json'))['spec']
        assert w['identity']==digest(dict(job=digest(job),time=w['time'],hub=w['hub'],q=w['q']))
        state,orders=restore(job['scenario'],job['parameters'],job['solver'],w['initial'])
    audit=replay_world(state,orders,w)
    assert audit['status']=='PASSED'
    assert abs(audit['total_difference'])<=1e-6
    assert w['independent_cash_max_residual']<=1e-6
    assert abs(w['total']-w['ledger']['realized_profit']+w['origin_realized_common'])<=1e-6
    assert not any(o.book_time>state.t for oid,o in orders.items() if oid in state.pool)
    for row in w['trace']:
        if not row.get('drain'):
            assert row['plan']['solver']['status']=='OPTIMAL'
            assert all(orders[a['order_id']].book_time<=row['period'] for a in row['plan']['assignments'])
    return dict(path=str(path),sha256=sha(path),steps=audit['periods'])

def audit_joint():
    from .analysis import load_rows
    from .actions import choose,verify,score,historical
    from .states import restore_row
    from .experiment import resource_pairs
    cfg=config();refs={};states=0
    for stage,count in [('pilot',24),('diagnostic',120),('formal',240)]:
        rows=load_rows(stage);assert len(rows)==count
        am=read_json(OUTPUT/'manifests'/f'{stage}_actions.json');sm=read_json(OUTPUT/'manifests'/f'{stage}_states.json')
        assert am['future_outcomes_seen'] is False and am['states_sha256']==sha(OUTPUT/'manifests'/f'{stage}_states.json')
        assert sm['created_at']<=am['created_at']
        seedgroups=defaultdict(set)
        for r in rows:
            states+=1;packet=read_json(r['action_ref']['path']);row=read_json(r['state_ref']['path'])
            assert sha(r['action_ref']['path'])==r['action_ref']['sha256']
            assert sha(r['state_ref']['path'])==r['state_ref']['sha256']
            assert row['state_hash']==digest(row['packet']) and row['hidden_orders_excluded'] is True
            assert all(o['book_time']<=row['time'] for o in row['scenario']['orders'])
            i=cfg[stage+'_seeds'].index(row['scenario_seed'])
            assert row['time']==cfg['query_times'][i%5]
            seedgroups[row['condition'],row['source']].add(row['scenario_seed'])
            assert sha(row['full_scenario_ref'])==row['full_scenario_sha256']
            full=read_json(row['full_scenario_ref'])['scenarios'][0]
            assert row['scenario']['orders']==[o for o in full['orders'] if o['book_time']<=row['time']]
            state,_=restore_row(row);policies,prior=historical(row['condition']);actions=packet['actions']
            assert 1<=len(actions)<=6 and any('M0' in a['labels'] for a in actions)
            assert len({a['action_hash'] for a in actions})==len(actions)
            for a in actions:
                verify(state,a['serial']);assert digest(a['physical'])==a['action_hash']
                for label in ['pi1','B2','D4','M0']:
                    actual=score(state,a['serial'],policies[label],prior);original=a['scores'][label]
                    for key in ['g_booked_USD','value_bonus_USD','L_V_USD','tie_break_USD','exact_optimizer_score_USD']:
                        assert abs(actual[key]-original[key])<=1e-6
                    assert actual['release_events']==original['release_events']
                    assert len(actual['release_events'])==len(state.vehs)
                    for coefficient in actual['occupied_coefficients']:
                        assert not coefficient['terminal'] or coefficient['value_USD']==0
            selected=read_json(OUTPUT/'raw/selections'/f'{r["state_id"]}.json')
            assert sha(OUTPUT/'raw/selections'/f'{r["state_id"]}.json')==r['selection_sha256']
            assert selected['assessment_seen'] is False and selected['choices']==r['selections']
            assert selected['selection_returns']==r['selection_returns']
            for label,n in [('select8',8),('select32',32)]:
                choice=choose(actions,{h:float(np.mean(v[:n])) for h,v in r['selection_returns'].items()})
                assert choice==r['selections'][label]
            for label,policy in [('price','pi1'),('bucket_price','B2')]:
                assert choose(actions,{a['action_hash']:a['scores'][policy]['exact_optimizer_score_USD'] for a in actions})==r['selections'][label]
            assert r['selections']['M0']['action_hash']==next(a['action_hash'] for a in actions if 'M0' in a['labels'])
            pair=read_json(OUTPUT/'raw/interactions'/f'{r["state_id"]}-pairs.json')
            assert pair['future_seen'] is False and pair['pairs']==resource_pairs(actions)
            streams={}
            for role,n in [('selection',32),('assessment',64),('interaction',64 if r['interactions'] else 0),('fixed_evaluation',64 if r['fixed'] else 0)]:
                streams[role]=set()
                for k in range(n):
                    f=read_json(OUTPUT/'raw/futures'/f'{stage}-{r["state_id"]}-{role}-{k}.json')
                    assert f['namespace']==dict(condition=r['condition'],state_id=r['state_id'],role=role,replicate=k)
                    assert digest(f['namespace'])==f['stream_hash'];streams[role].add(f['stream_hash'])
                    assert all(o['book_time']>state.t for o in f['future_orders'])
                    assert len({o['id'] for o in f['future_orders']})==len(f['future_orders'])
            for a,b in itertools.combinations(streams,2):assert not streams[a]&streams[b]
            indexed=defaultdict(dict);other_worlds={}
            for wref in r['worlds']:
                refs[wref['path']]=wref['sha256']
                w=read_compressed(wref['path']);assert w['identity']==wref['identity'] and abs(w['total']-wref['total_USD'])<=1e-6
                f=read_json(w['future_ref']);role=f['namespace']['role'];k=f['namespace']['replicate']
                other_worlds[digest(w['initial']),role,k,w['continuation_hash'],w['action_hash']]=w['total']
                if w['action_hash'] and w['continuation_hash']==policies['M0']['policy_hash']:
                    indexed[role,w['action_hash']][k]=w['total']
            for role,key,n in [('selection','selection_returns',32),('assessment','assessment_returns',64)]:
                for h,v in r[key].items():
                    assert len(v)==n and set(indexed[role,h])==set(range(n))
                    assert np.allclose(v,[indexed[role,h][k] for k in range(n)],rtol=0,atol=1e-6)
            from experiments.cfs_rollout_target_alignment.queries import pack_state
            from model_families.single.engine.mt_prototype import Veh
            from experiments.cfs_resource_policy_iteration.protocol import resource_world,pair_audit
            for x in r['interactions']:
                assert len(x['values'])==64 and not x['cross_time_injection']
                states4=[]
                for mask in range(4):
                    counterfactual=state.clone()
                    if mask&1:counterfactual.vehs.append(Veh(len(counterfactual.vehs),x['pair']['hub_i']))
                    if mask&2:counterfactual.vehs.append(Veh(len(counterfactual.vehs),x['pair']['hub_j']))
                    states4.append(digest(pack_state(counterfactual)))
                for v in x['values']:
                    assert abs(v['interaction_USD']-(v['J11']-v['J10']-v['J01']+v['J00']))<=1e-6
                    for initial,field in zip(states4,['J00','J10','J01','J11']):
                        actual=other_worlds[initial,'interaction',v['replicate'],policies['M0']['policy_hash'],None]
                        assert abs(actual-v[field])<=1e-6
            for x in r['fixed']:
                low,high=resource_world(state,x['hub'],0),resource_world(state,x['hub'],1);pair_audit(low,high,x['hub'])
                for k,value in enumerate(x['values_USD']):
                    lv=other_worlds[digest(pack_state(low)),'fixed_evaluation',k,policies[x['continuation']]['policy_hash'],None]
                    hv=other_worlds[digest(pack_state(high)),'fixed_evaluation',k,policies[x['continuation']]['policy_hash'],None]
                    assert abs(hv-lv-value)<=1e-6
            for x in r['sensitivity']:
                for h,value in x['returns'].items():
                    actual=other_worlds[digest(pack_state(state)),'assessment',x['replicate'],policies['pi1']['policy_hash'],h]
                    assert abs(actual-value)<=1e-6
        assert len(seedgroups)==6 and all(len(v)==len(cfg[stage+'_seeds']) for v in seedgroups.values())
    passed('joint_lineage',states=states,unique_worlds=len(refs),whole_current_actions_forced=True,
           no_hidden_futures=True,independent_roles=True,selectors_recomputed_from_selection_only=True)
    return refs

def audit_banks(refs):
    from experiments.cfs_resource_policy_iteration.targets import profiled_table_model
    from experiments.cfs_resource_policy_iteration.protocol import resource_world,pair_audit
    cfg=config();tablemanifest=read_json(OUTPUT/'manifests/banks_tables.json');targets=defaultdict(list);jobs=0
    for bank,seeds in enumerate(cfg['bank_seeds'],1):
        files=sorted((OUTPUT/'raw/banks'/f'bank{bank}').glob('*.json'));assert len(files)==180
        for p in files:
            r=read_json(p);job=r['spec'];jobs+=1;assert digest(job)==r['identity'];assert job['seed'] in seeds
            assert read_json(job['scenario_ref'])['scenarios'][0]==job['scenario']
            from model_families.single.data import matching_state
            from model_families.single.runner import serialize_plan,state_snapshot
            from experiments.cfs_rollout_target_alignment.queries import pack_state
            from .actions import verify
            source,orders=matching_state(job['scenario'],job['parameters'],job['solver']);natural={}
            for trace in r['source_trace']:
                t=trace['period'];source.begin_period([o for o in orders.values() if o.book_time==t])
                if t in job['times']:natural[t]=source.clone()
                plan,_=verify(source,trace['plan']);source.commit_plan(plan)
            assert len(natural)==len(job['times'])
            world={}
            for ref in r['worlds']:
                refs[ref['path']]=ref['sha256'];w=read_compressed(ref['path']);assert w['identity']==ref['identity']
                assert pack_state(resource_world(natural[w['time']],w['hub'],w['q']))==w['initial']
                world[w['time'],w['hub'],w['q']]=w
            for t in r['targets']:
                if t['terminal_anchor']:assert t['target']==0;continue
                lo=world[t['time'],t['hub'],t['q']-1];hi=world[t['time'],t['hub'],t['q']]
                s0,_=restore(job['scenario'],job['parameters'],job['solver'],lo['initial'])
                s1,_=restore(job['scenario'],job['parameters'],job['solver'],hi['initial'])
                pair_audit(s0,s1,t['hub'])
                assert abs(t['target']-hi['total']+lo['total'])<=1e-6
            targets[bank,job['condition']].extend(r['targets'])
    for tab in tablemanifest['tables']:
        assert sha(tab['path'])==tab['sha256']
        rr=targets[tab['bank'],tab['condition']]
        # Floating summation follows recorded job completion order in construction.
        # Independent arithmetic tolerates only roundoff, never changes a fitted table.
        fleet=next(c['fleet'] for c in cfg['conditions'] if f'n{c["orders"]}-k{c["fleet"]}' in tab['condition'])
        model,_,_=profiled_table_model(rr,2*fleet);original=read_json(tab['path'])
        assert model['values'].keys()==original['values'].keys()
        assert len(original['seeds'])==60 and original['seeds']==cfg['bank_seeds'][tab['bank']-1]
        for h in model['values']:
            for q in model['values'][h]:
                a,b=model['values'][h][q],original['values'][h][q]
                assert a['times']==b['times'] and np.allclose(a['values'],b['values'],rtol=0,atol=1e-8)
    deployment=read_json(OUTPUT/'manifests/banks_deployment.json');assert len(deployment['runs'])==600
    counts=Counter((r['condition'],r['policy']) for r in deployment['runs']);assert len(counts)==15 and set(counts.values())=={40}
    seen=set();scenarios={}
    for r in deployment['runs']:
        key=r['condition'],r['policy'],r['scenario_seed'];assert key not in seen;seen.add(key)
        w=read_compressed(r['world_ref']);assert abs(w['total']-r['profit_USD'])<=1e-6 and w['summary']==r
        assert r['scenario_seed'] in cfg['deployment_seeds'];refs[r['world_ref']]=sha(r['world_ref'])
        scene_key=r['condition'],r['scenario_seed'];scene_hash=digest(w['scenario'])
        if scene_key in scenarios:assert scenarios[scene_key]==scene_hash
        else:scenarios[scene_key]=scene_hash
        if r['policy']!='M0':
            tab=next(t for t in tablemanifest['tables'] if t['condition']==r['condition'] and t['bank']==int(r['policy'][-1]))
            assert w['continuation_hash']==tab['policy_hash'] and r['table_sha256']==tab['sha256']
    assert len(read_json(OUTPUT/'processed/bank_fixed_state_actions.json'))==480
    passed('bank_recalculation',construction_jobs=jobs,tables=len(tablemanifest['tables']),deployment_runs=600,
           target_pairs=sum(len(r) for r in targets.values()),no_bank_selection=True,table_roundoff_tolerance_USD=1e-8,
           table_recalculation='All raw targets in deterministic file order; recorded completion-order target_hash not independently reordered')

def worlds():
    refs=audit_joint();audit_banks(refs);steps=0
    for i,(path,expected) in enumerate(sorted(refs.items()),1):
        assert sha(path)==expected,path
        audit=replay_saved(path);steps+=audit['steps']
        if i%5000==0:print(f'Independent persisted-world replay {i}/{len(refs)}',flush=True)
    passed('world_lineage',unique_worlds=len(refs),independent_replay_steps=steps,
        sha256={str(Path(p).relative_to(OUTPUT)):v for p,v in refs.items()},all_current_cash_verified=True,full_cooldown=True)

def history():
    expected=dict(read_json(B/'manifests/historical_readonly.json')['files'])
    exact=read_json(B/'qa/complete_artifact_integrity.json')
    expected.update({(B/Path(k)).relative_to(ROOT).as_posix():v for k,v in exact['sha256'].items()})
    for folder in [A2,B]:
        m=read_json(folder/'manifests/final_manifest.json')
        expected.update({(folder/k).relative_to(ROOT).as_posix():v for k,v in m['deliverable_sha256'].items()})
    for i,(path,h) in enumerate(sorted(expected.items()),1):
        assert sha(ROOT/path)==h,path
        if i%20000==0:print(f'Readonly historical SHA {i}/{len(expected)}',flush=True)
    assert gate()==read_json(OUTPUT/'manifest.json')['historical_final_hashes']
    m=read_json(OUTPUT/'manifest.json');assert sha(m['source']['path'])==m['source']['sha256']
    passed('historical_integrity',files=len(expected),CFS_verified=True,A2_exactB_failedB_readonly=True,
        inherited_manifest_sha256=sha(B/'manifests/historical_readonly.json'),
        inherited_complete_index_sha256=sha(B/'qa/complete_artifact_integrity.json'))

def paired_audit():
    from .analysis import statistics,holm
    analysis=read_json(OUTPUT/'processed/analysis.json');rows=read_json(OUTPUT/'processed/paired_selector_differences.json');rebuilt=[]
    for r in analysis['primary']:
        vals=[x['mean_difference_USD'] for x in rows if (x['stage'],x['condition'],x['source'],x['comparison'])==('formal',r['condition'],r['source'],'select32-price')]
        s=statistics(vals);assert s['n']==40
        for k in s:
            if isinstance(s[k],(float,int)):assert np.isclose(s[k],r[k],rtol=0,atol=1e-9),(k,s[k],r[k])
        rebuilt.append(s|dict(condition=r['condition'],source=r['source']))
    holm(rebuilt)
    for a,b in zip(rebuilt,analysis['primary']):assert abs(a['p_Holm']-b['p_Holm'])<=1e-12
    passed('paired_statistics',six_Holm_tests=rebuilt,scenario_unit=True,conditional_path_variance_not_scenario_count=True)

def tests():
    p=OUTPUT/'qa/pytest_final.xml'
    assert p.exists(),'Run actual related tests first; preflight is not final QA'
    root=ET.parse(p).getroot();cases=root.findall('.//testcase')
    assert len(cases)>=27
    assert not root.findall('.//failure') and not root.findall('.//error') and not root.findall('.//skipped')
    names={r.get('classname','') for r in cases}
    assert any('test_joint_action_value_diagnostic' in n for n in names)
    assert any('test_integer_action_accounting' in n for n in names)
    passed('tests',test_count=len(cases),failures=0,errors=0,skipped=0,xml_sha256=sha(p),
           scope='round7 related tests and necessary historical controls only; not entire platform suite')

def integrity():
    m=read_json(OUTPUT/'manifest.json');assert numerical_hashes()==m['source_code_sha256']
    with zipfile.ZipFile(OUTPUT/'raw/source.snapshot.zip') as z:
        import hashlib
        for name,h in m['source_code_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    assert sha(OUTPUT/'raw/source.snapshot.zip')==m['snapshot_sha256']
    files={}
    for folder in ['raw','tables','manifests']:
        for p in (OUTPUT/folder).rglob('*'):
            if p.is_file() and p.name!='final_manifest.json':files[p.relative_to(OUTPUT).as_posix()]=sha(p)
    passed('complete_artifact_integrity',sha256=files,files=len(files),frozen_numerical_source_unchanged=True,
           frozen_analysis_unchanged=sha(OUTPUT/'analysis.py')==read_json(OUTPUT/'manifests/analysis_preregistration.json')['analysis_sha256'])

def report():
    from .figures import short
    a=read_json(OUTPUT/'processed/analysis.json');scores=read_json(OUTPUT/'processed/joint_action_scores.json')
    diffs=read_json(OUTPUT/'processed/paired_selector_differences.json');fallback=read_json(OUTPUT/'processed/fallback_economic_supplement.json')
    interaction=read_json(OUTPUT/'processed/resource_interactions.json') if (OUTPUT/'processed/resource_interactions.json').exists() else []
    fixed=read_json(OUTPUT/'processed/fixed_state_evaluation.json');supp=read_json(OUTPUT/'processed/supplement.json')
    lines=['# 第七轮：联合动作价值接入诊断与独立 MC 复核','',
        '本轮为 surrogate-distribution rollout 诊断，不是现实条件期望最优证明。数值已完成不等于交付完成；最终以 final_manifest 的 COMPLETED 和 SHA/QA 为准。',
        '所有旧实验与 CFS 只读。继承 exact-accounting B 的整数动作核算，不改变成本、物理、MILP容差、旧模型或平台默认值。没有提交、推送、RL训练或新PI链。','',
        '## 1. 资源底座、q、bucket 与时点：一致、代理和错误分别是什么？','',
        '训练资源标签是同一 pre-injection 状态移除查询hub全部idle，再注入 q/q−1 的全冷却边际；配对保留共同承诺、需求与未来。在线q则是同一(release time,hub)退出组的prefix序号，不是自然库存。两者采用同一标量lookup，但状态压缩/注入底座不能因此视为真实同状态Q。',
        'bucket采用冻结construction相对分位数，不是绝对供需平衡；q/q−1共同使用原始注入前桶，这属于明确声明的代理。当前桶应用于未来release时刻，不预测未来桶；终端严格0，负值保留。query_semantics和bucket_base_alignment逐值保存自然/移除/注入压力与支持。未将这些已声明近似预先认定为实现错误，未作单因素“修正”。',
        '完整当前动作经过独立编译核算，真实窗口目标=booked g+prefix bonus−delay tie-break。历史L报告列不含tie；真实价格selector保留原tie，不能假装其只优化L。人工例与回放通过情况见QA，不能替代机制性证据。','',
        '## 2. 可分价格能否在同一完整状态排对联合动作？','',
        '| 工况 / 状态来源 | 状态数 | 严格点排序一致率平均 | top分歧率 | tie-set Jaccard平均 |','|---|---:|---:|---:|---:|']
    for cid,src in sorted({(r['condition'],r['source']) for r in scores if r['stage']=='formal'}):
        by={r['state_id']:r for r in scores if (r['stage'],r['condition'],r['source'])==('formal',cid,src)}
        vals=[r['strict_point_ranking_agreement'] for r in by.values() if r['strict_point_ranking_agreement'] is not None]
        ranking='NA' if not vals else f'{np.mean(vals):.4f}'
        lines.append(f'|{short(cid)}/{src}|{len(by)}|{ranking}|{np.mean([r["point_top_disagreement"] for r in by.values()]):.4f}|{np.mean([r["tie_set_jaccard"] for r in by.values()]):.4f}|')
    lines+=['','点排序参考Q有64路径MC误差，未跨过路径区间的严格点顺序不能冒称真实顺序。候选合并物理等价动作，最多6；没有决策车辆或无其他动作仍保留，不按收益删除。','',
        '## 3. 独立 assessment 的收益差与精度','',
        '| 工况 / 状态来源 | 32−price均值 USD | SD | 中位数 | paired t95 | bootstrap95 | Holm p |','|---|---:|---:|---:|---|---|---:|']
    for r in a['primary']:
        lo,hi=r['t95_low_USD'],r['t95_high_USD'];bl,bh=r['bootstrap95_low_USD'],r['bootstrap95_high_USD']
        lines.append(f'|{short(r["condition"])}/{r["source"]}|{r["mean_USD"]:+.3f}|{r["SD_USD"]:.3f}|{r["median_USD"]:+.3f}|[{lo:.3f}, {hi:.3f}]|[{bl:.3f}, {bh:.3f}]|{r["p_Holm"]:.6g}|')
        direction='正向' if r['mean_USD']>0 else '负向' if r['mean_USD']<0 else '零差'
        conclusion=('有经六项Holm调整的'+direction+'差异证据' if r['p_Holm']<.05 else '未建立调整后差异证据；不是等价或无效证明')
        lines.append(f'\n{short(r["condition"])}/{r["source"]}：{conclusion}。效应范围按上述区间理解。\n')
    lines+=['','每行40独立scenario；状态来源分别报告，64未来路径只降低状态内MC误差，不能充当n=64/2560。区间未作多重区间校正，Holm是单列两侧p。收益是冻结两个选择的独立assessment差，不是selection最大值。','',
        '## 4. 32路径相对8路径：真实选择还是样本最大化？','',
        '| 工况 / 来源 | 32−8 USD | t95 | 独立单位 |','|---|---:|---|---|']
    for r in a['all_comparisons']:
        if r['stage']=='formal' and r['comparison']=='select32-select8':
            lines.append(f'|{short(r["condition"])}/{r["source"]}|{r["mean_USD"]:+.3f}|[{r["t95_low_USD"]:.3f},{r["t95_high_USD"]:.3f}]|40 scenario|')
    lines+=['','8/32使用嵌套selection CRN；表中用独立assessment评价预先冻结的动作，仍是预声明机制次比较，不拼为新的全局显著结论。候选assessment最大值构造的regret另列finite-sample maximum bias，不能当真实无偏regret或全动作最优。','',
        '## 5. 四世界交互与选择损失的关系','']
    grouped=defaultdict(list)
    for r in interaction:
        if r['stage']=='formal':grouped[r['condition'],r['source'],r['scenario_seed']].append(r)
    from .analysis import statistics
    if grouped:
        lines+=['| 工况 / 来源 | eligible scenario | I均值 USD | t95 | 平均绝对I USD |','|---|---:|---:|---|---:|']
        for cid,src in sorted({(c,s) for c,s,k in grouped}):
            keys=sorted(k for c,s,k in grouped if (c,s)==(cid,src))
            vals=[float(np.mean([r['mean_USD'] for r in grouped[cid,src,k]])) for k in keys]
            absval=[float(np.mean([abs(r['mean_USD']) for r in grouped[cid,src,k]])) for k in keys]
            st=statistics(vals);lines.append(f'|{short(cid)}/{src}|{len(vals)}|{st["mean_USD"]:+.3f}|[{st["t95_low_USD"]},{st["t95_high_USD"]}]|{np.mean(absval):.3f}|')
            match={r['scenario_seed']:r['mean_difference_USD'] for r in diffs if (r['stage'],r['condition'],r['source'],r['comparison'])==('formal',cid,src,'select32-price')}
            gain=[match[k] for k in keys]
            correlation=None if len(keys)<3 or np.std(absval)==0 or np.std(gain)==0 else float(np.corrcoef(absval,gain)[0,1])
            lines.append(f'\n{short(cid)}/{src}：同eligible scenario平均|I|与32−price gain的描述相关={correlation}；非因果识别、未校正次分析。\n')
    else:lines.append('无eligible资源对，明确未完成交互，不制造空洞图。')
    lines+=['','仅测当前可自由注入资源，每状态最多2对，先按pi1/M0 release差选择，再生成独立interaction角色未来。反事实总车队超过K被标记，不推广成固定车队政策；跨时刻注入未实现。I非零不证明导致政策损失，更不证明必须用neural。Q−L残差还混合标签误差、状态压缩、g口径及接入近似，不全归于nonseparability。','',
        '## 6. 回退是否出现在关键动作？','',
        '| 工况 / 来源 | selector | potential query回退率 | selected query回退率 | changed query回退率 | selected绝对价值回退占比 | changed绝对价值回退占比 |','|---|---|---:|---:|---:|---:|---:|']
    for cid,src,selector in sorted({(r['condition'],r['source'],r['selector']) for r in fallback if r['stage']=='formal' and r['selector'] in ['price','select32','bucket_price']}):
        rr=[r for r in fallback if (r['stage'],r['condition'],r['source'],r['selector'])==('formal',cid,src,selector)]
        def ratio(left,right):
            den=sum(r[right] for r in rr);return 'NA' if not den else f'{sum(r[left] for r in rr)/den:.4f}'
        rates=[ratio(p+'_fallback_count',p+'_count') for p in ['all','selected','changed']]
        values=[ratio(p+'_fallback_absolute_value_USD',p+'_absolute_value_USD') for p in ['selected','changed']]
        lines.append(f'|{short(cid)}/{src}|{selector}|'+ '|'.join(rates+values)+'|')
    lines+=['','potential分母来自B2完整原solve，即使物理动作和其他模型合并也保留其solver。selected是当前selector动作的occupied prefix；changed是相对price动作的occupied(time,hub,q)对称差，不是全部查询。绝对USD避免正负抵消；终端零、非终端分母及空分母均明列。回退率高不单独证明关键收益被回退：需要结合独立Q差与release机制图。','',
        '## 7. 独立构造银行的一次MC改进可重复性','',
        '| 工况 | 银行 | 相对M0部署均值 USD | t95 |','|---|---:|---:|---|']
    for r in a['bank_statistics']:lines.append(f'|{short(r["condition"])}|{r["bank"]}|{r["mean_USD"]:+.3f}|[{r["t95_low_USD"]:.3f},{r["t95_high_USD"]:.3f}]|')
    lines+=['','四个construction银行各60独立scenario，全部M0父策略、同一原A轮转hub/时间/q网格、插值及接入；每个模型冻结后在同40部署scenario对M0配对。银行间价格差与固定120状态动作hash保留在tables及bank_fixed_state_actions，不挑赢家、不替换模型。每个CI条件于其冻结表；4×40不是160独立部署样本，4银行不足以精确估总体训练方差。','',
        '## 8. 固定状态 continuation 与自然投影分布','',
        '固定diagnostic前4seed的相同状态/资源底座/64未来，分别评M0、pi1和CA pi3；每个continuation用同未来q/q−1，不混入主均值。matched marginal和natural projected V0/V1/V3差另存。自然表来自历史各自状态分布/有限样本，本轮独立未来又是代理分布，因此fixed−natural混合分布与采样器差，不能归为纯continuation、纯投影迁移或最优Bellman残差。','',
        f'固定状态评价记录数={len(fixed)}；pi1 continuation小样本次比较组数={len(supp["continuation_statistics"])}，详见supplement.json。没有独立E(D4)，未强制补做PI链。','',
        '## 9. 未来生成假设与外推边界','',
        '本轮在线代理仅用被冻结B construction8300–8359估计每期到达率和经验OD/load/fare/ready-delay/slack。未来订单数Poisson(rate×剩余可booking期数)，booking均匀，marks有放回独立采样；不知道正式场景隐藏未来shipment、剩余实际N或OD配额。公开订单/承诺不变，future IDs唯一。selection、assessment、interaction、fixed角色SHA namespace分离。',
        '这种 marked-Poisson SURROGATE 与历史固定N生成过程不同；正式结论只对该代理分布/冻结状态来源/有限候选成立。construction银行和部署保留原CFS场景过程，是另一层一次改进复核，不能把其现金与代理局部Q当同一种估计量。本轮没有用事后真实剩余路径选动作，也没有事后路径收益冒充在线可部署最优。','',
        '## 10. 下一步的证据驱动判断','',
        '若32−8独立收益为正而银行幅度差大，优先提高MC精度、支持和银行复核；若32−price经主family调整仍有正面证据，下一轮单独预注册有限候选联合rollout/少量coupling，而不是把所有残差解释成交互。若区间宽，报告可兼容效应和精度不足，不能声称等价、无效或安全非劣。',
        '已声明q/桶时点代理需要单因素语义实验才能识别其贡献；本轮不能据此选择更有利语义。代理表现还应对真实需求条件化作独立外推验证；只有充分复核后稳定状态依赖/政策损失证据，才讨论state-dependent price learning或Strategic RL对照，MILP继续负责可行性与承诺。没有自动进入RL、bucketPI或选择新alpha；RH4/RH8仅保留后续对照，不把Preview gap当完整日上界或可回收利润。','',
        '## 覆盖、预算、时间与复现','',
        f'正式缺失/退化状态记录={len([r for r in a["missing"] if r["stage"]=="formal"])}；完整列表在analysis.json。每个scenario只取循环t=3/8/12/16/20，不重选有利状态。工程pilot预算在diagnostic/formal前冻结，64assessment正式后未扩样。',
        '状态/完整动作集合先冻结，再生成selection，冻结选择后独立assessment；所有原始世界含initial、未来ref、整段trace、ledger、source SHA。世界与同动作future可内容寻址去重，index仍完整可重建。最终逐世界重放、现金、booking、冷却、历史SHA及真实PDF视觉QA均须PASSED。',
        '时间图报告阶段wall、候选、全MC、solver和replay。pipeline inclusive包含后两者，不能相加。当前数值记录没有将序列化与fit aggregation细分成精确独立计时，因此这两项未识别、明确不虚构；完整MC是离线诊断，不声称满足实时在线延迟。',
        '实际命令与恢复规则见README。统计采用Python NumPy/SciPy，绘图Matplotlib独立非数值层，源CSV/config、嵌入字体向量PDF/SVG、600dpiPNG和atlas保存。以final manifest及QA判定交付，不以报告存在判定完成。']
    (OUTPUT/'joint_action_value_diagnostic_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def freeze():
    m=writable()
    for name in QA:assert read_json(OUTPUT/'qa'/(name+'.json'))['status']=='PASSED',name
    integrity=read_json(OUTPUT/'qa/complete_artifact_integrity.json')
    for name,h in integrity['sha256'].items():assert sha(OUTPUT/name)==h,name
    figures=read_json(OUTPUT/'figures/figure_manifest.json');assert len(figures['names'])+len(figures['missing_groups'])==6
    for name,h in figures['sha256'].items():assert sha(OUTPUT/'figures'/name)==h,name
    assert gate()==m['historical_final_hashes']
    files={}
    for folder in ['config','processed','tables','figures','qa']:
        for p in (OUTPUT/folder).rglob('*'):
            if p.is_file():files[p.relative_to(OUTPUT).as_posix()]=sha(p)
    for p in list(OUTPUT.glob('*.py'))+list(OUTPUT.glob('*.md')):files[p.name]=sha(p)
    assert 'joint_action_value_diagnostic_report.md' in files
    m.update(status='COMPLETED',completed_at=now(),deliverable_sha256=files,
        complete_artifact_integrity_sha256=sha(OUTPUT/'qa/complete_artifact_integrity.json'),
        semantic_correction='NONE',future_distribution='SURROGATE',no_RL_or_new_PI=True)
    atomic_json(OUTPUT/'manifest.json',m);atomic_json(OUTPUT/'manifests/final_manifest.json',m)
    print('Round 7 COMPLETED and immutable; no RL/BPI launched.',flush=True)

def main(stage):
    writable()
    if stage=='analysis':
        marker('csv',20)
        from .analysis import run,load_rows
        run();supplement(load_rows('diagnostic')+load_rows('formal'));paired_audit();report()
        marker('pdf',7)
        from .figures import run as figures
        figures();return
    if stage=='worlds':worlds();integrity();return
    if stage=='history':history();return
    if stage=='tests':tests();return
    if stage=='freeze':freeze();return

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['analysis','worlds','history','tests','freeze'],required=True)
    main(p.parse_args().stage)
