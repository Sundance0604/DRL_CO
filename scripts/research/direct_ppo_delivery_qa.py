"""Read-only numerical audits; writes QA solely into active Direct attempt."""
import argparse
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import csv
import math
from model_families.single.strategic.artifacts import read_json,write_json
from experiments.cfs_ppo_strategic_control.study_common import sha,ROOT,CONDITIONS,key
from experiments.cfs_ppo_signal_action_diagnostic.run import verify_hashes
from experiments.cfs_direct_ppo_v3.run import load,cfg,scenes,PARENT


def integrity(target):
    from experiments.cfs_direct_ppo_v3.network import DirectPolicy
    from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
    from experiments.cfs_ppo_strategic_control.study_execution import replay as mc_replay
    from model_families.single.strategic.contracts import identity
    m=load(target)
    if m['status']!='NUMERICAL_COMPLETE_QA_PENDING':raise RuntimeError('numerical completion required')
    verify_hashes(Path('/'),m['history_hashes'])
    frozen=read_json(target/'protocol/frozen-policies.json');verify_hashes(target,frozen['hashes'])
    dimensions=[];counts=[];mc_count=0
    for c in CONDITIONS:
        name=key(c);records={s['seed']:s for split in ('train','validation','test') for s in scenes(target,c,split)}
        for seed in (30001,30002,30003):
            configuration=cfg(target,c,seed)
            choice=next(r for r in frozen['rows'] if r['condition']==name and r['seed']==seed)
            cp=target/choice['checkpoint'];meta=read_json(cp/'manifest.json')
            for field in ('action','feature_scales','parameters','solver'):
                if meta[field]!=configuration[field]:raise RuntimeError('checkpoint configuration mismatch '+field)
            if meta['update']!=choice['selected_update'] or not meta['no_baseline']:raise RuntimeError('checkpoint choice/MC mismatch')
            with no_mc_guard():
                policy=DirectPolicy(cp,m['source'])
                dimensions.append(dict(condition=name,seed=seed,inference_parameters=sum(p.numel() for p in policy.network.parameters()),
                    body_input=policy.network.body[0].in_features,MC_features=0,actor_dimensions=policy.network.actor.out_features))
            tr=list((target/'training'/name/str(seed)).glob('training-episode-*.json'))
            va=list((target/'training'/name/str(seed)).glob('validation-*-s*.json'))
            te=list((target/'evaluation').glob(f'{name}-{seed}-s*.json'))
            if (len(tr),len(va),len(te))!=(128,64,24):raise RuntimeError('model episode coverage mismatch')
            counts.append(dict(condition=name,seed=seed,training_episodes=len(tr),unique_training_scenarios=len({read_json(p)['scenario_seed'] for p in tr}),
                validation_episodes=len(va),test_episodes=len(te)))
            for update in range(1,33):
                r=read_json(target/'training'/name/str(seed)/f'update-{update:04d}.json')
                if r['updates']!=update or r['episodes']!=4*update or r['transitions']!=96:raise RuntimeError('PPO update budget mismatch')
        old=read_json(PARENT/'protocol'/(name+'.json'));old['baseline']=str(PARENT/old['baseline'])
        for split in ('validation','test'):
            for s in scenes(target,c,split):
                p=target/'mc'/f'{name}-{split}-s{s["seed"]}.json';r=read_json(p)
                if any(any(x!=0 for x in tr['signal']['values']) for tr in r['transitions']):raise RuntimeError('pure MC contains residual')
                mc_replay(s,old,r);mc_count+=1
    if mc_count!=120:raise RuntimeError('MC replay denominator')
    numerical_index={p.relative_to(target).as_posix():sha(p) for f in ('engineering','smoke','training','mc','evaluation')
        for p in (target/f).rglob('*') if p.is_file()}
    write_json(target/'qa/numeric-index.json',dict(status='PASSED',hashes=numerical_index))
    size=sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
    if size>8*2**30:raise RuntimeError('registered 8 GiB disk exceeded')
    calls=read_json(target/'qa/numerical-start.json')['solver_calls']
    if calls!=49959:raise RuntimeError('solver accounting mismatch: '+str(calls))
    write_json(target/'qa/integrity.json',dict(status='PASSED',code_sha256=sha(Path(__file__)),
        history_files=len(m['history_hashes']),all_history_and_CFS_SHA_checked=True,all_inputs_and_source_SHA_checked=True,
        numeric_files=len(numerical_index),model_counts=counts,model_dimensions=dimensions,
        additional_pure_MC_replays=mc_count,actual_solver_calls=calls,baseline_no_actor=True,
        output_bytes_before_final_QA=size,numerical_seconds=(datetime.fromisoformat(m['numerical_completed_at'])-
            datetime.fromisoformat(read_json(target/'qa/numerical-start.json')['started_at'])).total_seconds()))
    print('INTEGRITY PASSED',calls,'calls',size/2**30,'GiB',flush=True)


def tests(target):
    load(target)
    p=target/'qa/pytest-final.xml';suites=list(ET.parse(p).getroot().iter('testsuite'))
    counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts['tests']<56 or any(counts[k] for k in ('failures','errors','skipped')):raise RuntimeError('final tests did not pass')
    write_json(target/'qa/tests.json',dict(status='PASSED',**counts,passed=counts['tests'],xml_file='qa/pytest-final.xml',xml_sha256=sha(p),
        seconds=sum(float(s.get('time','0')) for s in suites),scope='Direct-v3, legacy Direct, PPO, classical value, rolling; not full-platform',
        code_sha256=sha(Path(__file__))))
    print('TESTS PASSED',counts,flush=True)


def visual(target):
    """Invoke only after actually displaying and reviewing every final plot."""
    from PIL import Image
    load(target);checks=[]
    for stem in ('01-profit','02-learning','03-seeds'):
        p=target/'figures'/(stem+'.png');im=Image.open(p)
        if abs(im.info['dpi'][0]-600)>.1:raise RuntimeError('PNG dpi wrong')
        ET.parse(target/'figures'/(stem+'.svg'))
        checks.append(dict(stem=stem,pixels=list(im.size),dpi=list(im.info['dpi'])))
    hashes={p.relative_to(target).as_posix():sha(p) for p in (target/'figures').glob('*')}
    write_json(target/'qa/visual.json',dict(status='PASSED',actual_visual_review=True,
        note='All three latest PNGs actually displayed and checked; corresponding SVG XML checked, not independently raster rendered.',
        technical=checks,reviewed_hashes=hashes))


def delivery(target):
    """Independent export reconciliation and byte checks before terminal freeze."""
    m=load(target)
    for name in ('data-isolation','engineering','smoke','training','mc','evaluation','final-replay','analysis','tests','visual','integrity','policy-recalculation'):
        if read_json(target/'qa'/(name+'.json'))['status']!='PASSED':raise RuntimeError('missing gate '+name)
    verify_hashes(Path('/'),m['history_hashes'])
    verify_hashes(target,read_json(target/'qa/numeric-index.json')['hashes'])
    verify_hashes(target,read_json(target/'protocol/frozen-policies.json')['hashes'])
    verify_hashes(target,read_json(target/'qa/visual.json')['reviewed_hashes'])
    tests_record=read_json(target/'qa/tests.json')
    if sha(target/tests_record['xml_file'])!=tests_record['xml_sha256']:raise RuntimeError('final XML changed')
    def rows(stem):
        with (target/(stem+'.csv')).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    def equal(actual,expected):
        if not math.isclose(float(actual),float(expected),rel_tol=1e-12,abs_tol=1e-7):raise RuntimeError('CSV numerical mismatch')
    paired=rows('paired_results');business=rows('business_metrics');curves=rows('learning_curves');summary=rows('summary')
    if tuple(map(len,(paired,business,curves,summary)))!=(216,288,288,3):raise RuntimeError('CSV denominator mismatch')
    ids={(r['condition'],int(r['training_seed']),int(r['scenario_seed'])) for r in paired}
    expected={(key(c),s,i) for c in CONDITIONS for s in (30001,30002,30003) for i in range(22020,22044)}
    if ids!=expected or len(ids)!=len(paired):raise RuntimeError('paired identities mismatch')
    for r in paired:
        direct=read_json(target/r['record_file']);mc=read_json(target/'mc'/f'{r["condition"]}-test-s{r["scenario_seed"]}.json')
        equal(r['Direct_profit_USD'],direct['ledger']['realized_profit']);equal(r['MC_profit_USD'],mc['ledger']['realized_profit'])
        equal(r['difference_USD'],direct['ledger']['realized_profit']-mc['ledger']['realized_profit'])
    for r in summary:
        group=[p for p in paired if p['condition']==r['condition']]
        mc=sum(float(p['MC_profit_USD']) for p in group)/72;direct=sum(float(p['Direct_profit_USD']) for p in group)/72
        equal(r['MC_mean_USD'],mc);equal(r['Direct_mean_USD'],direct);equal(r['difference_USD'],direct-mc)
        equal(r['difference_percent'],100*(direct-mc)/abs(mc))
        result=next(e for e in read_json(target/'analysis/results.json') if e['condition']==r['condition'])
        for name,value in [('CI95_low_USD',result['marginal95'][0]),('CI95_high_USD',result['marginal95'][1]),
            ('family_CI_low_USD',result['simultaneous98_333'][0]),('family_CI_high_USD',result['simultaneous98_333'][1])]:equal(r[name],value)
    for r in business:
        if r['method']=='Pure-MC':path=target/'mc'/f'{r["condition"]}-test-s{r["scenario_seed"]}.json'
        else:path=target/'evaluation'/f'{r["condition"]}-{r["training_seed"]}-s{r["scenario_seed"]}.json'
        raw=read_json(path);ledger=raw['ledger'];metrics=raw['metrics'];parameters=read_json(target/'protocol'/(r['condition']+'.json'))['parameters']
        checks={'profit_USD':ledger['realized_profit'],'revenue_USD':ledger['completed_fare'],'cost_USD':ledger['realized_cost'],
            'completed_orders':ledger['completed_orders'],'loaded_cost_USD':ledger['actual_loaded_distance']*parameters['loaded_cost'],
            'empty_cost_USD':ledger['actual_empty_distance']*parameters['empty_cost'],'waiting_cost_USD':ledger['actual_vehicle_waiting']*parameters['waiting_cost'],
            'backlog_penalty_USD':ledger['backlog_penalty'],'cancellation_penalty_USD':ledger['cancellation_penalty'],
            'solver_calls':metrics['solver_calls'],'solver_seconds':metrics['solver_seconds'],'MILP_failed_calls':0}
        for field,value in checks.items():equal(r[field],value)
        duration=sum(segment['end']-segment['start'] for event in metrics['execution_events'] for segment in event['segments'] if 'start' in segment and 'end' in segment)
        equal(r['vehicle_utilization_travel_fraction_full_cooldown'],duration/(len(metrics['execution_events'])*ledger['cutoff_exclusive']))
        equal(r['reposition_cost_USD'],sum(s['distance']*parameters['empty_cost'] for event in metrics['execution_events'] for s in event['segments'] if s['kind']=='reposition'))
    if len({(r['condition'],r['training_seed'],r['update']) for r in curves})!=288:raise RuntimeError('curve duplicate')
    for r in curves:
        raw=read_json(target/'training'/r['condition']/r['training_seed']/f'update-{int(r["update"]):04d}.json')
        for field,source in [('training_mean_USD','training_mean_USD'),('policy_loss','policy_loss'),('critic_MSE','value_loss'),('KL','approximate_kl'),('clip_fraction','clip_fraction')]:equal(r[field],raw[source])
        if raw['validation_mean_USD'] is not None:equal(r['validation_mean_USD'],raw['validation_mean_USD'])
    hashes={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file() and p.relative_to(target).as_posix() not in ('manifest.json','final_manifest.json','qa/delivery-index.json','qa/delivery.json')}
    write_json(target/'qa/delivery-index.json',dict(status='PASSED',hashes=hashes))
    verify_hashes(target,hashes)
    write_json(target/'qa/delivery.json',dict(status='PASSED',code_sha256=sha(Path(__file__)),export_rows=[216,288,288,3],
        all_numeric_visual_tests_history_sources_byte_verified=True,CSV_independently_reconciled_to_raw=True,
        delivery_index_sha256=sha(target/'qa/delivery-index.json'),deliverables=len(hashes)))
    print('DELIVERY PASSED',len(hashes),'files',flush=True)


def policy_recalculation(target):
    """Read-only deterministic policy audit, never optimizes or changes a checkpoint."""
    import torch
    from model_families.single.strategic.contracts import EncodedObservation,VehicleObservation
    from experiments.cfs_direct_ppo_v3.network import DirectPolicy
    from experiments.cfs_direct_ppo_v3.qa import no_mc_guard
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    m=load(target);count=0;episodes=0
    for c in CONDITIONS:
        name=key(c)
        for seed in (30001,30002,30003):
            folder=target/'training'/name/str(seed)
            groups=[(folder/f'inference-{update:04d}',list(folder.glob(f'validation-{update:04d}-s*.json'))) for update in (8,16,24,32)]
            groups.append((folder/'best-inference',list((target/'evaluation').glob(f'{name}-{seed}-s*.json'))))
            for checkpoint,paths in groups:
                with no_mc_guard():
                    policy=DirectPolicy(checkpoint,m['source'])
                    for path in paths:
                        for tr in read_json(path)['transitions']:
                            data=dict(tr['observation']);data['hubs']=tuple(data['hubs'])
                            data['vehicles']=tuple(VehicleObservation(**v) for v in data['vehicles'])
                            obs=EncodedObservation(**data)
                            computed=policy.act(obs)
                            if len(computed)!=10 or any(not math.isclose(a,b,rel_tol=0,abs_tol=1e-7) for a,b in zip(computed,tr['signal']['values'])):
                                raise RuntimeError('frozen deterministic actor output mismatch: '+str(path))
                            count+=1
                        episodes+=1
    if (episodes,count)!=(792,19008):raise RuntimeError('deterministic policy audit denominator')
    write_json(target/'qa/policy-recalculation.json',dict(status='PASSED',code_sha256=sha(Path(__file__)),
        deterministic_episodes=episodes,actor_actions=count,validation_checkpoint_and_selected_test_actor_recomputed=True,
        no_MC_guard=True,no_new_operating_experiment=True))
    print('POLICY RECALCULATION PASSED',episodes,count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('integrity','tests','visual','delivery','policy-recalculation'));p.add_argument('--attempt',required=True)
    a=p.parse_args();globals()[a.stage.replace('-','_')](Path(a.attempt).resolve())
