"""Independent replay and byte-level final gates, only writes recovery attempt."""
import argparse,csv,copy,math,statistics
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .run import TARGET,OLD,sources
from experiments.cfs_direct_ppo_diagnosis import common as c
from experiments.cfs_direct_ppo_v3.qa import replay,no_mc_guard

def qa():
    m=c.read_json(TARGET/'manifest.json')
    if m['status']!='NUMERICAL_COMPLETE_QA_PENDING' or m['source_files']!=sources():raise RuntimeError('recovery source/status')
    c.verify_parent(True);c.verify_hashes(Path('/'),m['recovered_history']);c.verify_hashes(TARGET,m['inputs'])
    from experiments.cfs_ppo_strategic_control.study_execution import replay as old_replay,action_from_plan
    from model_families.single.planning import compile_action
    from experiments.cfs_direct_ppo_v3.contracts import DirectSchema,DirectSignal
    from experiments.cfs_direct_ppo_v3.control import DirectProvider
    counts=dict(M0=0,zero=0,counterfactual=0,probes=0,actor_models=0,critic_steps=0)
    for name in c.NAMES:
        cfg=c.configuration(name);legacy=copy.deepcopy(cfg);legacy['action'].pop('version');legacy['baseline']=dict(method='zero')
        for s in c.scenes(name,'test'):
            a=c.read_json(OLD/'m0'/f'{name}-s{s["seed"]}.json');old_replay(s,legacy,a);counts['M0']+=1
            z=c.read_json(OLD/'zero'/f'{name}-s{s["seed"]}.json')
            with no_mc_guard():replay(s,cfg,z)
            if z['ledger']!=a['ledger']:raise RuntimeError('zero ledger regression')
            counts['zero']+=1
        for seed in c.SEEDS:
            d=c.read_json(TARGET/'diagnostics'/f'{name}-{seed}.json');counts['actor_models']+=1
            for actor in d['actor']:
                signals=np.array([p['signal'] for p in actor['points']]);mean=signals.mean(0)
                if len(signals)!=96 or not np.allclose(mean,actor['mean']) or not math.isclose(np.abs(signals-mean).mean(),actor['D_pi'],abs_tol=1e-12):raise RuntimeError('Actor mean/Dpi mismatch')
                if not np.allclose(signals.std(0),actor['std']):raise RuntimeError('Actor std mismatch')
                if np.any(np.abs(signals)>1):raise RuntimeError('Actor bounds')
                for point in actor['points']:
                    if not np.allclose(np.asarray(point['signal'])[::2]*cfg['action']['scales'],point['near_USD']):raise RuntimeError('Actor USD scale mismatch')
            folder=c.PARENT/'training'/name/str(seed);x=[];y=[]
            for path in sorted(folder.glob('training-episode-*.json')):
                r=c.read_json(path);carry=0.;returns=[]
                for tr in reversed(r['transitions']):carry+=tr['reward_USD']/1000;returns.append(carry)
                y.extend(reversed(returns));x.extend(t['value'] for t in r['learning']);counts['critic_steps']+=24
            difference=[a-b for a,b in zip(y,x)];ev=1-statistics.pvariance(difference)/statistics.pvariance(y)
            rmse=math.sqrt(sum(v*v for v in difference)/len(difference))
            if not math.isclose(ev,d['pooled_critic']['critic_MC_EV'],abs_tol=1e-10) or not math.isclose(rmse,d['pooled_critic']['critic_MC_RMSE_reward_units'],abs_tol=1e-8):raise RuntimeError('Critic independent realized target mismatch')
    for path in [*sorted((TARGET/'sensitivity').glob('state*.json')),*sorted((TARGET/'pilot').glob('probes-*.json'))]:
        result=c.read_json(path);spec=result['spec'];cfg=c.configuration(spec['condition'])
        s=next(s for s in c.scenes(spec['condition'],'validation') if s['seed']==spec['scenario_seed'])
        source=c.read_json(c.PARENT/spec['source']);ep,obs=c.restore(s,cfg,source,spec['period'])
        from model_families.single.runner import state_snapshot
        if c.identity(state_snapshot(ep._sim))!=c.identity(result['state']):raise RuntimeError('source physical state mismatch')
        for control in result['rows']:
            p=control['probe']['plan'];info=copy.deepcopy(p['solver']);compiled=compile_action(ep._sim,action_from_plan(p),info)
            if info['accounting']!=p['accounting'] or not math.isclose(compiled[0],p['operating_profit'],abs_tol=1e-7):raise RuntimeError('probe integer cash mismatch')
            if info['status']!='OPTIMAL' or control['canonical']!=c.canonical(p,result['state']):raise RuntimeError('probe status/identity')
            provider=DirectProvider(DirectSchema(**cfg['action']),DirectSignal(tuple(control['signal']),obs.period,obs.revision,obs.hash),ep._sim.T)
            for coeff in control['probe']['coefficients']:
                value=provider.marginal_release_value(coeff['time'],coeff['hub'],coeff['supply_slot'],{})
                if not math.isclose(value,coeff['value'],abs_tol=1e-7):raise RuntimeError('probe price mismatch')
            counts['probes']+=1
    cases=[*c.read_json(TARGET/'counterfactual/results.json'),*c.read_json(TARGET/'qa/pilot.json')['cases']]
    by_path={case[field]:case for case in cases for field in ('reference','variant_record')}
    for folder in ('counterfactual','pilot'):
        for path in (TARGET/folder).glob('case*-*.json'):
            r=c.read_json(path)
            meta=by_path[path.relative_to(TARGET).as_posix()];name=meta['condition']
            s=next(s for s in c.scenes(name,'validation')[:4] if s['seed']==meta['scenario_seed'])
            if s['id']!=r['scenario_id'] or s['seed']!=r['scenario_seed']:raise RuntimeError('counterfactual scenario mismatch')
            with no_mc_guard():replay(s,c.configuration(name),r)
            counts['counterfactual']+=1
    if counts['M0']!=72 or counts['zero']!=72 or counts['actor_models']!=9 or counts['critic_steps']!=27648:raise RuntimeError('audit coverage')
    numeric={p.relative_to(TARGET).as_posix():c.sha(p) for folder in ('pilot','diagnostics','sensitivity','counterfactual') for p in (TARGET/folder).rglob('*') if p.is_file()}
    c.write_json(TARGET/'qa/numeric-index.json',dict(status='PASSED',hashes=numeric))
    c.write_json(TARGET/'qa/final-qa.json',dict(status='PASSED',**counts,parent_and_failed_attempt_all_SHA_checked=True,
        no_parent_write=True,code_SHA=c.sha(Path(__file__))))
    print('QA PASSED',counts,flush=True)

def visual():
    """Run only AFTER actually displaying every final PNG."""
    from PIL import Image
    checks=[]
    for stem in ('01-m0','02-critic','03-switch','04-counterfactual'):
        p=TARGET/'figures'/(stem+'.png');im=Image.open(p);ET.parse(TARGET/'figures'/(stem+'.svg'))
        if abs(im.info['dpi'][0]-600)>.1:raise RuntimeError('dpi wrong')
        checks.append(dict(stem=stem,pixels=im.size,dpi=im.info['dpi']))
    c.write_json(TARGET/'qa/visual.json',dict(status='PASSED',actual_all_four_latest_PNG_viewed=True,SVG_XML_checked=True,
        checks=checks,hashes={p.relative_to(TARGET).as_posix():c.sha(p) for p in (TARGET/'figures').glob('*')}))

def tests():
    p=TARGET/'qa/pytest-final.xml';suites=list(ET.parse(p).getroot().iter('testsuite'))
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts['tests']<76 or any(counts[k] for k in ('failures','errors','skipped')):raise RuntimeError('final test failure')
    c.write_json(TARGET/'qa/tests.json',dict(status='PASSED',**counts,XML_SHA=c.sha(p),
        scope='diagnosis, Direct, PPO, classical, rolling related tests; not full platform'))

def exports():
    with (TARGET/'m0_comparison.csv').open(encoding='utf-8-sig',newline='') as f:paired=list(csv.DictReader(f))
    with (TARGET/'action_sensitivity.csv').open(encoding='utf-8-sig',newline='') as f:probes=list(csv.DictReader(f))
    with (TARGET/'actor_critic_diagnostics.csv').open(encoding='utf-8-sig',newline='') as f:critic=list(csv.DictReader(f))
    if (len(paired),len(probes),len(critic))!=(216,810,288):raise RuntimeError('CSV coverage wrong')
    if len({(r['condition'],r['training_seed'],r['scenario_seed']) for r in paired})!=216:raise RuntimeError('duplicate paired sample')
    for r in paired:
        name,seed,s=r['condition'],r['training_seed'],r['scenario_seed']
        z=c.read_json(OLD/'m0'/f'{name}-s{s}.json')['ledger']['realized_profit']
        mc=c.read_json(c.PARENT/'mc'/f'{name}-test-s{s}.json')['ledger']['realized_profit']
        direct=c.read_json(c.PARENT/'evaluation'/f'{name}-{seed}-s{s}.json')['ledger']['realized_profit']
        for field,value in [('M0_USD',z),('MC_USD',mc),('Direct_USD',direct),('Direct_minus_M0_USD',direct-z),('MC_minus_M0_USD',mc-z),('Direct_minus_MC_USD',direct-mc)]:
            if not math.isclose(float(r[field]),value,abs_tol=1e-7):raise RuntimeError('paired export mismatch')
    cache={}
    for row in probes:
        stem=row['state_id']
        if stem not in cache:cache[stem]=c.read_json(TARGET/'sensitivity'/(stem+'.json'))
        raw=cache[stem]['rows'][int(row['control_index'])];ref=cache[stem]['rows'][1]
        if (row['changed']=='True')!=(raw['canonical']!=ref['canonical']):raise RuntimeError('canonical switch export mismatch')
        if row['kind']!=raw['kind'] or row['solver_status']!='OPTIMAL':raise RuntimeError('probe identity/status mismatch')
        if not math.isclose(float(row['objective']),raw['probe']['plan']['augmented_solver_objective'],abs_tol=1e-7):raise RuntimeError('probe objective export mismatch')
        for dim,v in enumerate(raw['signal']):
            if not math.isclose(float(row[f'action{dim}']),v,abs_tol=1e-10):raise RuntimeError('probe signal export mismatch')
    for row in critic:
        d=c.read_json(TARGET/'diagnostics'/f'{row["condition"]}-{row["training_seed"]}.json')
        raw=next(r for r in d['critic'] if r['update']==int(row['update']))
        for field in ('critic_MC_EV','critic_MC_RMSE_reward_units','KL','gradient_norm_total_preclip','advantage_std'):
            if not math.isclose(float(row[field]),raw[field],abs_tol=1e-10):raise RuntimeError('critic export mismatch')
    c.write_json(TARGET/'qa/exports.json',dict(status='PASSED',paired=216,probes=810,critic_updates=288,
        primary_CSV_reconciled_to_raw=True,source_SHA=c.sha(Path(__file__))))

def freeze():
    m=c.read_json(TARGET/'manifest.json')
    if m['status']!='NUMERICAL_COMPLETE_QA_PENDING' or m['source_files']!=sources():raise RuntimeError('immutable source/stage')
    for gate in ('prepare','m0','pilot','diagnostics','final-qa','analysis','tests','visual','exports'):
        if c.read_json(TARGET/'qa'/(gate+'.json'))['status']!='PASSED':raise RuntimeError('missing gate '+gate)
    from experiments.cfs_direct_ppo_diagnosis.analysis import sha
    if sha(c.PACKAGE/'analysis.py')!=c.read_json(TARGET/'qa/analysis-freeze.json')['source_SHA']:raise RuntimeError('analysis changed')
    c.verify_parent(True);c.verify_hashes(Path('/'),m['recovered_history']);c.verify_hashes(TARGET,m['inputs'])
    c.verify_hashes(TARGET,c.read_json(TARGET/'qa/numeric-index.json')['hashes']);c.verify_hashes(TARGET,c.read_json(TARGET/'qa/visual.json')['hashes'])
    if c.sha(TARGET/'qa/pytest-final.xml')!=c.read_json(TARGET/'qa/tests.json')['XML_SHA']:raise RuntimeError('final XML changed')
    with (TARGET/'business_metrics.csv').open(encoding='utf-8-sig',newline='') as f:business=list(csv.DictReader(f))
    if len(business)!=360:raise RuntimeError('business export coverage')
    for r in business:
        name,s=r['condition'],r['scenario_seed']
        if r['method']=='M0':path=OLD/'m0'/f'{name}-s{s}.json'
        elif r['method']=='Pure-MC':path=c.PARENT/'mc'/f'{name}-test-s{s}.json'
        else:path=c.PARENT/'evaluation'/f'{name}-{r["training_seed"]}-s{s}.json'
        ledger=c.read_json(path)['ledger']
        for field,k in [('profit_USD','realized_profit'),('revenue_USD','completed_fare'),('cost_USD','realized_cost'),('completed_orders','completed_orders')]:
            if not math.isclose(float(r[field]),ledger[k],abs_tol=1e-7):raise RuntimeError('business raw reconciliation')
    with (TARGET/'counterfactual_results.csv').open(encoding='utf-8-sig',newline='') as f:counter=list(csv.DictReader(f))
    raw_cases=c.read_json(TARGET/'counterfactual/results.json')
    if len(counter)!=len(raw_cases):raise RuntimeError('counterfactual CSV coverage')
    for row,case in zip(counter,raw_cases,strict=True):
        a=c.read_json(TARGET/case['reference']);b=c.read_json(TARGET/case['variant_record'])
        if not math.isclose(float(row['difference_USD']),b['ledger']['realized_profit']-a['ledger']['realized_profit'],abs_tol=1e-7):raise RuntimeError('counterfactual final difference')
    c.write_json(TARGET/'qa/business-counter-export.json',dict(status='PASSED',business_rows=360,counterfactual_rows=len(counter),code_SHA=c.sha(Path(__file__))))
    hashes={p.relative_to(TARGET).as_posix():c.sha(p) for p in TARGET.rglob('*') if p.is_file() and p.relative_to(TARGET).as_posix() not in ('manifest.json','final_manifest.json')}
    m.update(status='COMPLETED',completed_at=c.now(),final_artifact_hashes=hashes,
        nonnumeric_source_files={p.relative_to(c.ROOT).as_posix():c.sha(p) for p in [*PACKAGE.glob('*.py'),c.PACKAGE/'analysis.py',c.ROOT/'tests/test_direct_ppo_diagnosis_recovery.py'] if p.name not in ('__init__.py','run.py')})
    c.write_json(TARGET/'manifest.json',m);c.write_json(TARGET/'final_manifest.json',m);c.verify_hashes(TARGET,hashes)
    print('COMPLETED',len(hashes),'files',flush=True)

PACKAGE=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('qa','visual','tests','exports','freeze'));args=p.parse_args();globals()[args.stage]()
