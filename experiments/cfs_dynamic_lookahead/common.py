"""Frozen CFS input, deterministic placement, resumable isolated run protocol."""
import copy
import itertools
import json
import importlib.metadata
import platform
import time
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from experiment_core.storage import atomic_json,read_json,digest
from experiment_core.plugins import resolve
from model_families.single.data import validate_payload
from model_families.single.plugin import validate_selection
from model_families.single.runner import execute
from model_families.single.cfs_adapter import read_pools,make_scenario
from model_families.single.engine.mt_prototype import Net
from experiments.cfs_rho_endofday.run_experiment import ROOT,file_hash,git_head,write_csv,code_hashes

DEFAULT_OUTPUT=ROOT/'results/cfs_dynamic_lookahead'
DEFAULT_CONFIG=ROOT/'experiments/cfs_dynamic_lookahead/configs/main.json'


def fingerprint():
    hashes=code_hashes()
    for p in (ROOT/'experiments/cfs_dynamic_lookahead').rglob('*.py'):
        if p.name in {'analysis.py','case_study.py'}:continue  # postprocessing has a separate reproducibility hash
        hashes[p.relative_to(ROOT).as_posix()]=file_hash(p)
    hashes['tests/test_dynamic_lookahead.py']=file_hash(ROOT/'tests/test_dynamic_lookahead.py')
    return hashes


def prepare(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    baseline=read_json(ROOT/config['baseline_config'])
    source=Path(baseline['data_source']).resolve(strict=True);stat=source.stat()
    metadata=dict(path=str(source),size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    hashes=fingerprint();identity=dict(config_hash=digest(config),source_code_digest=digest(hashes),source_metadata=metadata)
    path=output/'manifest.json'
    if path.exists():
        manifest=read_json(path)
        if any(manifest[k]!=v for k,v in identity.items()):
            raise RuntimeError('source/config/code differs: use a new output, never overwrite prior results')
        return baseline,manifest
    phase0=read_json(output/'phase0/regression.json')
    if phase0['status']!='PASSED':raise RuntimeError('Phase 0 reproduction required')
    print('Verifying CFS source SHA256 (read only)...',flush=True)
    sha=file_hash(source)
    historical=read_json(ROOT/'results/cfs_rho_endofday/manifest.json')
    if sha!=historical['source']['sha256']:raise RuntimeError('CFS source changed since baseline')
    raw=output/'raw';raw.mkdir(parents=True,exist_ok=True)
    snapshot=raw/'source.snapshot.zip'
    with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:z.write(ROOT/name,name)
        z.writestr('dynamic-config.json',json.dumps(config,indent=2))
        z.writestr('baseline-config.json',json.dumps(baseline,indent=2))
    manifest=dict(schema_version='cfs-dynamic-lookahead/v1',status='INITIALIZED',**identity,
                  config=config,baseline_config=baseline,baseline_config_hash=digest(baseline),source_code_sha256=hashes,
                  source=metadata|{'sha256':sha},git_commit=git_head(ROOT),dirty_worktree=True,
                  source_snapshot_sha256=file_hash(snapshot),python=platform.python_version(),
                  packages={p:importlib.metadata.version(p) for p in ['gurobipy','numpy','scipy','networkx','matplotlib','pydantic']},
                  baseline_phase0=phase0['source_code_digest'],baseline_reproduction_passed=True,
                  accounting_version='single-realized-v1',terminal_policy='cooldown_realized',training_seed=None,
                  forbidden_components_modified=False,protected_hashes={name:file_hash(ROOT/name) for name in
                    ['model_families/single/network.py','model_families/single/accounting.py','model_families/single/cfs_adapter.py']})
    atomic_json(path,manifest)
    return baseline,manifest


def rho_levels(scenario,baseline):
    import networkx as nx
    g=nx.Graph();g.add_weighted_edges_from(scenario['edges']);net=Net(g,0,baseline['speed'])
    distances=sorted({d for (a,b),d in net.d.items() if a!=b})
    formal=sorted({float(np.quantile(distances,q,method='nearest')) for q in baseline['rho_quantiles']})
    return dict(rho0=0.,rho_mid=formal[(len(formal)-1)//2],rho_max=formal[-1])


def placement(hubs,mass,total):
    ranked=sorted(hubs,key=lambda h:(-mass[h],h));hubs=sorted(hubs)
    if total<len(hubs):return ranked[:total]
    return hubs+[ranked[i%len(ranked)] for i in range(total-len(hubs))]


def freeze_data(config,baseline,output,conditions,seeds):
    """Same N/seed shipments and times across fleets/policies; only fleet changes."""
    raw=Path(output)/'raw/datasets';raw.mkdir(parents=True,exist_ok=True)
    data={};catalog={}
    needed={c['region'] for c in conditions}
    for region in baseline['regions']:
        if region['id'] not in needed:continue
        print('Reading CFS pools:',region['id'],flush=True)
        pools,counts,metadata=read_pools(baseline['data_source'],region['coordinates'])
        mass={h:0. for h in region['coordinates']}
        for directions in pools.values():
            for rows in directions:
                for r in rows:mass[r['ORIG_CFS_AREA']]+=float(r['WGT_FACTOR'])
        catalog[region['id']]=dict(outbound_weighted_shipment_mass=mass,directed_counts=counts,sqlite_metadata=metadata)
        for N,seed in itertools.product(sorted({c['orders'] for c in conditions if c['region']==region['id']}),seeds):
            cfg=copy.deepcopy(baseline);cfg['orders']=N
            s,provenance=make_scenario(region,pools,seed,cfg)
            for K in sorted({c['fleet'] for c in conditions if c['region']==region['id'] and c['orders']==N}):
                scenario=copy.deepcopy(s);sid=f'{region["id"]}-n{N}-k{K}-s{seed}';scenario['id']=sid
                scenario['hubs']=placement(s['nodes'],mass,K)
                provenance_here=copy.deepcopy(provenance)
                provenance_here.update(fleet_seed=None,placement_rule=config['placement_rule'],fleet=scenario['hubs'],outbound_mass=mass)
                payload=dict(schema_version='dataset/v1',family='single_level_matching',scenarios=[scenario],
                             provenance={'observed_generated_ref':sid+'.source.json','source_sha256':read_json(Path(output)/'manifest.json')['source']['sha256']})
                validate_payload(payload)
                path=raw/(sid+'.dataset.json')
                if path.exists() and digest(read_json(path))!=digest(payload):raise RuntimeError('frozen input mismatch')
                atomic_json(path,payload);atomic_json(raw/(sid+'.source.json'),provenance_here)
                data[(region['id'],N,K,seed)]=payload
        del pools
    catalog_path=Path(output)/'raw/placement-catalog.json'
    previous=read_json(catalog_path) if catalog_path.exists() else {}
    atomic_json(catalog_path,previous|catalog)
    return data


def run_job(job):
    output=Path(job['output']);target=output/'raw/runs'/job['run_id'];target.mkdir(parents=True,exist_ok=True)
    cfg=job['baseline'];H=job['H'];policy=job['policy'];scenario=job['data']['scenarios'][0]
    spec=resolve(dict(name=job['run_id'],family='single',framework_id='myopic' if policy=='myopic' else 'rolling' if policy=='known_supply' else 'oracle_preview',
        dataset=dict(dataset_id=scenario['id'],revision=digest(job['data']),scenario_ids=[scenario['id']]),
        model=dict(id='single_level_matching',parameters={k:cfg[k] for k in ['speed','capacity','loaded_cost','empty_cost','waiting_cost','expiry_cost','max_delay']}|
                   dict(cross_hub=True,reposition=False,pickup_radius=job['rho'],pickup_rule='network_radius')),
        controller=dict(id='myopic' if policy=='myopic' else 'rolling_horizon' if policy=='known_supply' else 'oracle_preview',
                        parameters={} if policy=='myopic' else dict(planning_horizon=H,commit_periods=1,completion_extension=0)),
        value_function=dict(id='zero'),solver=dict(backend='gurobi',parameters=cfg['solver']),
        evaluation=dict(accounting_version='single-realized-v1',terminal_policy='cooldown_realized',information_set='oracle' if policy=='oracle_preview' else 'online'),
        execution=dict(timeout_seconds=600,policy_seeds=[0],save_trace=True)))
    validate_selection(spec,job['data'],[scenario])
    identity=dict(data_hash=digest(job['data']),scenario_hash=digest(scenario),spec_hash=digest(spec.model_dump()),code_hash=job['code_hash'])
    base={k:job[k] for k in ['run_id','stage','region','orders','fleet','seed','rho_label','rho','policy','H']}
    base.update(identity,orders_per_vehicle=job['orders']/job['fleet'],requested_load_per_fleet_capacity=sum(o['passenger'] for o in scenario['orders'])/(job['fleet']*cfg['capacity']),
                info_mode=policy,policy_seed=0,training_seed=None,accounting_version='single-realized-v1',solver_version=importlib.metadata.version('gurobipy'))
    if (target/'manifest.json').exists():
        prior=read_json(target/'manifest.json')
        if any(prior[k]!=v for k,v in identity.items()):raise RuntimeError('existing run identity mismatch')
    else:
        atomic_json(target/'manifest.json',base|{'dataset_ref':'../../datasets/'+scenario['id']+'.dataset.json','source_snapshot_ref':'../../source.snapshot.zip'})
        atomic_json(target/'spec.resolved.json',spec.model_dump())
    start=time.perf_counter()
    try:
        if (target/'state.json').exists() and read_json(target/'state.json')['status']=='COMPLETED':
            m=read_json(target/'metrics.json')['rows'][0]['metrics'];runtime=read_json(target/'state.json')['runtime_seconds']
        else:
            m=execute(spec,job['data'],[scenario],target,0,lambda *a,**kw:None,lambda:False)[0]['metrics']
            runtime=time.perf_counter()-start
            if m['unfinished_committed'] or m['assigned']!=m['delivered']:raise RuntimeError('unfinished commitment')
            steps=read_json(target/'trace.json')[0]['steps']
            if not all(all(s['invariants'].values()) for s in steps):raise RuntimeError('trace invariant')
            atomic_json(target/'decision-ledger.json',[dict(period=s['period'],accounting=s['plan']['accounting']) for s in steps if not s.get('drain')])
            atomic_json(target/'realized-ledger.json',dict(final={k:v for k,v in m.items() if k!='execution_events'},execution_events=m['execution_events']))
            atomic_json(target/'state.json',dict(status='COMPLETED',runtime_seconds=runtime))
        windows=m.get('planning_windows',[]);times=[w['runtime'] for w in windows]
        base.update({k:v for k,v in m.items() if isinstance(v,(int,float)) and not isinstance(v,bool)})
        base.update(status='COMPLETED',runtime_seconds=runtime,service_rate=m['delivered']/job['orders'],
                    cancellation_rate=m['cancelled']/job['orders'],cross_hub_pickup_rate=m['cross']/m['assigned'] if m['assigned'] else 0,
                    average_occupancy=m['load_factor']*cfg['capacity'],RH_solve_count=len(windows),
                    mean_RH_solve_time=float(np.mean(times)) if times else 0,max_RH_solve_time=max(times,default=0),
                    p95_RH_solve_time=float(np.quantile(times,.95)) if times else 0,
                    max_window_gap=max((w['gap'] or 0 for w in windows),default=0),
                    limited_window_count=sum(w['status']!='OPTIMAL' for w in windows),
                    max_variable_count=max((w['variable_count'] for w in windows),default=0),
                    max_constraint_count=max((w['constraint_count'] for w in windows),default=0),
                    deferral_success_rate=m['deferral_success_rate'])
    except Exception as exc:
        runtime=time.perf_counter()-start
        base.update(status='FAILED',runtime_seconds=runtime,error=str(exc),error_code=getattr(exc,'code',None))
        atomic_json(target/'state.json',base)
    atomic_json(target/'summary.json',base)
    return base


def execute_jobs(config,output,baseline,manifest,conditions,seeds,stage,policies):
    data=freeze_data(config,baseline,output,conditions,seeds);jobs=[]
    for c,seed in itertools.product(conditions,seeds):
        payload=data[(c['region'],c['orders'],c['fleet'],seed)]
        levels=rho_levels(payload['scenarios'][0],baseline)
        for label,(policy,H) in itertools.product(config['formal_rho_labels'],policies):
            rid=f'{stage}-{c["region"]}-n{c["orders"]}-k{c["fleet"]}-s{seed}-{label}-{policy}-h{H}'
            jobs.append(dict(**c,seed=seed,rho_label=label,rho=levels[label],policy=policy,H=H,run_id=rid,stage=stage,
                             data=payload,baseline=baseline,output=str(output),code_hash=manifest['source_code_digest']))
    rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=config['max_parallel']) as pool:
        futures=[pool.submit(run_job,j) for j in jobs]
        for f in as_completed(futures):
            rows.append(f.result())
            if len(rows)%50==0 or len(rows)==len(jobs):
                print(f'{stage}: {len(rows)}/{len(jobs)}, failures={sum(r["status"]!="COMPLETED" for r in rows)}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
    rows.sort(key=lambda r:r['run_id'])
    write_csv(Path(output)/(stage+'_results.csv'),rows)
    manifest[stage]=dict(run_count=len(rows),failed_count=sum(r['status']!='COMPLETED' for r in rows),elapsed_seconds=time.perf_counter()-start,seeds=seeds)
    atomic_json(Path(output)/'manifest.json',manifest)
    if any(r['status']!='COMPLETED' for r in rows):raise RuntimeError('failed runs retained; resolve before inference')
    return rows
