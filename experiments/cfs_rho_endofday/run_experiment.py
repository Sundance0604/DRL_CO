"""Paired CFS study using the existing platform/single runner, not a second solver."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile
import numpy as np
import networkx as nx
from experiment_core.plugins import resolve
from experiment_core.runner import execute
from experiment_core.storage import atomic_json, digest, read_json
from model_families.single.data import validate_payload
from model_families.single.plugin import validate_selection
from model_families.single.cfs_adapter import read_pools, make_scenario
from model_families.single.network import eligible_pickups
from model_families.single.engine.mt_prototype import Net

ROOT=Path(__file__).resolve().parents[2]


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def code_hashes():
    files=list((ROOT/'model_families/single').rglob('*.py'))+list((ROOT/'experiment_core').glob('*.py'))
    files+=list((ROOT/'experiments/cfs_rho_endofday').rglob('*.py'))
    files += [ROOT/'tests/test_cfs_rho_endofday.py', ROOT/'pyproject.toml', ROOT/'uv.lock']
    return {str(p.relative_to(ROOT)).replace('\\','/'):file_hash(p) for p in sorted(files)}


def git_head(path):
    return subprocess.check_output(['git','-c','safe.directory='+str(Path(path).resolve()).replace('\\','/'),
                                    '-C',str(path),'rev-parse','HEAD'],text=True).strip()


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        raise ValueError('no experiment rows to save')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def run(config, output, limit_seeds=None, region_filter=None):
    output=Path(output).resolve()
    config=json.loads(json.dumps(config))
    if limit_seeds is not None:
        config['seeds']=config['seeds'][:limit_seeds]
    if region_filter:
        config['regions']=[r for r in config['regions'] if r['id']==region_filter]
    source=Path(config['data_source']).resolve(strict=True)
    raw=output/'raw';raw.mkdir(parents=True,exist_ok=True)
    hashes=code_hashes()
    stat=source.stat()
    identity={'path':str(source),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    previous=output/'manifest.json'
    if previous.exists():
        old=read_json(previous)
        if (old['config'] != config or old['source_code_digest'] != digest(hashes)
                or any(old['source'].get(k) != v for k,v in identity.items())):
            raise RuntimeError('output belongs to another source/config/code; choose a new --output directory')
    # Validate ownership before rewriting even the reproducibility archive/cache.
    with zipfile.ZipFile(raw/'source.snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(ROOT/name,name)
        archive.writestr('study-config.json',json.dumps(config,indent=2))
    source_hash=raw/'source-sha256.json'
    if source_hash.exists() and read_json(source_hash)['metadata']==identity:
        sha=read_json(source_hash)['sha256']
    else:
        print('Hashing local CFS source (read-only)...',flush=True)
        sha=file_hash(source)
        atomic_json(source_hash,{'metadata':identity,'sha256':sha})
    manifest={'schema_version':'cfs-rho-endofday-study/v1','status':'RUNNING','config':config,
              'source':identity|{'sha256':sha},'git_commit':git_head(ROOT),
              'source_code_sha256':hashes,'source_code_digest':digest(hashes),
              'source_snapshot_sha256':file_hash(raw/'source.snapshot.zip'),
              'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in
                  ['gurobipy','numpy','scipy','networkx','matplotlib','pydantic']},
              'reference':{'root':config['reference_root'],'commit':git_head(config['reference_root']),
                  'adapted_files':{p:file_hash(Path(config['reference_root'])/'intercity_delivery/data'/p)
                                   for p in ['cfs_catalog.py','cfs_processor.py','sqlite_store.py','loader.py']}},
              'policy_seed':config['policy_seed'],'training_seed':None,
              'output':str(output),'coordinate_note':config['coordinate_note'],
              'scope':'only physical network/rho and realized cooldown accounting',
              'cutoff_convention':'decisions t=0..T-1; events and fare at time T are processed in cooldown; snapshots use [0,cutoff)',
              'rows':0}
    if previous.exists():
        old=read_json(previous)
        if any(old[k]!=manifest[k] for k in ['config','source','source_code_digest']):
            raise RuntimeError('output belongs to another source/config/code; choose a new --output directory')
    atomic_json(previous,manifest)
    allrows=[];details=[];start=time.monotonic()
    for region in config['regions']:
        print('Loading supported CFS OD strata:',region['id'],flush=True)
        pools,counts,metadata=read_pools(source,region['coordinates'],maximum_weight=config['capacity']*config['pounds_per_unit'])
        manifest.setdefault('data_regions',{})[region['id']]={'directed_counts':counts,'sqlite_metadata':metadata,
                                                          'supported_pairs':list(pools)}
        for seed in config['seeds']:
            scenario,provenance=make_scenario(region,pools,seed,config)
            data={'schema_version':'dataset/v1','family':'single_level_matching','scenarios':[scenario],
                  'provenance':{'source':manifest['source'],'observed_generated_ref':scenario['id']+'.source.json'}}
            validate_payload(data)
            scenario_hash=digest(scenario)
            atomic_json(raw/(scenario['id']+'.source.json'),provenance)
            atomic_json(raw/(scenario['id']+'.dataset.json'),data)
            g=nx.Graph();g.add_weighted_edges_from(scenario['edges'])
            net=Net(g,seed,config['speed'])
            distances=sorted({d for (a,b),d in net.d.items() if a!=b})
            levels=sorted({0.0}|{float(np.quantile(distances,q,method='nearest')) for q in config['rho_quantiles']})
            variants=[('rho-'+str(i),'network_radius',rho) for i,rho in enumerate(levels)]
            variants.append(('adjacency','legacy_adjacency',0))
            for label,rule,rho in variants:
                rid=f'{region["id"]}-{seed}-{label}'
                target=raw/'runs'/rid;target.mkdir(parents=True,exist_ok=True)
                spec=resolve({'name':rid,'family':'single','framework_id':'myopic',
                    'dataset':{'dataset_id':region['id'],'revision':digest(data),'scenario_ids':[scenario['id']]},
                    'model':{'id':'single_level_matching','parameters':{k:config[k] for k in
                        ['speed','capacity','loaded_cost','empty_cost','waiting_cost','expiry_cost','max_delay']}|
                        {'cross_hub':True,'reposition':False,'pickup_radius':rho,'pickup_rule':rule}},
                    'solver':{'backend':'gurobi','parameters':config['solver']},
                    'evaluation':{'accounting_version':'single-realized-v1','terminal_policy':'cooldown_realized'},
                    'execution':{'timeout_seconds':600,'policy_seeds':[config['policy_seed']],'save_trace':True}})
                validate_selection(spec,data,[scenario])
                atomic_json(target/'spec.resolved.json',spec.model_dump())
                atomic_json(target/'dataset.snapshot.json',data)
                atomic_json(target/'manifest.json',{'run_id':rid,'batch_id':'cfs-rho-endofday','policy_seed':config['policy_seed'],
                    'code':{'content_hash':manifest['source_code_digest']},'condition_id':digest(spec.model_dump()['model'])})
                if (target/'state.json').exists() and read_json(target/'state.json').get('status')=='COMPLETED':
                    results=read_json(target/'metrics.json')['rows']
                else:
                    try:
                        results=execute(spec,data,[scenario],target,config['policy_seed'],lambda *a,**kw:None,lambda:False)
                        atomic_json(target/'state.json',{'status':'COMPLETED'})
                    except Exception as exc:
                        atomic_json(target/'state.json',{'status':'FAILED','error':str(exc)})
                        manifest.update(status='FAILED',failure_run=rid,error=str(exc))
                        atomic_json(previous,manifest)
                        raise
                metrics=results[0]['metrics']
                if metrics['unfinished_committed'] or metrics['pending_committed']:
                    raise RuntimeError('formal run did not complete all commitments')
                pickup=nx.Graph();pickup.add_nodes_from(g)
                sizes=[]
                for hub in net.H:
                    eligible=eligible_pickups(net,hub,True,rho,rule)
                    sizes.append(len(eligible)-1)
                    pickup.add_edges_from((hub,v) for v in eligible if v!=hub)
                components=list(nx.connected_components(pickup))
                scalar={k:v for k,v in metrics.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
                row=dict(run_id=rid,region=region['id'],seed=seed,policy_seed=config['policy_seed'],training_seed='',
                         variant=label,pickup_rule=rule,rho=rho,scenario_hash=scenario_hash,
                         mean_eligible_hubs=float(np.mean(sizes)),pickup_components=len(components),
                         largest_pickup_component=max(map(len,components)),
                         cross_pickup_rate=metrics['cross']/metrics['assigned'] if metrics['assigned'] else 0,
                         service_rate=metrics['completed_orders']/len(scenario['orders']),
                         cancellation_rate=metrics['cancelled_orders']/len(scenario['orders']),**scalar)
                cut=metrics['ledger_at_T']
                row.update(assigned_undelivered_at_T=cut['unfinished_committed'],credited_unrealized_fare_at_T=cut['credited_unrealized_fare'],
                           realized_profit_at_T=cut['realized_profit'],booked_minus_realized=metrics['booked_profit']-metrics['realized_profit'],
                           relative_bias=(metrics['booked_profit']-metrics['realized_profit'])/abs(metrics['realized_profit']) if abs(metrics['realized_profit'])>1e-9 else '')
                allrows.append(row)
                details.append(dict(run_id=rid,region=region['id'],seed=seed,variant=label,rho=rho,
                                    cooldown_scan=metrics['cooldown_scan'],ledger_at_T=cut))
                atomic_json(target/'decision-ledger.json',[dict(period=r['period'],**r['plan']['accounting'],
                    stage_objective=r['plan']['augmented_solver_objective']) for r in read_json(target/'trace.json')[0]['steps'] if not r.get('drain')])
                atomic_json(target/'realized-ledger.json',metrics)
                print(rid, 'completed', 'realized=',round(metrics['realized_profit'],2),flush=True)
            write_csv(output/'processed'/'runs.csv',allrows)
            atomic_json(output/'processed'/'details.json',details)
            manifest['rows']=len(allrows);atomic_json(previous,manifest)
    sampled=[r for p in raw.glob('*.source.json') for r in read_json(p)['observed_shipments']]
    manifest.update(status='COMPLETED',rows=len(allrows),elapsed_seconds=time.monotonic()-start,
                    scenario_count=len(config['regions'])*len(config['seeds']),
                    sample_size_per_condition=len(config['seeds']),
                    sampled_record_draws=len(sampled),unique_sampled_shipments=len({r['SHIPMT_ID'] for r in sampled}))
    atomic_json(previous,manifest)
    print('STUDY COMPLETED:',len(allrows),'runs; output:',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'experiments/cfs_rho_endofday/configs/main.json'))
    parser.add_argument('--output',default=str(ROOT/'results/cfs_rho_endofday'))
    parser.add_argument('--limit-seeds',type=int)
    parser.add_argument('--region')
    args=parser.parse_args()
    run(read_json(args.config),args.output,args.limit_seeds,args.region)
