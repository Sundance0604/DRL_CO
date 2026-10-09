"""New split, two within-split future paths, identical OLD/ACTION inputs."""
import zipfile
import numpy as np
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from experiments.cfs_classical_value.build_mc_targets import future_orders
from .common import *
from .queries import freeze_queries
from .phase2_targets import build_targets
from .mc_slots import build_targets as build_mc
from .fitting import fit_all,matrices,predict,prediction_scores,collapse
from .features import design
from .policies import run_ladder
from .protocol import read_compressed

def freeze_source(m):
    files=source_hashes()
    if 'phase2_code_hash' in m:
        assert all(file_hash(ROOT/n)==sha for n,sha in m['phase2_source_hashes'].items())
        return
    m['phase2_code_hash']=digest(files);m['phase2_source_hashes']=files
    path=OUTPUT/'raw/phase2-source.snapshot.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(ROOT/name,name)
        z.write(CONFIG,'fifth-config.json')
    m['phase2_snapshot_sha256']=file_hash(path)
    atomic_json(OUTPUT/'manifests/phase2_source_manifest.json',dict(code_hash=m['phase2_code_hash'],files=files,created_at=now()))
    atomic_json(OUTPUT/'manifest.json',m)

def replicate_queries(queries,data,seeds,cfg,label):
    path=OUTPUT/'manifests'/f'{label}_replication_manifest.json'
    if path.exists():return read_json(path)
    scenarios={key(c)+'-'+str(seed):data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0] for c in cfg['conditions'] for seed in seeds}
    ordercache={};rows=[]
    for q in queries['queries']:
        cid=q['condition'];seed=q['scenario_seed'];i=seeds.index(seed)
        original=scenarios[f'{cid}-{seed}']
        packet=read_compressed(OUTPUT/f'raw/query_states/{cid}-{seed}.json.gz')
        for rep,donor in enumerate([seed,seeds[(i+cfg['mc_donor_offset'])%len(seeds)]]):
            d=scenarios[f'{cid}-{donor}']
            if f'{cid}-{seed}' not in ordercache:ordercache[f'{cid}-{seed}']=matching_state(original,packet['parameters'],packet['solver'])[1]
            if f'{cid}-{donor}' not in ordercache:ordercache[f'{cid}-{donor}']=matching_state(d,packet['parameters'],packet['solver'])[1]
            future=future_orders(ordercache[f'{cid}-{seed}'],ordercache[f'{cid}-{donor}'],q['time'],rep)
            rows.append(q|dict(parent_query_id=q['query_id'],query_id=digest((q['query_id'],rep,donor))[:28],
                mc_replication=rep,donor_seed=donor,future_hash=digest([vars(future[k]) for k in sorted(future)])))
    result=dict(status='FROZEN_BEFORE_TARGETS',created_at=now(),label=label,queries=rows,donor_scenarios=scenarios,
        candidate_selection_digest=queries['query_digest'],query_digest=digest(rows),mc_paths=2,
        future_donors_selected_by='own scenario + fixed within-split offset7; no formal outcomes; no candidate reselection')
    atomic_json(path,result);return result

def heldout(cfg,m,priors):
    raw=read_json(OUTPUT/'targets/heldout.json');predictions=[];stats=[]
    for c in cfg['conditions']:
        cid=key(c);table=read_json(OUTPUT/'models/table'/f'{cid}.json')
        rows=matrices([r for r in raw if r['condition']==cid],table,priors[cid],c['fleet'])
        tablep=np.array([r['base_slot_value'] for r in rows])
        methods=[('Zero',np.zeros(len(rows))),('MC Table',tablep)]
        for method in ['linear','chebyshev','spline','bayesian']:
            for fs in ['OLD','ACTION']:
                model=read_json(OUTPUT/'models'/f'{method}_{fs}'/f'{cid}.json')
                values=predict(rows,model)
                values=np.where([r['terminal_release'] for r in rows],0.,values)
                methods.append((method+'-'+fs,values))
        for label,values in methods:
            for scope in ['all','nonterminal','terminal']:
                idx=[i for i,r in enumerate(rows) if scope=='all' or (r['terminal_release'] if scope=='terminal' else not r['terminal_release'])]
                if not idx:continue
                metrics=prediction_scores([rows[i] for i in idx],values[idx])
                stats.append(dict(condition=cid,method=label,scope=scope,**metrics))
            predictions.extend(r|dict(condition=cid,method=label) for r in collapse(rows,values))
    write_csv(OUTPUT/'processed/heldout_value_metrics.csv',stats)
    write_csv(OUTPUT/'processed/heldout_value_predictions.csv',predictions)
    atomic_json(OUTPUT/'processed/heldout_value_metrics.json',stats)
    atomic_json(OUTPUT/'targets/heldout_prediction_rows.json',predictions)
    return dict(independent_scenarios_per_condition=12,metrics=stats,model_selection_updated=False)

def run(cfg,base,m,through='formal'):
    assert m['phase1']['phase2_required'];freeze_source(m)
    def save():atomic_json(OUTPUT/'manifest.json',m)
    data=datasets(cfg,base,cfg['construction_seeds']+cfg['validation_seeds']+cfg['formal_seeds'],cfg['conditions'])
    if not (OUTPUT/'models/priors.json').exists():
        priors={}
        for c in cfg['conditions']:
            scenarios=[data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0] for seed in cfg['construction_seeds']]
            priors[key(c)]=dict(rate={h:sum(o['departure']==h for s in scenarios for o in s['orders'])/(40*base['horizon']) for h in scenarios[0]['nodes']},
                source='fifth construction-only',seeds=cfg['construction_seeds'])
        atomic_json(OUTPUT/'models/priors.json',priors)
    priors=read_json(OUTPUT/'models/priors.json')
    # All construction/validation queries freeze before their first outcome.
    prepared={}
    for split in ['construction','validation']:
        qs=freeze_queries(cfg,base,data,cfg[split+'_seeds'],cfg['conditions'],split)
        prepared[split]=replicate_queries(qs,data,cfg[split+'_seeds'],cfg,split)
    for split in ['construction','validation']:
        if split+'_mc' not in m:m[split+'_mc']=build_mc(cfg,base,data,priors,split+'_mc',cfg[split+'_seeds']);save()
        if split not in m:m[split]=build_targets(cfg,m,prepared[split],split);save()
    if through=='targets':return
    if 'fit' not in m:m['fit']=fit_all(cfg,m,priors);save()
    if through=='fit':return
    if 'heldout' not in m:
        qs=freeze_queries(cfg,base,data,cfg['heldout_value_seeds'],cfg['conditions'],'heldout')
        replicated=replicate_queries(qs,data,cfg['heldout_value_seeds'],cfg,'heldout')
        m['heldout']=build_targets(cfg,m,replicated,'heldout');save()
    if 'heldout_diagnostics' not in m:m['heldout_diagnostics']=heldout(cfg,m,priors);save()
    if through=='heldout':return
    if 'pilot' not in m:m['pilot']=run_ladder(cfg,base,data,priors,m,cfg['validation_seeds'][:2],'pilot');save()
    if through=='pilot':return
    if 'formal' not in m:m['formal']=run_ladder(cfg,base,data,priors,m,cfg['formal_seeds'],'formal');save()
    m['status']='NUMERICAL_RUNS_COMPLETED';save()
