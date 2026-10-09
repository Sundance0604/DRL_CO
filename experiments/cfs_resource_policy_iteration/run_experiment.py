"""Fixed four improvements; formal unseen until all tables/policies frozen."""
import argparse
from experiment_core.storage import read_json,atomic_json
from experiments.cfs_classical_value.build_fluid_value import demand_prior
from .common import OUTPUT,initialize,freeze,datasets,key,now
from .phase0 import run as phase0
from .targets import evaluate
from .policies import create_policy,load_policies,run_matrix,action_diagnostics

def main(stage):
    cfg,base,m=initialize();phase0(cfg)
    if stage=='phase0':return
    freeze(m)
    data=datasets(cfg,base,cfg['construction_seeds'])
    priors={key(c):demand_prior([data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0]
        for seed in cfg['construction_seeds']]) for c in cfg['conditions']}
    path=OUTPUT/'tables/priors.json'
    if path.exists():assert read_json(path)==priors
    else:atomic_json(path,priors)
    create_policy(cfg,m,0)
    for n in range(cfg['iterations']):
        result=evaluate(cfg,base,data,priors,m,n,load_policies(n));create_policy(cfg,m,n+1)
        archived=OUTPUT/'raw/failed_attempts/timing-v1/manifests/iteration_0_manifest.json'
        if n==0 and archived.exists():
            old=read_json(archived)
            assert old['tables']==result['tables'],'Observational timer repair changed V0 numerical tables'
            atomic_json(OUTPUT/'qa/timing_repair_parity.json',dict(status='PASSED',tables=4,
                original_table_hashes=old['tables'],profiled_table_hashes=result['tables'],
                exact_table_bytes_identical=True,archived_attempt='raw/failed_attempts/timing-v1',
                reason='complete separate context/lookup/build/solve/transition and offline aggregate/interpolate/serialize/QA timers',
                numerical_semantics_unchanged=True,formal_seen_before_repair=False))
        m[f'iteration_{n}']=result;m['status']=f'V{n}_FROZEN';atomic_json(OUTPUT/'manifest.json',m)
        print(f'V{n} frozen; pi{n+1} deployed; {result["worlds"]} audited worlds.',flush=True)
    atomic_json(OUTPUT/'manifests/policy_freeze.json',dict(status='FROZEN',created_at=now(),formal_seen=False,
        policies={f'pi{n}':{cid:{k:v for k,v in p.items() if k!='table'} for cid,p in load_policies(n).items()} for n in range(5)},code_hash=m['code_hash']))
    if stage=='construction':return
    for label in ['diagnostic','pilot','formal']:
        if label in m:continue
        stage_data=datasets(cfg,base,cfg[f'{label}_seeds'])
        m[label]=run_matrix(cfg,base,stage_data,priors,m,cfg[f'{label}_seeds'],label)
        m['status']=label.upper()+'_COMPLETED';atomic_json(OUTPUT/'manifest.json',m)
        if label=='diagnostic':action_diagnostics(cfg,base,priors)
    if not (OUTPUT/'processed/policy_action_disagreement.json').exists():action_diagnostics(cfg,base,priors)
    print('Numerical simulations complete; analysis and QA remain.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['phase0','construction','all'],default='all')
    main(p.parse_args().stage)
