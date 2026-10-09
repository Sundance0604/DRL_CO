"""Strict phased, resumable fourth study; historical artifacts never written."""
import argparse,time
from experiment_core.storage import atomic_json,read_json
from .common import *
from .targets import build_targets
from .fit_models import fit_all
from .run_policies import run_ladder

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--through',choices=['inputs','targets','fit','heldout','pilot','formal','all'],default='all')
    args=parser.parse_args();cfg,base,m=initialize()
    def save():atomic_json(OUTPUT/'manifest.json',m)
    data=datasets(cfg,base,cfg['construction_seeds']+cfg['validation_seeds']+cfg['formal_seeds'])
    if 'priors' not in m:
        priors={}
        for c in cfg['conditions']:
            scenarios=[data[c['region'],c['orders'],c['fleet'],seed]['scenarios'][0] for seed in cfg['construction_seeds']]
            rate={h:sum(o['departure']==h for s in scenarios for o in s['orders'])/(len(scenarios)*base['horizon']) for h in scenarios[0]['nodes']}
            priors[key(c)]={'rate':rate,'source':'construction-only booking counts','seeds':cfg['construction_seeds']}
        atomic_json(OUTPUT/'models/priors.json',priors);m['priors']=file_hash(OUTPUT/'models/priors.json');save()
    else:priors=read_json(OUTPUT/'models/priors.json')
    if args.through=='inputs':return
    for split in ['construction','validation']:
        if split not in m:m[split]=build_targets(cfg,base,data,priors,split,cfg[split+'_seeds']);save()
    if args.through=='targets':return
    if 'fit' not in m:m['fit']=fit_all(cfg,m);save()
    if args.through=='fit':return
    if 'heldout' not in m:
        m['heldout']=build_targets(cfg,base,data,priors,'heldout',cfg['heldout_value_seeds']);save()
    # Held-out diagnosis happens before any policy run and cannot update selected models.
    from .analysis import value_diagnostics
    if 'heldout_diagnostics' not in m:m['heldout_diagnostics']=value_diagnostics(cfg);save()
    if args.through=='heldout':return
    if 'pilot' not in m:m['pilot']=run_ladder(cfg,base,data,priors,m,cfg['validation_seeds'][:2],'pilot');save()
    if args.through=='pilot':return
    if 'formal' not in m:m['formal']=run_ladder(cfg,base,data,priors,m,cfg['formal_seeds'],'formal');save()
    if args.through=='formal':return
    if 'ablation' not in m:m['ablation']=run_ladder(cfg,base,data,priors,m,cfg['formal_seeds'][:20],'ablation',True);save()
    if 'sensitivity' not in m:m['sensitivity']=build_targets(cfg,base,data,priors,'sensitivity',cfg['secondary_seeds'],'RH4');save()
    m['status']='NUMERICAL_RUNS_COMPLETED';save()

if __name__=='__main__':main()
