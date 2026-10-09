"""Run the preregistered fifth study, with an explicit Phase-1 gate."""
import argparse
from experiment_core.storage import atomic_json,read_json
from .common import *
from .queries import freeze_queries
from .targets import build_targets
from .diagnostics import preregister,analyze_phase1
from .toys import audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--through',choices=['phase0','queries','phase1','all'],default='all')
    args=p.parse_args();cfg,base,m=initialize()
    def save():atomic_json(OUTPUT/'manifest.json',m)
    if 'baseline' not in m:m['baseline']=baseline_regression(cfg);save()
    if 'toy' not in m:m['toy']=audit();save()
    preregister(cfg)
    if args.through=='phase0':return
    freeze_numerical_source(m)
    c=cfg['phase1_condition'];data=datasets(cfg,base,cfg['phase1_seeds'],[c])
    queries=freeze_queries(cfg,base,data,cfg['phase1_seeds'],[c])
    m['phase1_queries']=dict(count=len(queries['queries']),missing=len(queries['missing']),digest=queries['query_digest']);save()
    if args.through=='queries':return
    if 'phase1_targets' not in m:m['phase1_targets']=build_targets(cfg,m,queries);save()
    if 'phase1' not in m:m['phase1']=analyze_phase1(cfg);save()
    print('PHASE 1 GATE:',m['phase1']['gate'],m['phase1']['triggers'],flush=True)
    if args.through=='phase1':return
    if m['phase1']['phase2_required']:
        from .phase2 import run
        run(cfg,base,m)
    else:m['status']='T0_NEGATIVE_FINDING';save()

if __name__=='__main__':main()
