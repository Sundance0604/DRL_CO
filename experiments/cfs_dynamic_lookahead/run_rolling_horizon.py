import argparse,csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiment_core.storage import read_json,atomic_json
from .common import DEFAULT_CONFIG,DEFAULT_OUTPUT,prepare,execute_jobs


def representatives(output):
    groups=defaultdict(dict)
    with (Path(output)/'regime_map_results.csv').open(encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['status']!='COMPLETED':raise RuntimeError('pilot contains failed run')
            groups[(r['region'],int(r['orders']),int(r['fleet']))].setdefault(r['rho_label'],{})[int(r['seed'])]=float(r['realized_profit'])
    rows=[]
    for (region,N,K),g in groups.items():
        if set(g['rho0'])!=set(g['rho_max']):raise RuntimeError('pilot not paired')
        delta=[g['rho_max'][s]-g['rho0'][s] for s in sorted(g['rho0'])]
        rows.append(dict(region=region,orders=N,fleet=K,pilot_gain=float(np.mean(delta)),roles=[]))
    chosen=[('positive',max(rows,key=lambda r:r['pilot_gain'])),
            ('failure',min((r for r in rows if r['region']=='western_multicluster'),key=lambda r:r['pilot_gain'])),
            ('tight',min((r for r in rows if r['orders']==80 and r['fleet']==3),key=lambda r:abs(r['pilot_gain']))),
            ('loose',min((r for r in rows if r['orders']==20 and r['fleet']==8),key=lambda r:abs(r['pilot_gain'])))]
    unique={}
    for role,r in chosen:
        key=(r['region'],r['orders'],r['fleet']);unique.setdefault(key,r)['roles'].append(role)
    selected=list(unique.values());atomic_json(Path(output)/'representative_regimes.json',selected)
    return [dict(region=r['region'],orders=r['orders'],fleet=r['fleet']) for r in selected]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--output',default=str(DEFAULT_OUTPUT));p.add_argument('--pilot',action='store_true');args=p.parse_args()
    config=read_json(args.config);output=Path(args.output);baseline,manifest=prepare(config,output);conditions=representatives(output)
    policies=[('myopic',0)]+[(policy,H) for policy in ['known_supply','oracle_preview'] for H in config['horizons']]
    seeds=config['pilot_seeds'] if args.pilot else config['formal_seeds'];stage='rolling_pilot' if args.pilot else 'rolling'
    execute_jobs(config,output,baseline,manifest,conditions,seeds,stage,policies)
    if not args.pilot:
        execute_jobs(config,output,baseline,manifest,conditions,config['full_preview_seeds'],'full_preview',[('oracle_preview',baseline['horizon'])])
        # H1 direct regression on the same held-out scenarios.
        execute_jobs(config,output,baseline,manifest,conditions,config['full_preview_seeds'],'H1_diagnostic',[('known_supply',1)])
        manifest['status']='DYNAMIC_RUNS_COMPLETED';atomic_json(output/'manifest.json',manifest)


if __name__=='__main__':main()
