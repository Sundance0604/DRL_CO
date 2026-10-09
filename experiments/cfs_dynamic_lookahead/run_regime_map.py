import argparse,itertools
from pathlib import Path
from experiment_core.storage import read_json,atomic_json
from .common import DEFAULT_CONFIG,DEFAULT_OUTPUT,prepare,execute_jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--output',default=str(DEFAULT_OUTPUT));args=p.parse_args()
    config=read_json(args.config);output=Path(args.output);baseline,manifest=prepare(config,output)
    conditions=[dict(region=r,orders=N,fleet=K) for r,N,K in itertools.product(config['regions'],config['order_levels'],config['fleet_levels'])]
    execute_jobs(config,output,baseline,manifest,conditions,config['pilot_seeds'],'regime_map',[('myopic',0)])
    # Two-hub sanity, not an additional pressure region.
    sanity=[dict(region='la_sf',orders=40,fleet=2)]
    execute_jobs(config,output,baseline,manifest,sanity,config['pilot_seeds'],'sanity',[('myopic',0)])
    manifest['status']='REGIME_MAP_COMPLETED';atomic_json(output/'manifest.json',manifest)


if __name__=='__main__':main()
