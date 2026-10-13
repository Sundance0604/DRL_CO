"""Pure-numpy path adapter; original scientific rules unchanged."""
import argparse
from pathlib import Path
from experiments.cfs_direct_ppo_diagnosis import analysis as a

OLD=a.TARGET
TARGET=a.ROOT/'reports/cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v2'
original_read=a.read
def read(path):
    if path.parent==TARGET/'m0':path=OLD/'m0'/path.name
    return original_read(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('data','figures'));args=p.parse_args()
    a.TARGET=TARGET;a.read=read
    getattr(a,args.stage)()
