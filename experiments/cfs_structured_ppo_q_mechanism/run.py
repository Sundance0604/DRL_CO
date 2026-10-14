import argparse
from .common import *


def main():
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','kl','select','probe','all'));a=p.parse_args()
    try:
        if a.stage=='prepare':prepare();return
        from .kl import run as kl
        from .states import select,probes
        functions=dict(kl=kl,select=select,probe=probes)
        if a.stage=='all':kl();select();probes()
        else:functions[a.stage]()
        if a.stage in ('all','probe'):
            m=load();m['status']='NUMERICAL_COMPLETE_QA_PENDING';write_json(TARGET/'manifest.json',m)
    except Exception as e:
        if (TARGET/'manifest.json').exists():
            write_json(TARGET/'qa/stop.json',dict(stage=a.stage,error=type(e).__name__,message=str(e),at=now()))
            m=read_json(TARGET/'manifest.json');m['status']='BUDGET_LIMITED' if isinstance(e,TimeoutError) else 'FAILED';write_json(TARGET/'manifest.json',m)
        raise


if __name__=='__main__':main()
