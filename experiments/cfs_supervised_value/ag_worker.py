"""Run only in the user's existing AutoGluon environment; no solver imports."""
import argparse
import contextlib
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time


def write(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['fit','serve']);parser.add_argument('--target',required=True)
    parser.add_argument('--method',choices=['Basic','State']);a=parser.parse_args();root=Path(a.target)
    if os.name=='nt':ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    import numpy as np
    import pandas as pd
    from autogluon.tabular import TabularPredictor
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    try:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(41002)
    except ImportError:pass
    np.random.seed(41002);random.seed(41002)
    manifest=json.loads((root/'dataset_manifest.json').read_text(encoding='utf-8'))
    if a.mode=='fit':
        cols=manifest['feature_columns'][a.method]
        train=pd.read_csv(root/'data/train.csv')[cols+['target']]
        val=pd.read_csv(root/'data/validation.csv')[cols+['target']]
        before=time.perf_counter()
        predictor=TabularPredictor(label='target',problem_type='regression',eval_metric='root_mean_squared_error',
            path=str(root/'models'/a.method),verbosity=2,log_to_file=True).fit(
                train_data=train,tuning_data=val,presets='medium_quality',time_limit=150,
                num_bag_folds=0,num_stack_levels=0,num_cpus=1,num_gpus=0,
                fit_strategy='sequential',dynamic_stacking=False,raise_on_model_failure=False)
        elapsed=time.perf_counter()-before
        if elapsed>180:raise TimeoutError('fit actual wall time exceeded 180 seconds')
        board=predictor.leaderboard(silent=True);board.insert(0,'method',a.method)
        board.to_csv(root/'training'/f'{a.method}_leaderboard.csv',index=False,encoding='utf-8-sig')
        info=predictor.info()
        result=dict(method=a.method,fit_wall_seconds=elapsed,model_best=predictor.model_best,
            autogluon_version=importlib.metadata.version('autogluon.tabular'),python=sys.version,
            training_rows=len(train),validation_rows=len(val),features=cols,
            info=info,models_bytes=sum(p.stat().st_size for p in (root/'models'/a.method).rglob('*') if p.is_file()))
        write(root/'training'/f'{a.method}_fit.json',result)
        # Only this final selected model is used for held-out predictions.
        test=pd.read_csv(root/'data/evaluation.csv');started=time.perf_counter()
        pred=predictor.predict(test[cols]).to_numpy();latency=time.perf_counter()-started
        write(root/'training'/f'{a.method}_evaluation.json',dict(predictions=pred.tolist(),seconds=latency,rows=len(pred)))
        sample=test[cols].iloc[:1];started=time.perf_counter();single=predictor.predict(sample).iloc[0]
        write(root/'training'/f'{a.method}_latency.json',dict(single_row_seconds=time.perf_counter()-started,
            batch_rows=len(test),batch_seconds=latency,single_prediction=float(single)))
        print('FIT_COMPLETE',a.method,elapsed,predictor.model_best,flush=True)
    else:
        with contextlib.redirect_stdout(sys.stderr):
            predictors={k:TabularPredictor.load(str(root/'models'/k),verbosity=0) for k in ['Basic','State']}
            for p in predictors.values():p.persist()
        print(json.dumps({'ready':True}),flush=True)
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('op')=='stop':break
            method=request['method'];cols=manifest['feature_columns'][method]
            frame=pd.DataFrame(request['rows'])[cols];started=time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):values=predictors[method].predict(frame).to_numpy()
            if not np.isfinite(values).all():raise ValueError('nonfinite online predictions')
            print(json.dumps(dict(values=values.tolist(),seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
