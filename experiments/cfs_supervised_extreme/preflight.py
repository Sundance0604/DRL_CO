"""Read-only asset checks and isolated GPU/model checks; no AutoGluon fit."""
import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/cfs_supervised_value/autogluon_extreme_gpu_v1'
PARENT = ROOT/'reports/cfs_supervised_value/autogluon_pilot_v3'


def save(name, value):
    p = OUT/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def medium():
    import numpy as np
    import pandas as pd
    import torch
    from autogluon.tabular import TabularPredictor
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(1)
    cols = ['release_time','hub','q']
    test = pd.read_csv(PARENT/'data/evaluation.csv')
    t = time.perf_counter()
    p = TabularPredictor.load(str(PARENT/'models/Basic'), verbosity=0)
    pred = p.predict(test[cols]).to_numpy()
    frozen = np.array(json.loads((PARENT/'training/Basic_evaluation.json').read_text())['predictions'])
    delta = float(np.max(np.abs(pred-frozen)))
    save('qa/medium_reproduction.json', dict(status='PASSED' if delta <= 1e-6 else 'FAILED',
         rows=len(pred), maximum_absolute_difference=delta, torch=torch.__version__,
         numpy=np.__version__, seconds=time.perf_counter()-t, model=p.model_best))
    assert delta <= 1e-6, 'Frozen Medium predictions differ'
    hubs=sorted(test.hub.unique())
    grid=pd.DataFrame([dict(release_time=t, hub=h, q=q) for t in range(24) for h in hubs for q in range(1,9)])
    t=time.perf_counter(); grid['value']=p.predict(grid[cols]); elapsed=time.perf_counter()-t
    grid.to_csv(OUT/'medium_price_grid.csv', index=False, encoding='utf-8-sig')
    save('qa/medium_grid.json', dict(rows=len(grid), seconds=elapsed, features=cols))
    print('MEDIUM_REPRODUCED', delta, flush=True)


def gpu():
    import torch
    versions={}
    for name in ['autogluon.tabular','autogluon.core','numpy','pandas','scikit-learn','torch','torchvision',
                 'synthefy-nori','tabicl','tabdpt','tabpfn','pytabkit','faiss-cpu','numba','llvmlite']:
        versions[name]=md.version(name)
    result=dict(platform=platform.platform(), python=platform.python_version(), versions=versions,
                cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
                gpu_count=torch.cuda.device_count(), devices=[], imports={}, probes={})
    save('environment/gpu_probe.json', result)
    try:
        assert torch.cuda.is_available()
        torch.set_num_threads(1)
        for i in range(torch.cuda.device_count()):
            prop=torch.cuda.get_device_properties(i)
            result['devices'].append(dict(name=prop.name, total_memory=prop.total_memory,
                capability=torch.cuda.get_device_capability(i)))
        a=torch.arange(256,device='cuda',dtype=torch.float32).reshape(16,16)
        b=a@a.T
        torch.cuda.synchronize()
        assert torch.isfinite(b).all().item()
        result['cuda_kernel_passed']=True
        for module in ['autogluon.tabular','synthefy_nori','tabicl','tabdpt','pytabkit']:
            importlib.import_module(module); result['imports'][module]='PASSED'
        # Package-provided download functions use their original official checkpoints.
        from huggingface_hub import hf_hub_download
        from synthefy_nori.hf import download_checkpoint
        from tabdpt.estimator import TabDPTEstimator
        weights_path=OUT/'environment/pretrained_weights.json'
        weights=json.loads(weights_path.read_text()) if weights_path.exists() else []
        for name, loader in [('Nori-30M', lambda: download_checkpoint(model='nori-30m')),
                             ('TabICLv2', lambda: hf_hub_download('jingang/TabICL','tabicl-regressor-v2-20260212.ckpt')),
                             ('TabDPT-Turbo', TabDPTEstimator.download_weights)]:
            if any(w['model']==name and sha(w['path'])==w['sha256'] for w in weights):
                continue
            print('WEIGHT_DOWNLOAD',name,flush=True)
            t=time.perf_counter(); path=Path(loader())
            weights.append(dict(model=name,path=str(path),bytes=path.stat().st_size,sha256=sha(path),
                                download_or_cache_wall_seconds=time.perf_counter()-t))
            save('environment/pretrained_weights.json',weights)
        # Small synthetic mechanical probes are not fits of an AutoGluon predictor or label training.
        # Use the installed AutoGluon adapters so device handling matches the formal preset.
        import numpy as np
        import pandas as pd
        from autogluon.tabular.models.nori.nori_model import NoriModel
        from autogluon.tabular.models.tabicl.tabicl_model import TabICLModel
        from autogluon.tabular.models.tabdpt.tabdpt_turbo_model import TabDPTTurboModel
        x=pd.DataFrame(dict(release_time=np.tile(np.arange(8),4),hub=pd.Categorical(['a','b']*16),q=np.repeat([1,2,3,4],8)))
        y=pd.Series(x.release_time.to_numpy()+x.q.to_numpy(),name='target')
        for cls,hps in [(NoriModel,{'model':'nori-30m'}),(TabICLModel,{}),(TabDPTTurboModel,{})]:
            print('GPU_MECHANICAL_PROBE',cls.__name__,flush=True)
            torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
            model=cls(path=str(ROOT/'workspace/extreme-mechanical-probes'), name=cls.__name__,
                      problem_type='regression',eval_metric='root_mean_squared_error',hyperparameters=hps)
            model.fit(X=x,y=y,num_cpus=1,num_gpus=1)
            pred=model.predict(x.iloc[:2]); torch.cuda.synchronize()
            assert np.isfinite(pred).all()
            device=str(model.get_device())
            assert 'cuda' in device, device
            result['probes'][cls.__name__]=dict(status='PASSED',device=device,
                seconds=time.perf_counter()-t,peak_cuda_bytes=torch.cuda.max_memory_allocated(),predictions=pred.tolist())
            save('environment/gpu_probe.json',result)
            del model
            import gc
            gc.collect();torch.cuda.empty_cache()
        result['status']='PASSED'
    except BaseException as e:
        result['status']='STOPPED_PREFLIGHT'
        result['error']=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
        raise
    finally:
        save('environment/gpu_probe.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['medium','gpu']);a=p.parse_args()
    if a.mode=='gpu':
        os.environ['HF_HOME']=str(ROOT/'workspace/extreme-cache/huggingface')
        os.environ['TORCH_HOME']=str(ROOT/'workspace/extreme-cache/torch')
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
        os.environ['HF_HUB_ETAG_TIMEOUT']='30'
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'
        gpu()
    else: medium()
