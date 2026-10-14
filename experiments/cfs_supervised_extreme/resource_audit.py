"""Static resource estimates and package metadata; never fits a model."""
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
import psutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/cfs_supervised_value/autogluon_extreme_gpu_v1'
before=psutil.virtual_memory()._asdict()
import pandas as pd
from autogluon.tabular.models.nori.nori_model import NoriModel
from autogluon.tabular.configs.presets_configs import tabular_presets_dict
from autogluon.tabular.configs.zeroshot.zeroshot_portfolio_commercial_2026_08_05 import hyperparameter_portfolio_commercial_2026_08_05 as portfolio
x=pd.read_csv(ROOT/'reports/cfs_supervised_value/autogluon_pilot_v3/data/train.csv')[['release_time','hub','q']]
x['hub']=x.hub.astype('category')
estimates={}
for rows in [32,8820,10080]:
    frame=x.iloc[:rows]
    estimates[str(rows)]=dict(system_memory_bytes=NoriModel._estimate_memory_usage_static(X=frame),
                              gpu_memory_bytes=NoriModel._estimate_gpu_memory_usage_static(X=frame),
                              required_available_system_memory_at_90_percent=NoriModel._estimate_memory_usage_static(X=frame)/.9)
value=dict(status='STATIC_ESTIMATES_ONLY',python=sys.executable,
           before_import_system_memory=before,after_import_system_memory=psutil.virtual_memory()._asdict(),
           process_rss_bytes=psutil.Process().memory_info().rss,nori_estimates=estimates,
           nori_requires_dist=md.requires('synthefy-nori'),extreme_preset=tabular_presets_dict['extreme_quality'],
           actual_portfolio=portfolio,
           nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu','--format=csv,noheader'],text=True))
(OUT/'environment/resource_estimates.json').write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps({k:value[k] for k in ['before_import_system_memory','after_import_system_memory','nori_estimates','nvidia_smi']},indent=2))
