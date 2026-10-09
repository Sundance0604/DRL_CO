"""Post-freeze scientific-export/statistics tests; never model selection."""
import csv,json
import numpy as np
from experiments.cfs_local_post_bayesian_value.analysis import ci
from experiments.cfs_local_post_bayesian_value.common import write_csv

def test_flat_csv_uses_valid_json_for_nested_provenance(tmp_path):
    path=tmp_path/'export.csv'
    write_csv(path,[{'seed':5000,'target':12.5,'context':{'hub':'A','time':3},'ids':['a','b']}])
    with path.open(encoding='utf-8',newline='') as f:row=next(csv.DictReader(f))
    assert json.loads(row['context'])=={'hub':'A','time':3}
    assert json.loads(row['ids'])==['a','b'] and float(row['target'])==12.5

def test_constant_sample_confidence_interval_is_not_nan():
    cfg={'bootstrap_seed':7,'bootstrap_replications':10000}
    result=ci(np.ones(40)*17,cfg)
    assert result['ci_lower']==result['ci_upper']==result['mean']==17 and result['skew']==0

def test_skewed_paired_bootstrap_is_fixed_seed_deterministic():
    cfg={'bootstrap_seed':7,'bootstrap_replications':10000};values=[0.]*39+[100.]
    a=ci(values,cfg);b=ci(values,cfg)
    assert a==b and a['skew']>1 and 'bootstrap_lower' in a
