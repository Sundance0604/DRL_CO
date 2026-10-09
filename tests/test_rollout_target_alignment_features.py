"""Shared transforms and q-average approximation are fixed across feature sets."""
import numpy as np
from experiments.cfs_rollout_target_alignment.features import OLD,ACTION,design,fields
from experiments.cfs_rollout_target_alignment.fitting import common_meta,select_features
from experiments.cfs_rollout_target_alignment.toys import fixture,order
from experiments.cfs_rollout_target_alignment.queries import choose_candidates
from experiments.cfs_classical_value.simulation import summary

def test_common_feature_columns_exact_for_old_action():
    s,o=fixture([order('a','A','B',20)]);cs,_,_=choose_candidates(s)
    online=summary(s,{'rate':{}})
    rows=[fields(c['context'],online,q) for c in cs for q in [1,2,3]]
    for method in ['linear','chebyshev','spline']:
        meta,P=common_meta(rows,method,3,4,'natural',['A','B','C'])
        a,_=select_features(meta,P,'OLD');b,_=select_features(meta,P,'ACTION')
        xa=design(rows,a,True);xb=design(rows,b,True)
        assert np.array_equal(xa,xb[:,:xa.shape[1]]) and a['qs']==b['qs'] and a['q_weights']==b['q_weights']

def test_deployed_q_mixture_is_invariant_and_no_postsolve_fields():
    s,o=fixture([order('a','A','B',20)]);cs,_,_=choose_candidates(s);online=summary(s,{'rate':{}})
    rows=[fields(cs[0]['context'],online,q) for q in [1,2,3]]
    meta,P=common_meta(rows,'chebyshev',3,4,'ordinary',['A','B','C']);m,_=select_features(meta,P,'ACTION')
    x=design(rows,m,True)
    assert np.array_equal(x[0],x[1]) and np.array_equal(x[1],x[2])
    assert all(not any('actual' in k or 'assignment' in k for k in r) for r in rows)
