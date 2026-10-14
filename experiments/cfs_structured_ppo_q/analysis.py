"""Frozen small-sample crossed bootstrap; raw paired axes remain distinct."""
import numpy as np


def crossed(x,replications=20000,seed=71000):
    x=np.asarray(x,float)
    if x.shape!=(3,8) or not np.isfinite(x).all():raise ValueError('requires3seed x8scenario')
    rng=np.random.default_rng(seed);out=[]
    for _ in range(replications//1000):
        a=rng.integers(0,3,(1000,3));b=rng.integers(0,8,(1000,8))
        d=x[a[:,:,None],b[:,None,:]].mean((1,2))
        ca=np.array([np.bincount(v,minlength=3)/3 for v in a]);cb=np.array([np.bincount(v,minlength=8)/8 for v in b])
        if not np.allclose(d,np.einsum('bi,ij,bj->b',ca,x,cb),rtol=1e-12,atol=1e-8):raise RuntimeError('bootstrap recomputation')
        out.extend(d)
    return dict(mean_USD=float(x.mean()),seed_means_USD=x.mean(1).tolist(),
        CI95=np.quantile(out,[.025,.975]).tolist(),Bonferroni3=np.quantile(out,[.05/6,1-.05/6]).tolist())
