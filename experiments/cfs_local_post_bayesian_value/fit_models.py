"""Shared features, additive bases, validation-only selection, Gaussian EB."""
import itertools,time
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import block_diag,null_space,cho_factor,cho_solve
from scipy.optimize import minimize
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.local_value.providers import FEATURES,design
from model_families.single.value.providers import MCTableValueProvider
from experiments.cfs_classical_value.fit_models import table_model,averaged,prediction_metrics
from .common import OUTPUT,OLD,key,file_hash,write_csv,now
from .targets import averaged_actions

def meta(rows,method,degree=2,knots=4,boundary='ordinary'):
    hubs=sorted({r['hub'] for r in rows});qs=sorted({int(r['q']) for r in rows})
    m=dict(method=method,hubs=hubs,qs=qs,q_weights=[sum(int(r['q'])==q for r in rows)/len(rows) for q in qs],bases={},features=FEATURES)
    penalties=[np.eye(1+len(hubs)+len(qs))]
    for name in FEATURES:
        x=np.array([r[name] for r in rows],float);lo,hi=(0.,1.) if name=='remaining' else (float(x.min()),float(x.max()))
        if method in ('linear','bayesian') or hi==lo:
            b=dict(kind='linear',center=float(x.mean()),scale=max(float(x.std()),1.));P=np.eye(1)
        elif method=='chebyshev':
            b=dict(kind='chebyshev',lower=lo,scale=hi-lo,degree=degree);P=np.eye(degree)
        else:
            internal=np.unique(np.quantile(x,np.arange(1,knots+1)/(knots+1)))
            internal=internal[(internal>lo)&(internal<hi)];kv=np.r_[np.repeat(lo,4),internal,np.repeat(hi,4)]
            n=len(kv)-4;raw=BSpline(kv,np.eye(n),3)
            Z=null_space(raw.derivative(2)([lo,hi])) if boundary=='natural' else np.eye(n)
            D=np.diff(np.eye(n),n=2,axis=0)
            P=Z.T@D.T@D@Z
            b=dict(kind='spline',lower=lo,upper=hi,knots=kv.tolist(),transform=Z.tolist(),center=(raw(x)@Z).mean(axis=0).tolist(),
                requested_internal_count=knots,internal_count=len(internal),boundary=boundary)
        m['bases'][name]=b;penalties.append(P)
    return m,block_diag(*penalties)

def gaussian_posterior(X,y,variance,prior_variance):
    D=np.asarray(prior_variance);assert np.all(D>0) and variance>0
    A=X.T@X/variance+np.diag(1/D);cf=cho_factor(A,lower=True,check_finite=False)
    cov=cho_solve(cf,np.eye(X.shape[1]),check_finite=False)
    beta=cho_solve(cf,X.T@y/variance,check_finite=False)
    cov=(cov+cov.T)/2
    assert np.linalg.eigvalsh(cov).min()>-1e-10
    return beta,cov

def bayes_fit(rows):
    m,_=meta(rows,'bayesian');X=design(rows,m);raw=np.array([r['residual'] for r in rows]);scale=max(float(raw.std()),1.)
    y=raw/scale;n,p=X.shape;nh,nq=len(m['hubs']),len(m['qs']);XX=X.T@X;xy=X.T@y;yy=y@y
    def prior(logs):
        global_var,hub_var,q_var=np.exp(2*np.array(logs[:3]))
        return np.r_[100.,np.repeat(hub_var,nh),np.repeat(q_var,nq),np.repeat(global_var,p-1-nh-nq)]
    def evidence(logs):
        D=prior(logs);var=float(np.exp(2*logs[3]));A=XX/var+np.diag(1/D)
        cf=cho_factor(A,lower=True,check_finite=False);v=cho_solve(cf,xy/var,check_finite=False)
        logdet=n*np.log(var)+np.log(D).sum()+2*np.log(np.diag(cf[0])).sum()
        quad=yy/var-(xy/var)@v
        return float(.5*(n*np.log(2*np.pi)+logdet+max(0,quad)))
    started=time.perf_counter()
    opt=minimize(evidence,np.log([1.,.5,.5,1.]),method='L-BFGS-B',bounds=[(-6,4)]*4,
                 options={'maxiter':250,'ftol':1e-10})
    assert np.isfinite(opt.fun) and np.isfinite(opt.x).all(),opt
    m.update(evidence_seconds=time.perf_counter()-started,evidence_optimizer_success=bool(opt.success),
        evidence_message=str(opt.message),negative_log_evidence=float(opt.fun),log_hyperparameters=opt.x.tolist(),
        response_scale=scale,evidence_prior_variances=prior(opt.x).tolist(),noise_variance=float(np.exp(2*opt.x[3])),
        inference='empirical Bayes evidence; closed-form Gaussian posterior; no MCMC',
        uncertainty_scope='conditional on fitted slot table and EB hyperparameters; iid Gaussian residual assumption, not cluster confidence intervals')
    return m,X,y

def scores(rows,pred):
    # Group by parent decision time, allowing action alternatives with different release times.
    return prediction_metrics([r|{'time':r['decision_time']} for r in rows],pred)

def residualize(rows,table):
    v=MCTableValueProvider(table);out=[]
    for r in rows:
        base=v.marginal_release_value(r['time'],r['hub'],r['q'],{'T':24})
        out.append(r|{'base_slot_value':base,'residual':r['target']-base})
    return out

def fit_all(cfg,manifest):
    train=read_json(OUTPUT/'targets/construction.json');val=read_json(OUTPUT/'targets/validation.json')
    candidates=[]; selections={}; runtimes=[];residual_rows=[]
    for c in cfg['conditions']:
        cid=key(c);start=time.perf_counter()
        table=table_model(averaged([r for r in train['base'] if r['condition']==cid]),c['fleet'])
        atomic_json(OUTPUT/'models/table'/f'{cid}.json',table)
        runtimes.append(dict(condition=cid,method='table',seconds=time.perf_counter()-start))
        a=residualize(averaged_actions([r for r in train['actions'] if r['condition']==cid]),table)
        b=residualize(averaged_actions([r for r in val['actions'] if r['condition']==cid]),table)
        residual_rows.extend(a+b)
        # Analytical terminal zeros are audited separately; they do not dominate model selection.
        a=[r for r in a if not r['terminal_anchor']];b=[r for r in b if not r['terminal_anchor']]
        assert len(a)>100 and len(b)>50
        selections[cid]={}
        for method in ['linear','chebyshev','spline','bayesian']:
            started=time.perf_counter(); options=[]
            if method=='bayesian':
                bm,X,ys=bayes_fit(a)
                for multiplier in [.25,1.,4.]:
                    m=dict(bm);D=np.array(m['evidence_prior_variances']);nh,nq=len(m['hubs']),len(m['qs'])
                    D[1:1+nh+nq]*=multiplier
                    ts=time.perf_counter();coef,cov=gaussian_posterior(X,ys,m['noise_variance'],D)
                    sc=m['response_scale'];m.update(coefficients=(coef*sc).tolist(),posterior_covariance=(cov*sc**2).tolist(),
                        predictive_noise_variance=m['noise_variance']*sc**2,pooling_variance_multiplier=multiplier,
                        posterior_seconds=time.perf_counter()-ts,prior_variances=(D*sc**2).tolist())
                    options.append(m)
            else:
                grid=(itertools.product([2],[4],['ordinary'],cfg['ridge_penalties']) if method=='linear' else
                      itertools.product(cfg['degrees'],[4],['ordinary'],cfg['ridge_penalties']) if method=='chebyshev' else
                      itertools.product([2],cfg['spline_internal_knots'],['ordinary','natural'],cfg['spline_penalties']))
                for degree,knots,boundary,penalty in grid:
                    m,P=meta(a,method,degree,knots,boundary);X=design(a,m);y=np.array([r['residual'] for r in a])
                    A=X.T@X+penalty*P+1e-6*np.eye(X.shape[1]);cf=cho_factor(A,lower=True)
                    coef=cho_solve(cf,X.T@y);edf=np.trace(cho_solve(cf,X.T@X))
                    m.update(coefficients=coef.tolist(),penalty=penalty,degree=degree,requested_knots=knots,boundary=boundary,
                        condition_number=float(np.linalg.cond(A)),effective_df=float(edf),roughness=float(coef@P@coef))
                    options.append(m)
            ranked=[]
            for i,m in enumerate(options):
                pred=design(b,m,True)@np.array(m['coefficients'])+np.array([r['base_slot_value'] for r in b])
                metrics=scores(b,pred);row=dict(condition=cid,method=method,candidate=i,
                    degree=m.get('degree'),knots=m.get('requested_knots'),boundary=m.get('boundary'),penalty=m.get('penalty'),
                    pooling=m.get('pooling_variance_multiplier'),condition_number=m.get('condition_number'),
                    effective_df=m.get('effective_df'),roughness=m.get('roughness'),**metrics)
                candidates.append(row);ranked.append((metrics['RMSE'],metrics['MAE'],i,m,row))
            best=min(ranked,key=lambda x:x[:3]);m=best[3]
            m.update(slot_model=table,selected_by='validation deployment RMSE, MAE, fixed candidate index',
                     training_hash=digest(a),validation_hash=digest(b),code_hash=manifest['code_hash'],frozen_at=now())
            atomic_json(OUTPUT/'models'/method/f'{cid}.json',m);selections[cid][method]=best[4]
            runtimes.append(dict(condition=cid,method=method,seconds=time.perf_counter()-started,
                evidence_seconds=m.get('evidence_seconds',0),posterior_seconds=m.get('posterior_seconds',0)))
        legacy=read_json(OLD/'models'/f'{cid}-fluid.json')
        atomic_json(OUTPUT/'models/legacy_fluid_reference'/f'{cid}.json',legacy)
        assert file_hash(OUTPUT/'models/legacy_fluid_reference'/f'{cid}.json')==file_hash(OLD/'models'/f'{cid}-fluid.json')
        print('Validation frozen:',cid, {m:round(r['RMSE'],2) for m,r in selections[cid].items()},flush=True)
    write_csv(OUTPUT/'processed/validation_candidates.csv',candidates)
    write_csv(OUTPUT/'processed/offline_fit_runtime.csv',runtimes)
    write_csv(OUTPUT/'targets/residual_targets.csv',residual_rows)
    atomic_json(OUTPUT/'models/selection.json',selections)
    modelhashes={p.relative_to(OUTPUT).as_posix():file_hash(p) for p in (OUTPUT/'models').rglob('*.json')}
    frozen=dict(models=modelhashes,code_hash=manifest['code_hash'],config_hash=manifest['config_hash'],
        frozen_at=now(),selection_split='validation',formal_seed_profit_seen=False,features=FEATURES,
        construction_seed_range=[4000,4039],validation_seed_range=[4100,4114],
        training_targets_hash=file_hash(OUTPUT/'targets/construction.json'),validation_targets_hash=file_hash(OUTPUT/'targets/validation.json'),
        q_integration='fixed construction-frequency marginalization, identical for all local methods',selection=selections)
    atomic_json(OUTPUT/'manifests/model_freeze.json',frozen)
    atomic_json(OUTPUT/'models/model_freeze.json',frozen)
    return dict(selection=selections,runtimes=runtimes)
