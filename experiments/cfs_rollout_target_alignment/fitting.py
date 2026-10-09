"""Strict paired feature ablation with shared bases and shared selected grid point."""
import itertools,time,copy
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import block_diag,null_space,cho_factor,cho_solve
from scipy.optimize import minimize
from experiment_core.storage import atomic_json,read_json,digest
from experiments.cfs_classical_value.fit_models import table_model,averaged
from experiments.cfs_local_post_bayesian_value.fit_models import gaussian_posterior
from model_families.single.value.providers import MCTableValueProvider
from .common import OUTPUT,key,now,file_hash,write_csv,FOURTH
from .features import OLD,ACTION,ALIASES,fields,design
from .queries import restore_state
from .protocol import read_compressed
from .diagnostics import ranking,correlation,sign
from experiments.cfs_classical_value.simulation import summary

def matrices(rows,table,prior,fleet):
    slots=sorted({q for q in [1,2,3,4,6,8] if q<=fleet}|{fleet});out=[];states={}
    groups={}
    for r in rows:groups.setdefault(r['parent_query_id'],[]).append(r)
    provider=MCTableValueProvider(table)
    for items in groups.values():
        r=items[0];statekey=(r['condition'],r['scenario_seed'],r['time'])
        if statekey not in states:
            packet=read_compressed(OUTPUT/f'raw/query_states/{r["condition"]}-{r["scenario_seed"]}.json.gz')
            s,_=restore_state(packet['scenario'],packet['parameters'],packet['solver'],packet['states'][str(r['time'])])
            states[statekey]=summary(s,prior)
        online=states[statekey];target=float(np.mean([a['Y_roll'] for a in items]))
        for q in slots:
            base=provider.marginal_release_value(r['release_time'],r['release_hub'],q,{'T':24})
            out.append({k:v for k,v in r.items() if k not in ['metadata_features','route']} | fields(r['context'],online,q) |
                dict(target=target,base_slot_value=base,residual=target-base,q=q,weight=1/len(slots),
                    terminal_release=r['release_time']>=24,mc_paths=len(items)))
    return out

def common_meta(rows,method,degree,knots,boundary,hubs):
    qs=sorted({int(r['q']) for r in rows});bases={};penalties={}
    for f in ACTION:
        x=np.array([r[f] for r in rows],float);lo,hi=(0.,1.) if f=='o_remaining' else (float(x.min()),float(x.max()))
        if method in ('linear','bayesian') or hi==lo:
            b=dict(kind='linear',center=float(x.mean()),scale=max(float(x.std()),1.));P=np.eye(1)
        elif method=='chebyshev':b=dict(kind='chebyshev',lower=lo,scale=hi-lo,degree=degree);P=np.eye(degree)
        else:
            inside=np.unique(np.quantile(x,np.arange(1,knots+1)/(knots+1)));inside=inside[(inside>lo)&(inside<hi)]
            kv=np.r_[np.repeat(lo,4),inside,np.repeat(hi,4)];n=len(kv)-4;raw=BSpline(kv,np.eye(n),3)
            Z=null_space(raw.derivative(2)([lo,hi])) if boundary=='natural' else np.eye(n)
            D=np.diff(np.eye(n),n=2,axis=0);P=Z.T@D.T@D@Z
            b=dict(kind='spline',lower=lo,upper=hi,knots=kv.tolist(),transform=Z.tolist(),
                center=(raw(x)@Z).mean(axis=0).tolist(),boundary=boundary,internal_count=len(inside))
        bases[f]=b;penalties[f]=P
    meta=dict(method=method,hubs=hubs,qs=qs,q_weights=[sum(r['q']==q for r in rows)/len(rows) for q in qs],
        bases=bases,features=ACTION,degree=degree,requested_knots=knots,boundary=boundary)
    return meta,penalties

def select_features(meta,penalties,feature_set):
    m=copy.deepcopy(meta);m['feature_set']=feature_set;m['features']=OLD if feature_set=='OLD' else ACTION
    m['bases']={f:m['bases'][f] for f in m['features']}
    P=block_diag(np.eye(1+len(m['hubs'])+len(m['qs'])),*[penalties[f] for f in m['features']])
    return m,P

def predict(rows,m):
    return design(rows,m,True)@np.array(m['coefficients'])+np.array([r['base_slot_value'] for r in rows])

def collapse(rows,pred):
    groups={}
    for r,p in zip(rows,pred):groups.setdefault(r['parent_query_id'],[]).append((r,float(p)))
    return [items[0][0]|dict(prediction=float(np.mean([p for _,p in items]))) for items in groups.values()]

def prediction_scores(rows,pred):
    rr=collapse(rows,pred);y=np.array([r['target'] for r in rr]);p=np.array([r['prediction'] for r in rr]);e=p-y
    rank=ranking(rr,'target','prediction');pairs=sum(r['all_pairs'] for r in rank)
    correct=0.;count=0
    groups={}
    for r in rr:groups.setdefault((r['scenario_seed'],r['time'],r['vehicle_id']),[]).append(r)
    for group in groups.values():
        for a,b in itertools.combinations(group,2):
            d=a['target']-b['target'];z=a['prediction']-b['prediction']
            if abs(d)<=1e-6:continue
            correct+=.5 if abs(z)<=1e-6 else float(d*z>0);count+=1
    return dict(n_candidates=len(rr),n_scenarios=len({r['scenario_seed'] for r in rr}),
        MAE=float(abs(e).mean()),RMSE=float(np.sqrt(np.mean(e*e))),R2=float(1-e@e/((y-y.mean())@(y-y.mean()))) if np.var(y)>0 else None,
        Spearman=correlation(y,p),sign_accuracy=float(np.mean(sign(y,1e-6)==sign(p,1e-6))),
        pairwise_ranking_accuracy=correct/count if count else None,
        top1_accuracy=float(np.mean([1-r['top1_disagreement'] for r in rank])) if rank else None,
        top1_set_accuracy=float(np.mean([1-r['top1_disjoint'] for r in rank])) if rank else None)

def bayes_pair(models,Xs,ys):
    """One shared EB noise/prior scale fitted jointly to OLD/ACTION construction data."""
    sc=max(float(np.std(np.concatenate(ys))),1.)
    def prior(m,logs):
        nh,nq=len(m['hubs']),len(m['qs']);p=Xs[m['feature_set']].shape[1]
        gv,hv,qv=np.exp(2*np.array(logs[:3]))
        return np.r_[100.,np.repeat(hv,nh),np.repeat(qv,nq),np.repeat(gv,p-1-nh-nq)]
    def evidence(logs):
        total=0.
        for m in models:
            fs=m['feature_set'];X=Xs[fs];y=ys[0 if fs=='OLD' else 1]/sc
            D=prior(m,logs);v=float(np.exp(2*logs[3]));A=X.T@X/v+np.diag(1/D)
            cf=cho_factor(A,lower=True);z=cho_solve(cf,X.T@y/v)
            # q rows are fractional likelihood augmentation, not independent samples.
            n_eff=len(y)/len(m['qs'])
            total+=.5*(n_eff*np.log(2*np.pi)+n_eff*np.log(v)+np.log(D).sum()+2*np.log(np.diag(cf[0])).sum()+max(0,y@y/v-(X.T@y/v)@z))
        return float(total)
    opt=minimize(evidence,np.log([1.,.5,.5,1.]),method='L-BFGS-B',bounds=[(-6,4)]*4,options={'maxiter':250,'ftol':1e-10})
    assert np.isfinite(opt.fun) and np.isfinite(opt.x).all()
    options={fs:[] for fs in ['OLD','ACTION']}
    for multiplier in [.25,1.,4.]:
        for m in models:
            fs=m['feature_set'];D=prior(m,opt.x);D[1:1+len(m['hubs'])+len(m['qs'])]*=multiplier
            var=float(np.exp(2*opt.x[3]));coef,cov=gaussian_posterior(Xs[fs],ys[0 if fs=='OLD' else 1]/sc,var,D)
            options[fs].append(m|dict(coefficients=(coef*sc).tolist(),posterior_covariance=(cov*sc**2).tolist(),
                predictive_noise_variance=var*sc**2,shared_EB_log_hyperparameters=opt.x.tolist(),
                shared_EB_success=bool(opt.success),shared_EB_message=str(opt.message),pooling_variance_multiplier=multiplier,
                conditional_uncertainty_only=True,inference='joint shared EB scales, closed-form Gaussian posterior, NO MCMC'))
    return options

def fit_all(cfg,manifest,priors):
    train=read_json(OUTPUT/'targets/construction.json');val=read_json(OUTPUT/'targets/validation.json')
    slot=read_json(OUTPUT/'targets/construction_mc.json');candidates=[];selection={};runtimes=[];allmatrices=[]
    for c in cfg['conditions']:
        cid=key(c);ts=time.perf_counter();table=table_model(averaged([r for r in slot if r['condition']==cid]),c['fleet'])
        atomic_json(OUTPUT/'models/table'/f'{cid}.json',table)
        a=matrices([r for r in train if r['condition']==cid],table,priors[cid],c['fleet'])
        b=matrices([r for r in val if r['condition']==cid],table,priors[cid],c['fleet'])
        allmatrices.extend(r|dict(split='construction') for r in a);allmatrices.extend(r|dict(split='validation') for r in b)
        a=[r for r in a if not r['terminal_release']];b=[r for r in b if not r['terminal_release']]
        assert a and b, f'No nonterminal training/validation queries: {cid}'
        selection[cid]={}
        for method in ['linear','chebyshev','spline','bayesian']:
            started=time.perf_counter()
            grid=(list(itertools.product([2],[4],['ordinary'],cfg['ridge_penalties'])) if method=='linear' else
                list(itertools.product(cfg['degrees'],[4],['ordinary'],cfg['ridge_penalties'])) if method=='chebyshev' else
                list(itertools.product([2],cfg['spline_internal_knots'],['ordinary','natural'],cfg['spline_penalties'])) if method=='spline' else [(2,4,'ordinary',None)])
            options={'OLD':[],'ACTION':[]}
            for degree,knots,boundary,penalty in grid:
                shared,P=common_meta(a,method,degree,knots,boundary,sorted(table['values']))
                ms=[];xs={};ys=[]
                for fs in ['OLD','ACTION']:
                    m,R=select_features(shared,P,fs);X=design(a,m)*np.sqrt([r['weight'] for r in a])[:,None]
                    y=np.array([r['residual'] for r in a])*np.sqrt([r['weight'] for r in a]);ms.append(m);xs[fs]=X;ys.append(y)
                    if method!='bayesian':
                        A=X.T@X+penalty*R+1e-6*np.eye(X.shape[1]);cf=cho_factor(A,lower=True);coef=cho_solve(cf,X.T@y)
                        options[fs].append(m|dict(coefficients=coef.tolist(),penalty=penalty,
                            condition_number=float(np.linalg.cond(A)),effective_df=float(np.trace(cho_solve(cf,X.T@X))),roughness=float(coef@R@coef)))
                if method=='bayesian':options=bayes_pair(ms,xs,ys)
            ranks=[]
            for i in range(len(options['OLD'])):
                stats={}
                for fs in ['OLD','ACTION']:
                    metrics=prediction_scores(b,predict(b,options[fs][i]));stats[fs]=metrics
                    candidates.append(dict(condition=cid,method=method,feature_set=fs,candidate=i,**metrics))
                rankloss=np.mean([1-(stats[fs]['pairwise_ranking_accuracy'] or 0) for fs in ['OLD','ACTION']])
                top_loss=np.mean([1-(stats[fs]['top1_accuracy'] or 0) for fs in ['OLD','ACTION']])
                rmse=np.mean([stats[fs]['RMSE'] for fs in ['OLD','ACTION']])
                ranks.append((rankloss,top_loss,rmse,i,stats))
            best=min(ranks,key=lambda x:x[:4]);index=best[3];selection[cid][method]=dict(shared_candidate=index,OLD=best[4]['OLD'],ACTION=best[4]['ACTION'])
            for fs in ['OLD','ACTION']:
                m=options[fs][index]
                m.update(slot_model=table,training_hash=digest(a),validation_hash=digest(b),
                    selected_by='SHARED grid point: mean OLD/ACTION validation rank loss, top1 loss, RMSE, candidate index',
                    same_transform_for_shared_features=True,selected_shared_candidate=index,code_hash=manifest['phase2_code_hash'],frozen_at=now(),aliases=ALIASES)
                atomic_json(OUTPUT/'models'/f'{method}_{fs}'/f'{cid}.json',m)
            runtimes.append(dict(condition=cid,method=method,seconds=time.perf_counter()-started))
        # Exact immutable Legacy Fluid copy; no new estimation or coefficient tuning.
        old=read_json(FOURTH/'models/legacy_fluid_reference'/f'{cid}.json')
        atomic_json(OUTPUT/'models/legacy_fluid_reference'/f'{cid}.json',old)
        assert file_hash(OUTPUT/'models/legacy_fluid_reference'/f'{cid}.json')==file_hash(FOURTH/'models/legacy_fluid_reference'/f'{cid}.json')
        print('Paired OLD/ACTION models frozen:',cid,flush=True)
    write_csv(OUTPUT/'targets/residual_targets.csv',allmatrices)
    write_csv(OUTPUT/'processed/validation_candidates.csv',candidates)
    write_csv(OUTPUT/'processed/offline_fit_runtime.csv',runtimes)
    atomic_json(OUTPUT/'models/selection.json',selection)
    hashes={p.relative_to(OUTPUT).as_posix():file_hash(p) for p in (OUTPUT/'models').rglob('*.json') if p.name not in ['model_freeze.json']}
    frozen=dict(models=hashes,code_hash=manifest['phase2_code_hash'],frozen_at=now(),formal_seen=False,selection=selection,
        construction_seeds=cfg['construction_seeds'],validation_seeds=cfg['validation_seeds'],feature_sets=dict(OLD=OLD,ACTION=ACTION),aliases=ALIASES,
        strict_ablation='same targets, samples, table, common transforms, jointly selected shared hyperparameters; only feature inclusion differs',
        boundary='all post-T queries retained in raw targets; fit excludes them; deployed original zero-release gate retained, not manufactured zero targets')
    atomic_json(OUTPUT/'manifests/model_freeze.json',frozen)
    return dict(selection=selection,runtimes=runtimes)
