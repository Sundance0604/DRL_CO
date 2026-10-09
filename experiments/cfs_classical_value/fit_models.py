"""Least-squares / ridge / penalized additive cubic fits, validation-only choice."""
import itertools
import math
import time
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import block_diag,null_space
from scipy.stats import spearmanr
from experiment_core.storage import atomic_json,read_json
from model_families.single.value.providers import FEATURES,design
from .common import OUTPUT,key,write_csv


def prediction_metrics(rows,pred):
    y=np.asarray([r["target"] for r in rows],float);pred=np.asarray(pred,float)
    assert np.isfinite(y).all() and np.isfinite(pred).all()
    e=y-pred
    corr=float(spearmanr(y,pred).statistic) if np.std(pred)>1e-10 and np.std(y)>1e-10 else None
    groups={}
    for i,r in enumerate(rows):
        groups.setdefault((r["scenario_seed"],r["time"]),[]).append(i)
    ranks=[];spatial=[]
    for ids in groups.values():
        for i,j in itertools.combinations(ids,2):
            if abs(y[i]-y[j])<1e-6:continue
            correct=0.5 if abs(pred[i]-pred[j])<1e-6 else float((y[i]-y[j])*(pred[i]-pred[j])>0)
            ranks.append(correct)
            if rows[i]["hub"]!=rows[j]["hub"] and rows[i]["q"]==rows[j]["q"]:spatial.append(correct)
    endpoint=[i for i,r in enumerate(rows) if r["time"]<=3 or r["time"]>=20]
    return dict(n=len(rows),MAE=float(np.mean(abs(e))),RMSE=float(np.sqrt(np.mean(e**2))),
        R2=1-float(e@e)/float((y-y.mean())@(y-y.mean())) if np.var(y)>0 else None,
        Spearman=corr,pairwise_ranking_accuracy=float(np.mean(ranks)) if ranks else None,
        spatial_same_q_ranking_accuracy=float(np.mean(spatial)) if spatial else None,
        sign_accuracy=float(np.mean(np.sign(y)==np.sign(pred))),endpoint_RMSE=float(np.sqrt(np.mean(e[endpoint]**2))),
        max_abs_error=float(np.max(abs(e))))


def averaged(rows):
    grouped={}
    for r in rows:
        k=(r["condition"],r["scenario_seed"],r["time"],r["hub"],r["q"])
        grouped.setdefault(k,[]).append(r)
    out=[]
    for items in grouped.values():
        out.append(items[0]|{"target":float(np.mean([r["target"] for r in items])),
            "mc_paths":len(items) if not items[0].get("terminal_anchor") else 0})
    return out


def real(rows):
    return [r for r in rows if not r.get("terminal_anchor",False)]


def basis_meta(rows,method,degree=2,knots=4,boundary="ordinary"):
    meta=dict(method=method,hubs=sorted(set(r["hub"] for r in rows)),bases={})
    penalties=[np.zeros((len(meta["hubs"]),len(meta["hubs"])))]
    for name in FEATURES:
        x=np.asarray([r[name] for r in rows],float)
        lower,upper=(0.,1.) if name=="remaining" else (float(x.min()),float(x.max()))
        scale=max(upper-lower,1.)
        if method=="linear" or upper==lower:
            b=dict(kind="linear",center=float(x.mean()),scale=max(float(x.std()),1.))
            penalty=np.eye(1)
        elif method=="chebyshev":
            b=dict(kind="chebyshev",lower=lower,scale=upper-lower,degree=degree)
            penalty=np.eye(degree)
        else:
            internal=np.unique(np.quantile(x,np.arange(1,knots+1)/(knots+1)))
            internal=internal[(internal>lower)&(internal<upper)]
            knot_vector=np.r_[np.repeat(lower,4),internal,np.repeat(upper,4)]
            n=len(knot_vector)-4
            raw=BSpline(knot_vector,np.eye(n),3)
            # Natural cubic f''(xmin)=f''(xmax)=0, exact null-space constraints.
            transform=null_space(raw.derivative(2)([lower,upper])) if boundary=="natural" else np.eye(n)
            center=(raw(x)@transform).mean(axis=0)
            penalty=transform.T@np.diff(np.eye(n),n=2,axis=0).T@np.diff(np.eye(n),n=2,axis=0)@transform
            b=dict(kind="spline",lower=lower,upper=upper,knots=knot_vector.tolist(),
                transform=transform.tolist(),center=center.tolist(),internal_count=len(internal),
                requested_internal_count=knots,boundary=boundary)
        meta["bases"][name]=b;penalties.append(penalty)
    return meta,block_diag(*penalties)


def fit(rows,method,penalty,degree=2,knots=4,boundary="ordinary",terminal="raw"):
    meta,P=basis_meta(rows,method,degree,knots,boundary)
    X=design(rows,meta);y=np.asarray([r["target"] for r in rows])
    # Centered spline blocks have null constant columns. Tiny isotropic ridge
    # stabilizes the nonidentified additive offsets without hiding conditioning.
    regularization=penalty*P+1e-6*np.eye(X.shape[1]);regularization[0,0]=0
    Z=np.eye(X.shape[1])
    if terminal=="anchored":
        median={f:float(np.median([r[f] for r in rows])) for f in FEATURES}
        anchors=[dict(median,remaining=0.,hub=h,q=1.) for h in meta["hubs"]]
        Z=null_space(design(anchors,meta))
    A=Z.T@(X.T@X+regularization)@Z
    beta=Z@np.linalg.solve(A,Z.T@X.T@y)
    smoother=np.linalg.solve(A,Z.T@X.T@X@Z)
    meta.update(coefficients=beta.tolist(),penalty=penalty,degree=degree if method=="chebyshev" else None,
        requested_knots=knots if method=="spline" else None,boundary=boundary if method=="spline" else None,
        terminal_constraint=terminal,effective_df=float(np.trace(smoother)),
        design_condition_number=float(np.linalg.cond(X)) if np.linalg.matrix_rank(X)==X.shape[1] else None,
        design_rank=int(np.linalg.matrix_rank(X)),design_columns=X.shape[1],
        regularized_condition_number=float(np.linalg.cond(A)),roughness=float(beta@P@beta),
        extrapolation="clip covariates to construction domain; exact free-supply gate at time>=T",
        terminal_anchor_semantics="exact zero at median other covariates, q=1 for each hub; does NOT impose zero for all additive covariate combinations")
    return meta


def table_model(rows,fleet):
    times=sorted(set(r["time"] for r in rows));hubs=sorted(set(r["hub"] for r in rows));values={}
    for h in hubs:
        values[h]={}
        for q in range(1,fleet+1):
            vals=[]
            for t in times:
                group=[r for r in rows if r["hub"]==h and r["time"]==t]
                byq={}
                for r in group:byq.setdefault(r["q"],[]).append(r["target"])
                if not byq:
                    raise RuntimeError("MC table missing time/hub cell")
                qs=sorted(byq)
                vals.append(float(np.interp(q,qs,[np.mean(byq[x]) for x in qs])))
            values[h][str(q)]=dict(times=times,values=vals)
    return dict(method="table",values=values,target="mean paired marginal by time/hub/q; time and unobserved integer q interpolated",
        state_dependence="current backlog/other idle/incoming ignored; construction-only surface")


def fit_all(cfg):
    train=averaged(read_json(OUTPUT/"targets/construction.json"))
    val=averaged(read_json(OUTPUT/"targets/validation.json"))
    candidate_rows=[];selection={};started=time.perf_counter()
    for c in cfg["conditions"]:
        cid=key(c);a=[r for r in train if r["condition"]==cid];b=real([r for r in val if r["condition"]==cid])
        atomic_json(OUTPUT/"models"/f"{cid}-table.json",table_model(a,c["fleet"]))
        selection[cid]={}
        for method in ["linear","chebyshev","spline"]:
            candidates=[]
            grid=(itertools.product([2],[4],["ordinary"],cfg["ridge_penalties"],["raw"]) if method=="linear" else
                  itertools.product(cfg["degrees"],[4],["ordinary"],cfg["ridge_penalties"],["raw","anchored"]) if method=="chebyshev" else
                  itertools.product([2],cfg["spline_internal_knots"],["ordinary","natural"],cfg["spline_penalties"],["raw","anchored"]))
            for index,(degree,knots,boundary,penalty,terminal) in enumerate(grid):
                model=fit(a,method,penalty,degree,knots,boundary,terminal)
                pred=design(b,model)@np.asarray(model["coefficients"])
                m=prediction_metrics(b,pred)
                name=f"{cid}-{method}-candidate-{index:03d}.json"
                atomic_json(OUTPUT/"models/candidates"/name,model)
                row=dict(condition=cid,method=method,candidate=name,degree=model["degree"],knots=model["requested_knots"],
                    boundary=model["boundary"],penalty=penalty,terminal=terminal,effective_df=model["effective_df"],
                    condition_number=model["regularized_condition_number"],roughness=model["roughness"],**m)
                candidate_rows.append(row);candidates.append((m["RMSE"],m["MAE"],-(m["pairwise_ranking_accuracy"] or 0),index,model,row))
            best=min(candidates,key=lambda x:x[:4]);model=best[4]
            model.update(selection_split="validation",selection_rule="minimum validation RMSE; MAE, ranking and index tie breaks",validation_metrics=best[5])
            atomic_json(OUTPUT/"models"/f"{cid}-{method}.json",model)
            selection[cid][method]=best[5]
        print("Frozen validation selection:",cid,selection[cid],flush=True)
    write_csv(OUTPUT/"processed/validation_candidates.csv",candidate_rows)
    atomic_json(OUTPUT/"models/selection.json",selection)
    return dict(wall_seconds=time.perf_counter()-started,selection=selection,
        interactions="none; first-stage additive diagnostic, no full tensor",formal_test_seen=False)
