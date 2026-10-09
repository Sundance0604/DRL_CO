"""Independent held-out prediction, paired policy inference and paper figures."""
import itertools
import importlib.metadata
import logging
import os
import platform
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment_core.storage import atomic_json,read_json,digest
from experiments.cfs_dynamic_lookahead.analysis import statistics
from model_families.single.value.providers import design,FEATURES
from .common import ROOT,OUTPUT,key,file_hash,write_csv,source_hashes
from .fit_models import averaged,real,prediction_metrics
from .run_value_policy import LADDER

REPORT=ROOT/"reports/cfs_classical_value"
METHODS=["fluid","table","linear","chebyshev","spline"]
COLORS=["#E69F00","#56B4E9","#999999","#0072B2","#D55E00"]
MARKERS=["v","s","^","o","D"]
LABELS=[p[0] for p in LADDER]


def usd_symlog(ax):
    ax.set_yscale("symlog",linthresh=1000)
    # Keep a true zero tick without overlapping +/-100 labels in the linear core.
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(
        [-10.**i for i in range(9,2,-1)]+[0]+[10.**i for i in range(3,10)]))


def display(cid):
    return cid.replace("california_corridor","California").replace("western_multicluster","Western").replace("-n"," N=").replace("-k",", K=")


def stats(a,cfg):
    return statistics(a,cfg["bootstrap_seed"],cfg["bootstrap_replications"])


def predict(rows,model):
    if model["method"] in ("table","fluid"):
        return np.asarray([np.interp(r["time"],model["values"][r["hub"]][str(int(r["q"]))]["times"],
            model["values"][r["hub"]][str(int(r["q"]))]["values"]) for r in rows])
    return design(rows,model)@np.asarray(model["coefficients"])


def value_evaluation(cfg):
    results=[];predictions=[];calibration=[];seed_scores=[]
    for split in ["construction","validation","heldout_value"]:
        rows=real(averaged(read_json(OUTPUT/"targets"/(split+".json"))))
        for c in cfg["conditions"]:
            cid=key(c);group=[r for r in rows if r["condition"]==cid]
            for method in METHODS:
                model=read_json(OUTPUT/"models"/f"{cid}-{method}.json")
                pred=predict(group,model)
                results.append(dict(condition=cid,split=split,method=method,**prediction_metrics(group,pred)))
                for r,p in zip(group,pred):
                    predictions.append(dict(r,prediction=float(p),error=float(p-r["target"]),method=method,split=split))
                for seed in sorted(set(r["scenario_seed"] for r in group)):
                    ids=[i for i,r in enumerate(group) if r["scenario_seed"]==seed]
                    m=prediction_metrics([group[i] for i in ids],pred[ids])
                    seed_scores.append(dict(condition=cid,split=split,method=method,seed=seed,**m))
                if split=="heldout_value":
                    for axis,fn in [("time_bucket",lambda r:str(int(r["time"])//6)),
                                    ("hub",lambda r:r["hub"]),("supply_slot",lambda r:str(int(r["q"])))]:
                        gs=defaultdict(list)
                        for r,p in zip(group,pred):gs[fn(r)].append((r["target"],p))
                        for level,items in gs.items():
                            y,p=np.asarray(items).T
                            calibration.append(dict(condition=cid,method=method,axis=axis,level=level,n=len(y),
                                mean_target=float(y.mean()),mean_prediction=float(p.mean()),bias=float((p-y).mean()),
                                RMSE=float(np.sqrt(np.mean((p-y)**2)))))
    intervals=[]
    for cid,method in itertools.product([key(c) for c in cfg["conditions"]],METHODS):
        a=[r for r in seed_scores if r["condition"]==cid and r["method"]==method and r["split"]=="heldout_value"]
        for metric in ["MAE","RMSE","pairwise_ranking_accuracy","spatial_same_q_ranking_accuracy","sign_accuracy"]:
            x=[r[metric] for r in a if r[metric] is not None]
            if x:intervals.append(dict(condition=cid,method=method,metric=metric,
                estimand="mean per-scenario prediction score, not pooled-state policy CI",**stats(x,cfg)))
    write_csv(OUTPUT/"processed/value_prediction_metrics.csv",results)
    write_csv(OUTPUT/"processed/value_prediction_seed_scores.csv",seed_scores)
    write_csv(OUTPUT/"processed/value_prediction_CI.csv",intervals)
    write_csv(OUTPUT/"processed/value_predictions.csv",predictions)
    write_csv(OUTPUT/"processed/value_calibration.csv",calibration)
    return results,predictions,intervals


def policy_statistics(cfg,rows):
    groups=defaultdict(list)
    for r in rows:
        if r["status"]!="COMPLETED":raise RuntimeError("failed sample: inference blocked")
        groups[r["condition"],r["policy"]].append(r)
    summaries=[];pairs=[]
    metrics=["realized_profit","completed_fare","realized_cost","service_rate","actual_loaded_distance",
        "actual_empty_distance","actual_order_waiting","pending_at_T","daily_wall_seconds",
        "daily_solver_seconds","online_value_eval_seconds","build_seconds"]
    for (cid,policy),a in groups.items():
        assert len(a)==len(cfg["formal_seeds"]),"duplicate/missing replication"
        assert {r["seed"] for r in a}==set(cfg["formal_seeds"])
        for metric in metrics:summaries.append(dict(condition=cid,policy=policy,metric=metric,**stats([r[metric] for r in a],cfg)))
    comparisons=[(p,"P0-Myopic") for p in LABELS if p!="P0-Myopic"]
    comparisons += [(p,"P6-RH4") for p in ["P7-RH4-Fluid","P8-RH4-Chebyshev","P9-RH4-Spline","P10-RH8"]]
    comparisons += [(p,"P10-RH8") for p in ["P7-RH4-Fluid","P8-RH4-Chebyshev","P9-RH4-Spline"]]
    # Exploratory basis comparison; never used to select frozen models.
    comparisons += [("P4-Myopic-Chebyshev","P5-Myopic-Spline"),
                    ("P8-RH4-Chebyshev","P9-RH4-Spline")]
    for c in cfg["conditions"]:
        cid=key(c)
        for a,b in comparisons:
            A={r["seed"]:r for r in groups[cid,a]};B={r["seed"]:r for r in groups[cid,b]}
            assert set(A)==set(B)
            assert all(A[s]["scenario_hash"]==B[s]["scenario_hash"] for s in A)
            for metric in ["realized_profit","service_rate","actual_empty_distance","daily_wall_seconds","daily_solver_seconds"]:
                pairs.append(dict(condition=cid,policy_a=a,policy_b=b,comparison=a+" minus "+b,metric=metric,
                    **stats([A[s][metric]-B[s][metric] for s in sorted(A)],cfg)))
    ratios=[]
    def pair(cid,a,b):
        return next(r for r in pairs if r["condition"]==cid and r["policy_a"]==a and r["policy_b"]==b and r["metric"]=="realized_profit")
    for c in cfg["conditions"]:
        cid=key(c)
        for kind,dena,denb,policies in [
            ("horizon_compression","P10-RH8","P6-RH4",["P7-RH4-Fluid","P8-RH4-Chebyshev","P9-RH4-Spline"]),
            ("oracle_gap_capture","D-OracleH8","P0-Myopic",[p for p in LABELS if p not in ("P0-Myopic","D-OracleH8")])]:
            denominator=pair(cid,dena,denb)
            for p in policies:
                numerator=pair(cid,p,denb)
                valid=denominator["CI_low"]>0
                ratios.append(dict(condition=cid,kind=kind,policy=p,numerator_mean=numerator["mean"],
                    denominator_mean=denominator["mean"],denominator_CI_low=denominator["CI_low"],
                    denominator_CI_high=denominator["CI_high"],eligible=valid,
                    ratio=numerator["mean"]/denominator["mean"] if valid else None,
                    reason="ratio of paired mean gains; descriptive, not information/prediction accuracy" if valid else "denominator t95CI includes zero; no ratio"))
    write_csv(OUTPUT/"processed/policy_summary.csv",summaries)
    write_csv(OUTPUT/"processed/paired_policy_differences.csv",pairs)
    write_csv(OUTPUT/"processed/performance_ratios.csv",ratios)
    return summaries,pairs,ratios


def controlled_slice(cfg,baseline,data,prior):
    """Fixed hub/q/initial other-hub supply, only observation time varies."""
    from model_families.single.data import matching_state
    from .common import parameters,rho_levels
    from .build_mc_targets import resource_world
    from .simulation import simulate
    c=cfg["secondary_condition"];cid=key(c);seeds=cfg["construction_seeds"][:10]
    scenarios=[read_json(OUTPUT/"raw/datasets"/f"{cid}-s{s}.dataset.json")["scenarios"][0] for s in seeds]
    hub=sorted(scenarios[0]["nodes"])[0];q=1;T=scenarios[0]["horizon"]
    path=OUTPUT/"targets/runge_controlled_slice.json"
    identity=digest(dict(condition=cid,seeds=seeds,hub=hub,q=q,scenarios=scenarios,baseline=baseline))
    if path.exists():
        prior_result=read_json(path);assert prior_result["identity"]==identity;return prior_result
    started=time.perf_counter();rows=[]
    for s in scenarios:
        for t in range(T):
            initial,all_orders=matching_state(s,parameters(baseline,rho_levels(s,baseline)["rho_max"]),baseline["solver"])
            initial.t=t
            orders={oid:o for oid,o in all_orders.items() if o.book_time>t}
            profits=[]
            for units in [0,1]:
                state=resource_world(initial,hub,units)
                m,_,_=simulate(state,orders,first_begun=True)
                profits.append(m["realized_profit"])
            rows.append(dict(condition=cid,hub=hub,q=1,time=t,seed=s["seed"],target=profits[1]-profits[0]))
    means=[float(np.mean([r["target"] for r in rows if r["time"]==t])) for t in range(T)]+[0.]
    result=dict(identity=identity,condition=cid,hub=hub,q=q,seeds=seeds,rows=rows,times=list(range(T+1)),
        mean_targets=means,wall_seconds=time.perf_counter()-started,
        analysis_source_sha256=file_hash(__file__),state="fixed initial other-hub idle supply; no current backlog or commitments; empirical future after t; same two perturbations",
        note="MC reference is noisy and correlated across times; not true population expectation; construction-only separate shape diagnostic")
    with zipfile.ZipFile(OUTPUT/"targets/runge-source.snapshot.zip","w",zipfile.ZIP_DEFLATED) as z:
        z.write(__file__,"experiments/cfs_classical_value/analysis.py")
    result["diagnostic_source_snapshot_sha256"]=file_hash(OUTPUT/"targets/runge-source.snapshot.zip")
    atomic_json(path,result);write_csv(OUTPUT/"targets/runge_controlled_slice.csv",rows)
    return result


def runge_diagnostic(slice_,cfg):
    x=np.asarray(slice_["times"])/12-1;y=np.asarray(slice_["mean_targets"])
    node_ids=np.unique(np.rint(np.linspace(0,24,13)).astype(int));z=x[node_ids];target=y[node_ids]
    mono=np.polynomial.polynomial.polyvander(z,12);a=np.linalg.lstsq(mono,target,rcond=None)[0]
    same_cheb=np.polynomial.chebyshev.chebvander(z,12)
    same_b=np.linalg.solve(same_cheb,target)
    degree=read_json(OUTPUT/"models"/(slice_["condition"]+"-chebyshev.json"))["degree"]
    cheb=np.polynomial.chebyshev.chebvander(z,degree)
    b=np.linalg.solve(cheb.T@cheb+np.eye(degree+1),cheb.T@target)
    knots=np.r_[np.repeat(-1.,4),np.linspace(-1,1,6)[1:-1],np.repeat(1.,4)]
    B=BSpline(knots,np.eye(len(knots)-4),3);X=B(z);D=np.diff(np.eye(X.shape[1]),n=2,axis=0)
    c=np.linalg.solve(X.T@X+10*D.T@D+1e-6*np.eye(X.shape[1]),X.T@target)
    funcs=dict(monomial=lambda u:np.polynomial.polynomial.polyval(u,a),
        cheb_same_degree=lambda u:np.polynomial.chebyshev.chebval(u,same_b),
        chebyshev=lambda u:np.polynomial.chebyshev.chebval(u,b),spline=lambda u:B(u)@c)
    results=[];grid=np.linspace(-1.15,1.15,400)
    values={k:fn(grid).tolist() for k,fn in funcs.items()}
    for name,fn in funcs.items():
        e=fn(x)-y
        matrix=mono if name=="monomial" else same_cheb if name=="cheb_same_degree" else cheb if name=="chebyshev" else X
        results.append(dict(method=name,degree=12 if name in ("monomial","cheb_same_degree") else degree if name=="chebyshev" else 3,
            RMSE=float(np.sqrt(np.mean(e**2))),max_abs_error=float(np.max(abs(e))),
            endpoint_RMSE=float(np.sqrt(np.mean(e[[0,1,-2,-1]]**2))),condition_number=float(np.linalg.cond(matrix)),
            left_extrapolation=float(fn(-1.1)),right_extrapolation=float(fn(1.1))))
    write_csv(OUTPUT/"processed/runge_numerical_diagnostics.csv",results)
    # Analytic Runge control isolates node design from representation.
    eq=np.linspace(-1,1,17);ch=np.cos((2*np.arange(1,18)-1)*np.pi/34)
    f=lambda u:1/(1+25*u*u)
    Am=np.polynomial.polynomial.polyvander(eq,16);Ac=np.polynomial.chebyshev.chebvander(eq,16)
    em=np.linalg.solve(Am,f(eq));ec=np.linalg.solve(Ac,f(eq))
    cc=np.polynomial.chebyshev.chebfit(ch,f(ch),16)
    inside=np.linspace(-1,1,1001)
    analytic=[]
    for name,pred,matrix in [
        ("equally-spaced monomial interpolation",np.polynomial.polynomial.polyval(inside,em),Am),
        ("SAME equally-spaced Chebyshev interpolation",np.polynomial.chebyshev.chebval(inside,ec),Ac),
        ("Chebyshev-nodes Chebyshev interpolation",np.polynomial.chebyshev.chebval(inside,cc),np.polynomial.chebyshev.chebvander(ch,16))]:
        analytic.append(dict(method=name,max_abs_error=float(np.max(abs(pred-f(inside)))),condition_number=float(np.linalg.cond(matrix))))
    atomic_json(OUTPUT/"processed/runge_canonical.json",analytic)
    atomic_json(OUTPUT/"processed/runge_plotting_config.json",dict(
        controlled_condition=slice_["condition"],hub=slice_["hub"],q=slice_["q"],
        fixed_state=slice_["state"],scenario_seeds=slice_["seeds"],
        mean_reference_sha256=file_hash(OUTPUT/"targets/runge_controlled_slice.json"),
        fitting_time_periods=node_ids.tolist(),monomial_degree=12,
        monomial_coefficients=a.tolist(),chebyshev_degree=degree,chebyshev_ridge_penalty=1,
        same_nodes_same_degree_chebyshev_coefficients=same_b.tolist(),
        same_nodes_same_degree_curve_max_difference=float(np.max(abs(funcs["monomial"](grid)-funcs["cheb_same_degree"](grid)))),
        chebyshev_coefficients=b.tolist(),spline_degree=3,spline_knots=knots.tolist(),
        spline_second_difference_penalty=10,spline_coefficients=c.tolist(),
        canonical_function="1/(1+25*z*z)",canonical_degree=16,canonical_equal_nodes=eq.tolist(),
        canonical_chebyshev_nodes=ch.tolist(),diagnostic_extrapolation_domain=[-1.15,1.15],
        note="Diagnostic only; none of these time-only diagnostic refits enters formal policy selection"))
    return dict(grid=grid,values=values,results=results,inside=inside,analytic_true=f(inside),
        analytic_eq=np.polynomial.polynomial.polyval(inside,em),analytic_ch=np.polynomial.chebyshev.chebval(inside,cc),analytic=analytic)


def audit_runs(cfg,rows):
    """Replay physical transitions without any MILP; regenerate segment events."""
    from model_families.single.data import matching_state
    from model_families.single.runner import state_snapshot
    from model_families.single.accounting import cooldown_tick,realized_ledger
    from experiments.cfs_dynamic_lookahead.case_study import replay_plan
    audits=[];total_steps=0;source=read_json(OUTPUT/"manifest.json")
    for i,r in enumerate(rows):
        target=OUTPUT/"raw/runs"/r["id"];spec=read_json(target/"spec.json")
        assert r["code_hash"]==source["code_hash"] and digest(spec["scenario"])==r["scenario_hash"]
        assert file_hash(target/"trace.json")==r["trace_hash"]
        sim,orders=matching_state(spec["scenario"],spec["parameters"],spec["solver"])
        steps=read_json(target/"trace.json");decision=[];seen=set()
        for row in steps:
            t=row["period"]
            if row.get("drain"):
                sim.pool={}
                assert row["plan"]["assignments"]==row["plan"]["repositions"]==row["plan"]["routes"]==[]
                assert digest(state_snapshot(sim))==row["plan"]["state_hash"]
                cooldown_tick(sim)
            else:
                sim.begin_period([o for o in orders.values() if o.book_time==t])
                assert digest(state_snapshot(sim))==row["plan"]["state_hash"]
                assert all(row["invariants"].values())
                for a in row["plan"]["assignments"]:
                    oid=a["order_id"];assert oid not in seen and orders[oid].book_time<=t
                    seen.add(oid)
                preview=row["plan"]["solver"].get("window",{}).get("preview_order_ids",[])
                if spec["policy"]!="OracleH8":assert not preview
                else:assert all(t<orders[oid].book_time<=t+8 for oid in preview)
                sim.commit_plan(replay_plan(sim,row));decision.append(row)
            assert state_snapshot(sim)==row["after"]
        final=realized_ledger(sim,orders,decision)
        assert abs(final["realized_profit"]-r["realized_profit"])<1e-6
        assert final["unfinished_committed"]==0 and r["assigned"]==r["delivered"]
        out=OUTPUT/"processed/execution_events"/(r["id"]+".json")
        atomic_json(out,dict(source_trace_sha256=r["trace_hash"],metrics=final,
            vehicles=[dict(vehicle_id=v.id,segments=v.segments,pickup_times=v.pickups,delivery_times=v.deliveries) for v in sim.vehs]))
        audits.append(dict(id=r["id"],status="PASSED",steps=len(steps),event_file_hash=file_hash(out)))
        total_steps+=len(steps)
        if (i+1)%100==0:print(f"Trace replay audit {i+1}/{len(rows)}",flush=True)
    result=dict(status="PASSED",runs=len(rows),steps=total_steps,checks=audits,
        tolerance=1e-6,replay="independent feasibility compilation then recorded monetary scalars within tolerance, no physical-state overwrites")
    atomic_json(OUTPUT/"processed/trace_audit.json",result);return result


def savefig(fig,name):
    fig.tight_layout()
    for extension in ["png","pdf","svg"]:
        fig.savefig(REPORT/"figures"/(name+"."+extension),dpi=600,bbox_inches="tight")
    plt.close(fig)


def figures(cfg,summary_rows,pairs,ratios,predictions,slice_,runge):
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":8,"axes.spines.top":False,
        "axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    (REPORT/"figures").mkdir(parents=True,exist_ok=True)
    conditions=[key(c) for c in cfg["conditions"]]
    train=real(averaged(read_json(OUTPUT/"targets/construction.json")))
    models={cid:{m:read_json(OUTPUT/"models"/f"{cid}-{m}.json") for m in METHODS} for cid in conditions}
    profiles={}
    for cid in conditions:
        a=[r for r in train if r["condition"]==cid]
        profiles[cid]={f:float(np.median([r[f] for r in a])) for f in ["remaining","other_idle","incoming","backlog","expected","q"]}
        profiles[cid].update(hub=sorted(models[cid]["fluid"]["values"])[0],q=1.,time=12)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.5))
    for ax,cid in zip(axes.flat,conditions):
        profile=profiles[cid];hub=profile["hub"]
        a=[r for r in train if r["condition"]==cid and r["hub"]==hub and r["q"]==1]
        ts=sorted(set(r["time"] for r in a));ys=[];es=[]
        for t in ts:
            x=np.array([r["target"] for r in a if r["time"]==t])
            ys.append(x.mean());es.append(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
        ax.errorbar(ts,ys,yerr=es,fmt="k.",capsize=2,label="MC mean, t95 CI")
        grid=list(range(25));rows=[dict(profile,time=t,remaining=(24-t)/24,expected=(24-t)/24*profile["expected"]/max(profile["remaining"],1e-6)) for t in grid]
        for m,color,mark in zip(METHODS,COLORS,MARKERS):
            values=predict(rows,models[cid][m]);values[-1]=0
            ax.plot(grid,values,color=color,marker=mark,markevery=5,markersize=3,label=m)
        usd_symlog(ax)
        ax.set(title=display(cid)+"\n"+hub+", q=1",xlabel="Release time (period)",ylabel="Marginal profit (USD, symlog)")
    axes.flat[0].legend(fontsize=6,ncol=2)
    savefig(fig,"01_mc_value_time_hub")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.5))
    for ax,cid in zip(axes.flat,conditions):
        fluid=models[cid]["fluid"]["values"];hubs=sorted(fluid)
        image=ax.imshow([fluid[h]["1"]["values"] for h in hubs],aspect="auto",origin="lower",cmap="cividis")
        ax.set_yticks(range(len(hubs)),hubs);ax.set(title=display(cid),xlabel="Release time",ylabel="Hub")
        fig.colorbar(image,ax=ax,label="LP dual (USD)")
    savefig(fig,"02_fluid_dual_heatmap")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5))
    for ax,cid in zip(axes.flat,conditions):
        m=models[cid]["chebyshev"];offset=len(m["hubs"]);beta=np.asarray(m["coefficients"])
        for i,name in enumerate(FEATURES):
            b=m["bases"][name]
            width=b["degree"] if b["kind"]=="chebyshev" else 1
            if b["kind"]=="chebyshev":
                ax.plot(range(1,width+1),np.maximum(abs(beta[offset:offset+width]),1e-6),
                    color=(COLORS+["#000000"])[i],marker=(MARKERS+["x"])[i],
                    linestyle=["-","--",":","-.","--",":"][i],markersize=3,label=name)
            offset+=width
        ax.set(title=display(cid)+f', degree={m["degree"]}',xlabel="Basis degree",ylabel="Absolute coefficient (USD)",yscale="log")
    axes.flat[0].legend(fontsize=6,ncol=2)
    savefig(fig,"03_chebyshev_coefficient_decay")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8))
    ax=axes[0,0];ax.plot(runge["inside"],runge["analytic_true"],"k",label="True Runge function")
    ax.plot(runge["inside"],runge["analytic_eq"],"--",color=COLORS[4],label="Equal nodes, degree 16")
    ax.plot(runge["inside"],runge["analytic_ch"],":",color=COLORS[3],label="Chebyshev nodes, degree 16")
    ax.set_yscale("symlog",linthresh=1)
    ax.set(title="Node design control (analytic)",xlabel="z",ylabel="f(z), symlog");ax.legend(fontsize=6)
    ax=axes[0,1];ax.plot(slice_["times"],slice_["mean_targets"],"k.",label="Paired MC reference")
    for name,color,style in zip(["monomial","cheb_same_degree","chebyshev","spline"],
            [COLORS[2],"#000000",COLORS[3],COLORS[4]],["-","--","-.",":"]):
        ax.plot((runge["grid"]+1)*12,runge["values"][name],color=color,linestyle=style,label=name)
    ax.axvspan(-2,0,color="grey",alpha=.1);ax.axvspan(24,26,color="grey",alpha=.1)
    usd_symlog(ax)
    ax.set(title="Fixed hub / q / other supply",xlabel="Observation time",ylabel="Marginal profit (USD, symlog)",xlim=(-1.8,25.8))
    ax.legend(fontsize=6)
    ax=axes[1,0];a=runge["results"]
    ticks=np.arange(len(a));names=["Mono12","Cheb12",f"Cheb{a[2]['degree']}","Spline3"]
    ax.bar(ticks-.17,[r["endpoint_RMSE"] for r in a],width=.34,color=COLORS[3],hatch="/",label="Endpoint RMSE")
    ax.bar(ticks+.17,[r["max_abs_error"] for r in a],width=.34,color=COLORS[4],hatch=".",label="Max absolute error")
    ax.set_xticks(ticks,names,fontsize=7)
    ax.set(title="MC-reference boundary / max error",ylabel="Error (USD)");ax.legend(fontsize=6)
    ax=axes[1,1];ax.bar(names,[r["condition_number"] for r in a],color=[COLORS[2],"#000000",COLORS[3],COLORS[4]])
    ax.set(title="Basis conditioning",ylabel="Design condition number",yscale="log")
    savefig(fig,"04_runge_stability")
    cid=conditions[1];model=models[cid]["spline"];profile=profiles[cid]
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.8))
    for ax,name in zip(axes.flat,["remaining","other_idle","incoming","backlog","expected","q"]):
        b=model["bases"][name]
        if b["kind"]=="spline":x=np.linspace(b["lower"],b["upper"],80)
        else:x=np.linspace(profile[name]-1,profile[name]+1,80)
        rows=[dict(profile,**{name:float(v)}) for v in x]
        y=predict(rows,model)-predict([profile],model)[0]
        ax.plot(x,y,color=COLORS[4]);ax.axhline(0,color="grey",lw=.5)
        ax.set(xlabel=name,ylabel="Partial difference (USD)",title=name)
    fig.suptitle(display(cid)+", spline additive components",y=1.01)
    savefig(fig,"05_spline_components")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.5))
    for ax,cid in zip(axes.flat,conditions):
        for method,color,mark in zip(["chebyshev","spline"],[COLORS[3],COLORS[4]],["o","x"]):
            a=[r for r in predictions if r["condition"]==cid and r["split"]=="heldout_value" and r["method"]==method]
            ax.scatter([r["target"] for r in a],[r["prediction"] for r in a],s=7,color=color,marker=mark,alpha=.25,label=method)
        lo,hi=ax.get_xlim();ax.plot([lo,hi],[lo,hi],"k--",lw=.8)
        ax.set(title=display(cid),xlabel="Held-out paired MC marginal (USD)",ylabel="Prediction (USD)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    axes.flat[0].legend(fontsize=7)
    savefig(fig,"06_heldout_prediction")
    def sumrow(cid,p,metric="realized_profit"):
        return next(r for r in summary_rows if r["condition"]==cid and r["policy"]==p and r["metric"]==metric)
    short=["M0","M+F","M+Tbl","M+Lin","M+Cheb","M+Spl","RH4","4+F","4+Cheb","4+Spl","RH8","Oracle (diag.)"]
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8))
    for ax,cid in zip(axes.flat,conditions):
        a=[sumrow(cid,p) for p in LABELS]
        ax.errorbar(range(12),[r["mean"]/1000 for r in a],yerr=[(r["CI_high"]-r["mean"])/1000 for r in a],fmt="o",capsize=2,color="#0072B2")
        ax.scatter([11],[a[-1]["mean"]/1000],marker="D",color="#999999",label="Oracle diagnostic")
        ax.set_xticks(range(12),short,rotation=65,ha="right",fontsize=6)
        ax.set(title=display(cid),ylabel="Full-cooldown profit (USD 1,000)")
    savefig(fig,"07_policy_ladder_profit")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.3))
    for ax,cid in zip(axes.flat,conditions):
        a=[r for r in pairs if r["condition"]==cid and r["policy_b"]=="P6-RH4" and r["metric"]=="realized_profit"]
        ax.errorbar(range(len(a)),[r["mean"] for r in a],yerr=[r["CI_high"]-r["mean"] for r in a],fmt="s",capsize=3,color=COLORS[4])
        ax.axhline(0,color="k",lw=.7)
        ax.set_xticks(range(len(a)),[r["policy_a"].replace("P7-","").replace("P8-","").replace("P9-","").replace("P10-","") for r in a],rotation=20,ha="right",fontsize=6)
        ax.set(title=display(cid),ylabel="Paired gain vs RH4 (USD), t95 CI")
    savefig(fig,"08_terminal_value_vs_RH8")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.3))
    for ax,cid in zip(axes.flat,conditions):
        a=[r for r in ratios if r["condition"]==cid and r["kind"]=="oracle_gap_capture"]
        if all(r["eligible"] for r in a):
            ax.bar(range(len(a)),[100*r["ratio"] for r in a],color=COLORS[3])
            ax.set_xticks(range(len(a)),[r["policy"].split("-",1)[1] for r in a],rotation=60,ha="right",fontsize=6)
        else:ax.text(.5,.5,"Denominator not clearly positive\nRatio not reported",transform=ax.transAxes,ha="center")
        ax.axhline(0,color="k",lw=.5);ax.set(title=display(cid),ylabel="Performance gap capture (%)\n(descriptive ratio)")
    savefig(fig,"09_oracle_gap_capture")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.3))
    annotations=[]
    for ax,cid in zip(axes.flat,conditions):
        items=[]
        for i,p in enumerate(LABELS):
            runtime=sumrow(cid,p,"daily_wall_seconds");profit=sumrow(cid,p)
            ax.scatter(runtime["mean"],profit["mean"]/1000,marker="D" if p=="D-OracleH8" else "o",s=22,color="#999999" if p=="D-OracleH8" else "#0072B2")
            items.append(("D (diag.)" if p=="D-OracleH8" else f"P{i}",
                (runtime["mean"],profit["mean"]/1000)))
        ax.set(title=display(cid),xlabel="Daily wall time (s)",ylabel="Profit (USD 1,000)")
        ax.margins(x=.12,y=.12)
        annotations.append((ax,items))
    fig.tight_layout();fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax,items in annotations:
        occupied=[]
        for label,point in items:
            annotation=ax.annotate(label,point,fontsize=6,xytext=(3,4),textcoords="offset points",
                arrowprops=dict(arrowstyle="-",lw=.4,color="grey"))
            candidates=[]
            for offset in [(3,4),(3,-12),(-18,4),(-18,-12),(3,14),(-24,14),
                           (3,-23),(-24,-23),(12,4),(-32,4),(12,23),(-32,23)]:
                annotation.set_position(offset)
                annotation.update_positions(renderer)
                box=matplotlib.text.Text.get_window_extent(annotation,renderer).expanded(1.15,1.15)
                outside=sum([box.x0<ax.bbox.x0,box.x1>ax.bbox.x1,box.y0<ax.bbox.y0,box.y1>ax.bbox.y1])
                score=1000*outside+sum(box.overlaps(old) for old in occupied)
                candidates.append((score,offset,box))
                if score==0:break
            _,offset,box=min(candidates,key=lambda item:item[0])
            annotation.set_position(offset);occupied.append(box)
    fig.text(.5,-.03,"P0 M0 | P1 M+Fluid | P2 M+Table | P3 M+Linear | P4 M+Cheb | P5 M+Spline\n"
        "P6 RH4 | P7 RH4+Fluid | P8 RH4+Cheb | P9 RH4+Spline | P10 RH8 | D Oracle diagnostic only",
        ha="center",fontsize=7)
    savefig(fig,"10_runtime_profit_frontier")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5))
    for ax,cid in zip(axes.flat,conditions):
        fleet=next(c["fleet"] for c in cfg["conditions"] if key(c)==cid)
        rows=[dict(profiles[cid],q=float(q)) for q in range(1,fleet+1)]
        for m,color,mark in zip(METHODS,COLORS,MARKERS):
            ax.plot(range(1,fleet+1),predict(rows,models[cid][m]),color=color,marker=mark,markersize=3,label=m)
        usd_symlog(ax)
        ax.set_xticks(range(1,fleet+1))
        ax.set(title=display(cid)+", release time 12",xlabel="q-th marginal free vehicle",ylabel="Marginal value (USD, symlog)")
    axes.flat[0].legend(fontsize=6,ncol=2)
    savefig(fig,"11_supply_slot_shape")
    atomic_json(OUTPUT/"processed/plotting_config.json",dict(font="DejaVu Serif",double_column_inches=7.2,
        dpi=600,formats=["PDF","SVG","PNG"],colorblind_palette=COLORS,markers=MARKERS,
        CI="scenario-level Student-t95; |skew|>1 paired 10000 bootstrap saved in tables",
        symmetric_log_axes={"01_mc_value_time_hub":1000,"04_runge_stability_MC":1000,
            "04_runge_stability_analytic":1,"11_supply_slot_shape":1000},
        symlog_units="USD except analytic f(z); linear below threshold, logarithmic outside; all values retained",
        coefficient_styles=dict(colors=COLORS+["#000000"],markers=MARKERS+["x"],linestyles=["-","--",":","-.","--",":"]),
        diagnostic="D/Oracle has future information; diagnostic only, not deployable",
        node_design="discrete projected Chebyshev periods recorded in manifest; basis alone does not eliminate Runge interpolation",
        plots=[p.name for p in sorted((REPORT/"figures").glob("*.png"))],
        input_hashes={p.relative_to(OUTPUT).as_posix():file_hash(p) for p in (OUTPUT/"processed").glob("*.csv")}))


def secondary_comparison(cfg,baseline,manifest):
    from .build_mc_targets import build_targets
    c=cfg["secondary_condition"];cid=key(c)
    data={(c["region"],c["orders"],c["fleet"],s):read_json(OUTPUT/"raw/datasets"/f"{cid}-s{s}.dataset.json") for s in cfg["secondary_seeds"]}
    priors=read_json(OUTPUT/"models/priors.json")
    if not (OUTPUT/"targets/secondary_Myopic.json").exists():
        info=build_targets(cfg|{"conditions":[c]},baseline,data,priors,"secondary_Myopic",cfg["secondary_seeds"],"myopic",[0,12,23])
    else:info=dict(reused=True)
    M=real(averaged(read_json(OUTPUT/"targets/secondary_Myopic.json")))
    R=real(averaged(read_json(OUTPUT/"targets/secondary_RH4.json")))
    keys=lambda r:(r["scenario_seed"],r["time"],r["hub"],r["q"])
    byM={keys(r):r for r in M};byR={keys(r):r for r in R}
    assert set(byM)==set(byR)
    rows=[]
    for k in sorted(byM):
        m,r=byM[k],byR[k];assert m["state_hash"]==r["state_hash"] and m["future_hash"]==r["future_hash"]
        rows.append(dict(seed=k[0],time=k[1],hub=k[2],q=k[3],myopic_target=m["target"],
            RH4_target=r["target"],difference=r["target"]-m["target"]))
    write_csv(OUTPUT/"processed/continuation_policy_targets.csv",rows)
    means=[np.mean([r["difference"] for r in rows if r["seed"]==s]) for s in cfg["secondary_seeds"]]
    return dict(count=len(rows),summary=stats(means,cfg),generation=info,
        estimand="mean fixed diagnostic query portfolio marginal difference per seed; only five construction seeds, not formal policy effect")


def residual_diagnostics(predictions,cfg):
    rows=[]
    for c in cfg["conditions"]:
        cid=key(c)
        for method in ["chebyshev","spline"]:
            a=[r for r in predictions if r["condition"]==cid and r["method"]==method and r["split"]=="heldout_value"]
            error=np.array([r["error"] for r in a])
            for name,feature in [("supply*demand",lambda r:r["q"]*r["expected"]),
                                 ("incoming*demand",lambda r:r["incoming"]*r["expected"])]:
                x=np.array([feature(r) for r in a])
                corr=float(np.corrcoef(x,error)[0,1]) if np.std(x)>0 else None
                rows.append(dict(condition=cid,method=method,interaction=name,residual_Pearson=corr,
                    interpretation="post-fit descriptive diagnostic, no model tuned on test"))
    write_csv(OUTPUT/"processed/residual_interactions.csv",rows)
    return rows


def report(cfg,manifest,summaries,pairs,ratios,value_metrics,secondary,audit,runge,slice_):
    def s(cid,p):
        return next(r for r in summaries if r["condition"]==cid and r["policy"]==p and r["metric"]=="realized_profit")
    def d(cid,a,b="P0-Myopic"):
        return next(r for r in pairs if r["condition"]==cid and r["policy_a"]==a and r["policy_b"]==b and r["metric"]=="realized_profit")
    def ci(r):
        return f'{r["mean"]:+,.2f} [{r["CI_low"]:+,.2f}, {r["CI_high"]:+,.2f}]'
    conditions=[key(c) for c in cfg["conditions"]]
    rows=read_json(OUTPUT/"processed/formal_policies.json")
    modelsel=read_json(OUTPUT/"models/selection.json")
    prediction_rows=[r for r in value_metrics if r["split"]=="heldout_value"]
    suites=list(ET.parse(OUTPUT/"processed/pytest-full.xml").getroot().iter("testsuite"))
    qa={name:sum(int(s.get(name,0)) for s in suites) for name in ["tests","failures","errors","skipped"]}
    assert not qa["failures"] and not qa["errors"]
    repetition=read_json(OUTPUT/"processed/independent_reproduction.json")
    assert repetition["status"]=="PASSED" and repetition["identical_physical_trajectory"]
    atomic_json(OUTPUT/"processed/qa_validation.json",dict(test_suite=qa,
        junit_sha256=file_hash(OUTPUT/"processed/pytest-full.xml"),independent_reproduction=repetition,
        numerical_code_hash=manifest["code_hash"],environment=dict(PYTHONHASHSEED="0")))
    lines=[
        "# 第三次数值实验：经典边际价值的 Chebyshev 与 Spline 逼近",
        "",
        "所有策略收益均为 **full-cooldown realized profit**。未使用 neural、deep、DRL、SAC 或其他强化学习。",
        "正式模型只使用当前在线信息与构造集统计先验；OraclePreview H8 仅为不可部署的信息放松诊断，不是全日最优上界。",
        "",
        "## 1. 完成情况与研究问题",
        "",
        f'已完成 {manifest["formal"]["count"]} 次正式策略运行（四工况 × 12策略/诊断 × 40场景），失败 {manifest["formal"]["failures"]}。每条件 40 paired seeds 3000–3039；pilot 240次只作实现检查，不混入正式CI。',
        f'构造集配对边际目标 {manifest["construction"]["paired_target_count"]} 条，验证集 {manifest["validation"]["paired_target_count"]} 条；解析终端锚点另存，不当作独立MC样本。held-out目标 {manifest["heldout_value"]["paired_target_count"]} 条来自12个正式场景。',
        "问题是低维价值的正则性、局部结构以及 tail value 能否替代部分显式前视。不能仅凭最低拟合误差宣布方法胜出，不能从Oracle潜力直接推出DRL必要性。",
        "",
        "## 2. 冻结基线与数据",
        "",
        "完整阅读第一、第二次报告；原 network-rho、CFS adapter、日终核算、Myopic、RH-Supply H4/H8、runner源码均保持原hash。四工况seed1010的Myopic/RH4/RH8共12次历史利润复现误差均小于1e-6，详见manifests/baseline_regression.json。旧报告和旧结果未覆盖。",
        f'CFS只读源 SHA256：`{manifest["source"]["sha256"]}`。原WGT_FACTOR加权无放回抽样、双向OD配平、metro坐标/网络、capacity40、成本/运费、时间生成规则和确定性fleet placement保持不变。',
        "CFS是货运记录。货值不是运费；booking/ready/deadline是生成假设；geometry-informed network不是观测高速公路。各seed可能抽中相同原shipment，仿真CI不含调查误差，不能外推全国商业效果。",
        "本次只研究California80/3、Western40/8、Western80/3、Western20/8，**rho固定第二次的rho_max**。没有重跑压力map，也未按正式结果选择半径。第二次1010–1039只作历史基线，不参与价值拟合或新out-of-sample主张。",
        "",
        "## 3. 种子、目标和边界",
        "",
        "construction：2000–2039（40/工况）；validation：2100–2114（15/工况）；formal policy：3000–3039（40/工况）；held-out value：3000–3011（12/工况）。三个split互不重叠。policy_seed=0，无随机训练seed。经典岭/样条拟合是离线数值回归，不是强化学习。",
        f'主动时点采用7个连续Chebyshev根节点映射到0..23，四舍五入并去重，**实际整数节点为 {manifest["actual_projected_chebyshev_nodes"]}**。这是控制观察时刻、但车辆/订单状态来自自然Myopic轨迹的设计，绝不是连续精确Chebyshev插值网格。',
        "每个节点轮换查询hub；第0、3、6个节点增加相邻hub，支持同状态下空间排序检查。K3计算q=1/2/3；K8计算q=1/2/3/4/6/8；q5、7的表格估值按q插值。两条MC路径先分别保存J(q)、J(q−1)，再平均成拟合观测，不将两路径或不同q当独立统计replication。",
        "",
        r"`Y(s,t,u,q)=J^{continuation}(s^{-u}+q辆自由车)-J^{continuation}(s^{-u}+(q-1)辆自由车)`。",
        "",
        "s^-u只移除查询hub当前自由idle车，保留其他idle车和全部承诺；车辆ID重新连续编号。两世界使用相同网络、成本、需求实现、continuation与seed，并执行到全部承诺结束。主目标continuation=原Myopic V0，不是全日最优值。增加车可能改变贪心决策，因此边际值无需强制非负或递减。",
        "第一条未来路径是该构造/验证/held-out场景的剩余订单；第二条从**同split**固定offset donor取booking>t的剩余未来，并重命名ID避免撞单。已观察当前订单保留。此经验重采样没有严格条件化剩余订单总数，也不是正确现实需求后验。donor复用导致价值误差CI仅为探索性的scenario-block口径；正式策略的40个场景CI与此分开。",
        "当前pool、未揭示订单不能进入在线V。统计expected demand只来自构造集均匀booking率及OD/货重/运费均值；正式未来记录仅在调度器中按booking释放，Oracle专用接口除外。",
        "边界是**新自由资源在T时不能发起任何新控制**，不是原承诺价值为零：两世界commitment相同，冷却收入/费用会抵消，故V(T,u,q)=0成立。T前发起、T后完成的任务仍完整记收入/费用。raw拟合与terminal-anchored候选都保存；anchored只在各hub、q1、其他特征中位数组合精确约束零，不能冒称对所有additive组合都连续趋零。所有provider在time>=T使用准确自由资源零门限。",
        "",
        "## 4. 统一接口、线性目标与模型",
        "",
        "`ValueProvider.marginal_release_value(time, hub, supply_slot, online_state_summary)` 返回固定数值；solver不认识具体Chebyshev/spline表达式。候选释放事件在求解前评价，只有线性常系数进入目标。",
        "同(time,hub)使用binary z_q，sum z_q=该处sink release流量，并强制z_q≤z_(q−1)。因此两辆只能取q1+q2，不能重复q1，也不能因为q3特别高而跳过q2。测试包含非单调/负系数。每辆车只在窗口最后进sink时领取tail，不能在中间hub反复领取。RH4内已完成再复用的车辆只计最终资源。",
        "M+V用H1窗口的同一单期可行集；V0仍调用原Myopic。RH4+V只改独立value planner，原RH4/RH8不变；窗口[t,min(t+H,T))，k=1；路线macro-action限制沿用第二次，不是所有逐期动作序列的精确展开。",
        "",
        "| 方法 | 表示与限制 |",
        "|---|---|",
        "| Zero | 原Myopic / 原RH-Supply，V=0 |",
        "| FluidDual | 构造集需求的fractional时空LP shadow-price surface；在q−1供给处取dual，离线初始其他hub供给冻结；不响应当前backlog；放松整车/锚点，允许空调位。它是简单、乐观surrogate，对相同MC离散realized目标测误差，而不是宣称LP dual恰好等于该目标 |",
        "| MC Table | time/hub/q配对MC均值；时间与缺失整数q线性插值；忽略在线backlog等 |",
        "| Linear Ridge | hub截距与标准化remaining、other idle、incoming、backlog、expected demand、q |",
        "| Chebyshev | 同六特征的additive Chebyshev LS/ridge；remaining映射[-1,1]；p=2..6；无full tensor |",
        "| Cubic Spline/GAM | 六特征additive cubic B-spline；二阶系数差分惩罚；ordinary/natural；4/6/8/10 requested internal knots，quantile去重后actual数量逐特征记录 |",
        "",
        "所有回归候选使用同一paired realized marginal标签。Natural spline用端点二阶导数零约束的null-space；不是exact cubic interpolation。Spline块中心化，微小isotropic ridge稳定非识别offset；部分设计仍rank-deficient、正则化条件数很高，必须谨慎解释原始系数。",
        "online摘要包含查询hub之外idle、已知最终idle release incoming、当前hub-origin backlog load、构造先验remaining demand、q。RH-terminal在当前摘要上评价未来释放时刻，**没有联合更新候选动作之后的backlog/其他hub supply**；属于冻结摘要tail近似。未来估计状态与自然构造状态存在distribution shift，不能将策略退化直接归因于函数类失败。不得把它称为完整post-decision value或准确future-demand预测。",
        "",
        "## 5. 验证选择与数值稳定性",
        "",
        "模型只按validation RMSE最小选择，MAE、ranking、索引用于tie-break；所有raw/anchored和ordinary/natural候选在models/candidates/。选定后models/selection.json及model_freeze.json保存hash，代码先冻结再生成formal利润。pilot使用验证集前5seed，只检查运行和预算；没有删除低收益方法，完整ladder保留到formal。",
        "",
        "| 工况 | Cheb p / lambda / boundary | Spline requested knots / lambda / boundary | val RMSE Cheb / Spline | 正则矩阵cond Cheb / Spline |",
        "|---|---|---|---:|---:|"
    ]
    for cid in conditions:
        ch,sp=modelsel[cid]["chebyshev"],modelsel[cid]["spline"]
        lines.append(f'| {display(cid)} | {ch["degree"]} / {ch["penalty"]} / {ch["terminal"]} | {sp["knots"]} / {sp["penalty"]} / {sp["boundary"]}, {sp["terminal"]} | {ch["RMSE"]:,.2f} / {sp["RMSE"]:,.2f} | {ch["condition_number"]:.3g} / {sp["condition_number"]:.3g} |')
    lines += [
        "",
        "高q多项式可只有3或6个distinct样本值；time、expected等也相关。rank、EDF、regularized conditioning、roughness和actual knots在候选/最终模型中。不能仅从某一组不唯一additive系数下降推断真实V spectral smooth；图3是所选正则化表示的系数，不是唯一经济分解。未在测试集上增设interaction或调惩罚。Fluid使用q−1处的infinitesimal LP dual，可能显著高于整辆车的finite difference；需先对统一paired target校准，不能把本次简化Fluid失败概括为所有流体价值方法失败。",
        "",
        "### Runge专项",
        "",
        f'独立控制切片固定Western40/8、hub {slice_["hub"]}、q1、初始其他hub供给及零当前backlog，只有观察时点变化；10个构造seed，paired未来，完整cooldown。MC参考均值仍有噪声与跨时点相关，不是真实无噪population函数。',
        "在同一13个离散等距节点的MC参考上，对比degree12 monomial与同degree12 Chebyshev插值（隔离basis影响），再对比低阶Cheb ridge及penalized cubic spline（正则化模型对照）；未用于正式policy。前两者表示同一多项式，均可能产生端点振荡，不能将低阶ridge的改善全部归因于换basis。外推[-1.15,1.15]只用于诊断，正式provider对特征clip、T准确gate。max error、RMSE、endpoint RMSE、condition number与±1.1外推保存于runge_numerical_diagnostics.csv。",
        "另用解析Runge函数1/(1+25z²)隔离节点设计：**同等距节点、同degree的monomial和Chebyshev exact interpolation给出同一个数学多项式**，换basis只能改善conditioning，不能消除节点引发的振荡。Chebyshev nodes改善节点设计；正式方法采用低阶LS/ridge而非高阶等距exact interpolation。",
        "",
        "| MC切片方法 | RMSE | max abs error | endpoint RMSE | design cond |",
        "|---|---:|---:|---:|---:|"
    ]
    for r in runge["results"]:
        lines.append(f'| {r["method"]} | {r["RMSE"]:,.2f} | {r["max_abs_error"]:,.2f} | {r["endpoint_RMSE"]:,.2f} | {r["condition_number"]:.3g} |')
    lines += ["","## 6. 独立held-out价值预测","","以下每条件12个scenario，MC两路径先在同state/q平均，解析T锚点不计评分。time/hub/q calibration、sign、endpoint/max error、每seed误差CI分别保存；不是把所有状态当独立replication。sign_accuracy按浮点exact sign计算，接近零的MC噪声可影响该分数。空间同q排序更直接评价“释放到A还是B”，总pairwise还包括同hub不同q，不能互相混称。","","| 工况 | 方法 | MAE | RMSE | R² | Spearman | pair rank | 同q空间 rank | sign |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def f(v):
        return "n.a." if v is None else f"{v:.3f}"
    for r in prediction_rows:
        lines.append(f'| {display(r["condition"])} | {r["method"]} | {r["MAE"]:,.2f} | {r["RMSE"]:,.2f} | {f(r["R2"])} | {f(r["Spearman"])} | {f(r["pairwise_ranking_accuracy"])} | {f(r["spatial_same_q_ranking_accuracy"])} | {f(r["sign_accuracy"])} |')
    lines += ["","## 7. 正式策略利润与配对CI","","每个条件40个场景；参数、算法、q、MC路径都不是独立统计样本。下表为mean、sample SD、median、Student-t95% CI；|skew|>1补10000次固定seed paired bootstrap，完整值在CSV。主要比较预先设定；另列Cheb−Spline探索性比较，不用于冻结模型选择。均为marginal区间，未多重比较校正。","","| 工况 | 策略 | n | mean profit | SD | median | t95% CI |","|---|---|---:|---:|---:|---:|---|"]
    for cid in conditions:
        for p in LABELS:
            r=s(cid,p)
            lines.append(f'| {display(cid)} | {p} | {r["n"]} | {r["mean"]:,.2f} | {r["SD"]:,.2f} | {r["median"]:,.2f} | [{r["CI_low"]:,.2f}, {r["CI_high"]:,.2f}] |')
    lines += ["","### 同场景增益：Myopic+V","","| 工况 | Fluid−M0 | Table−M0 | Linear−M0 | Cheb−M0 | Spline−M0 |","|---|---|---|---|---|"]
    for cid in conditions:
        lines.append("| "+display(cid)+" | "+" | ".join(ci(d(cid,p)) for p in ["P1-Myopic-Fluid","P2-Myopic-Table","P3-Myopic-Linear","P4-Myopic-Chebyshev","P5-Myopic-Spline"])+" |")
    lines += ["","### Chebyshev−Spline探索性配对比较","",
        "事后补充的解释性对照，正值表示Cheb利润较高。不能把两个方法分别对Myopic的显著性不同，当作彼此显著不同。没有预定等价界限，CI跨0也不能证明等效。",
        "","| 工况 | Myopic+Cheb−Myopic+Spline [t95 CI] | RH4+Cheb−RH4+Spline [t95 CI] |","|---|---|---|"]
    for cid in conditions:
        lines.append(f'| {display(cid)} | {ci(d(cid,"P4-Myopic-Chebyshev","P5-Myopic-Spline"))} | {ci(d(cid,"P8-RH4-Chebyshev","P9-RH4-Spline"))} |')
    lines += ["","## 8. RH4 terminal value 与 horizon compression","","| 工况 | RH8−RH4 [CI] | RH4+Fluid−RH4 [CI] | RH4+Cheb−RH4 [CI] | RH4+Spline−RH4 [CI] |","|---|---|---|---|"]
    for cid in conditions:
        lines.append("| "+display(cid)+" | "+" | ".join(ci(d(cid,p,"P6-RH4")) for p in ["P10-RH8","P7-RH4-Fluid","P8-RH4-Chebyshev","P9-RH4-Spline"])+" |")
    lines += ["","compression=(RH4+V−RH4)/(RH8−RH4)，只在**分母的t95CI下界>0**时报告。>100%只是本配置均值收益比，不表示更多未来信息。RH4+V−RH8差值与CI在paired_policy_differences.csv；CI跨0只是不确定，不能宣布等价/达到论文要求的noninferiority，后者需预定equivalence margin。","","| 工况 | 方法 | horizon compression | Oracle-gap capture |","|---|---|---:|---:|"]
    for cid in conditions:
        for p in ["P7-RH4-Fluid","P8-RH4-Chebyshev","P9-RH4-Spline"]:
            r1=next(r for r in ratios if r["condition"]==cid and r["policy"]==p and r["kind"]=="horizon_compression")
            r2=next(r for r in ratios if r["condition"]==cid and r["policy"]==p and r["kind"]=="oracle_gap_capture")
            v=lambda r:f'{100*r["ratio"]:.1f}%' if r["eligible"] else "不报告（分母未明确为正）"
            lines.append(f'| {display(cid)} | {p} | {v(r1)} | {v(r2)} |')
    lines += ["","## 9. Oracle gap 与 continuation敏感性","","Oracle-gap capture=(策略−M0)/(OracleH8−M0)，同样只在分母CI明确为正时报告；是performance gap ratio，不是信息恢复率、预测准确率或可部署收益率。剩余gap可能来自不可预测需求、受限动作族、不同目标/预算、target/integration偏差，不能全部归于模型维数。"]
    for cid in conditions:
        lines.append(f'- {display(cid)}：Oracle−M0 {ci(d(cid,"D-OracleH8"))}；Oracle减RH4+Cheb均值 {s(cid,"D-OracleH8")["mean"]-s(cid,"P8-RH4-Chebyshev")["mean"]:,.2f}；Oracle减RH4+Spline {s(cid,"D-OracleH8")["mean"]-s(cid,"P9-RH4-Spline")["mean"]:,.2f}。')
    lines += ["",f'Western40/8额外以5个construction seeds、t=0/12/23构造MC-RH4 target，并用完全相同查询、状态hash与未来hash复跑MC-Myopic：{secondary["count"]}个平均目标配对。每seed对固定query组合的mean marginal变化为 {ci(secondary["summary"])}。这是target-policy sensitivity小样本诊断，不能称正式RH4训练或policy改善CI；详见continuation_policy_targets.csv。',"","## 10. 时间、失败与审计","",
        f'离线Fluid surface wall {manifest["fluid"]["wall_seconds"]:.2f}s；MC construction wall {manifest["construction"]["wall_seconds"]:.2f}s（worker累计 {manifest["construction"]["summed_worker_seconds"]:.2f}s）；validation wall {manifest["validation"]["wall_seconds"]:.2f}s；全部候选fit wall {manifest["fit"]["wall_seconds"]:.2f}s。',
        f'正式矩阵wall {manifest["formal"]["wall_seconds"]:.2f}s，3并行worker、每Gurobi一线程。失败0；非OPTIMAL窗口计数 {sum(r["limited_window_count"] for r in rows)}，最大记录window gap {max(r["max_window_gap"] for r in rows):.6g}。窗口30s限制、每日600s wall限制，无incumbent会保留失败而非静默换算法/删样本。',
        "offline样本生成、offline fit、online value preparation、MILP solver、daily wall分开记录。online_value_eval_seconds实际包含provider评价与slot变量/约束建立，不是单纯浮点函数算术微基准；build_seconds也包含该段，**不能将build与value preparation再相加为互斥时间**。daily_solver_seconds只为Gurobi Runtime合计，daily_wall包含摘要/构建/仿真/ledger/trace计算，输出文件写入不在sim daily计时。CFS冻结输入时间与resume读取时间不混称solver。",
        f'逐t独立物理重放审计 {audit["runs"]}次run、{audit["steps"]}时点：{audit["status"]}。核对source/scenario/trace hash、同场景配对、预订边界、无提前/重复分配、capacity、clock、state hash、原始after状态、冷却无控制、realized利润与全部commitment完成。独立编译金额误差≤1e-6后仅保留原记录monetary scalars处理浮点加法顺序，不覆盖物理状态。',
        f'固定PYTHONHASHSEED=0的全量测试 {qa["tests"]}项，failure {qa["failures"]}、error {qa["errors"]}、skip {qa["skipped"]}；存在一条既有Starlette/AnyIO依赖弃用警告，不影响通过。独立重跑Western40/8 seed3000 RH4+Spline，利润差 {repetition["profit_difference"]:.6g}，物理轨迹完全一致。pytest-full.xml、independent_reproduction.json与qa_validation.json留存证据。',
        "最后复测曾把pytest临时目录误放在非忽略的仓库根层，平台代码身份扫描碰到Windows已锁定的coordinator.lock，出现6项PermissionError；失败记录保留在pytest-placement-check-failed.xml。将本次测试临时文件移至被忽略的workspace并按README重跑后通过，未修改平台扫描器、冻结数值源码或模型；这不属于1920次正式仿真的失败。",
        "运行环境为Windows、Python3.11.14、Gurobi10.0.3；处理器、逻辑核数、NumPy/SciPy/Matplotlib等版本写入最终manifest。并行worker中的wall time含资源竞争，不是标准化硬件上的单进程微基准。MC目标文件未逐个保存扰动求解的status/gap，尤其RH4 continuation仅作小样本敏感性诊断；不能把正式策略全OPTIMAL结论扩展成全部MC子问题的逐项最优证明。",
        "逐t trace保留当前订单/车辆、assignment/route/reposition、常数释放系数、solver/status/gap、各账本与实际利润序列。processed/execution_events/另由无求解重放生成逐车segments、load、pickup与delivery，供检查实际时间序列和Python多实验绘图。raw未覆盖。",
        "字段语义特别说明：raw plan.augmented_solver_objective沿用执行编译器的当前阶段booked profit−delay tie-break，value控制时**不包含tail项**；真正包含V的优化目标在plan.solver.window_objective。H4/H8的window目标还含未执行未来阶段，不能当作当前利润或把每天window目标相加称经营收益。各t实际收益应读取realized_ledger.realized_profit及period_realized_profit，详见trace_field_semantics.json。",
        "",
        "## 11. 正则性与Go / No-Go判断",
        "",
        "验证RMSE非常接近，不能仅据几美元差距声称Spline具有统计优势或Chebyshev已证明spectral smooth。必须联合held-out误差、空间ranking、系数识别性、端点、配对profit、真实post-decision integration判断。以下为逐工况证据："
    ]
    for cid in conditions:
        ch=next(r for r in prediction_rows if r["condition"]==cid and r["method"]=="chebyshev")
        sp=next(r for r in prediction_rows if r["condition"]==cid and r["method"]=="spline")
        gains=[d(cid,p) for p in ["P8-RH4-Chebyshev","P9-RH4-Spline"]]
        choice="Spline" if sp["RMSE"]<ch["RMSE"] else "Chebyshev"
        lines.append(f'- {display(cid)}：held-out RMSE Cheb {ch["RMSE"]:,.2f} / Spline {sp["RMSE"]:,.2f}，数值较低为{choice}；同q空间rank {f(ch["spatial_same_q_ranking_accuracy"])} / {f(sp["spatial_same_q_ranking_accuracy"])}。相对Myopic，RH4+Cheb {ci(gains[0])}，RH4+Spline {ci(gains[1])}。')
    positive=sum(d(cid,p,"P6-RH4")["CI_low"]>0 for cid in conditions for p in ["P8-RH4-Chebyshev","P9-RH4-Spline"])
    lines += ["",
        f'四工况、两个RH4 classical tail的8个预定增益比较中，{positive}个相对RH4的CI下界>0（未多重校正）。这说明某些低维tail确有用，不能保证全工况。Fluid的分数、无锚点与冻结初始供给会高估未来资源价值；若政策退化，应先修正surrogate和integration，而不是将数值回归的“高级程度”作为解释。',
        "当前不宜单独宣布Case A（低阶全局spectral smooth）或Case B（明确piecewise-smooth胜出）：多数p不是很低，局部q/near-zero行为、终端gate、自然state变化与矩阵非识别混在一起；表格和Spline/Cheb的收益差也不能仅用MSE解释。受控切片与slot图可以显示局部弯曲/非单调，但有限MC噪声下不证明真实值可微阶数。",
        "**下一阶段选择Chebyshev作为主基线，Spline作为局部形状对照；优先修正post-decision摘要与continuation target，再验证少量有经济含义的交互。** 这是稳定性、解释性与增益的综合工程选择，不是证明Cheb利润普遍显著优于Spline。应先区分candidate后backlog/供给变化、route anchor与实际费用，再考虑更丰富空间状态。residual_interactions.csv仅诊断supply×demand与incoming×demand，未用formal target调模型；不直接上full tensor或神经网络。",
        "",
        "## 12. 图表与可复现文件",
        "",
        "11张图均为7.2英寸双栏、DejaVu Serif、色盲友好颜色+不同markers、600dpi PNG与PDF/SVG矢量。图1的MC误差带按相同时点/hub/q的scenario均值，不把两条路径混作样本；图9ratio是描述性点值，表中利润CI是独立的配对推断。绘图配置及输入CSV hash一并保留。",
        "图1、图4受控MC切片和图11采用symlog纵轴：绝对值1000 USD内线性，外侧对数；图4解析Runge控制的阈值为1。全部极端值保留，避免Fluid/单项式放大值压平其他曲线，不可按图上的等距直接解释绝对利润差。图10用P0–P10及D代码防止文本堆叠，底部给出算法映射；D明确为不可部署Oracle诊断。",
        "",
        "核心文件：targets/*.csv/json（路径目标/终端锚点与独立切片）；models/最终与candidates/全部候选；manifests/model_freeze.json；raw/datasets/冻结CFS与source字段；raw/runs/逐t双账；processed/value_predictions.csv与calibration/CI；policy_summary.csv与paired_policy_differences.csv；performance_ratios.csv；plotting_config.json；trace_audit.json。",
        "",
        "数值源码逐文件hash、原始基线报告hash、CFS hash、source.snapshot.zip、ValueProvider版本、seed、solver、actual nodes、模型超参数与时间在manifest/模型中；analysis-source.snapshot.zip独立保存后处理来源，避免把报告修订假称新数值运行。源码/config/model变动会阻止续跑，不允许覆盖旧结果。运行命令见experiments/cfs_classical_value/README.md。",
        "",
        "局限汇总：经验需求并非真实条件后验；只两个MC paths，target有噪声且donor复用；少数q离散支持及强共线性；低维摘要缺失完整空间/订单结构；候选后摘要未联合更新；Fluid为乐观简化surface；baseline是受限macro-action窗口；只wide rho、四个事先选定工况；没有equivalence margin、多重比较控制或全国外推。新ValueProvider目前用于本次独立实验runner，**尚未新增平台UI framework/schema入口**，不是前端已经可选的新训练流程。",
        "",
        "## 13. Implications for legacy DRL",
        "",
        "1. 简单classical方法已捕获的收益必须以上面40-seed配对gain/ratio为准；ratio超100%不意味着更多未来信息。RH4与tail有改善时，旧DRL应先与这个合法在线、同预算基准比较。",
        "2. 目前明显非线性候选是资源槽、需求-供给稀缺性、终端/路线完成边界；自然样本与冻结summary混淆使其不能被精确识别。不要把原始非唯一系数当经济因果。",
        "3. Cheb与Spline未得到足以预设某一类函数正则性的统一胜负；局部piecewise结构值得保留对照，但更高阶不是默认方向。",
        "4. 剩余Oracle gap不等于可预测/可学习的收益，更不等于DRL可达到收益；oracle不是全日最优解，且信息不可部署。",
        "5. 当前证据优先支持修正post-decision state/target、少量经济交互与空间摘要。只有这些公平控制仍不足、且独立样本确认可部署预测价值，才评估更高维统计学习，最后考虑neural/DRL。此次不修改或训练旧DRL/SAC与BHH。"
    ]
    appendix=["","### 论文图集",""]
    for name in sorted((REPORT/"figures").glob("*.png")):
        appendix += ["",f'#### {name.stem}',"",f'![{name.stem}](figures/{name.name})']
    final_section=lines.index("## 13. Implications for legacy DRL")
    lines[final_section:final_section]=appendix+[""]
    (REPORT/"classical_value_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    started=time.perf_counter()
    manifest=read_json(OUTPUT/"manifest.json");cfg=manifest["config"];baseline=manifest["baseline_config"]
    assert manifest["status"] in ("NUMERICAL_RUNS_COMPLETED","COMPLETED")
    assert manifest["code_hash"]==digest(source_hashes()),"numerical source changed"
    for p,h in read_json(OUTPUT/"manifests/model_freeze.json")["models"].items():
        assert file_hash(OUTPUT/"models"/p)==h,"frozen model changed"
    rows=read_json(OUTPUT/"processed/formal_policies.json")
    metrics,predictions,intervals=value_evaluation(cfg)
    summaries,pairs,ratios=policy_statistics(cfg,rows)
    write_csv(OUTPUT/"processed/paper_value_predictions.csv",[
        dict(condition=r["condition"],method=r["method"],split=r["split"],
            scenario_seed=r["scenario_seed"],release_time_period=r["time"],hub=r["hub"],
            supply_slot_vehicles=r["q"],true_MC_marginal_value_usd=r["target"],
            predicted_marginal_value_usd=r["prediction"],prediction_error_usd=r["error"])
        for r in predictions])
    conditions={key(c):c for c in cfg["conditions"]}
    policy_meta={label:(framework,method) for label,framework,method in LADDER}
    policy_data=[]
    for r in rows:
        c=conditions[r["condition"]];framework,method=policy_meta[r["policy"]]
        spec=read_json(OUTPUT/"raw/runs"/r["id"]/"spec.json")
        policy_data.append(dict(model_family="single",condition=r["condition"],framework=framework,
            value_method=method or "zero",scenario_seed=r["seed"],policy_seed=0,
            training_seed=None,scenario_hash=r["scenario_hash"],orders=c["orders"],fleet_vehicles=c["fleet"],
            pickup_radius_miles=spec["parameters"]["pickup_radius"],information=r["information"],
            realized_profit_usd=r["realized_profit"],realized_cost_usd=r["realized_cost"],
            completed_fare_usd=r["completed_fare"],service_rate_fraction=r["service_rate"],
            empty_distance_miles=r["actual_empty_distance"],loaded_distance_miles=r["actual_loaded_distance"],
            daily_wall_seconds=r["daily_wall_seconds"],daily_solver_seconds=r["daily_solver_seconds"],
            value_preparation_seconds=r["online_value_eval_seconds"]))
    write_csv(OUTPUT/"processed/paper_policy_samples.csv",policy_data)
    residual_diagnostics(predictions,cfg)
    secondary=secondary_comparison(cfg,baseline,manifest)
    prior=read_json(OUTPUT/"models/priors.json")
    slice_=controlled_slice(cfg,baseline,None,prior)
    runge=runge_diagnostic(slice_,cfg)
    audit=audit_runs(cfg,rows)
    figures(cfg,summaries,pairs,ratios,predictions,slice_,runge)
    atomic_json(OUTPUT/"processed/trace_field_semantics.json",dict(
        current_stage_booked_objective="plan.augmented_solver_objective: compiled CURRENT-stage booked profit minus departure-delay tie-break; excludes V even for value controllers",
        planning_objective_with_tail="plan.solver.window_objective: full restricted window objective including pre-evaluated marginal slot values; H1 is single-period+tail",
        actual_cumulative_profit="realized_ledger.realized_profit; deliveries credited only after physical execution",
        actual_period_profit="measurements.period_realized_profit",
        warning="Do not sum receding-window objectives or repeatedly booked continuation estimates as realized revenue. Raw legacy field names retained and explicitly qualified."))
    report(cfg,manifest,summaries,pairs,ratios,metrics,secondary,audit,runge,slice_)
    for name,h in manifest["protected_hashes"].items():assert file_hash(ROOT/name)==h
    assert file_hash(ROOT/"results/cfs_dynamic_lookahead/manifest.json")==manifest["baseline_manifest_sha256"]
    assert file_hash(baseline["data_source"])==manifest["source"]["sha256"]
    with zipfile.ZipFile(OUTPUT/"analysis-source.snapshot.zip","w",zipfile.ZIP_DEFLATED) as z:
        for name in ["experiments/cfs_classical_value/analysis.py","experiments/cfs_dynamic_lookahead/analysis.py",
                     "experiments/cfs_dynamic_lookahead/case_study.py","tests/test_classical_analysis.py"]:
            z.write(ROOT/name,name)
    manifest.update(status="COMPLETED",analysis_wall_seconds=time.perf_counter()-started,
        analysis_sha256=file_hash(__file__),analysis_snapshot_sha256=file_hash(OUTPUT/"analysis-source.snapshot.zip"),
        trace_audit=dict(status=audit["status"],runs=audit["runs"],steps=audit["steps"]),
        secondary_target_comparison=secondary,runge_controlled_slice=dict(seeds=slice_["seeds"],wall_seconds=slice_["wall_seconds"]))
    manifest.update(final_CFS_content_reverified=True,
        input_seconds_semantics="latest frozen-dataset retrieval wall time; first SQLite pool scan duration not separately retained; excluded from MILP timings",
        packages={p:importlib.metadata.version(p) for p in ["gurobipy","numpy","scipy","matplotlib","networkx","pydantic"]},
        hardware=dict(OS=platform.platform(),logical_processors=os.cpu_count(),
            processor=os.environ.get("PROCESSOR_IDENTIFIER",platform.processor())))
    files=list((OUTPUT/"processed").glob("*.csv"))+list((OUTPUT/"processed").glob("*.json"))+list((OUTPUT/"models").glob("*.json"))+list((REPORT/"figures").glob("*"))+[REPORT/"classical_value_report.md",OUTPUT/"processed/pytest-full.xml"]
    manifest["analysis_output_hashes"]={p.relative_to(ROOT).as_posix():file_hash(p) for p in files}
    atomic_json(OUTPUT/"manifest.json",manifest)
    print("Analysis/report COMPLETED:",REPORT/"classical_value_report.md",flush=True)


if __name__=="__main__":main()
