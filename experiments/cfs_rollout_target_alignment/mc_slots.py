"""Paired q-vs-q-1 resources and within-split empirical future realizations."""
import copy
import time
from pathlib import Path
import numpy as np
from experiment_core.storage import atomic_json,read_json,digest
from model_families.single.data import matching_state
from model_families.single.engine.mt_prototype import Veh
from .common import OUTPUT,key,parameters,map_jobs,rho_levels,write_csv
from experiments.cfs_classical_value.common import nodes
from experiments.cfs_classical_value.simulation import simulate,summary,feature_row


def resource_world(state,hub,q):
    c=state.clone()
    c.vehs=[v for v in c.vehs if not(v.phase=="idle" and v.loc==hub)]
    for i in range(q):c.vehs.append(Veh(-1,hub))
    for i,v in enumerate(c.vehs):v.id=i
    return c


def future_orders(original,donor,time,rep):
    # Keep observed current and past records; donor-only future IDs never collide.
    known={oid:o for oid,o in original.items() if o.book_time<=time}
    for o in donor.values():
        if o.book_time>time:
            n=copy.copy(o);n.id=f"mc{rep}-"+str(o.id);known[n.id]=n
    return known


def target_job(job):
    target=OUTPUT/"targets/raw"/(job["id"]+".json")
    identity=digest({k:v for k,v in job.items() if k!="prior"}|{"prior":job["prior"]})
    if target.exists():
        saved=read_json(target);assert saved["identity"]==identity
        return saved
    started=time.perf_counter();scenario=job["scenario"];T=scenario["horizon"]
    sim,orders=matching_state(scenario,job["parameters"],job["solver"])
    times=job["times"];_,_,states=simulate(sim,orders,collect_times=times)
    donors=[matching_state(s,job["parameters"],job["solver"])[1] for s in job["donors"]]
    rows=[]
    hubs=sorted(sim.net.H)
    for index,t in enumerate(times):
        state=states[t]
        # Active projected Chebyshev periods with natural on-policy states.
        query_hubs=[hubs[(scenario["seed"]+index)%len(hubs)]]
        if index%3==0:
            query_hubs.append(hubs[(scenario["seed"]+index+1)%len(hubs)])
        for hub in query_hubs:
            online=summary(state,job["prior"]);qmax=job["fleet"]
            slots=sorted(set(q for q in job["slots"] if q<=qmax)|{qmax})
            allq=sorted(set([q for q in slots]+[q-1 for q in slots]))
            outcomes={}
            for rep,donor in enumerate(donors):
                future=future_orders(orders,donor,t,rep)
                profits={}
                for q in allq:
                    c=resource_world(state,hub,q)
                    m,_,_=simulate(c,future,policy=job["continuation"],first_begun=True)
                    profits[q]=m["realized_profit"]
                for q in slots:
                    outcomes.setdefault(q,[]).append(profits[q]-profits[q-1])
                    rows.append(dict(condition=job["condition"],split=job["split"],scenario_seed=scenario["seed"],
                        donor_seed=job["donors"][rep]["seed"],mc_replication=rep,time=t,supply_slot=q,
                        state_hash=digest(__import__("model_families.single.runner",fromlist=["state_snapshot"]).state_snapshot(state)),
                        future_hash=digest([vars(o) for o in future.values()]),continuation_policy=job["continuation"],
                        profit_q=profits[q],profit_previous=profits[q-1],target=profits[q]-profits[q-1],
                        **feature_row(t,hub,q,online)))
    # Correct terminal anchor is analytically zero for NEW free resource.
    for hub in hubs:
        for q in sorted(set(q for q in job["slots"] if q<=job["fleet"])|{job["fleet"]}):
            online=summary(states[times[-1]],job["prior"])
            rows.append(dict(condition=job["condition"],split=job["split"],scenario_seed=scenario["seed"],
                donor_seed=None,mc_replication=-1,time=T,supply_slot=q,state_hash=None,future_hash=None,
                continuation_policy=job["continuation"],profit_q=None,profit_previous=None,target=0.,
                **feature_row(T,hub,q,online),terminal_anchor=True))
    saved=dict(identity=identity,rows=rows,seconds=time.perf_counter()-started)
    target.parent.mkdir(parents=True,exist_ok=True);atomic_json(target,saved)
    return saved


def build_targets(cfg,baseline,data,priors,split,seeds,continuation="myopic",times=None):
    conditions=cfg["conditions"] if continuation=="myopic" else [cfg["secondary_condition"]]
    jobs=[]
    for c in conditions:
        cid=key(c)
        for i,seed in enumerate(seeds):
            s=data[c["region"],c["orders"],c["fleet"],seed]["scenarios"][0]
            donor_seeds=[seed,seeds[(i+cfg["mc_donor_offset"])%len(seeds)]]
            jobs.append(dict(id=f"{split}-{cid}-s{seed}-{continuation}",condition=cid,scenario=s,
                donors=[data[c["region"],c["orders"],c["fleet"],d]["scenarios"][0] for d in donor_seeds],
                parameters=parameters(baseline,rho_levels(s,baseline)["rho_max"]),solver=baseline["solver"],
                times=times or nodes(s["horizon"],cfg["chebyshev_node_count"]),fleet=c["fleet"],
                slots=cfg["target_supply_slots"],prior=priors[cid],split=split,continuation=continuation))
    saved,wall=map_jobs(target_job,jobs,"MC "+split,cfg["max_parallel"])
    rows=[r for item in saved for r in item["rows"]]
    rows.sort(key=lambda r:(r["condition"],r["scenario_seed"],r["time"],r["hub"],r["supply_slot"],r["mc_replication"]))
    write_csv(OUTPUT/"targets"/(split+".csv"),rows)
    atomic_json(OUTPUT/"targets"/(split+".json"),rows)
    return dict(jobs=len(jobs),paired_target_count=sum(r["mc_replication"]>=0 for r in rows),
        terminal_anchor_count=sum(r["mc_replication"]<0 for r in rows),
        wall_seconds=wall,summed_worker_seconds=sum(item["seconds"] for item in saved),
        continuation=continuation)
