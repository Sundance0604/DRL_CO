"""Time-expanded fractional load/vehicle relaxation, construction-only demand."""
import time
import gurobipy as gp
from gurobipy import GRB
from experiment_core.storage import atomic_json
from model_families.single.data import matching_state
from .common import OUTPUT,key,parameters,rho_levels


def demand_prior(scenarios,T=24):
    hubs=scenarios[0]["nodes"];all_orders=[o for s in scenarios for o in s["orders"]]
    rate={h:sum(o["departure"]==h for o in all_orders)/(len(scenarios)*T) for h in hubs}
    od={}
    for a in hubs:
        for b in hubs:
            items=[o for o in all_orders if o["departure"]==a and o["destination"]==b]
            if not items:continue
            od[a+"|"+b]=dict(rate=len(items)/(len(scenarios)*T),
                mean_fare=sum(o["revenue"] for o in items)/len(items),
                mean_load=sum(o["passenger"] for o in items)/len(items))
    return dict(rate=rate,od=od,source_seeds=[s["seed"] for s in scenarios],
        assumption="uniform future booking mean; empirical OD and cargo/fare from construction only; ready delay rounded mean=1; fractional full-capacity consolidation relaxation")


def fluid_dual(sim,prior,t,hub,q):
    """Derivative at q-1 free vehicles; base other-hub fleet retained.

    No backlog here: this deliberately simple OFFLINE surface is a classical
    forecast/geometry comparator, not an oracle or state-aware reoptimization.
    Fractional fleet/cargo and empty transfers relax indivisible trip anchors.
    """
    net=sim.net;T=sim.T;p=sim.parameters
    with gp.Model("classical_fluid") as m:
        m.Params.OutputFlag=0;m.Params.Threads=1;m.Params.Method=1
        inflow={};outflow={};objective=[]
        sink=("sink",)
        def arc(a,b,profit,upper=GRB.INFINITY):
            x=m.addVar(lb=0,ub=upper)
            outflow.setdefault(a,[]).append(x);inflow.setdefault(b,[]).append(x)
            objective.append(profit*x)
        for when in range(t,T):
            for a in net.H:
                node=(when,a)
                arc(node,(when+1,a) if when+1<T else sink,0)
                for b in net.H:
                    if a==b:continue
                    finish=when+net.tau[a,b]
                    dest=(finish,b) if finish<T else sink
                    # Relaxed empty transfer, not a statement that formal policy
                    # enables reposition. Explicit optimism is disclosed.
                    if finish<T:arc(node,dest,-p["empty_cost"]*net.d[a,b])
                    od=prior["od"].get(a+"|"+b)
                    if od and when>=t+1:  # no unbooked demand instantly available
                        vehicles=od["rate"]*od["mean_load"]/p["capacity"]
                        fare_per_vehicle=od["mean_fare"]*p["capacity"]/od["mean_load"]
                        arc(node,dest,fare_per_vehicle-p["loaded_cost"]*net.d[a,b],vehicles)
        balances={}
        for node in sorted(set(inflow)|set(outflow),key=str):
            if node==sink:continue
            when,a=node
            supply=0
            if when==t:
                supply=sum(v.phase=="idle" and v.loc==a for v in sim.vehs if a!=hub)
                if a==hub:supply=q-1
            balances[node]=m.addConstr(gp.quicksum(outflow.get(node,[]))-gp.quicksum(inflow.get(node,[]))==supply)
        m.setObjective(gp.quicksum(objective),GRB.MAXIMIZE);m.optimize()
        if m.Status!=GRB.OPTIMAL:raise RuntimeError("fluid LP failed")
        return float(balances[t,hub].Pi),m.Runtime


def build_fluid(cfg,baseline,data):
    priors={};diagnostics=[];started=time.perf_counter()
    for c in cfg["conditions"]:
        cid=key(c);scenarios=[data[c["region"],c["orders"],c["fleet"],seed]["scenarios"][0] for seed in cfg["construction_seeds"]]
        prior=demand_prior(scenarios);priors[cid]=prior
        sim,_=matching_state(scenarios[0],parameters(baseline,rho_levels(scenarios[0],baseline)["rho_max"]),baseline["solver"])
        values={};solver_time=0.
        for h in sorted(sim.net.H):
            values[h]={}
            for q in range(1,c["fleet"]+1):
                v=[]
                for t in range(sim.T):
                    value,runtime=fluid_dual(sim,prior,t,h,q)
                    v.append(value);solver_time+=runtime
                values[h][str(q)]=dict(times=list(range(sim.T+1)),values=v+[0.])
        atomic_json(OUTPUT/"models"/f"{cid}-fluid.json",dict(method="fluid",values=values,prior=prior,
            target="local LP dual at q-1; approximation to discrete paired realized marginal, not identical LP optimization target",
            state_dependence="offline frozen initial other-hub supply; no live backlog",
            relaxation="fractional capacity pooling and empty reposition arcs, relaxed endpoint anchors",
            boundary="free resource at T=0"))
        diagnostics.append(dict(condition=cid,solver_seconds=solver_time))
    atomic_json(OUTPUT/"models/priors.json",priors)
    return priors,dict(wall_seconds=time.perf_counter()-started,conditions=diagnostics)
