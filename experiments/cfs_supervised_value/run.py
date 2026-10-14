"""Two fits and 24 full-cooldown H1 days under an explicit numerical budget."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .common import *


class Budget:
    def __init__(self,seconds=900,solves=1500):
        self.started=time.monotonic();self.deadline=self.started+seconds;self.limit=solves;self.solves=0
    def check(self):
        if time.monotonic()>self.deadline:raise TimeoutError('900-second numerical budget reached')
        return False
    def claim(self):
        self.check()
        if self.solves>=self.limit:raise TimeoutError('1500-call budget reached')
        self.solves+=1


class Worker:
    def __init__(self,target,budget):
        self.budget=budget;self.executor=ThreadPoolExecutor(max_workers=1)
        self.err=(target/'training/online_worker.log').open('w',encoding='utf-8')
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='41002',PYTHONIOENCODING='utf-8')
        self.proc=subprocess.Popen([str(AG_PYTHON),str(Path(__file__).with_name('ag_worker.py')),'serve','--target',str(target)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,encoding='utf-8',env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        response=self._read()
        if not response.get('ready'):raise RuntimeError('AG worker not ready')
    def _read(self):
        left=min(60,max(0.1,self.budget.deadline-time.monotonic()))
        line=self.executor.submit(self.proc.stdout.readline).result(timeout=left)
        if not line:raise RuntimeError('AG worker died; see online_worker.log')
        return json.loads(line)
    def predict(self,method,rows):
        self.budget.check();self.proc.stdin.write(json.dumps({'method':method,'rows':rows})+'\n');self.proc.stdin.flush()
        return self._read()
    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write('{"op":"stop"}\n');self.proc.stdin.flush()
            try:self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:self.proc.terminate();self.proc.wait(timeout=5)
        self.executor.shutdown(wait=True);self.err.close()


class Provider:
    def __init__(self,method,state,orders,prior,worker,baseline,extreme):
        self.method=method;self.cache={};self.queries=[];self.pred_seconds=0.;self.batch_rows=0;self.extreme=extreme
        self.T=state.T;self.baseline=baseline
        if method in ['Basic','State']:
            known={k:o for k,o in orders.items() if o.book_time<=state.t}
            keys=[(t,h,q) for t in range(state.t+1,state.T) for h in sorted(state.net.H) for q in range(1,len(state.vehs)+1)]
            if keys:
                rows=[features(state,known,h,t,q,prior) for t,h,q in keys]
                response=worker.predict(method,rows);self.pred_seconds=response['seconds'];self.batch_rows=len(rows)
                assert len(response['values'])==len(keys)
                self.cache=dict(zip(keys,response['values']))
    def marginal_release_value(self,time,hub,supply_slot,online_state_summary):
        if time>=self.T:value=0.
        elif self.method=='M0':value=0.
        elif self.method=='PureMC':value=self.baseline.marginal_release_value(time,hub,supply_slot,online_state_summary)
        else:
            if time!=int(time):raise ValueError('noninteger release time outside registered batch grid')
            value=self.cache[int(time),hub,supply_slot]
        self.queries.append(dict(time=time,hub=hub,q=supply_slot,value=value,extreme=abs(value)>self.extreme))
        return float(value)


def business(plan,before):
    from model_families.single.strategic.contracts import identity
    vehicles={str(v['id']):{k:v[k] for k in ['hub','phase','route','dead','arrive','hold','assigned_orders','onboard']} for v in before['vehicles']}
    assignments={}
    for x in plan['assignments']:assignments.setdefault(str(x['vehicle_id']),[]).append(x['order_id'])
    routes=[dict(before=vehicles[str(r['vehicle_id'])],route={k:v for k,v in r.items() if k!='vehicle_id'},
        orders=sorted(assignments.get(str(r['vehicle_id']),[]))) for r in plan['routes']]
    repos=[dict(before=vehicles[str(r['vehicle_id'])],hub=r['hub']) for r in plan['repositions']]
    return identity(dict(period=plan['period'],routes=sorted(routes,key=identity),repositions=sorted(repos,key=identity),expired=sorted(plan['expired'])))


def one_day(scene,cfg,method,prior,worker,budget,extreme):
    from model_families.single.data import matching_state
    from model_families.single.parameters import MatchingParameters
    from experiment_core.contracts import SolverParameters
    from model_families.single.rolling.contracts import WindowRequest
    from model_families.single.rolling.controller import observed_state
    from model_families.single.planning import compile_action,PeriodAction
    from model_families.single.value.route_milp import ValueRouteMILPPlanner
    from model_families.single.value.providers import MCTableValueProvider
    from model_families.single.runner import state_snapshot,serialize_plan
    from model_families.single.strategic.contracts import identity
    from model_families.single.strategic.observation import free_release
    from model_families.single.accounting import completion_bound,cooldown_tick,realized_ledger
    from experiments.cfs_ppo_strategic_control.study_execution import replay_rh4
    sim,orders=matching_state(copy.deepcopy(scene),MatchingParameters.model_validate(cfg['parameters']).model_dump(),SolverParameters.model_validate(cfg['solver']).model_dump())
    base=MCTableValueProvider(read(cfg['baseline']));trace=[];rows=[];slot_rows=[];queries=[]
    started=time.perf_counter();prediction=solver_seconds=batch_wall=0.;batch_rows=0
    for t in range(sim.T):
        budget.check();sim.begin_period([o for o in orders.values() if o.book_time==t]);before=state_snapshot(sim)
        tick=time.perf_counter();provider=Provider(method,observed_state(sim),orders,prior,worker,base,extreme);batch_wall+=time.perf_counter()-tick
        prediction+=provider.pred_seconds;batch_rows+=provider.batch_rows
        request=WindowRequest(observed_state(sim),t,t+1,t+1,0,budget.check)
        budget.claim();solution=ValueRouteMILPPlanner(provider,dict(T=sim.T)).plan(request)
        meta=solution.diagnostics
        if meta['status']!='OPTIMAL':raise RuntimeError('nonoptimal daily solve')
        info=dict(meta);plan=compile_action(sim,solution.actions.get(t,PeriodAction(t)),info)
        serial=serialize_plan(plan,t,identity(before),info);sim.commit_plan(plan)
        solver_seconds+=meta['runtime'];rows.append(dict(period=t,plan=serial))
        after=state_snapshot(sim);trace.append(dict(period=t,before=before,after=after,plan=serial))
        # Read the selected FREE sink resources from the committed physical routes.
        projection=sim.clone();projection.begin_period([])
        mip_query_count=len(provider.queries)
        groups={}
        for v in projection.vehs:
            h,when=free_release(projection,v);groups[when,h]=groups.get((when,h),0)+1
        bonus=0.
        for (when,h),count in groups.items():
            if when>=sim.T:continue
            for q in range(1,count+1):
                value=provider.marginal_release_value(when,h,q,dict(T=sim.T));bonus+=value
                slot_rows.append(dict(scenario_seed=scene['seed'],method=method,period=t,release_time=when,hub=h,q=q,value=value,occupied=True))
        if not math.isclose(plan[1]+bonus,meta['window_objective'],abs_tol=1e-5,rel_tol=1e-10):
            raise RuntimeError(f'occupied prefix bonus mismatch {method} t{t}: {plan[1]+bonus} != {meta["window_objective"]}')
        serial['business_hash']=business(serial,before)
        for q in provider.queries[:mip_query_count]:queries.append(dict(scenario_seed=scene['seed'],method=method,period=t,**q))
    decision=list(rows);sim.pool={};cap=completion_bound(sim)
    while any(v.phase!='idle' for v in sim.vehs):
        budget.check()
        if sim.t>=cap:raise RuntimeError('unfinished cooldown')
        before=state_snapshot(sim);period=sim.t;cooldown_tick(sim)
        trace.append(dict(period=period,before=before,after=state_snapshot(sim),drain=True,plan={}))
    ledger=realized_ledger(sim,orders,decision)
    assert ledger['unfinished_committed']==0
    record=dict(scenario_id=scene['id'],scenario_seed=scene['seed'],method=method,
        ledger={k:v for k,v in ledger.items() if isinstance(v,(int,float))},full_ledger=ledger,
        original_trace=trace,metrics=dict(**ledger,**sim.S),episode_seconds=time.perf_counter()-started)
    for tr in trace:tr['realized_ledger']=realized_ledger(sim,orders,decision,tr['period']+1)
    replay_rh4(scene,cfg,record)
    for v in sim.vehs:
        assert all(seg['load']<=cfg['parameters']['capacity'] for seg in v.segments)
        assert all(t>=orders[oid].start_time for oid,t in v.pickups.items())
        assert all(t<=orders[oid].end_time for oid,t in v.deliveries.items())
    runtime=dict(scenario_seed=scene['seed'],method=method,episode_seconds=record['episode_seconds'],
        model_prediction_seconds=prediction,feature_RPC_seconds=batch_wall,model_batch_rows=batch_rows,
        solver_seconds=solver_seconds,queries=len(queries),q2_queries=sum(q['q']>=2 for q in queries),
        q2_occupied=sum(q['q']>=2 for q in slot_rows),extreme_queries=sum(q['extreme'] for q in queries),
        empty_cost_USD=ledger['actual_empty_distance']*cfg['parameters']['empty_cost'])
    return record,runtime,queries,slot_rows


def run(target=TARGET):
    target=Path(target).resolve();protocol=read(target/'protocol.json')
    if protocol['status']!='PREPARED':raise RuntimeError('new prepared attempt required')
    import psutil,torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if os.name=='nt':psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    budget=Budget();protocol['status']='RUNNING';write(target/'protocol.json',protocol)
    own_sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    own_sources[str(ROOT/'tests/test_supervised_value.py')]=sha(ROOT/'tests/test_supervised_value.py')
    write(target/'qa/experiment_source_hashes.json',own_sources)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='41002',PYTHONIOENCODING='utf-8')
    worker=None
    try:
        for method in ['Basic','State']:
            budget.check();log=(target/'training'/f'{method}_console.log').open('w',encoding='utf-8')
            with log:
                subprocess.run([str(AG_PYTHON),str(Path(__file__).with_name('ag_worker.py')),'fit','--target',str(target),'--method',method],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=min(210,budget.deadline-time.monotonic()),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            result=read(target/'training'/f'{method}_fit.json')
            print('FIT',method,'best',result['model_best'],'seconds',round(result['fit_wall_seconds'],2),flush=True)
        from .analyse import predictions
        predictions(target)
        model_hashes={str(p):sha(p) for p in (target/'models').rglob('*') if p.is_file()}
        write(target/'qa/frozen_model_hashes.json',model_hashes)
        worker=Worker(target,budget);cfg=read(target/'data/runtime_config.json');prior=read(target/'data/train_demand_prior.json')
        results=[];runtime=[];queries=[];occupied=[];switches=[]
        for scene in read(target/'data/development_scenarios.json'):
            days={}
            for method in ['M0','PureMC','Basic','State']:
                r,c,q,z=one_day(scene,cfg,method,prior,worker,budget,protocol['extreme_flag_USD'])
                write(target/'raw'/f's{scene["seed"]}-{method}.json',r);days[method]=r
                runtime.append(c);queries.extend(q);occupied.extend(z)
                results.append(dict(scenario_seed=scene['seed'],method=method,profit_USD=r['ledger']['realized_profit'],
                    completed_fare=r['ledger']['completed_fare'],realized_cost=r['ledger']['realized_cost'],delivered=r['metrics']['delivered'],
                    empty_cost_USD=c['empty_cost_USD']))
            reference=days['PureMC']['original_trace'][:24]
            for method in ['M0','Basic','State']:
                for a,b in zip(days[method]['original_trace'][:24],reference):
                    switches.append(dict(scenario_seed=scene['seed'],period=a['period'],method=method,
                        business_changed=a['plan']['business_hash']!=b['plan']['business_hash'],
                        exact_same_before_state=a['before']==b['before']))
            print('DAY',scene['seed'],{m:round(r['ledger']['realized_profit'],2) for m,r in days.items()},'solves',budget.solves,flush=True)
            if sum(p.stat().st_size for p in target.rglob('*') if p.is_file())>protocol['max_output_bytes']:raise TimeoutError('2GiB budget')
        csv_write(target/'policy_profit_comparison.csv',results);csv_write(target/'runtime_comparison.csv',runtime)
        csv_write(target/'decision_switches.csv',switches);csv_write(target/'data/resource_queries.csv',queries)
        csv_write(target/'data/occupied_slots.csv',occupied)
        assert len(results)==24 and budget.solves==576
        worker.close();worker=None
        for p,expected in {**read(target/'qa/readonly_inputs.json'),**read(target/'qa/model_source_hashes.json'),**model_hashes,**own_sources}.items():
            if sha(p)!=expected:raise RuntimeError('protected source/input/model changed '+p)
        write(target/'qa/numerical.json',dict(status='PASSED',solves=budget.solves,full_days=24,all_optimal=True,
            all_trajectories_replayed=True,occupied_bonus_recalculated=True,historical_readonly=True,
            wall_seconds=time.monotonic()-budget.started,
            peak_working_set_bytes=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)))
        protocol['status']='NUMERICAL_COMPLETE_QA_PENDING';write(target/'protocol.json',protocol)
        from .analyse import finish
        finish(target)
    except BaseException as e:
        write(target/'qa/stop_condition_failure.json',dict(error=type(e).__name__,message=str(e),solves=budget.solves,wall_seconds=time.monotonic()-budget.started))
        protocol['status']='STOP_CONDITION_TRIGGERED';write(target/'protocol.json',protocol)
        raise
    finally:
        if worker is not None:worker.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,default=TARGET);run(p.parse_args().target)
