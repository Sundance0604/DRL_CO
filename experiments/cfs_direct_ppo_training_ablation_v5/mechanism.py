"""Small preregistered common-state probes, never used for checkpoint selection."""
import torch
from .common import *
from .run import policy
from experiments.cfs_direct_ppo_diagnosis.common import restore,canonical,finish
from experiments.cfs_direct_ppo_v3.qa import no_mc_guard,replay
from model_families.single.runner import state_snapshot
from model_families.single.strategic.contracts import identity
from model_families.single.learning.ppo.networks import sample_action


def run():
    m=load()
    if (TARGET/'qa/mechanism.json').exists():raise RuntimeError('mechanism complete')
    budget=Budget();counts=0;cases=[]
    try:
        for name in NAMES:
            refs=references(name)
            # Fixed state identity, not selected by result or learned-model visitation.
            specs=[next(r for r in refs if r['period']==t and r['scenario_seed']==21008+i)
                   for i,t in enumerate((4,12,20))]
            for variant in ('A','B','C','D'):
                chosen=None
                for seed in SEEDS:
                    cfg=configuration(name,variant,seed)
                    with no_mc_guard():p=policy(name,variant,seed,m)
                    for i,spec in enumerate(specs):
                        scene=next(s for s in scenes(name,'validation') if s['seed']==spec['scenario_seed'])
                        record=read_json(spec['source'])
                        with no_mc_guard():
                            ep,o=restore(scene,cfg,record,spec['period'],budget)
                            before=state_snapshot(ep._sim);det=p.act(o)
                            gen=torch.Generator().manual_seed(63000+seed+100*i+NAMES.index(name)*1000)
                            controls=[('zero',[0.]*10),('det',list(det))]
                            controls += [('sample'+str(j),sample_action(p.network,o,gen,False)[0].tolist()) for j in range(2)]
                            rows=[]
                            for control,signal in controls:
                                probe=ep.probe(signal)
                                if before!=state_snapshot(ep._sim):raise RuntimeError('probe mutated common state')
                                rows.append(dict(control=control,signal=signal,probe=probe,canonical=canonical(probe['plan'],before)))
                            counts+=4
                        payload=dict(condition=name,variant=variant,seed=seed,spec=spec,state=before,rows=rows)
                        file=TARGET/'mechanism'/f'{name}-{variant}-{seed}-t{spec["period"]}.json'
                        write_json(file,payload)
                        changed=next((r for r in rows if r['canonical']!=rows[1]['canonical']),None)
                        if chosen is None and changed is not None:chosen=(spec,scene,record,cfg,rows[1],changed,file)
                if chosen is not None:
                    spec,scene,record,cfg,det,change,file=chosen;profits=[]
                    # Same A30001 policy from next period for both branches and allvariants.
                    with no_mc_guard():
                        continuation=policy(name,'A',30001,m)
                        for label,action in (('det',det),('changed',change)):
                            ep,o=restore(scene,cfg,record,spec['period'],budget)
                            r=finish(ep,action['signal'],continuation);r['replay']=replay(scene,cfg,r)
                            path=TARGET/'mechanism'/f'{name}-{variant}-counterfactual-{label}.json'
                            write_json(path,r);profits.append(r['ledger']['realized_profit'])
                    cases.append(dict(condition=name,variant=variant,source_probe=file.relative_to(TARGET).as_posix(),
                        control=change['control'],scenario_seed=scene['seed'],period=spec['period'],
                        difference_USD=profits[1]-profits[0],common_continuation='A30001 selected',selected_by='first changed, not profit'))
                else:cases.append(dict(condition=name,variant=variant,missing_reason='no canonical switch in frozen finite probes'))
        write_json(TARGET/'mechanism/cases.json',cases)
        if counts!=432:raise RuntimeError('probe denominator mismatch')
        write_json(TARGET/'qa/mechanism.json',dict(status='PASSED',probe_count=counts,
            fixed_states=9,model_state_pairs=108,counterfactual_cases=sum('difference_USD' in r for r in cases),
            same_future_common_continuation=True))
    finally:budget.save()
