"""Canonical, order-insensitive serialization of SAME-state action comparisons."""
import math
from numbers import Integral,Real
from experiment_core.storage import read_json,atomic_json,digest
from .common import OUTPUT,map_jobs,now,write_csv,file_hash
from .protocol import read_compressed,write_compressed,restore,decide_same_state,continuation,action_hash
from .policies import load_policies

ORIGINAL_DEFINITION='Sort assignments/routes/repositions/expired before hashing; vehicle IDs retained; same pi0 state bank, all t, no future input.'
DEFINITION='Sort assignments/routes/repositions/expired before hashing; ordered sequences normalized to lists and exactly equal scalar representations normalized without rounding; vehicle IDs retained; same pi0 state bank, all t, no future input.'

def normalized_json(value):
    if isinstance(value,dict):return {k:normalized_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [normalized_json(v) for v in value]
    if isinstance(value,Integral):return int(value)
    if isinstance(value,Real):
        number=float(value);assert math.isfinite(number)
        return int(number) if number.is_integer() else number
    return value

def canonical(plan):
    return {k:sorted(normalized_json(plan[k]),key=digest) for k in ['assignments','routes','repositions','expired']}

def verify_declaration():
    declaration=read_json(OUTPUT/'manifests/action_canonicalization_preregistration.json')
    if declaration['definition']==DEFINITION:return
    assert declaration['definition']==ORIGINAL_DEFINITION
    revision=read_json(OUTPUT/'manifests/action_serialization_normalization_revision.json')
    assert revision['previous_definition']==ORIGINAL_DEFINITION and revision['revised_definition']==DEFINITION
    assert not revision['diagnostic_seen'] and not revision['formal_seen']

def control_modes(plan):
    """Idle/wait is absence; delayed accepted routes must not look immediate."""
    modes={r['vehicle_id']:('route',r.get('delay',0)) for r in plan['routes']}
    modes.update({r['vehicle_id']:('reposition',r['hub']) for r in plan['repositions']})
    return modes

def state_bank(request,policy):
    saved=read_compressed(OUTPUT/request['trace_ref'])
    if 'state_packets' in saved:return saved['state_packets']
    # Formal traces intentionally contain no extra state collection. Independently
    # reconstruct pi0's identical deterministic trajectory; never change a policy.
    path=OUTPUT/'raw/analysis/formal_state_banks'/(request['id']+'.json.gz')
    identity=digest(dict(trace_hash=file_hash(OUTPUT/request['trace_ref']),pi0_hash=policy['policy_hash']))
    if path.exists():
        packet=read_compressed(path);assert packet['identity']==identity
        return packet['state_packets']
    from model_families.single.data import matching_state
    from experiments.cfs_rollout_target_alignment.queries import pack_state
    s,orders=matching_state(request['scenario'],request['parameters'],request['solver'])
    s.begin_period([o for o in orders.values() if o.book_time==0])
    reproduced=continuation(s,orders,policy,request['prior'],collect_times=range(s.T))
    assert reproduced['final_hash']==saved['final_hash']
    assert abs(reproduced['metrics']['realized_profit']-saved['metrics']['realized_profit'])<1e-6
    assert [action_hash(r['plan']) for r in reproduced['trace'] if not r.get('drain')]==[action_hash(r['plan']) for r in saved['trace'] if not r.get('drain')]
    states={str(t):pack_state(v) for t,v in reproduced['states'].items()}
    write_compressed(path,dict(identity=identity,state_packets=states,physical_actions_and_profit_identical=True,
        method='independent pi0 resolve, same frozen code/parameters/seeds; analysis-only state collection'))
    return states

def job(request):
    policies=[load_policies(n)[request['condition']] for n in range(5)];rows=[]
    states=state_bank(request,policies[0])
    for t in range(request['scenario']['horizon']):
        s,_=restore(request['scenario'],request['parameters'],request['solver'],states[str(t)])
        plans=[canonical(decide_same_state(s,p,request['prior'])) for p in policies]
        for n in range(4):
            a,b=plans[n],plans[n+1]
            assert (a!=b)==(digest(a)!=digest(b)),'Canonical equality and recorded action hash disagree'
            rows.append(dict(stage=request['stage'],condition=request['condition'],scenario_seed=request['scenario']['seed'],time=t,iteration=n+1,
                reference_state_policy='pi0',state_hash=digest(states[str(t)]),policy_a=f'pi{n}',policy_b=f'pi{n+1}',
                action_hash_a=digest(a),action_hash_b=digest(b),full_action_disagree=int(a!=b),
                routes_disagree=int(a['routes']!=b['routes']),assignments_disagree=int(a['assignments']!=b['assignments']),
                wait_reposition_disagree=int(control_modes(a)!=control_modes(b)),
                release_hub_disagree=int({r['vehicle_id']:r['e'] for r in a['routes']}!={r['vehicle_id']:r['e'] for r in b['routes']})))
    return rows

def preregister():
    path=OUTPUT/'manifests/action_canonicalization_preregistration.json'
    if path.exists():verify_declaration();return
    assert not (OUTPUT/'processed/diagnostic_runs.json').exists() and not (OUTPUT/'processed/formal_runs.json').exists()
    atomic_json(path,dict(created_at=now(),definition=DEFINITION,diagnostic_seen=False,formal_seen=False,
        purpose='serialization ordering is not a physical action change; no symmetry quotient or profit-based choices'))

def run(cfg):
    qa=OUTPUT/'qa/canonical_same_state_action_audit.json'
    if qa.exists() and read_json(qa).get('formal_states',0)==160*24:
        assert read_json(qa)['status']=='PASSED';return
    verify_declaration()
    rawold=OUTPUT/'raw/analysis/unsorted_action_disagreement.json.gz'
    if rawold.exists():old=read_compressed(rawold)
    else:
        old=read_json(OUTPUT/'processed/policy_action_disagreement.json')
        write_compressed(rawold,old)
    jobs=[]
    for r in read_json(OUTPUT/'processed/diagnostic_runs.json'):
        if r['policy']!='P0':continue
        spec=read_json(OUTPUT/'raw/runs'/r['id']/'spec.json');jobs.append(spec|dict(trace_ref=r['trace_ref']))
    results,wall=map_jobs(job,jobs,'canonical same-state action QA',cfg['max_parallel'])
    rows=[r for rr in results for r in rr]
    oldmap={(r['condition'],r['scenario_seed'],r['time'],r['iteration']):r for r in old}
    ordering_only=sum(r['full_action_disagree']!=oldmap[r['condition'],r['scenario_seed'],r['time'],r['iteration']]['full_action_disagree'] for r in rows)
    waitflagchanges=sum(r['wait_reposition_disagree']!=oldmap[r['condition'],r['scenario_seed'],r['time'],r['iteration']]['wait_reposition_disagree'] for r in rows)
    write_csv(OUTPUT/'processed/policy_action_disagreement.csv',rows);atomic_json(OUTPUT/'processed/policy_action_disagreement.json',rows)
    formaljobs=[]
    for r in read_json(OUTPUT/'processed/formal_runs.json'):
        if r['policy']!='P0':continue
        spec=read_json(OUTPUT/'raw/formal_runs'/r['id']/'spec.json')
        formaljobs.append(spec|dict(trace_ref=r['trace_ref']))
    formalresults,formalwall=map_jobs(job,formaljobs,'formal canonical same-state action comparison',cfg['max_parallel'])
    formalrows=[r for rr in formalresults for r in rr]
    write_csv(OUTPUT/'processed/formal_policy_action_disagreement.csv',formalrows)
    atomic_json(OUTPUT/'processed/formal_policy_action_disagreement.json',formalrows)
    atomic_json(qa,dict(status='PASSED',states=len(jobs)*24,comparisons=len(rows),definition=DEFINITION,
        serialization_only_full_flag_changes=ordering_only,raw_unsorted_preserved='raw/analysis/unsorted_action_disagreement.json.gz',wall_seconds=wall,
        formal_states=len(formaljobs)*24,formal_comparisons=len(formalrows),formal_wall_seconds=formalwall,
        secondary_delay_aware_wait_flag_changes=waitflagchanges,wait_definition='vehicle route participation and accepted departure delay, or reposition hub; absent vehicle means wait',
        formal_pi0_reproduction_exact=True,formal_used_for_policy_selection=False))
