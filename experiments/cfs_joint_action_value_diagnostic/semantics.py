"""Per-query evidence for resource base, slot counts, support and observation time."""
from experiment_core.storage import digest
from experiments.cfs_resource_policy_iteration.protocol import resource_world,pair_audit
from experiments.cfs_resource_value_state_bucket_exact_accounting.pressure import summary
from .actions import historical,score,releases
from .states import restore_row

def audit(row,actions):
    state,_=restore_row(row);policies,prior=historical(row['condition']);online=summary(state,prior)
    queries=[];alignment=[];fallback=[]
    for a in actions:
        bscore=score(state,a['serial'],policies['B2'],prior)
        occupied={(r['time'],r['hub'],r['q']):r for r in bscore['occupied_coefficients']}
        for qrow in bscore['lookup_queries']:
            h=qrow['hub'];q=qrow['q'];pp=online['pressure'][h]
            low=resource_world(state,h,q-1);high=resource_world(state,h,q);pair_audit(low,high,h)
            recomputed=[summary(x,prior) for x in (resource_world(state,h,0),low,high)]
            semantic=dict(state_id=row['state_id'],action_hash=a['action_hash'],condition=row['condition'],
                scenario_seed=row['scenario_seed'],source=row['source'],decision_time=state.t,value_time=qrow['release_time'],
                release_time=qrow['release_time'],bucket_observation_time=state.t,state_hash=row['state_hash'],hub=h,q=q,
                removed_idle=online['idle'][h],injected_resources=q,backlog=pp['Bload'],expected_demand=pp['Dnear_prior'],
                known_near_demand=pp['Dnear_known_bookings'],idle=pp['Midle'],incoming=pp['Inear'],
                incoming_vehicle_ids=pp['near_incoming_vehicle_ids'],pressure_ratio=pp['R'],thresholds=prior['thresholds'],
                bucket=qrow['bucket'],native_cell_counts=qrow['source_corners'],interpolation_corners=qrow['source_corners'],
                fallback_reason=qrow['fallback_reason'],fallback=qrow['fallback'],value_USD=qrow['value_USD'],
                selected_occupied=True,online_q='count in this (release_time,hub) group, NOT whole inventory',
                counterfactual_fleet=len(high.vehs),fleet_exceeds_K=len(high.vehs)>len(state.vehs),
                time_proxy_status='confirmed_consistent_declared_proxy' if state.t!=qrow['release_time'] else 'same_time',
                bucket_label='relative construction quantile, NOT absolute surplus/balance/scarcity')
            queries.append(semantic)
            alignment.append(dict(state_id=row['state_id'],action_hash=a['action_hash'],hub=h,q=q,
                natural_R=pp['R'],removed_R=recomputed[0]['pressure'][h]['R'],
                qminus1_R=recomputed[1]['pressure'][h]['R'],q_R=recomputed[2]['pressure'][h]['R'],
                natural_bucket=online['bucket'][h],removed_bucket=recomputed[0]['bucket'][h],
                qminus1_bucket=recomputed[1]['bucket'][h],q_bucket=recomputed[2]['bucket'][h],
                inherited_label_used_for_both=True,pair_status='confirmed_consistent_original_pre_injection_label',
                projection_querybase_status='unresolved_approximation_not_an_engine_bug'))
        potential=a['serial']['solver'].get('release_coefficients',[])
        fallback.append(dict(state_id=row['state_id'],action_hash=a['action_hash'],labels=a['labels'],
            selected_coefficient_queries=len(bscore['lookup_queries']),
            selected_fallback_count=sum(r['fallback'] for r in bscore['lookup_queries']),
            selected_absolute_bonus_USD=sum(abs(r['value_USD']) for r in bscore['occupied_coefficients']),
            selected_fallback_absolute_value_USD=sum(abs(r['value_USD']) for r in bscore['lookup_queries'] if r['fallback']),
            optimizer_all_coefficients=len(potential),optimizer_all_coefficient_source='release_coefficients (only original generating solve)',
            query_vs_economic_denominators_not_equated=True,
            terminal_release_count=sum(r['terminal'] for r in bscore['release_events'])))
    return queries,alignment,fallback
