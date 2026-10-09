"""Execute the original selected actions, but book their independently compiled cash.

No second solve, tolerances, optimization variables, action selection, or physics
are changed. Raw floating solver values remain diagnostics, not executable cash.
"""
import copy
import math
from experiment_core.storage import digest
from model_families.single.engine.mt_prototype import DecisionPlan
from model_families.single.planning import PeriodAction, VehicleAction, compile_action
from model_families.single.runner import serialize_plan, state_snapshot
from experiments.cfs_resource_policy_iteration.action_analysis import canonical


def reconcile_integer_action(sim, raw, info):
    if info.get('status') != 'OPTIMAL':
        raise RuntimeError('Exact accounting requires an OPTIMAL original solve')
    if raw.period != sim.t or raw.revision != sim.revision:
        raise RuntimeError('Exact accounting refuses a stale decision plan')
    before = digest(state_snapshot(sim))
    actions = []
    sol, routes = raw[2], raw[3]
    assignments = {}
    for oid, vid in sol['x']:
        assignments.setdefault(vid, []).append(oid)
    for vid, rid in sol['y']:
        actions.append(VehicleAction(vid, routes[vid][rid], tuple(assignments.pop(vid, ()))))
    for vid, hub in sol['z']:
        actions.append(VehicleAction(vid, reposition_to=hub))
    if assignments:
        raise RuntimeError('Original integer assignment has no executable route')
    independent_info = {}
    checked = compile_action(sim, PeriodAction(sim.t, tuple(actions)), independent_info)
    original_action = canonical(serialize_plan(raw, sim.t, before, {}))
    checked_action = canonical(serialize_plan(checked, sim.t, before, {}))
    if original_action != checked_action or raw[5] != checked[5]:
        raise RuntimeError('Independent cash compilation changed the original actions')
    if digest(state_snapshot(sim)) != before:
        raise RuntimeError('Independent cash compilation mutated the input state')
    if not all(math.isfinite(float(x)) for x in (raw[0], raw[1], checked[0], checked[1])):
        raise RuntimeError('Nonfinite solver or integer-action amount')
    info['integer_action_accounting'] = dict(
        method='independent compile_action, identical selected integer actions',
        raw_solver_profit_USD=raw[0], raw_solver_objective=raw[1],
        raw_accounting=copy.deepcopy(info.get('accounting', {})),
        integer_action_profit_USD=checked[0], integer_action_objective=checked[1],
        integer_minus_raw_profit_USD=checked[0]-raw[0],
        integer_minus_raw_objective=checked[1]-raw[1],
        original_integer_action_hash=digest(original_action),
        independently_compiled_action_hash=digest(checked_action),
        integer_action_preserved=True)
    info['accounting'] = independent_info['accounting']
    # Preserve the ORIGINAL objects and list order used by the frozen physics.
    return DecisionPlan((checked[0], checked[1], *raw[2:]), raw.period, raw.revision)


def solve_myopic_exact(sim, info):
    raw = sim.solve_plan(sim.parameters['cross_hub'], sim.parameters['empty_cost'],
                         sim.parameters['reposition'], None, info)
    return reconcile_integer_action(sim, raw, info)
