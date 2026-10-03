import numpy as np
from gurobipy import GRB
from .engine.optimization.lower_layer import Lower_Layer
from .engine.simulation.tools import basic_cost
from .engine.simulation.transitions import self_update, update_order, update_var, update_vehicle

def solve_lower_layer(graph, cities, vehicles, orders, time, cost_matrix, solver_parameters=None, solve_info=None):
    """Solve one dispatch step and advance fleet/order state exactly once."""
    available = [vehicle.id for vehicle in vehicles.values() if vehicle.whether_city]
    travelling = [vehicle.id for vehicle in vehicles.values() if not vehicle.whether_city]
    objective = -float(basic_cost(vehicles, orders))
    solved = None

    if available:
        lower = Lower_Layer(graph, cities, vehicles, orders, "dispatch", [available, travelling], time)
        lower.get_decision()
        lower.constrain_1()
        lower.constrain_2()
        lower.constrain_3()
        lower.constrain_4()
        lower.constrain_5()
        lower.set_objective(cost_matrix)
        lower.model.setParam("OutputFlag", 0)
        lower.model.setParam("Threads", 1)
        for key, value in (solver_parameters or {}).items():
            lower.model.setParam({"time_limit_seconds": "TimeLimit", "mip_gap": "MIPGap", "threads": "Threads", "seed": "Seed"}[key], value)
        lower.model.optimize()
        solved = lower.model.status == GRB.OPTIMAL
        if solve_info is not None:
            solve_info.update(status="OPTIMAL" if solved else "FEASIBLE_LIMIT" if lower.model.SolCount else
                              "INFEASIBLE" if lower.model.status == GRB.INFEASIBLE else "NO_SOLUTION_LIMIT",
                              runtime=lower.model.Runtime, incumbent=float(lower.model.ObjVal) if lower.model.SolCount else None)
            if not lower.model.SolCount:
                # Restore IDs but never mutate dispatch state using absent X values.
                update_var(lower, vehicles, orders)
                lower.model.dispose()
                raise RuntimeError("legacy solver has no incumbent")
            solved = True
        if solved:
            objective = float(lower.model.objVal)
        else:
            self_update(vehicles, graph)
        # Also restores the real order ids when no optimum was found.
        update_var(lower, vehicles, orders, accept_incumbent=solve_info is not None)
        lower.model.dispose()
    else:
        self_update(vehicles, graph)

    update_vehicle(vehicles, battery_consume=10, battery_add=300, speed=20, G=graph)
    cancelled = update_order(orders, time, speed=20)
    return objective, solved, cancelled


def _supply_actions(mask, cities, capacity):
    """Deterministic supply baseline used by evaluation and reward control."""
    actions = []
    for row in np.asarray(mask, dtype=bool):
        candidates = np.flatnonzero(row).tolist()
        actions.append(max(
            candidates,
            key=lambda city_id: (cities[city_id].city_seat_count(capacity), -city_id),
        ))
    return actions
