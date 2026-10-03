def metric(key, label, unit, sense, population, accounting, models):
    return {"key":key, "label":label, "unit":unit, "sense":sense, "population":population,
            "aggregation":"scenario_mean", "time_scope":"configured terminal policy",
            "accounting_version":accounting, "compatible_model_families":models}
DEFINITIONS = [metric(k,l,u,s,p,"legacy-assignment-v1",["single_level_matching"]) for k,l,u,s,p in [
("operating_profit","Operating profit","currency","maximize","run"),
("business_cost","Assignment-accounted cost","currency","minimize","run"),
("augmented_solver_objective","Augmented objective","currency","diagnostic","period"),
("assigned","Assigned orders","orders","maximize","orders"),("delivered","Delivered orders","orders","maximize","orders"),
("pending","Pending orders","orders","minimize","orders"),
("assigned_order_rate","Assignment rate","fraction","maximize","orders"),
("delivered_order_rate","Delivery rate","fraction","maximize","orders"),
("upper_bound","Perfect-information upper bound","currency","upper-bound","full horizon")]]
TRACE_SCHEMA = {"version":"single-trace/v2", "granularity":"period", "fields":["before","observation","plan","after","measurements","invariants"]}
