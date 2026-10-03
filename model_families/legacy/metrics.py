def metric(key, label, unit, sense, population, accounting, models):
    return {"key":key, "label":label, "unit":unit, "sense":sense, "population":population,
            "aggregation":"scenario_mean", "time_scope":"configured terminal policy",
            "accounting_version":accounting, "compatible_model_families":models}
DEFINITIONS = [metric(k,l,u,s,p,"legacy-assignment-v1",["legacy_dispatch"]) for k,l,u,s,p in [
("operating_profit","Legacy dispatch profit","currency","maximize","run"),("assigned","Assigned orders","orders","maximize","orders"),("pending","Pending orders","orders","minimize","orders")]]
TRACE_SCHEMA={"version":"legacy-trace/v1","granularity":"period","verified_delivery_events":False}
