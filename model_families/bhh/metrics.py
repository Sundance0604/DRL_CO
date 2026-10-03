def metric(key, label, unit, sense, population, accounting, models):
    return {"key":key, "label":label, "unit":unit, "sense":sense, "population":population,
            "aggregation":"scenario_mean", "time_scope":"configured terminal policy",
            "accounting_version":accounting, "compatible_model_families":models}
DEFINITIONS = [metric("business_cost","BHH cost","currency","minimize","run/load","bhh-cost-v1",["bhh_finite","bhh_steady"]),
metric("delivered_load","Delivered stops","stops","maximize","divisible commodities","bhh-cost-v1",["bhh_finite"])]
TRACE_SCHEMA={"version":"bhh-trace/v1","granularity":"window-plan","steady_has_time_series":False}
