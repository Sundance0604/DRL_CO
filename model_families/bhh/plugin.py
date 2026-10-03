from experiment_core.contracts import Contract, PlatformError
from experiment_core.storage import read_json, run_dir
from . import parameters as p
from . import data
from .generation import Generation
FAMILY_ID="bhh"
DATASET_FAMILY="bhh"
ACCOUNTING="bhh-cost-v1"
class Empty(Contract):
    pass
REGISTRY = {
"model":{"bhh_steady":p.BHHSteadyParameters,"bhh_finite":p.BHHParameters,"bhh_spatial":p.SpatialParameters},
"controller":{"myopic":Empty,"rolling_horizon":p.RollingParameters},
"value_function":{"zero":Empty}}
FRAMEWORKS = [{"id":"steady","label":"Stationary analytic study","model":"bhh_steady","controller":"myopic","value_function":"zero","training_required":False},{"id":"finite","label":"Finite-horizon oracle","model":"bhh_finite","controller":"myopic","value_function":"zero","information_set":"oracle","training_required":False},{"id":"rolling","label":"Rolling-horizon commitments","model":"bhh_finite","controller":"rolling_horizon","value_function":"zero","training_required":False},{"id":"spatial","label":"Spatial calibration","model":"bhh_spatial","controller":"myopic","value_function":"zero","training_required":False}]
from .mathematics import PROFILES
MATH = PROFILES["steady"]
def is_training(spec):
    return spec.controller.id in set()
def backend(model_id):
    return "cpu" if model_id in {"bhh_steady","bhh_spatial"} else "gurobi"
def execute(*args, **kwargs):
    from .runner import execute as run
    return run(*args, **kwargs)
def metric_definitions():
    from .metrics import DEFINITIONS
    return DEFINITIONS

def checkpoint_reference(spec):
    return None

def supports_samples(spec):
    return spec.model.id == "bhh_finite"

def validate_selection(spec, data, scenarios):
    if data["family"] != DATASET_FAMILY:
        raise PlatformError("DATASET_MODEL_MISMATCH","dataset belongs to another research family")
    if spec.evaluation.warmup_periods:
        raise PlatformError("UNSUPPORTED_ACCOUNTING","warm-up measurement is not implemented")
    if spec.evaluation.accounting_version != ACCOUNTING:
        raise PlatformError("ACCOUNTING_MISMATCH","accounting version does not match research family")
    if any(s["nodes"] != ["0","1"] for s in scenarios):
        raise PlatformError("BHH_TWO_CITY","BHH requires exactly cities 0 and 1")
    allowed = {"bhh_steady":{"myopic"},"bhh_spatial":{"myopic"},"bhh_finite":{"myopic","rolling_horizon"}}
    if spec.controller.id not in allowed[spec.model.id]:
        raise PlatformError("UNSUPPORTED_COMBINATION","controller unavailable for BHH variant")
    if spec.model.id == "bhh_finite" and spec.controller.id == "myopic" and spec.evaluation.information_set != "oracle":
        raise PlatformError("ORACLE_REQUIRED","full-horizon optimization requires oracle declaration")

def manifest():
    from .mathematics import PROFILES
    from .templates import templates
    from .metrics import TRACE_SCHEMA
    return {"id":FAMILY_ID,"name":"BHH","accent":"#bc6e20","dataset_family":DATASET_FAMILY,"models":list(REGISTRY["model"]),
    "frameworks":FRAMEWORKS,"math":MATH,"math_profiles":PROFILES,"trace":TRACE_SCHEMA,"analysis_supported":False,
    "accounting":ACCOUNTING,"backends":{m:backend(m) for m in REGISTRY["model"]},"generation_schema":Generation.model_json_schema(),"templates":templates(),"metrics":metric_definitions()}
