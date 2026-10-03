from experiment_core.contracts import Contract, PlatformError
from experiment_core.storage import read_json, run_dir
from . import parameters as p
from . import data
from .generation import Generation
FAMILY_ID="single"
DATASET_FAMILY="single_level_matching"
ACCOUNTING="legacy-assignment-v1"
class Empty(Contract):
    pass
REGISTRY = {
"model":{"single_level_matching":p.MatchingParameters},
"controller":{"myopic":Empty,"rollout":p.RolloutParameters,"train_value":p.TrainParameters,"oracle_lp":Empty,"oracle_mip":Empty},
"value_function":{"zero":Empty,"fluid_dual":p.FluidParameters,"learned_hub_time":p.LearnedParameters}}
FRAMEWORKS = [{"id":"myopic","label":"Myopic","controller":"myopic","value_function":"zero","training_required":False},{"id":"fluid","label":"Analytical / fluid value","controller":"myopic","value_function":"fluid_dual","training_required":False},{"id":"rollout","label":"Sampled rollout","controller":"rollout","value_function":"zero","training_required":False},{"id":"oracle_lp","label":"Perfect-information LP bound","controller":"oracle_lp","value_function":"zero","information_set":"oracle","training_required":False},{"id":"oracle_mip","label":"Perfect-information MIP benchmark","controller":"oracle_mip","value_function":"zero","information_set":"oracle","training_required":False},{"id":"learned","label":"Learned hub value (optional supervised component)","controller":"myopic","value_function":"learned_hub_time","training_controller":"train_value","training_required":True}]
from .mathematics import PROFILES
MATH = PROFILES["myopic"]
def is_training(spec):
    return spec.controller.id in {"train_value"}
def backend(model_id):
    return "gurobi"
def execute(*args, **kwargs):
    from .runner import execute as run
    return run(*args, **kwargs)
def metric_definitions():
    from .metrics import DEFINITIONS
    return DEFINITIONS

def checkpoint_reference(spec):
    return spec.value_function if spec.value_function.id == "learned_hub_time" else None

def supports_samples(spec):
    return True

def analysis_adapter():
    from . import analysis
    return analysis

def validate_selection(spec, data, scenarios):
    if data["family"] != DATASET_FAMILY:
        raise PlatformError("DATASET_MODEL_MISMATCH","dataset belongs to another research family")
    if spec.evaluation.warmup_periods:
        raise PlatformError("UNSUPPORTED_ACCOUNTING","warm-up measurement is not implemented")
    if spec.evaluation.accounting_version != ACCOUNTING:
        raise PlatformError("ACCOUNTING_MISMATCH","accounting version does not match research family")
    if spec.controller.id == "rollout" and spec.value_function.id != "zero":
        raise PlatformError("UNSUPPORTED_COMBINATION","rollout currently requires the zero-value base policy")
    if spec.controller.id in {"oracle_lp","oracle_mip"} and (spec.evaluation.information_set != "oracle" or spec.value_function.id != "zero" or spec.evaluation.terminal_policy != "report_pending"):
        raise PlatformError("BOUND_PROFILE","oracle benchmarks require oracle, zero value and report_pending")
    if spec.evaluation.information_set == "oracle" and spec.controller.id not in {"oracle_lp","oracle_mip"}:
        raise PlatformError("UNSUPPORTED_INFORMATION","online framework cannot claim oracle information")
    if is_training(spec) and (spec.dataset.split != "train" or spec.value_function.id != "zero"):
        raise PlatformError("TRAIN_SPLIT_REQUIRED","supervised fitting requires train split and zero reference")

    if spec.value_function.id == "learned_hub_time":
        ref = spec.value_function
        meta = read_json(run_dir(ref.parameters["checkpoint_run"])/"checkpoint.json")
        expected = "hub-time-value/v1"
        if meta.get("feature_schema") != expected:
            raise PlatformError("CHECKPOINT_FAMILY","checkpoint belongs to another component")
        if meta["model_parameters"] != spec.model.parameters:
            raise PlatformError("CHECKPOINT_PARAMETERS","physical parameter mismatch")
        if meta["dataset_hash"] == spec.dataset.revision and set(meta["train_scenario_ids"]) & {s["id"] for s in scenarios}:
            raise PlatformError("TRAIN_TEST_LEAKAGE","training scenarios are not held-out evaluation")

def manifest():
    from .mathematics import PROFILES
    from .templates import templates
    from .metrics import TRACE_SCHEMA
    return {"id":FAMILY_ID,"name":"Single-level model","accent":"#138579","dataset_family":DATASET_FAMILY,"models":list(REGISTRY["model"]),
    "frameworks":FRAMEWORKS,"math":MATH,"math_profiles":PROFILES,"trace":TRACE_SCHEMA,"analysis_supported":True,
    "accounting":ACCOUNTING,"backends":{m:backend(m) for m in REGISTRY["model"]},"generation_schema":Generation.model_json_schema(),"templates":templates(),"metrics":metric_definitions()}
