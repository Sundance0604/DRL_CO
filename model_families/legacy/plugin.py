from experiment_core.contracts import Contract, PlatformError
from experiment_core.storage import read_json, run_dir
from . import parameters as p
from . import data
from .generation import Generation
FAMILY_ID="legacy"
DATASET_FAMILY="legacy_dispatch"
ACCOUNTING="legacy-assignment-v1"
class Empty(Contract):
    pass
REGISTRY = {
"model":{"legacy_dispatch":p.LegacyParameters},
"controller":{"myopic":Empty,"train_sac":p.SacTrainParameters,"candidate_sac":p.SacParameters},
"value_function":{"zero":Empty}}
FRAMEWORKS = [{"id":"supply","label":"Supply heuristic","controller":"myopic","value_function":"zero","training_required":False},{"id":"sac","label":"Candidate SAC","controller":"candidate_sac","value_function":"zero","training_controller":"train_sac","training_required":True}]
from .mathematics import PROFILES
MATH = PROFILES["supply"]
def is_training(spec):
    return spec.controller.id in {"train_sac"}
def backend(model_id):
    return "gurobi"
def execute(*args, **kwargs):
    from .runner import execute as run
    return run(*args, **kwargs)
def metric_definitions():
    from .metrics import DEFINITIONS
    return DEFINITIONS

def checkpoint_reference(spec):
    return spec.controller if spec.controller.id == "candidate_sac" else None

def supports_samples(spec):
    return True

def validate_selection(spec, data, scenarios):
    if data["family"] != DATASET_FAMILY:
        raise PlatformError("DATASET_MODEL_MISMATCH","dataset belongs to another research family")
    if spec.evaluation.warmup_periods:
        raise PlatformError("UNSUPPORTED_ACCOUNTING","warm-up measurement is not implemented")
    if spec.evaluation.accounting_version != ACCOUNTING:
        raise PlatformError("ACCOUNTING_MISMATCH","accounting version does not match research family")
    if spec.evaluation.terminal_policy != "report_pending":
        raise PlatformError("UNSUPPORTED_TERMINAL","legacy supports report_pending only")
    if spec.evaluation.information_set != "online":
        raise PlatformError("UNSUPPORTED_INFORMATION","legacy controllers are online")
    if is_training(spec) and spec.dataset.split != "train":
        raise PlatformError("TRAIN_SPLIT_REQUIRED","SAC fitting requires train split")

    if spec.controller.id == "candidate_sac":
        ref = spec.controller
        meta = read_json(run_dir(ref.parameters["checkpoint_run"])/"checkpoint.json")
        expected = "candidate-sac/v1"
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
    return {"id":FAMILY_ID,"name":"DRL / SAC","accent":"#3468b2","dataset_family":DATASET_FAMILY,"models":list(REGISTRY["model"]),
    "frameworks":FRAMEWORKS,"math":MATH,"math_profiles":PROFILES,"trace":TRACE_SCHEMA,"analysis_supported":False,
    "accounting":ACCOUNTING,"backends":{m:backend(m) for m in REGISTRY["model"]},"generation_schema":Generation.model_json_schema(),"templates":templates(),"metrics":metric_definitions()}
