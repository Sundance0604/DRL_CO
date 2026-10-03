"""Generic plugin negotiation across separately owned research families."""
from . import contracts as c
from .families import all_families, get_family

def parameter_schema(cls):
    schema=cls.model_json_schema()
    for key,field in cls.model_fields.items():
        if not field.is_required():
            schema["properties"][key]["default"]=field.get_default(call_default_factory=True)
    return schema

def descriptors(family=None):
    entries=[]
    for f in ([get_family(family)] if family else all_families()):
        for kind, plugins in f.REGISTRY.items():
            for name, cls in plugins.items():
                entries.append({"family":f.FAMILY_ID,"kind":kind,"id":name,"version":"1", "parameters_schema":parameter_schema(cls),
                "capabilities":{"resume":False,"training":name.startswith("train_"),"arbitrary_imports":False}, "ui_schema":{"section":kind}})
    return entries

def resolve(spec):
    spec=c.RunSpec.model_validate(spec)
    f=get_family(spec.model.id)
    if spec.family and spec.family != f.FAMILY_ID:
        raise c.PlatformError("FAMILY_MISMATCH","model does not belong to selected workspace","/family")
    spec.family=f.FAMILY_ID
    for kind in ("model","controller","value_function"):
        ref=getattr(spec,kind)
        cls=f.REGISTRY[kind].get(ref.id)
        if cls is None:
            raise c.PlatformError("UNSUPPORTED_COMBINATION","component is not registered in this research family",f"/{kind}/id")
        ref.parameters=cls.model_validate(ref.parameters).model_dump()
    if spec.solver.backend != f.backend(spec.model.id):
        raise c.PlatformError("SOLVER_MISMATCH","solver does not match model", "/solver/backend")
    matches=[x for x in f.FRAMEWORKS if x.get("model", spec.model.id)==spec.model.id and
             ((x["controller"]==spec.controller.id and x["value_function"]==spec.value_function.id) or
              (x.get("training_controller")==spec.controller.id and spec.value_function.id=="zero"))]
    canonical=matches[0]["id"] if matches else spec.controller.id+"."+spec.value_function.id
    if spec.framework_id and spec.framework_id != canonical:
        raise c.PlatformError("FRAMEWORK_MISMATCH","framework label does not match components","/framework_id")
    spec.framework_id=canonical
    if spec.execution.training_seeds is not None and not f.is_training(spec):
        raise c.PlatformError("TRAINING_NOT_APPLICABLE","training seeds only apply to a fitting framework")
    return spec

def schemas():
    from .analysis_contracts import AnalysisSpec
    return {"run-spec":c.RunSpec.model_json_schema(),"batch-spec":c.BatchSpec.model_json_schema(),"analysis-spec":AnalysisSpec.model_json_schema(),"application-config":c.ApplicationConfig.model_json_schema(),"solver_backend.gurobi":c.SolverParameters.model_json_schema(),"solver_backend.cpu":c.SolverParameters.model_json_schema(),
    **{f"generation.{f.FAMILY_ID}":f.Generation.model_json_schema() for f in all_families()},
    **{f"{d['family']}.{d['kind']}.{d['id']}":d["parameters_schema"] for d in descriptors()}}
