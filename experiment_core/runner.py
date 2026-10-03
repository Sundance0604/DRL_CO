"""Dispatch only. No model mathematics or branch-specific metrics live here."""
def execute(spec, *args, **kwargs):
    from .families import get_family
    return get_family(spec.model.id).execute(spec, *args, **kwargs)

def __getattr__(name):
    if name == "matching":
        from model_families.single.runner import matching
        return matching
    raise AttributeError(name)
