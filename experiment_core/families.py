"""Trusted registration of independent model packages. Never import user-supplied modules."""
from importlib import import_module
from .contracts import PlatformError
PACKAGES = {"single":"model_families.single.plugin", "legacy":"model_families.legacy.plugin", "bhh":"model_families.bhh.plugin"}
OWNERS = {"single_level_matching":"single", "legacy_dispatch":"legacy", "bhh_steady":"bhh","bhh_finite":"bhh","bhh_spatial":"bhh"}
def get_family(value):
    key = OWNERS.get(value, value)
    if key not in PACKAGES:
        raise PlatformError("FAMILY_NOT_FOUND","unknown model family", "/family")
    return import_module(PACKAGES[key])
def all_families():
    return [get_family(k) for k in PACKAGES]
