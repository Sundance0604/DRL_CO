"""Compatibility import; implementation belongs to the legacy family."""
import importlib as _importlib
import sys as _sys
if __name__ == "__main__":
    from runpy import run_module
    run_module("model_families.legacy.engine.domain.vehicle", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("model_families.legacy.engine.domain.vehicle")
    
    
