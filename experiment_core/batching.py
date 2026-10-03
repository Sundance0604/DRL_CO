"""Finite grids and immutable experimental conditions, independent of physics."""
from decimal import Decimal
from .contracts import PlatformError


def expand_ranges(ranges):
    result = {}
    for path, item in ranges.items():
        start, stop, step = map(lambda v: Decimal(str(v)), (item.start, item.stop, item.step))
        if stop < start or (item.scale == "log" and (start <= 0 or step <= 1)):
            raise PlatformError("SWEEP_RANGE", "ascending bounds required; logarithmic ratio must exceed one", path)
        values, value = [], start
        while value <= stop:
            values.append(int(value) if value == int(value) else float(value))
            if len(values) > 1000:
                raise PlatformError("BATCH_SIZE", "range exceeds 1000 values", path)
            value = value + step if item.scale == "linear" else value * step
        result[path] = values
    return result


def assign_parameter(draft, path, value):
    from .families import get_family
    parts = path.split(".")
    if len(parts) < 3 or parts[0] not in {"model", "controller", "value_function", "solver"} or parts[1] != "parameters":
        raise PlatformError("SWEEP_PATH", "scan a declared component parameter; choose data and seeds separately", path)
    if parts[0] != "solver":
        f = get_family(draft["model"]["id"])
        cls = f.REGISTRY[parts[0]].get(draft[parts[0]]["id"])
        if cls is None or parts[2] not in cls.model_fields:
            raise PlatformError("SWEEP_PATH", "parameter is not declared by selected family/component", path)
        field = cls.model_fields[parts[2]]
        if len(parts) > 3 and parts[2] not in draft[parts[0]]["parameters"]:
            draft[parts[0]]["parameters"][parts[2]] = field.get_default(call_default_factory=True)
    node = draft
    for part in parts[:-1]:
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                raise PlatformError("SWEEP_PATH", "invalid array index", path) from None
        else:
            if part not in node:
                if part == "parameters":
                    node[part] = {}
                else:
                    raise PlatformError("SWEEP_PATH", "invalid parameter path", path)
            node = node[part]
    if isinstance(node, list):
        try:
            node[int(parts[-1])] = value
        except (ValueError, IndexError):
            raise PlatformError("SWEEP_PATH", "invalid array index", path) from None
    else:
        node[parts[-1]] = value
