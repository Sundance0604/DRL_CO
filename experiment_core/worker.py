"""Spawn-safe worker: immutable file inputs, no SQLite or inherited solver state."""

import sys
import traceback
from .contracts import PlatformError
from .datasets import load_dataset
from .plugins import resolve
from .runner import execute
from .storage import atomic_json, event, read_json, run_dir


def main():
    rid, seed = sys.argv[1], int(sys.argv[2])
    target = run_dir(rid)
    atomic_json(target / "state.json", {"status": "RUNNING"})
    try:
        spec = resolve(read_json(target / "spec.resolved.json"))
        data, scenarios = load_dataset(spec.dataset)
        event(target, "started", policy_seed=seed)
        execute(
            spec,
            data,
            scenarios,
            target,
            seed,
            lambda kind, **kw: event(target, kind, **kw),
            lambda: (target / "cancel.json").exists(),
        )
        atomic_json(target / "state.json", {"status": "COMPLETED"})
        event(target, "completed")
        return 0
    except PlatformError as exc:
        status = (
            "CANCELLED"
            if exc.code == "CANCELLED"
            else "TIMEOUT"
            if exc.code == "TIMEOUT"
            else "FAILED"
        )
        if (target / "cancel.json").exists() and read_json(target / "cancel.json").get(
            "timeout"
        ):
            status = "TIMEOUT"
        atomic_json(target / "state.json", {"status": status, **exc.as_dict()})
        event(target, "error", **exc.as_dict())
        return exc.exit_code
    except Exception as exc:
        traceback.print_exc()
        atomic_json(
            target / "state.json",
            {
                "status": "FAILED",
                "error": {"code": "BACKEND_ERROR", "message": str(exc), "path": ""},
            },
        )
        event(
            target,
            "error",
            error={"code": "BACKEND_ERROR", "message": str(exc), "path": ""},
        )
        return 4


if __name__ == "__main__":
    sys.exit(main())
