from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from pydantic import ValidationError
from .contracts import PlatformError, DatasetRef
from .datasets import (
    generate,
    list_datasets,
    load_dataset,
    validate_payload,
    save_dataset,
)
from .plugins import descriptors, schemas
from .service import Coordinator, plan, environment, TERMINAL
from .storage import atomic_json, read_json, run_dir, workspace


def remote(path, body=None):
    server = read_json(workspace() / "server.json")
    request = urllib.request.Request(
        f"http://127.0.0.1:{server['port']}/api/v1/{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Workspace-Token": server["token"],
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        error = json.load(exc).get("error", {})
        raise PlatformError(
            error.get("code", "HTTP_ERROR"), error.get("message", str(exc))
        ) from exc


def active_server():
    if not (workspace() / "server.json").exists():
        return False
    try:
        return remote("health").get("workspace") == str(workspace())
    except (OSError, PlatformError):
        return False


def read_config(path):
    config = read_json(Path(path).resolve())
    # Resolve an explicit convenience reference before persisting spec.resolved.
    specs = [config["base_spec"]] if "base_spec" in config else [config]
    for spec in specs:
        ref = spec.get("dataset", {})
        if ref.get("revision") == "latest":
            matches = [
                d
                for d in list_datasets()
                if d["dataset_id"] == ref["dataset_id"] and not d["archived"]
            ]
            if not matches:
                raise PlatformError(
                    "NOT_FOUND", "generate the configured dataset first"
                )
            if len(matches) != 1:
                raise PlatformError(
                    "AMBIGUOUS_REVISION",
                    "choose an exact hash: multiple dataset revisions exist",
                )
            ref["revision"] = matches[0]["revision"]
    return config


def execute_command(args):
    cmd = args.command
    if cmd == "doctor":
        env = environment()
        env.update(
            node=__import__("shutil").which("node"),
            workspace=str(workspace()),
            api_running=active_server(),
            native_macos_verified=False,
        )
        try:
            import gurobipy as gp

            m = gp.Model()
            m.Params.OutputFlag = 0
            m.addVar()
            m.optimize()
            env["gurobi_license"] = True
            m.dispose()
        except Exception as exc:
            env.update(gurobi_license=False, gurobi_error=str(exc))
        return env
    if cmd == "plugins":
        return descriptors()
    if cmd == "analysis":
        from .analysis import create_analysis, export_results, list_analyses
        if args.action == "list":
            return list_analyses()
        cfg = read_json(args.config)
        if args.action == "export":
            return remote("results/export", cfg) if active_server() else export_results(cfg["run_ids"])
        return remote("analyses", cfg) if active_server() else create_analysis(cfg)
    if cmd == "schema":
        target = Path(args.output)
        target.mkdir(parents=True, exist_ok=True)
        for key, value in schemas().items():
            atomic_json(target / (key + ".json"), value)
        from platform_api.app import create_app

        atomic_json(target / "openapi.json", create_app().openapi())
        return {"exported": len(schemas()) + 1, "output": str(target.resolve())}
    if cmd == "dataset":
        if args.action == "import":
            cfg = read_json(args.config)
            if active_server():
                return remote("datasets/freeze", cfg)
            co = Coordinator()
            try:
                return save_dataset(cfg["dataset_id"], cfg["payload"])
            finally:
                co.close()
        if args.action == "generate":
            cfg = read_json(args.config)
            if active_server():
                return remote("datasets/generate", cfg)
            co = Coordinator()
            try:
                return generate(cfg)
            finally:
                co.close()
        if args.action == "list":
            return list_datasets()
        data, _ = load_dataset(
            DatasetRef(dataset_id=args.dataset, revision=args.revision, split="all")
        )
        return validate_payload(data)
    if cmd == "experiment":
        planned = plan(read_config(args.config))
        return (
            {"valid": True, "plan": planned} if args.action == "validate" else planned
        )
    if cmd in ("run", "reproduce"):
        if cmd == "reproduce":
            from .reproduction import reproduce

            config = reproduce(Path(args.bundle).read_bytes())["plan"]["runs"][0][
                "spec"
            ]
        else:
            config = read_config(args.config)
        if active_server():
            batch = remote("batches", config)
            while True:
                result = remote("batches/" + batch["batch_id"])
                if all(r["status"] in TERMINAL for r in result["runs"]):
                    break
                time.sleep(0.2)
            return result
        coordinator = Coordinator()
        try:
            batch = coordinator.submit(config)
            while True:
                coordinator.tick()
                result = coordinator.batch(batch["batch_id"])
                if all(r["status"] in TERMINAL for r in result["runs"]):
                    break
                if args.json_stream:
                    print(
                        json.dumps({"event": "progress", "batch": result}),
                        file=sys.stderr,
                    )
                time.sleep(0.15)
            return result
        finally:
            coordinator.close()
    if cmd == "runs":
        if args.action == "logs":
            target = run_dir(args.run)
            return {
                name: (target / name)
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()[-args.tail :]
                if (target / name).exists()
                else []
                for name in ("stdout.log", "stderr.log")
            }
        if args.action == "replay":
            trace = read_json(run_dir(args.run) / "trace.json")
            return [
                {
                    "scenario_id": s["scenario_id"],
                    "steps": [
                        row for row in s["steps"] if row["period"] == args.period
                    ],
                }
                for s in trace
            ]
        if active_server():
            return remote(
                f"runs/{args.run}" + ("/cancel" if args.action == "cancel" else ""),
                {} if args.action == "cancel" else None,
            )
        coordinator = Coordinator()
        try:
            return (
                coordinator.cancel(args.run)
                if args.action == "cancel"
                else coordinator.run(args.run)
            )
        finally:
            coordinator.close()
    if cmd == "debug":
        if args.action == "step":
            from .plugins import resolve
            from .runner import matching

            spec = resolve(read_config(args.config))
            _, selected = load_dataset(spec.dataset)
            if spec.model.id != "single_level_matching":
                raise PlatformError("DEBUG_MODEL", "matching single-step only")
            result = matching(
                spec,
                selected[0],
                spec.execution.policy_seeds[0],
                lambda: None,
                stop_period=args.period,
            )
            atomic_json(Path(args.output) / "step.json", result)
            return result
        from .diagnostics import diagnose

        return diagnose(args.run, args.output)
    if cmd == "export":
        from .reproduction import export_bundle

        return export_bundle(args.run)
    if cmd == "serve":
        import uvicorn
        from platform_api.app import create_app

        if active_server():
            return {
                "reused": True,
                "url": f"http://127.0.0.1:{read_json(workspace() / 'server.json')['port']}",
            }
        app = create_app(port=args.port)
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="info")
        )
        app.state.server = server
        server.run()
        return {"stopped": True}
    raise PlatformError("CLI_COMMAND", "unknown command")


def main():
    parser = argparse.ArgumentParser(
        prog="exp", description="Local DRL CO experiment platform"
    )
    parser.add_argument(
        "command",
        choices=[
            "doctor",
            "plugins",
            "schema",
            "dataset",
            "experiment",
            "run",
            "runs",
            "debug",
            "reproduce",
            "export",
            "serve",
            "analysis",
        ],
    )
    parser.add_argument("action", nargs="?", default="list")
    parser.add_argument("--config")
    parser.add_argument("--output", default="schemas/generated")
    parser.add_argument("--dataset")
    parser.add_argument("--revision")
    parser.add_argument("--run")
    parser.add_argument("--bundle")
    parser.add_argument("--period", type=int, default=0)
    parser.add_argument("--tail", type=int, default=200)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-stream", action="store_true")
    parser.add_argument(
        "--redact", action="store_true", help="diagnostic output is always redacted"
    )
    args = parser.parse_args()
    code = 0
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = execute_command(args)
        if args.command in ("run", "reproduce"):
            failed = [r for r in result["runs"] if r["status"] != "COMPLETED"]
            if failed:
                code = (
                    6
                    if all(r["status"] == "CANCELLED" for r in failed)
                    else 5
                    if any(r["status"] == "TIMEOUT" for r in failed)
                    else 4
                )
    except PlatformError as exc:
        result = exc.as_dict()
        code = exc.exit_code
    except ValidationError as exc:
        result = {
            "error": {
                "code": "SCHEMA_VALIDATION",
                "message": "invalid configuration",
                "path": "",
                "details": [
                    {"path": "/" + "/".join(map(str, e["loc"])), "message": e["msg"]}
                    for e in exc.errors()
                ],
            }
        }
        code = 2
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        result = {"error": {"code": "INVALID_INPUT", "message": str(exc), "path": ""}}
        code = 2
    except ImportError as exc:
        result = {
            "error": {"code": "DEPENDENCY_MISSING", "message": str(exc), "path": ""}
        }
        code = 3
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        result = {"error": {"code": "INTERNAL_ERROR", "message": str(exc), "path": ""}}
        code = 7
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
