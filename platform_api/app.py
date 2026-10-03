from __future__ import annotations

import asyncio
import secrets
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from experiment_core.contracts import (
    PlatformError,
    DatasetRef,
    strict_json,
    ApplicationConfig,
)
from experiment_core.datasets import (
    generate,
    import_orders,
    list_datasets,
    load_dataset,
    validate_payload,
    save_dataset,
)
from experiment_core.plugins import descriptors, schemas
from experiment_core.service import Coordinator, plan, environment
from experiment_core.storage import (
    ROOT,
    atomic_json,
    events,
    identifier,
    read_json,
    run_dir,
    workspace,
)
from experiment_core.reproduction import comparison, export_bundle, validate_bundle


def create_app(coordinator=None, token=None, port=8765):
    config = ApplicationConfig.model_validate(read_json(ROOT / "platform.local.json"))
    token = token or secrets.token_urlsafe(32)
    holder = {"coordinator": coordinator}

    @asynccontextmanager
    async def lifespan(app):
        holder["coordinator"] = holder["coordinator"] or Coordinator(
            max_parallel=config.max_parallel, cpu_budget=config.cpu_budget
        )
        quit = threading.Event()

        def work():
            while not quit.wait(0.15):
                holder["coordinator"].tick()

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        atomic_json(
            workspace() / "server.json",
            {
                "port": port,
                "token": token,
                "workspace": str(workspace()),
                "pid": __import__("os").getpid(),
            },
        )
        try:
            yield
        finally:
            quit.set()
            thread.join(timeout=2)
            holder["coordinator"].close()

    app = FastAPI(
        title="DRL CO Experiment Platform", version="1.0.0", lifespan=lifespan
    )
    app.state.write_token = token

    @app.middleware("http")
    async def local_guard(request, call_next):
        origin = request.headers.get("origin")
        allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
        if origin and origin not in allowed:
            return JSONResponse(
                {
                    "error": {
                        "code": "ORIGIN_DENIED",
                        "message": "cross-origin request rejected",
                        "path": "",
                    }
                },
                status_code=403,
            )
        if request.headers.get("host", "").split(":")[0] not in (
            "127.0.0.1",
            "localhost",
            "testserver",
        ):
            return JSONResponse(
                {
                    "error": {
                        "code": "HOST_DENIED",
                        "message": "local hosts only",
                        "path": "",
                    }
                },
                status_code=403,
            )
        if (
            request.method in ("POST", "PUT", "DELETE", "PATCH")
            and request.headers.get("x-workspace-token") != token
        ):
            return JSONResponse(
                {
                    "error": {
                        "code": "WRITE_TOKEN",
                        "message": "workspace write token required",
                        "path": "",
                    }
                },
                status_code=403,
            )
        if int(request.headers.get("content-length", "0")) > 15 * 1024 * 1024:
            return JSONResponse(
                {
                    "error": {
                        "code": "UPLOAD_SIZE",
                        "message": "request exceeds 15 MiB",
                        "path": "",
                    }
                },
                status_code=413,
            )
        if request.method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            if len(body) > 15 * 1024 * 1024:
                return JSONResponse(
                    {
                        "error": {
                            "code": "UPLOAD_SIZE",
                            "message": "request too large",
                            "path": "",
                        }
                    },
                    status_code=413,
                )
            if "application/json" in request.headers.get("content-type", ""):
                try:
                    strict_json(body.decode())
                except PlatformError as exc:
                    return JSONResponse(exc.as_dict(), status_code=422)
        return await call_next(request)

    @app.exception_handler(PlatformError)
    async def platform_error(request, exc):
        return JSONResponse(
            exc.as_dict(), status_code=404 if exc.code == "NOT_FOUND" else 422
        )

    def errors(exc):
        return {
            "error": {
                "code": "SCHEMA_VALIDATION",
                "message": "configuration validation failed",
                "path": "",
                "details": [
                    {
                        "path": "/"
                        + "/".join(
                            str(p).replace("~", "~0").replace("/", "~1")
                            for p in e["loc"]
                        ),
                        "message": e["msg"],
                        "code": e["type"],
                    }
                    for e in exc.errors()
                ],
            }
        }

    @app.exception_handler(ValidationError)
    async def validation(request, exc):
        return JSONResponse(errors(exc), status_code=422)

    @app.exception_handler(RequestValidationError)
    async def request_validation(request, exc):
        return JSONResponse(errors(exc), status_code=422)

    @app.exception_handler(KeyError)
    async def missing_field(request, exc):
        return JSONResponse(
            {
                "error": {
                    "code": "MISSING_FIELD",
                    "message": f"missing field {exc.args[0]}",
                    "path": "/" + str(exc.args[0]),
                }
            },
            status_code=422,
        )

    def co():
        return holder["coordinator"]

    @app.post("/api/v1/shutdown")
    def shutdown():
        if not hasattr(app.state, "server"):
            raise PlatformError(
                "NO_MANAGED_SERVER", "server cannot be stopped through this instance"
            )
        app.state.server.should_exit = True
        return {"stopping": True}

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "version": "0.1.0", "workspace": str(workspace())}

    @app.get("/api/v1/session")
    def session():
        return {"write_token": token}

    @app.get("/api/v1/capabilities")
    def capabilities():
        env = environment()
        gurobi = {
            "installed": bool(env["dependencies"]["gurobipy"]),
            "license_verified": False,
        }
        try:
            import gurobipy as gp

            m = gp.Model()
            m.Params.OutputFlag = 0
            m.addVar()
            m.optimize()
            gurobi["license_verified"] = True
            m.dispose()
        except Exception as exc:
            gurobi["error"] = str(exc)
        return {
            "environment": env,
            "gurobi": gurobi,
            "resume": False,
            "native_macos_verified": False,
            "max_parallel": co().max_parallel,
            "cpu_budget": co().cpu_budget,
        }

    @app.get("/api/v1/plugins")
    def plugins():
        return descriptors()

    @app.get("/api/v1/metrics")
    def metrics():
        from experiment_core.metrics import DEFINITIONS

        return DEFINITIONS

    @app.get("/api/v1/schemas/{schema_id}")
    def schema(schema_id: str):
        if schema_id not in schemas():
            raise PlatformError("NOT_FOUND", "schema not registered")
        return schemas()[schema_id]

    @app.get("/api/v1/datasets")
    def datasets():
        return list_datasets()

    @app.post("/api/v1/datasets/generate")
    def dataset_generate(config: dict):
        with co().guard:
            return generate(config)

    @app.post("/api/v1/datasets")
    def dataset_create(config: dict):
        with co().guard:
            return generate(config)

    @app.post("/api/v1/datasets/import")
    def dataset_import(config: dict):
        with co().guard:
            return import_orders(
                config["dataset_id"],
                config["content"],
                config["format"],
                config["mapping"],
                config["template"],
            )

    @app.post("/api/v1/datasets/freeze")
    def dataset_freeze(config: dict):
        with co().guard:
            return save_dataset(config["dataset_id"], config["payload"])

    @app.post("/api/v1/datasets/validate")
    def dataset_validate(config: dict):
        data, _ = load_dataset(DatasetRef.model_validate(config))
        return validate_payload(data)

    @app.get("/api/v1/datasets/{did}/revisions/{revision}")
    def dataset_revision(did: str, revision: str):
        data, _ = load_dataset(
            DatasetRef(dataset_id=did, revision=revision, split="all")
        )
        return data

    @app.post("/api/v1/datasets/{did}/archive")
    def dataset_archive(did: str):
        path = workspace() / "datasets" / identifier(did)
        if not path.exists():
            raise PlatformError("NOT_FOUND", "dataset not found")
        atomic_json(path / "archived.json", {"archived": True})
        return {"archived": True}

    @app.post("/api/v1/experiments/validate")
    def validate(config: dict):
        return {"valid": True, "plan": plan(config)}

    @app.post("/api/v1/experiments/plan")
    def experiment_plan(config: dict):
        return plan(config)

    @app.post("/api/v1/batches", status_code=202)
    def submit(config: dict, request: Request):
        return co().submit(config, request.headers.get("idempotency-key"))

    @app.get("/api/v1/batches/{bid}")
    def batch(bid: str):
        return co().batch(identifier(bid))

    @app.get("/api/v1/runs")
    def runs():
        return co().list_runs()

    @app.get("/api/v1/runs/{rid}")
    def run(rid: str):
        return co().run(rid)

    @app.post("/api/v1/runs/{rid}/cancel")
    def cancel(rid: str):
        return co().cancel(rid)

    @app.post("/api/v1/runs/{rid}/rerun", status_code=202)
    def rerun(rid: str, request: Request):
        return co().rerun(rid, request.headers.get("idempotency-key"))

    @app.get("/api/v1/runs/{rid}/events")
    async def event_list(
        rid: str, request: Request, after: int = 0, stream: bool = False
    ):
        co().run(rid)
        if not stream:
            return events(rid, after)

        async def stream_events():
            cursor = max(after, int(request.headers.get("last-event-id", "0")))
            while not await request.is_disconnected():
                for item in events(rid, cursor):
                    cursor = item["sequence"]
                    yield f"id: {cursor}\ndata: {__import__('json').dumps(item)}\n\n"
                if co().run(rid)["status"] in {
                    "COMPLETED",
                    "FAILED",
                    "CANCELLED",
                    "TIMEOUT",
                    "INTERRUPTED",
                }:
                    break
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    @app.get("/api/v1/runs/{rid}/artifacts/{artifact_id}")
    def artifact(rid: str, artifact_id: str):
        run = co().run(rid)
        if artifact_id not in run["artifacts"]:
            raise PlatformError("NOT_FOUND", "artifact not registered")
        target = (run_dir(rid) / identifier(artifact_id)).resolve()
        if not target.is_relative_to(run_dir(rid).resolve()):
            raise PlatformError("ARTIFACT_PATH", "artifact escapes run directory")
        return FileResponse(target, filename=artifact_id)

    @app.post("/api/v1/comparisons")
    def compare(config: dict):
        return comparison(config["run_ids"], config.get("metric", "operating_profit"))

    @app.post("/api/v1/reproduction/export")
    def export(config: dict):
        co().run(config["run_id"])
        return export_bundle(config["run_id"])

    @app.post("/api/v1/reproduction/validate")
    async def validate_reproduction(request: Request):
        result = validate_bundle(await request.body())
        return {k: v for k, v in result.items() if k != "dataset"}

    @app.post("/api/v1/debug/step")
    def debug_step(config: dict):
        from experiment_core.plugins import resolve
        from experiment_core.runner import matching

        spec = resolve(config["spec"])
        _, scenarios = load_dataset(spec.dataset)
        if spec.model.id != "single_level_matching":
            raise PlatformError(
                "DEBUG_MODEL", "single-step currently supports matching only"
            )
        return matching(
            spec,
            scenarios[0],
            spec.execution.policy_seeds[0],
            lambda: None,
            stop_period=config["period"],
        )

    @app.post("/api/v1/diagnostics")
    def diagnostics(config: dict):
        from experiment_core.diagnostics import diagnose

        run = co().run(config["run_id"])
        return diagnose(run["run_id"])

    dist = ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app
