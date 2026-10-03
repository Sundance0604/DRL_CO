"""Owned-instance launcher. Never terminates a process merely owning our port."""

import json
import os
import secrets
import subprocess
import sys
import time
import psutil
from .cli import active_server
from .contracts import PlatformError, ApplicationConfig
from .storage import ROOT, atomic_json, read_json, workspace


def main():
    config = ApplicationConfig.model_validate(
        read_json(ROOT / "platform.local.json")
    ).model_dump()
    port = config["port"]
    if config["host"] != "127.0.0.1":
        raise PlatformError("LOCAL_ONLY", "nonlocal binding is not supported")
    marker = workspace() / "launcher.json"
    if sys.argv[1] == "stop":
        owner = read_json(marker)
        try:
            process = psutil.Process(owner["pid"])
            if (
                abs(process.create_time() - owner["create_time"]) > 0.01
                or process.environ().get("DRL_SERVER_TOKEN") != owner["token"]
            ):
                raise PlatformError(
                    "OWNERSHIP_MISMATCH", "refusing to stop an unrelated process"
                )
            if os.name == "nt":
                # Graceful server shutdown HTTP marks cancellation and then exits;
                # no console signal or blanket port/PID kill.
                from .cli import remote

                remote("shutdown", {})
                process.wait(timeout=10)
            else:
                process.terminate()
                process.wait(timeout=10)
        except psutil.NoSuchProcess:
            pass
        print(json.dumps({"stopped": True}))
        return
    if active_server():
        server = read_json(workspace() / "server.json")
        print(json.dumps({"reused": True, "url": f"http://127.0.0.1:{server['port']}"}))
        return
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        raise PlatformError("UI_NOT_BUILT", "run setup before start")
    token = secrets.token_urlsafe(32)
    env = os.environ.copy()
    env.update(DRL_SERVER_TOKEN=token, PYTHONDONTWRITEBYTECODE="1")
    with (
        (workspace() / "server.stdout.log").open("ab") as stdout,
        (workspace() / "server.stderr.log").open("ab") as stderr,
    ):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-m",
                "experiment_core.cli",
                "serve",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
    atomic_json(
        marker,
        {
            "pid": proc.pid,
            "create_time": psutil.Process(proc.pid).create_time(),
            "token": token,
        },
    )
    limit = time.monotonic() + 15
    while time.monotonic() < limit:
        if active_server():
            print(json.dumps({"started": True, "url": f"http://127.0.0.1:{port}"}))
            return
        if proc.poll() is not None:
            raise PlatformError(
                "START_FAILED", "server failed; inspect workspace/server.stderr.log"
            )
        time.sleep(0.2)
    raise PlatformError("START_TIMEOUT", "server not ready; inspect logs")


if __name__ == "__main__":
    try:
        main()
    except PlatformError as exc:
        print(json.dumps(exc.as_dict()))
        sys.exit(exc.exit_code)
