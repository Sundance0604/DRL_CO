"""Owned-instance launcher. Never terminates a process merely owning our port."""

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import webbrowser
import psutil
from .cli import active_server
from .contracts import PlatformError, ApplicationConfig
from .storage import ROOT, atomic_json, read_json, workspace


def report_ready(result, open_browser=False):
    """Open only the verified local service, after health readiness or reuse."""
    if open_browser:
        try:
            result["browser_opened"] = bool(webbrowser.open(result["url"], new=2))
        except Exception:
            result["browser_opened"] = False
        if not result["browser_opened"]:
            result["browser_warning"] = "Open the local URL manually in your browser."
    print(json.dumps(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Start or stop this local platform.")
    parser.add_argument("action", choices=("start", "stop"), nargs="?", default="start")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "stop" and args.open_browser:
        parser.error("--open-browser is only available for start")
    config = ApplicationConfig.model_validate(
        read_json(ROOT / "platform.local.json")
    ).model_dump()
    port = config["port"]
    if config["host"] != "127.0.0.1":
        raise PlatformError("LOCAL_ONLY", "nonlocal binding is not supported")
    marker = workspace() / "launcher.json"
    if args.action == "stop":
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
        port = ApplicationConfig.model_validate({"port": server["port"]}).port
        report_ready({"reused": True, "url": f"http://127.0.0.1:{port}"}, args.open_browser)
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
            report_ready({"started": True, "url": f"http://127.0.0.1:{port}"}, args.open_browser)
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
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": {"code": "LAUNCH_ERROR", "message": str(exc)}}))
        sys.exit(2)
