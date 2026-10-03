"""Readiness, browser handoff and real Windows batch error-path regressions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from experiment_core import launch
from experiment_core.contracts import PlatformError
from experiment_core.storage import atomic_json, read_json


@pytest.fixture
def local_launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    monkeypatch.setenv("DRL_WORKSPACE", str(tmp_path / "workspace"))
    atomic_json(tmp_path / "platform.local.json", {"host": "127.0.0.1", "port": 8765})
    return tmp_path


def existing_server(root, monkeypatch):
    atomic_json(root / "workspace/server.json", {"port": 8766})
    monkeypatch.setattr(launch, "active_server", lambda: True)
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: pytest.fail("must reuse"))


def test_reuse_opens_actual_local_url(local_launcher, monkeypatch, capsys):
    existing_server(local_launcher, monkeypatch)
    opened = []
    monkeypatch.setattr(launch.webbrowser, "open", lambda url, new: opened.append((url, new)) or True)
    launch.main(["start", "--open-browser"])
    result = json.loads(capsys.readouterr().out)
    assert result == {"reused": True, "url": "http://127.0.0.1:8766", "browser_opened": True}
    assert opened == [(result["url"], 2)]


def test_old_cli_does_not_open_browser(local_launcher, monkeypatch, capsys):
    existing_server(local_launcher, monkeypatch)
    monkeypatch.setattr(launch.webbrowser, "open", lambda *a, **k: pytest.fail("no browser flag"))
    launch.main(["start"])
    assert json.loads(capsys.readouterr().out) == {"reused": True, "url": "http://127.0.0.1:8766"}


@pytest.mark.parametrize("raises", [False, True])
def test_browser_failure_preserves_ready_service(local_launcher, monkeypatch, capsys, raises):
    existing_server(local_launcher, monkeypatch)

    def unavailable(*args, **kwargs):
        if raises:
            raise OSError("browser unavailable")
        return False

    monkeypatch.setattr(launch.webbrowser, "open", unavailable)
    launch.main(["start", "--open-browser"])
    result = json.loads(capsys.readouterr().out)
    assert result["reused"] and not result["browser_opened"]
    assert "manually" in result["browser_warning"]


def fake_cold_start(root, monkeypatch, readiness):
    target = root / "frontend/dist/index.html"
    target.parent.mkdir(parents=True)
    target.write_text("ready", encoding="utf-8")
    states = iter(readiness)
    monkeypatch.setattr(launch, "active_server", lambda: next(states))
    proc = SimpleNamespace(pid=42, poll=lambda: None)
    calls = []
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: calls.append((a, k)) or proc)
    monkeypatch.setattr(launch.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 123.0))
    monkeypatch.setattr(launch.time, "sleep", lambda seconds: None)
    return proc, calls


def test_cold_start_opens_only_after_readiness(local_launcher, monkeypatch, capsys):
    proc, calls = fake_cold_start(local_launcher, monkeypatch, [False, False, True])
    opened = []
    monkeypatch.setattr(launch.webbrowser, "open", lambda url, new: opened.append(url) or True)
    launch.main(["start", "--open-browser"])
    result = json.loads(capsys.readouterr().out)
    assert result["started"] and opened == ["http://127.0.0.1:8765"]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0][2:5] == ["-m", "experiment_core.cli", "serve"]
    if os.name == "nt":
        assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
    assert read_json(local_launcher / "workspace/launcher.json")["pid"] == proc.pid


def test_missing_ui_never_opens_browser(local_launcher, monkeypatch):
    monkeypatch.setattr(launch, "active_server", lambda: False)
    monkeypatch.setattr(launch.webbrowser, "open", lambda *a, **k: pytest.fail("not ready"))
    with pytest.raises(PlatformError, match="run setup") as error:
        launch.main(["start", "--open-browser"])
    assert error.value.code == "UI_NOT_BUILT"


def test_failed_process_never_opens_browser(local_launcher, monkeypatch):
    proc, _ = fake_cold_start(local_launcher, monkeypatch, [False, False])
    proc.poll = lambda: 1
    monkeypatch.setattr(launch.webbrowser, "open", lambda *a, **k: pytest.fail("not ready"))
    with pytest.raises(PlatformError) as error:
        launch.main(["start", "--open-browser"])
    assert error.value.code == "START_FAILED"


def test_timeout_never_opens_browser(local_launcher, monkeypatch):
    fake_cold_start(local_launcher, monkeypatch, [False])
    times = iter([0.0, 16.0])
    monkeypatch.setattr(launch.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(launch.webbrowser, "open", lambda *a, **k: pytest.fail("not ready"))
    with pytest.raises(PlatformError) as error:
        launch.main(["start", "--open-browser"])
    assert error.value.code == "START_TIMEOUT"


def test_nonlocal_config_rejected(local_launcher, monkeypatch):
    atomic_json(local_launcher / "platform.local.json", {"host": "example.com", "port": 8765})
    monkeypatch.setattr(launch.webbrowser, "open", lambda *a, **k: pytest.fail("must stay local"))
    with pytest.raises(ValidationError):
        launch.main(["start", "--open-browser"])


@pytest.mark.parametrize("arguments", [["unexpected"], ["stop", "--open-browser"]])
def test_invalid_arguments_rejected(arguments):
    with pytest.raises(SystemExit) as error:
        launch.main(arguments)
    assert error.value.code == 2


@pytest.mark.skipif(os.name != "nt", reason="real cmd.exe regression")
@pytest.mark.parametrize("ui_missing", [False, True])
def test_batch_errors_from_unicode_space_path(tmp_path, ui_missing):
    root = tmp_path / "启动 platform with spaces"
    root.mkdir()
    batch = root / "start.bat"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "start.bat", batch)
    if ui_missing:
        python = root / ".venv/Scripts/python.exe"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"not executed: UI preflight must fail first")
    # Pass the full Windows command line: list2cmdline would backslash-escape
    # cmd.exe's nested quotes, which cmd does not interpret like C argv parsing.
    command = f'cmd.exe /d /s /c ""{batch}" --no-browser --no-pause"'
    result = subprocess.run(
        command, cwd=tmp_path,
        capture_output=True, timeout=10,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    expected = b"Built web interface is missing" if ui_missing else b"Project Python environment is missing"
    assert expected in result.stdout
    assert b"setup.ps1" in result.stdout
