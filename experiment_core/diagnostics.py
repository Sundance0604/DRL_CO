import re
import zipfile
from pathlib import Path
from .storage import ROOT, canonical, read_json, run_dir, events
from .service import environment


def redact(value):
    if isinstance(value, dict):
        return {
            key: "<redacted>"
            if re.search(r"(?i)password|token|secret", key)
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        for sensitive, label in (
            (str(ROOT), "<project>"),
            (str(Path.home()), "<home>"),
        ):
            value = value.replace(sensitive, label).replace(
                sensitive.replace("\\", "/"), label
            )
        return re.sub(
            r"(?i)((?:password|token|secret)\s*[:=]\s*)[^\s,;]+", r"\1<redacted>", value
        )
    return value


def diagnose(rid, output=None):
    target = run_dir(rid)
    report = {
        "run_id": rid,
        "state": read_json(target / "state.json"),
        "environment": environment(),
        "errors": [e for e in events(rid) if e["type"] == "error"],
        "redacted": True,
        "raw_data_included": False,
    }
    text = canonical(redact(report)).decode()
    destination = Path(output) if output else target / "diagnostic.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("diagnostic.json", text)
    return {
        "run_id": rid,
        "artifact_id": destination.name,
        "redacted": True,
        "raw_data_included": False,
    }
