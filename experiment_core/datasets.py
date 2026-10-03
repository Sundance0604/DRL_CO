"""Versioned immutable storage; domain interpretation belongs to each family."""
from .contracts import PlatformError
from .storage import atomic_json, digest, identifier, read_json, workspace

def validate_payload(data):
    from .families import get_family
    return get_family(data.get("family")).data.validate_payload(data)

def save_dataset(dataset_id, payload, raw=None):
    identifier(dataset_id)
    validate_payload(payload)
    revision = digest(payload)
    target = workspace() / "datasets" / dataset_id / revision
    if not target.exists():
        target.mkdir(parents=True)
        atomic_json(target / "normalized.json", payload)
        atomic_json(target / "raw.json", raw if raw is not None else payload)
        atomic_json(target / "derived.json", {"validation": validate_payload(payload)})
        atomic_json(
            target / "manifest.json",
            {
                "dataset_id": dataset_id,
                "revision": revision,
                "hash": revision,
                "family": payload["family"],
                "scenarios": [
                    {k: s[k] for k in ("id", "split", "horizon")}
                    for s in payload["scenarios"]
                ],
            },
        )
    return read_json(target / "manifest.json")


def load_dataset(ref):
    target = (
        workspace() / "datasets" / identifier(ref.dataset_id) / identifier(ref.revision)
    )
    data = read_json(target / "normalized.json")
    if digest(data) != ref.revision:
        raise PlatformError(
            "DATASET_HASH_MISMATCH",
            "frozen dataset content hash changed",
            "/dataset/revision",
        )
    validate_payload(data)
    scenarios = [
        s
        for s in data["scenarios"]
        if (not ref.scenario_ids or s["id"] in ref.scenario_ids)
        and (ref.split == "all" or s["split"] == ref.split)
    ]
    if not scenarios or (
        ref.scenario_ids and set(ref.scenario_ids) != {s["id"] for s in scenarios}
    ):
        raise PlatformError(
            "DATASET_SELECTION",
            "split/scenario selection is empty or missing",
            "/dataset/scenario_ids",
        )
    return data, scenarios


def list_datasets():
    root = workspace() / "datasets"
    return [
        read_json(p) | {"archived": (p.parent.parent / "archived.json").exists()}
        for p in sorted(root.glob("*/*/manifest.json"))
    ]


def generate(config):
    from .families import get_family
    return get_family(config.get("family", "single_level_matching")).data.generate(config)

def import_orders(dataset_id, content, format, mapping, template):
    from .families import get_family
    return get_family(template.get("family", "single_level_matching")).data.import_orders(dataset_id, content, format, mapping, template)

def matching_state(*args, **kwargs):
    # Compatibility only; shared orchestration never imports a physical adapter.
    from model_families.single.data import matching_state as restore
    return restore(*args, **kwargs)
