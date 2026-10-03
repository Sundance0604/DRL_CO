import csv
import io
import math
import random
from types import SimpleNamespace
import networkx as nx
from experiment_core.contracts import PlatformError
from experiment_core.storage import identifier
from .generation import Generation

# Storage remains a platform concern; domain interpretation stays in this package.
def save_dataset(*args, **kwargs):
    from experiment_core.datasets import save_dataset as save
    return save(*args, **kwargs)

def validate_payload(data):
    if (
        set(data) != {"schema_version", "family", "scenarios", "provenance"}
        or data["schema_version"] != "dataset/v1"
    ):
        raise PlatformError("DATASET_SCHEMA", "expected dataset/v1 envelope")
    if data["family"] != "legacy_dispatch":
        raise PlatformError("DATASET_FAMILY", "unsupported dataset family")
    if not data["scenarios"]:
        raise PlatformError("EMPTY_DATASET", "at least one scenario is required")
    if (
        len(data["scenarios"]) > 100
        or sum(len(s.get("orders", [])) for s in data["scenarios"]) > 200000
    ):
        raise PlatformError(
            "DATASET_SIZE", "dataset exceeds 100 scenarios or 200000 orders"
        )
    ids = set()
    for si, s in enumerate(data["scenarios"]):
        pointer = f"/scenarios/{si}"
        sid = identifier(s["id"])
        if sid in ids:
            raise PlatformError("DUPLICATE_SCENARIO", "duplicate scenario", pointer)
        ids.add(sid)
        if (
            s["split"] not in ("train", "validation", "test")
            or not isinstance(s["horizon"], int)
            or not 1 <= s["horizon"] <= 200
        ):
            raise PlatformError("SCENARIO_SCHEMA", "invalid split or horizon", pointer)
        nodes = [identifier(str(n)) for n in s["nodes"]]
        if any(not isinstance(n, str) for n in s["nodes"]):
            raise PlatformError(
                "NODE_ID_TYPE", "normalized node IDs must be strings", pointer
            )
        if len(nodes) != len(set(nodes)) or len(nodes) < 2:
            raise PlatformError(
                "NETWORK_NODES", "at least two unique nodes required", pointer
            )
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for e in s["edges"]:
            if (
                e[0] not in nodes
                or e[1] not in nodes
                or e[0] == e[1]
                or not math.isfinite(e[2])
                or e[2] <= 0
            ):
                raise PlatformError(
                    "NETWORK_EDGE", "invalid endpoint or distance", pointer
                )
            graph.add_edge(e[0], e[1], weight=e[2])
        if not nx.is_connected(graph):
            raise PlatformError(
                "DISCONNECTED_NETWORK", "network must be connected", pointer
            )
        if any(h not in nodes for h in s["hubs"]):
            raise PlatformError("VEHICLE_HUB", "initial hub not in network", pointer)
        if data["family"] == "legacy_dispatch" and len(
            s.get("legacy_vehicles", [])
        ) != len(s["hubs"]):
            raise PlatformError(
                "LEGACY_INITIAL_STATE",
                "legacy dataset requires frozen vehicle states",
                pointer,
            )
        orders = set()
        for oi, o in enumerate(s["orders"]):
            op = pointer + f"/orders/{oi}"
            oid = identifier(o["id"])
            if oid in orders:
                raise PlatformError(
                    "DUPLICATE_ORDER", "order IDs must be unique within scenario", op
                )
            orders.add(oid)
            if o.get("book_time", 0) >= s["horizon"]:
                raise PlatformError(
                    "BOOKING_HORIZON",
                    "booking must be before scenario horizon cutoff",
                    op + "/book_time",
                )
            if data["family"] == "legacy_dispatch" and any(
                k not in o
                for k in ("battery", "distance", "least_time_consume", "matched")
            ):
                raise PlatformError(
                    "LEGACY_ORDER_FIELDS",
                    "legacy orders require battery, distance, least_time_consume and matched",
                    op,
                )
            if (
                o["departure"] not in nodes
                or o["destination"] not in nodes
                or o["departure"] == o["destination"]
            ):
                raise PlatformError("ORDER_OD", "invalid origin/destination", op)
            numeric = (
                "passenger",
                "book_time",
                "start_time",
                "end_time",
                "revenue",
                "penalty",
            )
            if any(
                isinstance(o[k], bool)
                or not isinstance(o[k], (int, float))
                or not math.isfinite(o[k])
                for k in numeric
            ):
                raise PlatformError(
                    "ORDER_NUMBER", "finite numeric order fields required", op
                )
            if (
                o["passenger"] <= 0
                or not float(o["passenger"]).is_integer()
                or o["book_time"] < 0
                or not float(o["book_time"]).is_integer()
            ):
                raise PlatformError(
                    "ORDER_LOAD",
                    "positive integral load and nonnegative integral booking required",
                    op,
                )
            if (
                o["start_time"] < o["book_time"]
                or o["end_time"] < o["start_time"]
                or o["penalty"] < 0
            ):
                raise PlatformError(
                    "ORDER_TIME",
                    "booking <= ready <= deadline and nonnegative penalty required",
                    op,
                )
    return {
        "valid": True,
        "scenarios": len(ids),
        "orders": sum(len(s["orders"]) for s in data["scenarios"]),
    }


def generate(config):
    cfg = Generation.model_validate(config)
    from .engine.simulation.scenarios import generate_scenario

    scenarios = []
    for i, seed in enumerate(cfg.seeds):
        vehicles, original, graph = generate_scenario(
            seed,
            horizon=cfg.horizon,
            num_cities=cfg.num_cities,
            num_vehicles=cfg.num_vehicles,
            orders_per_step=cfg.orders_per_step,
        )
        radii = {}
        orders = [
            dict(vars(o), book_time=o.start_time) for o in original.values()
        ]
        for o in orders:
            o["id"], o["departure"], o["destination"] = (
                str(o["id"]),
                str(o["departure"]),
                str(o["destination"]),
            )
        scenario = dict(
            nodes=[str(n) for n in graph.G],
            edges=[[str(a), str(b), w] for a, b, w in graph.G.edges(data="weight")],
            hubs=[str(v.intercity) for v in vehicles.values()],
            orders=orders,
            radii={str(k): v for k, v in radii.items()},
        )
        scenario["legacy_vehicles"] = [vars(v) for v in vehicles.values()]
        scenario.update(
            id=f"s-{seed}",
            seed=seed,
            split=cfg.splits[0] if len(cfg.splits) == 1 else cfg.splits[i],
            horizon=cfg.horizon,
        )
        scenarios.append(scenario)
    payload = dict(
        schema_version="dataset/v1",
        family=cfg.family,
        scenarios=scenarios,
        provenance={"generator": "scenario-factory/v1", "config": cfg.model_dump()},
    )
    return save_dataset(cfg.dataset_id, payload)


def import_orders(dataset_id, content, format, mapping, template):
    """Imports tabular orders into an explicitly supplied frozen network/scenario."""
    from experiment_core.contracts import strict_json

    if len(content.encode()) > 10 * 1024 * 1024:
        raise PlatformError("UPLOAD_SIZE", "upload exceeds 10 MiB")
    if format == "csv":
        rows = list(csv.DictReader(io.StringIO(content)))
    elif format == "jsonl":
        rows = [strict_json(line) for line in content.splitlines() if line.strip()]
    elif format == "json":
        rows = strict_json(content)
    else:
        raise PlatformError("IMPORT_FORMAT", "only csv/json/jsonl are supported")
    if not isinstance(rows, list) or len(rows) > 20000:
        raise PlatformError("IMPORT_ROWS", "expected at most 20000 order rows")
    orders = []
    for row in rows:
        order = {target: row[source] for target, source in mapping.items()}
        for key in (
            "passenger",
            "book_time",
            "start_time",
            "end_time",
            "revenue",
            "penalty",
        ):
            order[key] = float(order[key])
        for key in ("id", "departure", "destination"):
            order[key] = str(order[key])
        orders.append(order)
    payload = dict(
        schema_version="dataset/v1",
        family=template.get("family", "legacy_dispatch"),
        scenarios=[{k:v for k,v in template.items() if k != "family"} | {"orders": orders}],
        provenance={"import": format, "mapping": mapping},
    )
    return save_dataset(
        dataset_id, payload, {"format": format, "content": content, "mapping": mapping}
    )
