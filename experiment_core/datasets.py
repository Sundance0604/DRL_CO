from __future__ import annotations

import csv
import io
import math
import random
from types import SimpleNamespace
import networkx as nx
from .contracts import Generation, PlatformError
from .storage import atomic_json, digest, identifier, read_json, workspace


def validate_payload(data):
    if (
        set(data) != {"schema_version", "family", "scenarios", "provenance"}
        or data["schema_version"] != "dataset/v1"
    ):
        raise PlatformError("DATASET_SCHEMA", "expected dataset/v1 envelope")
    if data["family"] not in ("single_level_matching", "legacy_dispatch", "bhh"):
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
    cfg = Generation.model_validate(config)
    from drl_co.simulation.scenarios import generate_scenario
    from model.mt_prototype import Net
    from model.demand import city_times, convert, convert_batches

    scenarios = []
    for i, seed in enumerate(cfg.seeds):
        if cfg.family == "bhh":
            rng = random.Random(seed)
            orders = [
                dict(
                    id=f"o-{t}-{j}",
                    departure=str(j % 2),
                    destination=str(1 - j % 2),
                    passenger=rng.randint(1, 3),
                    book_time=t,
                    start_time=t,
                    end_time=t + 12,
                    revenue=100,
                    penalty=100,
                )
                for t in range(cfg.horizon)
                for j in range(cfg.orders_per_step)
            ]
            scenario = dict(
                nodes=["0", "1"], edges=[["0", "1", 40]], hubs=["0", "1"], orders=orders
            )
        else:
            vehicles, original, graph = generate_scenario(
                seed,
                horizon=cfg.horizon,
                num_cities=cfg.num_cities,
                num_vehicles=cfg.num_vehicles,
                orders_per_step=cfg.orders_per_step,
            )
            if cfg.family == "single_level_matching":
                net = Net(graph.G, seed)
                radii = city_times(net, seed)
                converted = (
                    convert_batches(original, net, radii, seed)
                    if cfg.first_mile == "batch"
                    else convert(original, net, radii, seed, cfg.first_mile == "direct")
                )
                orders = [dict(vars(o)) for o in converted.values()]
            else:
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
            if cfg.family == "legacy_dispatch":
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
    from .contracts import strict_json

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
        family=template.pop("family", "single_level_matching"),
        scenarios=[template | {"orders": orders}],
        provenance={"import": format, "mapping": mapping},
    )
    return save_dataset(
        dataset_id, payload, {"format": format, "content": content, "mapping": mapping}
    )


def matching_state(scenario, parameters, solver_parameters):
    from model.mt_prototype import Net, Sim

    graph = nx.Graph()
    graph.add_weighted_edges_from(scenario["edges"])
    net = Net(graph, scenario["seed"], parameters["speed"])
    net.radii = scenario.get("radii", {n: 1 for n in graph})
    sim = Sim(net, scenario["hubs"], scenario["horizon"], parameters, solver_parameters)
    orders = {o["id"]: SimpleNamespace(**o) for o in scenario["orders"]}
    return sim, orders
