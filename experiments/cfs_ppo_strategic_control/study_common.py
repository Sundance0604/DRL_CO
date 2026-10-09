"""Round-eight immutable CFS preparation and preregistered identities."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from model_families.single.strategic.artifacts import read_json, write_json, source_identity
from model_families.single.strategic.contracts import identity
from model_families.single.learning.ppo.config import PPOConfig

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
TASK = ROOT/"reports/PPO首次数值实验任务书_第八轮.md"
HISTORY = ROOT/"experiments/cfs_joint_action_value_diagnostic"
BASE_CONFIG = ROOT/"experiments/cfs_rho_endofday/configs/main.json"
CONDITIONS = [dict(region="california_corridor", orders=80, fleet=3),
              dict(region="western_multicluster", orders=40, fleet=8),
              dict(region="western_multicluster", orders=80, fleet=3)]
SPLITS = dict(engineering=list(range(18000, 18004)), pilot_train=list(range(18100, 18108)),
              pilot_validation=list(range(18108, 18116)), train=list(range(20000, 20256)),
              validation=list(range(21000, 21032)), test=list(range(22000, 22080)),
              diagnostic=list(range(23000, 23016)))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def key(condition):
    return f'{condition["region"]}-n{condition["orders"]}-k{condition["fleet"]}'


def study_sources():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(PACKAGE.glob("*.py"))}


def load(attempt):
    target = Path(attempt).resolve()
    manifest = read_json(target/"manifest.json")
    if (target/"qa/stop_condition_failure.json").exists() or manifest["status"] == "COMPLETED":
        raise RuntimeError("failed/completed scientific attempts are immutable")
    if manifest["source_identity"] != source_identity() or manifest["study_sources"] != study_sources():
        raise RuntimeError("frozen numerical or study source changed; create a new attempt")
    for relative, expected in manifest["input_hashes"].items():
        if sha(target/relative) != expected:
            raise RuntimeError(f"frozen input changed: {relative}")
    return target, manifest


def payload(scenarios):
    return dict(schema_version="dataset/v1", family="single_level_matching", scenarios=scenarios,
                provenance=dict(study="round-eight PPO", real_days=False))


def data(attempt, condition, split):
    return read_json(Path(attempt)/"data"/key(condition)/(split+".json"))


def run_config(attempt, condition, seed, pilot=False):
    cfg = read_json(Path(attempt)/"protocol"/(key(condition)+".json"))
    cfg["baseline"] = str((Path(attempt)/cfg["baseline"]).resolve())
    cfg["ppo"]["training_seed"] = seed
    if pilot:
        cfg["ppo"].update(updates=4, episodes_per_update=4, validation_every=1)
    return cfg


def prepared_train_data(attempt, condition, pilot=False):
    training = data(attempt, condition, "pilot_train" if pilot else "train")["scenarios"]
    validation = data(attempt, condition, "pilot_validation" if pilot else "validation")["scenarios"]
    return payload(training+validation)


def seed_values(value):
    result = set()
    if isinstance(value, dict):
        for name, entry in value.items():
            if "seed" in name:
                def integers(x):
                    if isinstance(x, int) and not isinstance(x, bool):
                        result.add(x)
                    elif isinstance(x, list):
                        for item in x:
                            integers(item)
                integers(entry)
            result.update(seed_values(entry))
    elif isinstance(value, list):
        for entry in value:
            result.update(seed_values(entry))
    return result


def prepare(attempt):
    from model_families.single.cfs_adapter import read_pools, make_scenario
    from model_families.single.engine.mt_prototype import Net
    from model_families.single.strategic.baseline import FrozenBaseline
    from model_families.single.learning.ppo.runner import scenario_content, prepare as check_config
    import networkx as nx
    target = Path(attempt).resolve()
    target.mkdir(parents=True, exist_ok=False)
    for folder in ("protocol", "data", "sealed", "baselines", "preflight", "training", "validation",
                   "evaluation", "diagnostics", "analysis", "figures", "qa"):
        (target/folder).mkdir()
    base, historical = read_json(BASE_CONFIG), read_json(HISTORY/"manifests/final_manifest.json")
    if historical["status"] != "COMPLETED":
        raise RuntimeError("historical round seven is not frozen")
    history_hashes = {str(BASE_CONFIG.relative_to(ROOT)): sha(BASE_CONFIG),
                      str((HISTORY/"manifests/final_manifest.json").relative_to(ROOT)): sha(HISTORY/"manifests/final_manifest.json")}
    for relative, expected in historical["source_code_sha256"].items():
        if sha(ROOT/relative) != expected:
            raise RuntimeError(f"historical numerical source mismatch: {relative}")
        history_hashes[relative] = expected
    print("Verifying original CFS SQLite SHA256 (read only)", flush=True)
    source_hash = sha(base["data_source"])
    if source_hash != historical["source"]["sha256"]:
        raise RuntimeError("original CFS source changed")
    historical_seeds, historical_contents, historical_config_files = set(), set(), {}
    for parent in (ROOT/"experiments", ROOT/"results"):
        for family in parent.iterdir():
            if not family.is_dir() or family == PACKAGE:
                continue
            for folder in ("config", "configs"):
                for path in (family/folder).glob("*.json"):
                    obj = read_json(path)
                    historical_seeds.update(seed_values(obj))
                    historical_config_files[str(path.relative_to(ROOT))] = sha(path)
            for path in (family/"raw/datasets").glob("*.dataset.json"):
                for scenario in read_json(path).get("scenarios", []):
                    historical_contents.add(scenario_content(scenario))
                    if "seed" in scenario:
                        historical_seeds.add(scenario["seed"])
    proposed = {s for seeds in SPLITS.values() for s in seeds}
    if proposed & historical_seeds:
        raise RuntimeError(f"proposed scenario seeds conflict with historical data: {sorted(proposed & historical_seeds)}")
    manifest = dict(schema_version="cfs-first-ppo-study/v1", status="PREPARING", created_at=now(),
                    source=dict(path=base["data_source"], size_bytes=Path(base["data_source"]).stat().st_size,
                                sha256=source_hash), task_sha256=sha(TASK), source_identity=source_identity(),
                    study_sources=study_sources(), input_hashes={}, history_hashes=history_hashes,
                    historical_config_hashes=historical_config_files)
    write_json(target/"manifest.json", manifest)
    protocol = dict(task_sha256=sha(TASK), conditions=CONDITIONS, splits=SPLITS,
                    training_seeds=list(range(30001, 30006)), pilot_training_seed=29999,
                    bootstrap_seed=31000, bootstrap_replications=20000, family_size=6,
                    inference="deterministic", CAL_grid=[-.5, 0., .5],
                    CAL_tie="zero, smaller squared norm, near/far lexicographic",
                    success="both Bonferroni lower bounds positive, >=4/5 seed gains over MC positive, all audits pass",
                    feature_scales=dict(money=1000., distance=100., time=24., load=40.),
                    placement="eligible outbound WGT_FACTOR; K3 top3; K8 one per hub plus top3; ties hub ID",
                    pickup_radius="historical rho_max=max network distance; cross_hub true",
                    tolerance=dict(USD_abs=1e-6, USD_rel=1e-10),
                    formal_episode_count=22224, workers=1,
                    env=dict(PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                             OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
                    budget_status="requires pilot-based human confirmation before formal training")
    write_json(target/"protocol/preregistration.json", protocol)
    shutil.copyfile(TASK, target/"protocol/task.md")
    all_identities, overlap, catalog = [], [], {}
    for region in base["regions"]:
        selected = [c for c in CONDITIONS if c["region"] == region["id"]]
        if not selected:
            continue
        print("Reading CFS pools:", region["id"], flush=True)
        pools, counts, metadata = read_pools(base["data_source"], region["coordinates"])
        mass = {h: 0. for h in region["coordinates"]}
        for directions in pools.values():
            for records in directions:
                for record in records:
                    mass[record["ORIG_CFS_AREA"]] += float(record["WGT_FACTOR"])
        ranked = sorted(mass, key=lambda h: (-mass[h], h))
        catalog[region["id"]] = dict(mass=mass, rank=ranked, eligible_counts=counts, metadata=metadata)
        for c in selected:
            name = key(c)
            (target/"data"/name).mkdir()
            (target/"sealed"/name).mkdir()
            source_table = HISTORY/"tables/bank1"/(name+".json")
            expected = historical["deliverable_sha256"]["tables/bank1/"+name+".json"]
            if sha(source_table) != expected:
                raise RuntimeError("bank1 table hash mismatch")
            history_hashes[str(source_table.relative_to(ROOT))] = expected
            model = read_json(source_table)
            frozen = FrozenBaseline(model)
            hubs = sorted(region["coordinates"])
            frozen.validate(hubs, c["fleet"])
            if set(model["seeds"]) & proposed:
                raise RuntimeError("construction/RL data overlap")
            shutil.copyfile(source_table, target/"baselines"/(name+".json"))
            scales, grids = [], {}
            for hub in hubs:
                unique = {}
                for q in range(1, c["fleet"]+1):
                    group = model["values"][hub][str(q)]
                    for time, value in zip(group["times"], group["values"]):
                        if time < base["horizon"]:
                            unique[time, q] = abs(value)
                scales.append(max(100., float(np.median(list(unique.values())))))
                grids[hub] = [dict(time=t, q=q, abs_USD=v) for (t, q), v in sorted(unique.items())]
            write_json(target/"baselines"/(name+"-scales.json"), dict(hubs=hubs, scales=scales,
                table_file_sha256=expected, table_identity=frozen.hash, grids=grids,
                rule="max(100 USD, median absolute unique nonterminal grid values over q=1..K)"))
            datasets = {}
            for split, seeds in SPLITS.items():
                scenarios, provenance = [], []
                for seed in seeds:
                    generation = copy.deepcopy(base)
                    generation["orders"] = c["orders"]
                    scenario, source = make_scenario(region, pools, seed, generation)
                    scenario["id"] = f"ppo-{name}-{split}-s{seed}"
                    scenario["split"] = "validation" if "validation" in split else "test" if split in ("test", "diagnostic") else "train"
                    scenario["hubs"] = ranked[:c["fleet"]] if c["fleet"] < len(hubs) else hubs+ranked[:c["fleet"]-len(hubs)]
                    content = scenario_content(scenario)
                    if content in historical_contents:
                        overlap.append(scenario["id"])
                    all_identities.append(dict(condition=name, role=split, id=scenario["id"], seed=seed,
                                               content_hash=content, scenario_hash=identity(scenario)))
                    source.update(fleet=scenario["hubs"], fleet_seed=None, outbound_mass=mass,
                                  placement_rule=protocol["placement"])
                    scenarios.append(scenario)
                    provenance.append(source)
                datasets[split] = payload(scenarios)
                folder = "sealed" if split == "test" else "data"
                write_json(target/folder/name/(split+".json"), datasets[split])
                write_json(target/folder/name/(split+"-provenance.json"), provenance)
            sample = datasets["engineering"]["scenarios"][0]
            graph = nx.Graph()
            graph.add_weighted_edges_from(sample["edges"])
            net = Net(graph, 0, base["speed"])
            physics = {k: base[k] for k in ("speed", "capacity", "loaded_cost", "empty_cost", "waiting_cost", "expiry_cost", "max_delay")}
            physics.update(cross_hub=True, reposition=False, pickup_rule="network_radius",
                           pickup_radius=max(net.d.values()), tie_break=1.)
            ppo = PPOConfig(updates=128, episodes_per_update=8, epochs=4, minibatch_size=64,
                            hidden_size=64, training_seed=30001, validation_every=16).model_dump()
            cfg = dict(parameters=physics, solver=base["solver"], action=dict(hubs=hubs, scales=scales),
                       feature_scales=protocol["feature_scales"], baseline="baselines/"+name+".json", ppo=ppo)
            write_json(target/"protocol"/(name+".json"), cfg)
            check_cfg = copy.deepcopy(cfg)
            check_cfg["baseline"] = str(target/cfg["baseline"])
            check_config(check_cfg, payload(datasets["train"]["scenarios"]+datasets["validation"]["scenarios"]))
            print("Prepared", name, sum(len(v["scenarios"]) for v in datasets.values()), "scenarios", flush=True)
        del pools
    if overlap:
        raise RuntimeError(f"historical content overlap: {overlap[:5]}")
    for name in [key(c) for c in CONDITIONS]:
        identities = [r for r in all_identities if r["condition"] == name]
        hashes = [r["content_hash"] for r in identities]
        if len(hashes) != len(set(hashes)):
            raise RuntimeError("cross-split content overlap")
    write_json(target/"data/identities.json", all_identities)
    write_json(target/"data/placement-catalog.json", catalog)
    write_json(target/"qa/data-isolation.json", dict(status="PASSED", scenarios=len(all_identities),
        historical_seeds=sorted(historical_seeds), proposed_seeds=sorted(proposed),
        historical_content_count=len(historical_contents), content_collisions=overlap))
    manifest["input_hashes"] = {p.relative_to(target).as_posix(): sha(p) for folder in ("protocol", "data", "sealed", "baselines")
                                 for p in sorted((target/folder).rglob("*")) if p.is_file()}
    with zipfile.ZipFile(target/"protocol/source.snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        paths = [*sorted((ROOT/"model_families/single").rglob("*.py")),
                 *sorted((ROOT/"experiment_core").glob("*.py")), *sorted(PACKAGE.glob("*.py")),
                 ROOT/"pyproject.toml", ROOT/"uv.lock", TASK]
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    manifest["input_hashes"]["protocol/source.snapshot.zip"] = sha(target/"protocol/source.snapshot.zip")
    manifest.update(status="PREPARED", prepared_at=now(), study_sources=study_sources(),
                    historical_content_count=len(historical_contents), scenarios=len(all_identities))
    write_json(target/"manifest.json", manifest)
    return manifest
