"""Owned output paths, immutable history, and outcome-blind protocol."""
import os
import zipfile
from pathlib import Path
from experiment_core.storage import atomic_json, read_json, digest
from experiments.cfs_resource_policy_iteration.common import (
    ROOT, OUTPUT as HISTORY, key, parameters, rho_levels, nodes, map_jobs,
    write_csv, now, file_hash, INTERPOLATION, TERMINAL)
from experiments.cfs_dynamic_lookahead.common import freeze_data

OUTPUT = ROOT / 'experiments/cfs_resource_policy_iteration_damping'
ALPHAS = {'025': .25, '050': .5, '075': .75}
TASK = Path('D:/download/Codex_第六轮后续_A2_Damping与B_MinimalStateExpansion.md')

def configuration():
    cfg = read_json(HISTORY/'config/main.json')
    cfg.update(schema_version='cfs-self-consistent-damping/v1',
        diagnostic_seeds=list(range(8200, 8220)), pilot_seeds=[8950, 8951],
        formal_seeds=list(range(9100, 9140)), alphas=ALPHAS,
        clarification='User confirmed true self-consistent damping on 2026-10-07. '
            'Historical A reconstruction and common V0/V1 match A; subsequent raw '
            'V2/V3 evaluate ACTUAL damped D2/D3, not frozen undamped policies.',
        damping='Vtilde0=V0; Vraw_n=E(Dn); Vtilde_n=(1-alpha)*Vtilde_(n-1)+alpha*Vraw_n',
        selection=dict(scope='one global alpha; equal-weight CA80/3 and W40/8; W80 safety screen',
            collapse='D4-D1 mean < -5% abs(D1 mean) AND paired t95 upper < 0 in any key regime',
            table_increase='last relativeL2 > 1.10 * first nonzero relativeL2 + 1e-12 in CA or W40',
            action_increase='step4 full disagreement > step2 + 0.02 in CA or W40',
            primary_action_tie=.01, secondary_table_tie=.02, profit_tie_fraction=.01,
            tie_preference=['050','025','075'],
            no_eligible='select among fewest failed screens using same lexicographic rule; '
                'mark exploratory selection, ineligible for claimed stabilization'),
        effectiveness=dict(profit_margin_fraction=.01, action_improvement_ratio=.8,
            action_trend_tolerance=.01, table_worsening_ratio=1.1,
            W80_noninferiority_margin_fraction=.01,
            meaning='joint evidence, not just profit; diagnostic/formal gain directions must agree; '
                'W80 paired t95 lower must exceed negative 1% of abs(U4 mean); '
                'a failed evidence threshold is not proof of no effect'),
        comparisons=[['D4','U4'],['D2','U2'],['D3','U3'],
            ['D2','D1'],['D3','D2'],['D4','D3'],['D4','D1'],['U4','U1']],
        sample_unit='scenario seed; 40 formal independent paired scenarios per condition',
        action_comparison='canonical exact action on U0 same-state bank at ALL periods; '
            'diagnostic n20 selection, formal n40 corroboration; IDs retained',
        historical_policy_alias='D1=U1; cached duplicate trace only with explicit alias and exact identity checks')
    return cfg

def code_hashes():
    old = read_json(HISTORY/'manifests/source_manifest.json')['source_code_sha256']
    out = dict(old)
    for name, sha in old.items():
        assert file_hash(ROOT/name) == sha, f'Frozen numerical source changed: {name}'
    # These imports are PURE helpers; no historical writer is ever called.
    for name in ['analysis.py', 'action_analysis.py']:
        p = HISTORY/name; out[p.relative_to(ROOT).as_posix()] = file_hash(p)
    for p in OUTPUT.glob('*.py'):
        if p.name not in ['analysis.py', 'figures.py', 'delivery.py', 'pdf_qa.py']:
            out[p.relative_to(ROOT).as_posix()] = file_hash(p)
    p=ROOT/'tests/test_resource_policy_iteration_damping.py'
    out[p.relative_to(ROOT).as_posix()]=file_hash(p)
    return out

def initialize():
    assert os.environ.get('PYTHONHASHSEED') == '0'
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    cfg = configuration(); base = read_json(ROOT/cfg['baseline_config'])
    path = OUTPUT/'manifest.json'
    if path.exists():
        m = read_json(path)
        assert m['config_hash'] == digest(cfg)
        if m.get('code_hash'): assert m['source_code_sha256'] == code_hashes()
        return cfg, base, m
    assert read_json(HISTORY/'manifests/final_manifest.json')['status'] == 'COMPLETED'
    for d in ['config','manifests','raw','targets','tables','policies','processed','qa','figures']:
        (OUTPUT/d).mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT/'config/main.json', cfg)
    atomic_json(OUTPUT/'config/formal.json', dict(created_at=now(), formal_seen=False,
        seeds=cfg['formal_seeds'], conditions=cfg['conditions'], policies=[f'U{i}' for i in range(5)]+[f'D{i}' for i in range(1,5)],
        primary_Holm_family='four D4-U4 comparisons across conditions',
        supporting_Holm_family='20 within-path comparisons across conditions',
        exploratory_Holm_family='eight D2-U2 / D3-U3 comparisons',
        statistic='paired t95, 10000 paired bootstrap, mean SD median skewness; no cell pseudoreplication'))
    atomic_json(OUTPUT/'manifests/analysis_preregistration.json', dict(created_at=now(),
        diagnostic_seen=False, formal_seen=False, config=cfg,
        canonicalization='sort top-level collections; normalize exact scalar/sequence representations; '
            'no rounding, no ID quotient, route node order retained',
        figures='nine task-specified figure groups; 600dpi PNG/vector PDF/SVG/source CSV/config',
        oscillation_selection='largest five DEPLOYED cell ranges per condition and alpha; all cells also exported',
        no_fixed_point_claim='no E(D4), only four fixed improvements',
        interval_limit='t95/bootstrap intervals unadjusted; Holm p separately reported; MC_SE conditional on empirical parent'))
    m = dict(status='INITIALIZED', created_at=now(), config_hash=digest(cfg),
        source=read_json(HISTORY/'manifest.json')['source'], baseline_config=base,
        historical_final_sha256=file_hash(HISTORY/'manifests/final_manifest.json'),
        task_document_sha256=file_hash(TASK), policy_seed=0, training_seed=None)
    atomic_json(path, m)
    return cfg, base, m

def freeze(m):
    hashes = code_hashes()
    if m.get('code_hash'):
        assert hashes == m['source_code_sha256']; return
    snap = OUTPUT/'raw/source.snapshot.zip'
    with zipfile.ZipFile(snap,'w',zipfile.ZIP_DEFLATED) as z:
        for name in hashes: z.write(ROOT/name, name)
        z.write(OUTPUT/'config/main.json','A2-config.json'); z.write(TASK,'task.md')
    m.update(code_hash=digest(hashes), source_code_sha256=hashes,
        source_frozen_at=now(), snapshot_sha256=file_hash(snap))
    atomic_json(OUTPUT/'manifests/source_manifest.json',m)
    atomic_json(OUTPUT/'manifest.json',m)

def datasets(cfg, base, seeds):
    data={}
    for c in cfg['conditions']:
        for seed in seeds:
            p = OUTPUT/'raw/datasets'/f'{key(c)}-s{seed}.dataset.json'
            if seed in cfg['construction_seeds']:
                p = HISTORY/'raw/datasets'/f'{key(c)}-s{seed}.dataset.json'
            if p.exists(): data[c['region'],c['orders'],c['fleet'],seed]=read_json(p)
    if len(data)==len(cfg['conditions'])*len(seeds): return data
    assert not set(seeds) & set(cfg['construction_seeds']), 'Historical construction bank must already exist'
    return freeze_data(cfg,base,OUTPUT,cfg['conditions'],seeds)

def history_index():
    """Snapshot all six prior studies without calling any historical writer."""
    path=OUTPUT/'manifests/historical_readonly.json'
    if path.exists(): return read_json(path)
    old=read_json(HISTORY/'manifests/historical_readonly.json')['files']
    index={}
    print('Read-only history validation: previous five studies...',flush=True)
    for name,sha in old.items():
        assert file_hash(ROOT/name)==sha, name; index[name]=sha
    print('Read-only history freeze: sixth A...',flush=True)
    for p in sorted(HISTORY.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            index[p.relative_to(ROOT).as_posix()]=file_hash(p)
    source=read_json(HISTORY/'manifest.json')
    assert file_hash(Path(source['baseline_config']['data_source']))==source['source']['sha256']
    result=dict(created_at=now(),files=index,CFS_sha256=source['source']['sha256'])
    atomic_json(path,result); return result
