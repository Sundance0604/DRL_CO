"""Post-experiment packaging and independent integrity checks, never model selection."""
import argparse
import shutil
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import numpy as np
from experiment_core.storage import atomic_json, read_json, digest
from .common import OUTPUT, ROOT, FOURTH, file_hash, write_csv, verify_historical, now
from .protocol import read_compressed


def reproduction():
    from .policies import job_run
    from .protocol import assert_replay_snapshot, action_hash
    cfg = read_json(OUTPUT / 'config/main.json')
    checks = []
    for c in cfg['conditions']:
        from .common import key
        for label in ['P2', 'P10']:
            name = f'pilot-{key(c)}-6300-{label}'
            old = read_json(OUTPUT / 'raw/runs' / name / 'summary.json')
            spec = read_json(OUTPUT / 'raw/runs' / name / 'spec.json')
            spec |= dict(id='qa-reproduction-' + name, stage='independent-reproduction')
            new = job_run(spec)
            assert new['status'] == 'COMPLETED'
            assert abs(new['realized_profit'] - old['realized_profit']) < 1e-6
            a = read_compressed(OUTPUT / 'raw/runs' / name / 'observations.json.gz')['trace']
            b = read_compressed(OUTPUT / 'raw/runs' / spec['id'] / 'observations.json.gz')['trace']
            assert len(a) == len(b)
            for x, y in zip(a, b):
                assert_replay_snapshot(x['before'], y['before'])
                assert_replay_snapshot(x['after'], y['after'])
                assert action_hash(x['plan']) == action_hash(y['plan'])
            checks.append(dict(condition=key(c), policy=label, seed=6300,
                profit_difference=new['realized_profit']-old['realized_profit'],
                physical_and_action_trace_identical=True, replay_steps=len(a), run_id=spec['id']))
    result = dict(status='PASSED', fresh_solver_runs=len(checks), checks=checks,
        model_selection_updated=False, historical_outputs_written=False)
    atomic_json(OUTPUT / 'qa/independent_reproduction.json', result)
    return result


def fit_audit():
    """Backward error and independent augmented-SVD prediction check; no refitting for deployment."""
    from scipy.linalg import lstsq
    from .fitting import matrices, common_meta, select_features
    from .features import design
    from .common import key
    cfg = read_json(OUTPUT / 'config/main.json')
    raw = read_json(OUTPUT / 'targets/construction.json')
    heldout = read_json(OUTPUT / 'targets/heldout_prediction_rows.json')
    priors = read_json(OUTPUT / 'models/priors.json')
    rows = []
    for c in cfg['conditions']:
        cid = key(c)
        table = read_json(OUTPUT / f'models/table/{cid}.json')
        training = [r for r in matrices([r for r in raw if r['condition'] == cid], table, priors[cid], c['fleet']) if not r['terminal_release']]
        for method in ['linear', 'chebyshev', 'spline', 'bayesian']:
            for fs in ['OLD', 'ACTION']:
                model = read_json(OUTPUT / f'models/{method}_{fs}/{cid}.json')
                coef = np.array(model['coefficients'])
                assert np.isfinite(coef).all()
                rr = [r for r in heldout if r['condition'] == cid and r['method'] == method + '-' + fs]
                prediction_matrix = design(rr, model, True)
                assert np.isfinite(prediction_matrix @ coef).all()
                check = dict(condition=cid, method=method, feature_set=fs,
                    parameter_count=len(coef), condition_number=model.get('condition_number'),
                    coefficients_finite=True, deployment_predictions_finite=True,
                    used_to_change_model_or_select_policy=False)
                if method == 'bayesian':
                    cov = np.array(model['posterior_covariance'])
                    eig = np.linalg.eigvalsh((cov + cov.T) / 2)
                    assert np.isfinite(cov).all() and eig.min() >= -1e-6
                    check |= dict(posterior_covariance_min_eigenvalue=float(eig.min()), conditional_uncertainty_only=True)
                else:
                    shared, P = common_meta(training, method, model['degree'], model.get('requested_knots', 4), model['boundary'], model['hubs'])
                    meta, R = select_features(shared, P, fs)
                    assert meta['bases'] == model['bases'] and meta['q_weights'] == model['q_weights']
                    weights = np.sqrt([r['weight'] for r in training])
                    X = design(training, model) * weights[:, None]
                    y = np.array([r['residual'] for r in training]) * weights
                    A = X.T @ X + model['penalty'] * R + 1e-6 * np.eye(len(coef))
                    b = X.T @ y
                    backward = np.linalg.norm(A @ coef - b) / max(1., np.linalg.norm(A) * np.linalg.norm(coef) + np.linalg.norm(b))
                    eig, vec = np.linalg.eigh((R + R.T) / 2)
                    assert eig.min() >= -1e-8
                    penalty_root = np.sqrt(model['penalty'] * np.maximum(eig, 0))[:, None] * vec.T
                    augmented = np.vstack((X, penalty_root, .001 * np.eye(len(coef))))
                    independent = lstsq(augmented, np.r_[y, np.zeros(2 * len(coef))], lapack_driver='gelsd')[0]
                    deviation = float(np.max(np.abs(prediction_matrix @ (independent - coef))))
                    assert backward < 1e-10 and deviation < .01, (cid, method, fs, backward, deviation)
                    check |= dict(relative_backward_error=float(backward),
                        max_heldout_prediction_difference_augmented_SVD_usd=deviation,
                        independent_check='same fixed penalized objective via augmented-design SVD; frozen deployment coefficients unchanged')
                rows.append(check)
    write_csv(OUTPUT / 'processed/fitting_numerical_stability.csv', rows)
    result = dict(status='PASSED', models=len(rows), checks=rows,
        max_condition_number=max(r.get('condition_number') or 0 for r in rows),
        max_SVD_prediction_difference_usd=max(r.get('max_heldout_prediction_difference_augmented_SVD_usd', 0) for r in rows),
        caveat='Centered spline/category redundant directions cause large normal-equation condition numbers; this is a finite heldout numerical check, not proof of global/extrapolation robustness.')
    atomic_json(OUTPUT / 'qa/fitting_numerical_stability.json', result)
    return {k: v for k, v in result.items() if k != 'checks'}


def prepare():
    folder = OUTPUT / 'phase1_identification'
    folder.mkdir(exist_ok=True)
    q = read_json(OUTPUT / 'manifests/phase1_query_manifest.json')
    write_csv(folder / 'queries.csv', q['queries'])
    copies = {
        'query_manifest.json': 'manifests/phase1_query_manifest.json',
        'freeze_targets.csv': 'targets/freeze_targets.csv',
        'rolling_targets.csv': 'targets/rolling_targets.csv',
        'paired_target_differences.csv': 'processed/paired_target_differences.csv',
        'ranking_flips.csv': 'processed/candidate_ranking_metrics.csv',
        'top1_disagreement.csv': 'processed/candidate_ranking_metrics.csv',
        'phase1_summary.json': 'processed/phase1_summary.json',
    }
    for name, source in copies.items():
        shutil.copyfile(OUTPUT / source, folder / name)
    shutil.copyfile(OUTPUT / 'targets/construction_mc.csv', OUTPUT / 'targets/mc_slot_targets.csv')
    shutil.copyfile(OUTPUT / 'targets/residual_targets.csv', OUTPUT / 'targets/rollout_residual_targets.csv')
    rows = []
    for split in ['construction', 'validation', 'heldout']:
        rows.extend(r | {'split': split} for r in read_json(OUTPUT / f'targets/{split}.json'))
    write_csv(OUTPUT / 'targets/rollout_action_targets.csv', rows)
    runtime = []
    m = read_json(OUTPUT / 'manifest.json')
    for split in ['phase1', 'construction', 'validation', 'heldout']:
        q = read_json(OUTPUT / f'manifests/{split}_query_manifest.json')
        runtime.append(dict(stage='query_generation', split=split,
            wall_seconds=q['selection_wall_seconds'], worker_seconds=q['selection_worker_seconds'],
            includes='natural M0 trajectory + current query feasibility + state serialization'))
        result = m['phase1_targets' if split == 'phase1' else split]
        runtime.append(dict(stage='target_pipeline', split=split, wall_seconds=result['wall_seconds'],
            worker_seconds=result['worker_seconds'], includes='rollouts + independent replay + raw serialization'))
        for mode in ['roll', 'freeze']:
            if result['runtime'][mode]:
                runtime.append(dict(stage=mode+'_worlds', split=split, wall_seconds=None,
                    worker_seconds=result['runtime'][mode], includes='unique cached world calls; excludes replay/storage; unequal mode world counts'))
    for split in ['construction_mc', 'validation_mc']:
        result = m[split]
        runtime.append(dict(stage='MC_slot_generation', split=split, wall_seconds=result['wall_seconds'],
            worker_seconds=result['summed_worker_seconds'], includes='normal-rolling finite resource marginals + analytical terminal anchors'))
    for r in m['fit']['runtimes']:
        runtime.append(dict(stage=r['method']+'_fit_pair', condition=r['condition'], split='construction+validation',
            wall_seconds=r['seconds'], worker_seconds=r['seconds'],
            includes='both OLD/ACTION construction fits, shared grid and joint validation selection'))
    write_csv(OUTPUT / 'processed/offline_runtime_breakdown.csv', runtime)
    return {'phase1_convenience_files': 8, 'combined_action_rows': len(rows)}


def numerical_integrity():
    m = read_json(OUTPUT / 'manifest.json')
    source_counts = {}
    for label, hashes, expected in [
        ('phase1', m['source_code_sha256'], m['code_hash']),
        ('phase2', m['phase2_source_hashes'], m['phase2_code_hash']),
    ]:
        assert digest(hashes) == expected
        assert all(file_hash(ROOT / name) == sha for name, sha in hashes.items()), label
        source_counts[label] = len(hashes)
    assert file_hash(OUTPUT / 'raw/source.snapshot.zip') == m['source_snapshot_sha256']
    assert file_hash(OUTPUT / 'raw/phase2-source.snapshot.zip') == m['phase2_snapshot_sha256']
    frozen = read_json(OUTPUT / 'manifests/model_freeze.json')
    assert not frozen['formal_seen']
    assert all(file_hash(OUTPUT / name) == sha for name, sha in frozen['models'].items())
    result = dict(status='PASSED', numerical_source_files=source_counts,
                  frozen_model_files=len(frozen['models']), models_unchanged=True,
                  fifth_config_hash=digest(read_json(OUTPUT / 'config/main.json')),
                  no_training_or_formal_selection_updates=True)
    assert result['fifth_config_hash'] == m['config_hash']
    atomic_json(OUTPUT / 'qa/numerical_identity_audit.json', result)
    return result


def history():
    result = verify_historical()
    old = read_json(FOURTH / 'manifests/final_manifest.json')['deliverable_sha256']
    assert all(file_hash(FOURTH / name) == sha for name, sha in old.items())
    m = read_json(OUTPUT / 'manifest.json')
    source = Path(m['baseline_config']['data_source'])
    sha = file_hash(source)
    assert sha == m['source']['sha256']
    result |= dict(fourth_deliverables=len(old), CFS_content_reverified=True,
                   CFS_sha256=sha, numerical_identity=numerical_integrity())
    atomic_json(OUTPUT / 'qa/historical_integrity.json', result)
    return result


def target_audit():
    results = {}
    for split in ['phase1', 'construction', 'validation', 'heldout']:
        qa = read_json(OUTPUT / f'qa/{split}_target_audit.json')
        assert qa['status'] == 'PASSED'
        assert all(all(r[name] for name in ['next_period_reoptimized', 'current_reward_count_once',
            'pair_hash_identical', 'candidate_current_action_identical_between_modes']) for r in qa['queries'])
        results[split] = {name: qa[name] for name in ['rows', 'unique_worlds', 'replay_periods']}
    runtimes = []
    for path in sorted((OUTPUT / 'raw/worlds/phase1').rglob('*.json.gz')):
        packet = read_compressed(path)
        w = packet['world']
        assert packet['replay']['status'] == 'PASSED'
        assert abs(packet['replay']['total_difference']) < 1e-6
        mode = 'freeze' if '-freeze' in path.name else 'rolling'
        runtimes.append(dict(mode=mode, world=path.relative_to(OUTPUT).as_posix(),
            wall_seconds=w['wall_seconds'], solve_count=len(w['solver_periods'])))
    write_csv(OUTPUT / 'processed/phase1_world_runtime.csv', runtimes)
    runtime = {}
    for mode in ['freeze', 'rolling']:
        selected = [r for r in runtimes if r['mode'] == mode]
        runtime[mode] = dict(unique_worlds=len(selected), total_seconds=sum(r['wall_seconds'] for r in selected),
            mean_world_seconds=float(np.mean([r['wall_seconds'] for r in selected])),
            median_world_seconds=float(np.median([r['wall_seconds'] for r in selected])),
            total_solves=sum(r['solve_count'] for r in selected))
    runtime['total_rolling_to_freeze_ratio'] = runtime['rolling']['total_seconds'] / runtime['freeze']['total_seconds']
    runtime['mean_world_rolling_to_freeze_ratio'] = runtime['rolling']['mean_world_seconds'] / runtime['freeze']['mean_world_seconds']
    runtime['interpretation'] = 'Unequal cached world counts: total ratio is not a same-world cost ratio. Mean-world ratio is descriptive; states/threshold mix also differs.'
    result = dict(status='PASSED', splits=results, phase1_runtime=runtime)
    atomic_json(OUTPUT / 'qa/target_delivery_audit.json', result)
    return result


def world_index():
    rows = []
    for path in sorted((OUTPUT / 'raw/worlds').rglob('*.json.gz')):
        packet = read_compressed(path)
        w = packet['world']
        replay = packet['replay']
        assert replay['status'] == 'PASSED' and abs(replay['total_difference']) < 1e-6
        assert len(w['trace']) == replay['periods']
        assert w['fare_count_once'] and w['completed_commitments']
        for x in w['trace']:
            if not x.get('drain'):
                assert abs(x['plan']['accounting']['profit_identity_residual']) < 1e-6
        rows.append(dict(world=path.relative_to(OUTPUT).as_posix(), sha256=file_hash(path),
            identity=packet['identity'], stage=path.relative_to(OUTPUT / 'raw/worlds').parts[0],
            trace_steps=len(w['trace']), source_state_hash=w['source_state_hash'], future_hash=w['future_hash'],
            current_action_hash=w['current_action_hash'], current_stage=w['current_stage'], total=w['total'],
            replay_status=replay['status']))
    write_csv(OUTPUT / 'processed/raw_target_world_index.csv', rows)
    result = dict(status='PASSED', worlds=len(rows), trace_steps=sum(r['trace_steps'] for r in rows),
        unique_worlds_per_split=dict(Counter(r['stage'] for r in rows)),
        index_sha256=file_hash(OUTPUT / 'processed/raw_target_world_index.csv'))
    atomic_json(OUTPUT / 'qa/raw_target_world_index_audit.json', result)
    return result


def run_audit():
    indices = []
    total_steps = 0
    statuses = Counter()
    local_contexts = 0
    coefficient_checks = 0
    for stage, expected in [('pilot', 112), ('formal', 2240)]:
        rows = read_json(OUTPUT / f'processed/{stage}_policies.json')
        assert len(rows) == expected
        assert all(r['status'] == 'COMPLETED' for r in rows)
        for r in rows:
            folder = OUTPUT / 'raw/runs' / r['id']
            path = folder / 'observations.json.gz'
            assert file_hash(path) == r['observations_sha256']
            packet = read_compressed(path)
            spec = read_json(folder / 'spec.json')
            assert digest(spec) == r['identity']
            assert digest(spec['scenario']) == r['scenario_hash']
            replay = packet['replay']
            assert replay['status'] == 'PASSED' and abs(replay['total_difference']) < 1e-6
            assert packet['metrics']['unfinished_committed'] == 0
            assert packet['metrics']['assigned'] == packet['metrics']['delivered']
            trace = packet['trace']
            assert replay['periods'] == len(trace) == r['replay_steps']
            assert all(all(x['invariants'].values()) for x in trace)
            assert [x['period'] for x in trace] == list(range(len(trace)))
            bookings = {o['id']: o['book_time'] for o in spec['scenario']['orders']}
            for x in trace:
                if x.get('drain'):
                    continue
                assert all(bookings[a['order_id']] <= x['period'] for a in x['plan']['assignments'])
                info = x['plan']['solver']
                statuses[str(info.get('status', 'not_recorded'))] += 1
                assert info.get('quadratic_terms', 0) == 0
                assert info.get('quadratic_constraints', 0) == 0
                if 'context_hash_before' in info:
                    assert info['context_hash_before'] == info['context_hash_after']
                    coefficient_checks += 1
                from model_families.single.local_value.providers import CONTEXT_FIELDS
                assert r['policy'] == 'D' or info.get('information_mode') != 'oracle_preview'
                for entry in info.get('selected_local_contexts', []):
                    assert set(entry['context']) == CONTEXT_FIELDS
                    assert entry['context']['decision_time'] == x['period']
                    assert not any('actual_post' in f for f in entry['features'])
                    local_contexts += 1
            total_steps += len(trace)
            indices.append(dict(stage=stage, run_id=r['id'], condition=r['condition'], seed=r['seed'],
                policy=r['policy'], scenario_hash=r['scenario_hash'], model_hash=r['model_hash'],
                observations=path.relative_to(OUTPUT).as_posix(), observations_sha256=r['observations_sha256'],
                trace_steps=len(trace), replay_status=replay['status']))
    write_csv(OUTPUT / 'processed/raw_run_index.csv', indices)
    result = dict(status='PASSED', audited_runs=len(indices), audited_steps=total_steps,
        formal_runs=2240, pilot_runs=112, failures=0, information_leakage_violations=0,
        incomplete_commitments=0, cash_tolerance_usd=1e-6, physical_replay_tolerance='exact',
        solver_status_counts=dict(statuses), frozen_context_hash_checks=coefficient_checks,
        audited_selected_pre_solve_context_records=local_contexts, nonlinear_value_terms=0,
        raw_run_index_sha256=file_hash(OUTPUT / 'processed/raw_run_index.csv'))
    atomic_json(OUTPUT / 'qa/raw_run_delivery_audit.json', result)
    return result


def tests():
    p = OUTPUT / 'qa/pytest-full.xml'
    suites = ET.parse(p).getroot()
    totals = Counter()
    for suite in suites.iter('testsuite'):
        for field in ['tests', 'failures', 'errors', 'skipped']:
            totals[field] += int(suite.get(field, 0))
    assert totals['failures'] == totals['errors'] == 0, dict(totals)
    return dict(totals) | dict(xml_sha256=file_hash(p),
        initial_attempt='152 passed / 2 failed: source identity during artifact creation and Windows metrics-file sharing race; both isolated retries passed',
        warnings=1, warning='Existing Starlette anyio BlockingPortal deprecation',
        retry_xml_sha256=file_hash(OUTPUT / 'qa/pytest-retry.xml'))


def finalize():
    m = read_json(OUTPUT / 'manifest.json')
    assert m['status'] == 'NUMERICAL_RUNS_COMPLETED'
    prepared = prepare()
    targets = target_audit()
    worlds = world_index()
    runs = run_audit()
    numerical = numerical_integrity()
    historical = read_json(OUTPUT / 'qa/historical_integrity.json')
    assert historical['status'] == 'PASSED' and historical['CFS_content_reverified']
    pytest = tests()
    reproduction_qa = read_json(OUTPUT / 'qa/independent_reproduction.json')
    assert reproduction_qa['status'] == 'PASSED'
    fit_qa = read_json(OUTPUT / 'qa/fitting_numerical_stability.json')
    assert fit_qa['status'] == 'PASSED'
    dependency = read_json(OUTPUT / 'qa/heldout_donor_dependency_audit.json')
    assert dependency['status'] == 'COMPLETED' and not dependency['model_selection_updated']
    pdf = read_json(OUTPUT / 'qa/vector_pdf_audit.json')
    assert pdf['status'] == 'PASSED' and pdf['atlas_pages'] == 16
    assert (OUTPUT / 'deployment_aligned_rollout_target_report.md').exists()
    analysis = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in [
        OUTPUT / 'statistics.py', OUTPUT / 'figures.py', OUTPUT / 'delivery.py', OUTPUT / 'pdf_qa.py']}
    snapshot = OUTPUT / 'raw/analysis-source.snapshot.zip'
    with zipfile.ZipFile(snapshot, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in analysis:
            z.write(ROOT / name, name)
    final = dict(schema_version='cfs-target-alignment-delivery/v1', status='COMPLETED', created_at=now(),
        prepared=prepared, targets=targets, raw_target_worlds=worlds, raw_runs=runs, numerical=numerical, historical=historical,
        pytest=pytest, vector_pdf_qa=pdf, original_model_freeze_sha256=file_hash(OUTPUT / 'manifests/model_freeze.json'),
        independent_reproduction=reproduction_qa,
        fitting_numerical_stability={k:v for k,v in fit_qa.items() if k!='checks'},
        heldout_donor_dependence={k:v for k,v in dependency.items() if k not in ['summaries','paired']},
        analysis_source_sha256=analysis, analysis_snapshot_sha256=file_hash(snapshot),
        phase1_gate='T1', heldout_requested_scenarios=12,
        heldout_actual_scenarios={'CA80/3': 9, 'W40/8': 12, 'W80/3': 6, 'W20/8': 12},
        not_run=['DRL', 'SAC', 'neural value', 'MCMC', 'RH4+V', 'route-slot coupling upgrade', 'full nonlinear joint post-state V'],
        git_action='none; current request did not authorize branch switch/commit/push',
        statistical_limitations=['system-relative action advantage is not additive resource marginal',
            'original terminal gate and q-average retained', 'reduced legal heldout coverage',
            'per-row descriptive metrics are not independent-sample inference', 'shared heldout donor dependence: primary parent-block intervals approximate; independent own-future sensitivity post-hoc', 'unadjusted t95 intervals; Holm p separately',
            'OraclePreview H8 is not a certified global bound'])
    atomic_json(OUTPUT / 'qa/delivery_integrity_audit.json', dict(status='PASSED',
        formal_runs=2240, pilot_runs=112, audited_steps=runs['audited_steps'],
        numerical_and_model_identity_unchanged=True, historical_files=historical['files'],
        figure_groups=16, pytest=pytest))
    final['deliverable_sha256'] = {p.relative_to(OUTPUT).as_posix(): file_hash(p)
        for p in sorted(OUTPUT.rglob('*')) if p.is_file()
        and '__pycache__' not in p.parts and not any(part in ['raw', 'pdf_render'] for part in p.relative_to(OUTPUT).parts)
        and p != OUTPUT / 'manifests/final_manifest.json'}
    atomic_json(OUTPUT / 'manifests/final_manifest.json', final)
    print(dict(status=final['status'], formal_runs=2240, pilot_runs=112, trace_steps=runs['audited_steps'],
        deliverables=len(final['deliverable_sha256']), tests=pytest['tests']), flush=True)
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'history', 'targets', 'world_index', 'runs', 'reproduction', 'fit_audit', 'finalize'])
    args = parser.parse_args()
    print(globals()[{'targets': 'target_audit', 'runs': 'run_audit'}.get(args.stage, args.stage)]())
