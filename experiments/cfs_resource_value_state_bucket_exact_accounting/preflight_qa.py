"""Record evidence from actual pytest XML, never manufacture test outcomes."""
from pathlib import Path
import xml.etree.ElementTree as ET
from experiment_core.storage import atomic_json
from .common import OUTPUT,FAILED_B,A2,ROOT,file_hash,now,failed_attempt_gate


def xml_evidence(path):
    root=ET.parse(path).getroot()
    suites=list(root.iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert counts['tests']>0 and counts['failures']==counts['errors']==counts['skipped']==0
    return dict(status='PASSED',checked_at=now(),**counts,
        XML_ref=path.relative_to(OUTPUT).as_posix(),XML_sha256=file_hash(path))


def main():
    original=failed_attempt_gate()
    final=__import__('experiment_core.storage',fromlist=['read_json']).read_json(A2/'manifests/final_manifest.json')
    for name,sha in final['deliverable_sha256'].items():assert file_hash(A2/name)==sha,name
    assert file_hash(OUTPUT/'analysis.py')==file_hash(FAILED_B/'analysis.py')
    unchanged=['pressure.py','provider.py','analysis.py','figures.py','policies.py']
    for name in unchanged:assert file_hash(OUTPUT/name)==file_hash(FAILED_B/name),name
    tests=xml_evidence(OUTPUT/'qa/pytest_amendment.xml')
    names={case.attrib['name'] for case in ET.parse(OUTPUT/'qa/pytest_amendment.xml').getroot().iter('testcase')}
    assert {'test_real_stopped_seed8405_q3_passes_strict_full_cooldown_replay',
        'test_near_integer_solver_cash_is_not_executed_and_raw_values_are_retained',
        'test_nonoptimal_stale_and_invalid_original_actions_are_not_accepted'}<=names
    atomic_json(OUTPUT/'qa/accounting_amendment.json',dict(status='PASSED',checked_at=now(),
        tests=tests,real_seed8405_q3_full_cooldown_replay=True,
        same_selected_actions=True,strict_original_tolerances=True,
        unchanged_sources={n:file_hash(OUTPUT/n) for n in unchanged},
        A2_deliverables_verified=len(final['deliverable_sha256']),original_attempt=original))
    atomic_json(OUTPUT/'qa/tests_preflight.json',xml_evidence(OUTPUT/'qa/pytest_preflight.xml'))
    print('Actual XML, original failure regression, unchanged statistics and historical source/deliverable gates PASSED.',flush=True)


if __name__=='__main__':main()
