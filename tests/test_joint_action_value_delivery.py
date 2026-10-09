"""Non-numerical delivery controls; all file writes use pytest temporary paths."""
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from experiment_core.storage import atomic_json
from experiments.cfs_joint_action_value_diagnostic import delivery,figures

def xml(tmp_path,skipped=False):
    (tmp_path/'qa').mkdir();root=ET.Element('testsuites');suite=ET.SubElement(root,'testsuite')
    for i in range(27):
        case=ET.SubElement(suite,'testcase',classname='tests.test_joint_action_value_diagnostic' if i<26 else 'tests.test_integer_action_accounting',name=str(i))
        if skipped and i==0:ET.SubElement(case,'skipped')
    ET.ElementTree(root).write(tmp_path/'qa/pytest_final.xml')

def test_final_qa_requires_actual_final_xml_not_preflight(tmp_path,monkeypatch):
    monkeypatch.setattr(delivery,'OUTPUT',tmp_path)
    with pytest.raises(AssertionError,match='actual related tests'):delivery.tests()

def test_final_qa_rejects_skipped_controls(tmp_path,monkeypatch):
    monkeypatch.setattr(delivery,'OUTPUT',tmp_path);xml(tmp_path,True)
    with pytest.raises(AssertionError):delivery.tests()
    assert not (tmp_path/'qa/tests.json').exists()

def test_final_qa_parses_independent_xml_and_keeps_its_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(delivery,'OUTPUT',tmp_path);xml(tmp_path)
    delivery.tests()
    from experiment_core.storage import read_json
    qa=read_json(tmp_path/'qa/tests.json')
    assert qa['test_count']==27 and qa['xml_sha256']==delivery.sha(tmp_path/'qa/pytest_final.xml')

def test_completed_delivery_never_writes_or_reopens_numeric_entry(tmp_path,monkeypatch):
    monkeypatch.setattr(delivery,'OUTPUT',tmp_path);atomic_json(tmp_path/'manifest.json',dict(status='COMPLETED'))
    with pytest.raises(AssertionError,match='immutable'):delivery.writable()

def test_zero_width_scientific_interval_keeps_exact_zero():
    figures.style();fig,ax=figures.plt.subplots()
    r=dict(mean_USD=0.,t95_low_USD=0.,t95_high_USD=0.)
    figures.interval(ax,1,r,'black','o');figures.interval(ax,2,r,'black','s',True)
    assert np.array_equal(ax.lines[0].get_xdata(),[0.])
    figures.plt.close(fig)
