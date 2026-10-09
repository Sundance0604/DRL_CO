"""Toy statistical checks; never execute B datasets before A2 completion."""
import pytest
from experiment_core.storage import atomic_json,read_json
from experiments.cfs_resource_value_state_bucket_exact_accounting import analysis as a

def bank(tmp_path,monkeypatch,heldout_mode):
    monkeypatch.setattr(a,'OUTPUT',tmp_path)
    c=dict(region='toy',orders=2,fleet=1);cid=a.key(c)
    cfg=dict(conditions=[c],diagnostic_seeds=[8400,8401],bootstrap_seed=17,bootstrap_replications=10000)
    surf=lambda v:dict(method='table',values={'A':{'1':dict(times=[0,3,24],values=[v,v,0.])}})
    model=dict(base=surf(10),buckets={str(b):surf(-10+20*b) for b in range(3)},
        support={'A':{str(b):{'1':{'0':12,'3':12}} for b in range(3)}},measured_q=[1],observed_max_q=1,
        base_and_bucket_same_targets_sha256='frozen-construction-toy')
    train=[];diag=[]
    for b in range(3):
        for i in range(12):train.append(dict(condition=cid,time=0,hub='A',q=1,bucket=b,target=-10+20*b+(-1 if i%2 else 1),terminal_anchor=False))
        for seed in cfg['diagnostic_seeds']:
            diag.append(dict(condition=cid,time=0,hub='A',q=1,bucket=b,
                target=(-10+20*b if heldout_mode=='informative' else 10)+(.2 if seed==8400 else -.3),
                scenario_seed=seed,terminal_anchor=False,pre_injection_pressure=dict(R=b)))
    atomic_json(tmp_path/'tables/bucket_tuqb'/(cid+'.json'),model)
    atomic_json(tmp_path/'targets/construction/targets.json',train)
    atomic_json(tmp_path/'targets/diagnostic/targets.json',diag)
    return cfg

def test_bucket_variance_has_independent_predictive_corroboration(tmp_path,monkeypatch):
    cfg=bank(tmp_path,monkeypatch,'informative');result=a.variance_analysis(cfg)
    assert result[0]['n']==2 and result[0]['corroborated']
    assert result[0]['mean']==pytest.approx(800/3)
    assert len(read_json(tmp_path/'processed/diagnostic_MSE_daily.json'))==2

def test_mechanical_train_variance_reduction_is_not_confirmatory(tmp_path,monkeypatch):
    cfg=bank(tmp_path,monkeypatch,'uninformative');result=a.variance_analysis(cfg)
    var=read_json(tmp_path/'processed/within_cell_variance.json')[0]
    assert var['variance_reduction_USD2']>0
    assert result[0]['mean']<0 and not result[0]['corroborated']

def test_b_initialization_is_gated_on_completed_A2(monkeypatch,tmp_path):
    from experiments.cfs_resource_value_state_bucket_exact_accounting import common
    monkeypatch.setattr(common,'A2',tmp_path)
    atomic_json(tmp_path/'manifests/final_manifest.json',dict(status='NUMERICAL_COMPLETE_ANALYSIS_QA_PENDING'))
    with pytest.raises(AssertionError,match='final frozen A2'):common.initialize()

def test_joint_report_covers_task_twelve_questions_without_new_selection(tmp_path,monkeypatch):
    """Pure report fixture in pytest tmp, not a real numerical experiment output."""
    from experiments.cfs_resource_value_state_bucket_exact_accounting import delivery as d
    monkeypatch.setattr(d,'OUTPUT',tmp_path)
    monkeypatch.setattr(d,'A2',tmp_path/'readonly_a2_fixture')
    cfg=dict(conditions=[dict(region='california_corridor',orders=80,fleet=3),
        dict(region='western_multicluster',orders=40,fleet=8),
        dict(region='western_multicluster',orders=80,fleet=3),
        dict(region='western_multicluster',orders=20,fleet=8)])
    cids=[d.key(c) for c in cfg['conditions']]
    arows=[dict(condition=cid,profit_difference=1.,paired_t95=[-2.,4.],Holm_p=.8,
        joint_stabilization_supported=False,profit_evidence=False,action_evidence=True,table_evidence=True,
        late_damped_action=.02,late_undamped_action=.1,late_damped_relativeL2=.1,late_undamped_relativeL2=.2)
        for cid in cids]
    brows=[dict(condition=cid,formal_gain_USD=1.,t95=[-1.,3.],Holm_p=.8,diagnostic_gain_USD=2.,
        stable_gain_supported=False,query_fallback_daily_mean=.6,query_fallback_pooled=.6,
        nonterminal_query_fallback_pooled=.7,fallback_supported=False) for cid in cids]
    ae=dict(regimes=arows,damping_effective=False,W80_noninferiority_supported=False,
        W80_margin_USD=5.,W80_t95_lower=-10.)
    be=dict(regimes=brows,bucket_effective=False,W80_noninferiority_supported=False,
        W80_noninferiority_margin_USD=5.)
    selection=dict(alpha_value=.25,selected_alpha='025',exploratory_selection=False,
        records=[dict(alpha_value=v,conditions=[dict(condition=cid,D4_mean=100.+v) for cid in cids[:3]])
            for v in [.25,.5,.75]])
    variance=[dict(condition=cid,mean=-1.,CI95_lower=-3.,CI95_upper=1.,Holm_p=.8,corroborated=False)
        for cid in cids]
    osc=[dict(condition=cid,alpha=alpha,range_USD=1.,update_direction_reversal_count=1)
        for cid in cids for alpha in ['undamped','025']]
    priors={cid:dict(thresholds=dict(c33=1.,c67=2.,tied_thresholds=False)) for cid in cids}
    payloads={'evidence.json':ae,'alpha_freeze.json':selection,'table_metrics.json':[],
        'oscillations.json':osc,'priors.json':priors,'variance_corroboration.json':variance,
        'formal_statistics.json':dict(paired=[dict(condition=cids[-1],comparison='B1-B0',
            mean=1.,CI95_lower=-1.,CI95_upper=3.,Holm_p=.8)]),
        'bucket_contrast.json':[dict(condition=cid,base_value_USD=1.) for cid in cids]}
    monkeypatch.setattr(d,'read_json',lambda path:payloads[path.name])
    d.joint_report(cfg,be)
    report=(tmp_path/'A2_B_joint_diagnostic_report.md').read_text(encoding='utf-8')
    for number in range(1,13):assert f'### {number}. ' in report
    assert '最优alpha是否跨工况一致' in report and 'W20是否仍表现为资源宽松' in report
    assert '没有各工况独立alpha选择' in report and '不能确定剩余问题主要是哪一个' in report
    assert '进入bucketed self-consistent PI' in report and '进入Strategic RL' in report
    assert '不能据此选出更强机制' in report and 'Case IV' in report
    assert not (tmp_path/'readonly_a2_fixture').exists()

