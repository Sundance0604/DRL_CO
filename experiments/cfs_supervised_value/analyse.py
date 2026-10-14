"""Independent cash/metric summaries and scientific plots for a single pilot."""
import csv
from collections import defaultdict
import math
from pathlib import Path
import statistics
import numpy as np
from .common import *


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def concordance(y,p,keys):
    groups=defaultdict(list)
    for i,k in enumerate(keys):groups[k].append(i)
    values=[];pairs=0;ties=0
    for ids in groups.values():
        scores=[]
        for n,i in enumerate(ids):
            for j in ids[n+1:]:
                if abs(y[i]-y[j])<=1e-6:continue
                dy=y[i]-y[j];dp=p[i]-p[j]
                scores.append(.5 if abs(dp)<=1e-6 else float(dy*dp>0));ties+=int(abs(dp)<=1e-6);pairs+=1
        if scores:values.append(float(np.mean(scores)))
    return dict(mean_group_concordance=float(np.mean(values)) if values else None,
        groups_with_non_tied_labels=len(values),pairs=pairs,prediction_ties=ties)


def predictions(target):
    from model_families.single.value.providers import MCTableValueProvider
    rows=csv_read(target/'data/evaluation.csv');cfg=read(target/'data/runtime_config.json')
    table=MCTableValueProvider(read(cfg['baseline']));y=np.array([float(r['target']) for r in rows])
    preds={'PureMC':np.array([table.marginal_release_value(int(r['release_time']),r['hub'],int(r['q']),dict(T=24)) for r in rows])}
    for method in ['Basic','State']:preds[method]=np.array(read(target/'training'/f'{method}_evaluation.json')['predictions'])
    report=[];records=[];qa=[]
    for method,p in preds.items():
        keys=[(r['scenario_seed'],r['decision_time'],r['release_time']) for r in rows]
        result=metric(y,p);result.update(method=method,scope='all nonterminal',
            candidate_rank=concordance(y,p,keys),hub_rank=concordance(y,p,[k+(r['q'],) for k,r in zip(keys,rows)]))
        report.append(result);report.append(dict(method=method,scope='q>=2',**metric(y[[int(r['q'])>=2 for r in rows]],p[[int(r['q'])>=2 for r in rows]])))
        errors=[float(a-b) for a,b in zip(p,y)]
        rmse=math.sqrt(statistics.fmean(e*e for e in errors));mae=statistics.fmean(abs(e) for e in errors)
        assert math.isclose(rmse,result['RMSE'],abs_tol=1e-8) and math.isclose(mae,result['MAE'],abs_tol=1e-8)
        qa.append(dict(method=method,RMSE_stdlib=rmse,MAE_stdlib=mae))
        groups=defaultdict(list)
        for r,v in zip(rows,p):groups[r['release_time'],r['hub'],r['q']].append(float(v))
        result['mean_across_state_prediction_SD']=float(np.mean([np.std(a) for a in groups.values()]))
        for r,v in zip(rows,p):records.append(dict(scenario_seed=r['scenario_seed'],decision_time=r['decision_time'],
            release_time=r['release_time'],hub=r['hub'],q=r['q'],method=method,target=float(r['target']),prediction=float(v)))
    csv_write(target/'prediction_comparison.csv',records)
    write(target/'qa/prediction_metrics_independent.json',dict(status='PASSED',results=qa))
    write(target/'prediction_summary.json',report)
    boards=[]
    for method in ['Basic','State']:boards.extend(csv_read(target/'training'/f'{method}_leaderboard.csv'))
    csv_write(target/'autogluon_leaderboards.csv',boards)
    print('PREDICTIONS',[(r['method'],round(r['RMSE'],2)) for r in report if r['scope']=='all nonterminal'],flush=True)


def finish(target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments.cfs_ppo_strategic_control.study_execution import replay_rh4
    cfg=read(target/'data/runtime_config.json');scenes=read(target/'data/development_scenarios.json')
    independent=[]
    for scene in scenes:
        for method in ['M0','PureMC','Basic','State']:
            record=read(target/'raw'/f's{scene["seed"]}-{method}.json')
            check=replay_rh4(scene,cfg,record);independent.append(dict(scenario_seed=scene['seed'],method=method,**check))
    write(target/'qa/independent_cash_replay.json',dict(status='PASSED',days=24,results=independent))
    prediction=read(target/'prediction_summary.json');profit=csv_read(target/'policy_profit_comparison.csv')
    runtime=csv_read(target/'runtime_comparison.csv');switch=csv_read(target/'decision_switches.csv')
    methods=['M0','PureMC','Basic','State'];seeds=[s['seed'] for s in scenes]
    values={m:np.array([float(next(r for r in profit if r['method']==m and int(r['scenario_seed'])==s)['profit_USD']) for s in seeds]) for m in methods}
    qa=read(target/'qa/numerical.json');dataset=read(target/'dataset_manifest.json')
    fit={m:read(target/'training'/f'{m}_fit.json') for m in ['Basic','State']}
    cost=csv_read(target/'qa/historical_label_cost.csv')
    costs={stage:sum(float(r['total_job_seconds']) for r in cost if r['stage']==stage) for stage in ['construction','diagnostic']}
    summary=dict(prediction=prediction,profit={m:dict(mean=float(v.mean()),scenarios=v.tolist(),
        paired_minus_MC=(v-values['PureMC']).tolist(),mean_gain_USD=float((v-values['PureMC']).mean())) for m,v in values.items()},
        seeds=seeds,fit_seconds={m:fit[m]['fit_wall_seconds'] for m in fit},model_best={m:fit[m]['model_best'] for m in fit},
        historical_W40_label_job_seconds=costs,new_MC_solves=0,all_new_solves=qa['solves'],
        state_versus_basic_mean_gain_USD=float((values['State']-values['Basic']).mean()),
        model_disk_bytes=sum(p.stat().st_size for p in (target/'models').rglob('*') if p.is_file()))
    write(target/'analysis.json',summary)
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    def save(fig,name):
        for ext in ['png','svg','pdf']:fig.savefig(target/'figures'/f'{name}.{ext}',dpi=600,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    rr=[r for r in prediction if r['scope']=='all nonterminal']
    axes[0].bar([r['method'] for r in rr],[r['RMSE'] for r in rr]);axes[0].set(title='Held-out resource prediction',ylabel='RMSE (USD)')
    axes[1].bar([r['method'] for r in rr],[r['candidate_rank']['mean_group_concordance'] for r in rr]);axes[1].set(title='Within-state candidate ranking',ylabel='Group mean concordance',ylim=(0,1))
    save(fig,'01-supervised-quality')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    for m in methods:axes[0].plot(range(6),values[m],marker='o',label=m)
    axes[0].set(xlabel='Development scenario',ylabel='Full-cooldown profit (USD)',xticks=range(6),xticklabels=[str(x) for x in seeds]);axes[0].legend(fontsize=8)
    for m in ['Basic','State']:axes[1].plot(range(6),values[m]-values['PureMC'],marker='o',label=m)
    axes[1].axhline(0,color='grey');axes[1].set(xlabel='Development scenario',ylabel='Paired gain vs PureMC (USD)',xticks=range(6),xticklabels=[str(x) for x in seeds]);axes[1].legend()
    save(fig,'02-policy-profit')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    for m in methods:
        samples=[r for r in runtime if r['method']==m]
        axes[0].bar(m,np.mean([float(r['solver_seconds']) for r in samples]),label=None)
        axes[0].bar(m,np.mean([float(r['feature_RPC_seconds']) for r in samples]),bottom=np.mean([float(r['solver_seconds']) for r in samples]),color='orange')
        axes[1].bar(m,np.mean([int(r['q2_occupied']) for r in samples]))
    axes[0].set(title='Mean day cost: MILP + feature/RPC',ylabel='Seconds');axes[1].set(title='Occupied nonterminal q >= 2 slots',ylabel='Mean occupied slot events / day')
    save(fig,'03-cost-and-slots')
    lines=['# AutoGluon监督资源价值轻量实验 I','',
        '本轮只运行W40/8，监督学习而非强化学习；训练两个AutoGluon，不追加PPO或MC数据生成。标签预检及语义对齐见supervised_data_audit.md。',
        '', '## 数据、时间与资源语义','',
        f'使用第六轮精确核算的 {dataset["nonterminal_supervised_records"]:,} 个非终端配对边际记录，80个完整场景：48训练、12验证、20独立监督评价。每标签一个共同未来，无法估计状态内MC标准误。跨场景误差包含条件方差，不把标签当无噪声Bellman真值。',
        '', '旧标签在释放τ执行自由车辆数量替换；输入使用同一M0自然轨迹较早t公开快照。两世界[t,τ)前缀与现金共同抵消，标签不变。没有使用τ时刻状态作t时刻特征，也没有把表值复制到不同状态。该目标包含共同M0前缀，不是候选动作条件价值；资源替换q与在线释放slot q沿用旧近似映射，不宣称严格等价。',
        '', 'PureMC为冻结的第六轮base_tuq，使用全60构建场景；AG只有48训练/12选择，监督评价20组均未参与三种模型构建。Basic按真实表接口只含τ/h/q；State增至20项公开摘要。需求强度仅来自48训练组，旧bucket/prior及未来信息不入模型。',
        '', '## 训练与预测','',
        '现有pavane环境AutoGluon 1.6.3；medium_quality、CPU1/GPU0、显式tuning_data、无bagging/stacking/HPO。两个fit各请求150秒，预留30秒开销，实际fit上限180秒；与任务示例180秒的差异预先登记。默认轻量ensemble允许但不强选。官方参数参考[fit文档](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html)，并核对本机1.6.3签名（官网stable显示1.6.1）。',
        '', '| 模型 | 最佳模型 | fit秒 | 磁盘MiB |','|---|---|---:|---:|']
    for m in fit:lines.append(f'| {m} | {fit[m]["model_best"]} | {fit[m]["fit_wall_seconds"]:.2f} | {fit[m]["models_bytes"]/2**20:.2f} |')
    lines+=['','完整leaderboard、各模型分数及最佳模型info（含ensemble依赖/权重，如适用）保存在training/和autogluon_leaderboards.csv。缺少依赖的模型失败按AutoGluon原日志保留，未临时安装或替换基础学习器。','',
        '| 方法／样本 | RMSE USD | MAE USD | R² | 符号错误率 |','|---|---:|---:|---:|---:|']
    for r in prediction:lines.append(f'| {r["method"]} / {r["scope"]} | {r["RMSE"]:.2f} | {r["MAE"]:.2f} | {r["R2"]:.4f} | {r["sign_error"]:.2%} |')
    lines+=['','排序以同场景、同公开时刻、同释放时刻为组；目标相等的对不比较，预测平手计0.5，详细候选及同q跨hub concordance见analysis.json。符号错误用1e−6美元定义零，零目标占比较高时需连同RMSE理解。','']
    for r in rr:lines.append(f'- {r["method"]}：候选排序concordance={r["candidate_rank"]["mean_group_concordance"]:.4f}，同q跨hub={r["hub_rank"]["mean_group_concordance"]:.4f}；相同τ/h/q跨公开状态预测标准差均值={r["mean_across_state_prediction_SD"]:.4f} USD。')
    lines+=['','## 六个新开发日的完整冷却利润','',
        '从21028开始按递增规则跳过已登记/冻结历史seed，在训练之前固定六日。不是正式显著性检验；不按收益选择场景或调整价格。四方法使用相同原H1 MILP、约束、物理与现金账本，仅provider不同，负值/非单调q保留，τ≥T价值为零。','',
        '| 场景seed | M0 | Pure MC | AG Basic | AG State |','|---|---:|---:|---:|---:|']
    for i,s in enumerate(seeds):lines.append('| '+str(s)+' | '+' | '.join(f'{values[m][i]:.2f}' for m in methods)+' |')
    lines.append('| 均值 | '+' | '.join(f'{values[m].mean():.2f}' for m in methods)+' |')
    lines+=['','配对AG−MC（按上表seed顺序）：']
    for m in ['Basic','State']:lines.append(f'- {m}：'+', '.join(f'{x:.2f}' for x in values[m]-values['PureMC'])+f' USD；均值 {(values[m]-values["PureMC"]).mean():.2f} USD。')
    lines.append(f'- State−Basic：均值 {(values["State"]-values["Basic"]).mean():.2f} USD。')
    lines+=['','## 决策、槽位与运行成本','',
        '| 方法 | 完成订单均值 | 每日线上query均值 | q≥2 query均值 | q≥2实际占用均值 | solver秒/日 | 特征+RPC秒/日 |','|---|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        a=[r for r in runtime if r['method']==m];b=[r for r in profit if r['method']==m]
        lines.append(f'| {m} | {np.mean([int(r["delivered"]) for r in b]):.2f} | {np.mean([int(r["queries"]) for r in a]):.1f} | {np.mean([int(r["q2_queries"]) for r in a]):.1f} | {np.mean([int(r["q2_occupied"]) for r in a]):.1f} | {np.mean([float(r["solver_seconds"]) for r in a]):.3f} | {np.mean([float(r["feature_RPC_seconds"]) for r in a]):.3f} |')
    for m in ['M0','Basic','State']:
        a=[r for r in switch if r['method']==m];changed=sum(r['business_changed']=='True' for r in a);same=[r for r in a if r['exact_same_before_state']=='True']
        lines.append(f'- {m}相对PureMC：规范化业务决策不同{changed}/144期；完整before相同期{len(same)}，其中决策不同{sum(r["business_changed"]=="True" for r in same)}期。')
    lines+=['','业务等价保留每条路线/订单及车辆决策前承诺签名，仅允许同质车辆编号互换。闭环后续状态会分叉，所以全144期差异不是同状态因果效应，也不是144个独立日。占用数据由已提交物理路线投影逐q前缀重建，所有步的slot奖金与原MILP目标核对；query计数不含独立奖金复算的额外查价。',
        '', '批量及单行模型延迟见training/*_latency.json，线上实际worker批量推理及特征/RPC耗时分开记录。子进程使AutoGluon与原Gurobi环境隔离，两者均CPU1，不修改原依赖。',
        '',f'新增主求解 {qa["solves"]} 次、24完整日，新增MC标签求解0次；数值阶段 {qa["wall_seconds"]:.2f} 秒（{qa["wall_seconds"]/60:.2f}分钟）。原W40 construction标签作业累计 {costs["construction"]:.2f} worker秒，diagnostic {costs["diagnostic"]:.2f} worker秒，不是本次新增耗时，也不把复用的MC构建成本记为零。其他三工况共享预处理成本未按W40分摊。',
        '', '所有24日通过独立整数动作回放与现金重算，完整冷却、容量、ready/deadline检查；标签来源、源代码与模型hash运行后核验。数据与模型留本地，不同步GitHub。',
        '', '## 研究判断与限制','']
    basic=next(r for r in rr if r['method']=='Basic');state=next(r for r in rr if r['method']=='State');mc=next(r for r in rr if r['method']=='PureMC')
    lines.append(f'加入状态使独立监督RMSE从 {basic["RMSE"]:.2f} 变为 {state["RMSE"]:.2f} USD；是否下降：{state["RMSE"]<basic["RMSE"]}。State相对PureMC的六日平均利润差为 {(values["State"]-values["PureMC"]).mean():.2f} USD，正差日数 {int(np.sum(values["State"]-values["PureMC"]>1e-6))}/6。预测误差与调度利润分别评价，不能用RMSE改善宣称策略成功。')
    lines+=['','本轮仅一次轻量AutoGluon配置、单一W40、六个开发日，不代表集成方法性能上限或总体优越。有限配对标签含一次实际未来的噪声；固定共同M0前缀与资源替换不是本次候选动作的后继状态价值。后续状态分布会随新provider改变，监督训练覆盖不到全部在线τ/t/q组合，q=5/7依赖模型泛化。',
        '', '对于MILP关键区域，本次证据限于实际占用q、闭环决策分叉和独立监督排序，尚未估计每个整数最优边界附近的局部预测误差，不能宣称“关键区域误差已改善”。',
        '', '若状态模型既改善独立预测又呈现跨开发日较一致现金趋势，可下一步单独预登记更多独立场景确认；若二者方向不一致，应先检查时刻/资源替换近似与决策区域误差。当前数据足以开展这次监督试验，但不支持自动升级Deep Sets、残差模型或更多训练；本轮结束不启动第二轮。',
        '', '图形：figures/01-supervised-quality、02-policy-profit、03-cost-and-slots，各PDF/SVG/600dpi PNG。完整逐预测、逐政策、逐查询、逐实际占用和运行时间文件见目录；最终状态以final_manifest.json为准。']
    (target/'autogluon_mc_value_pilot_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(target/'qa/analysis.json',dict(status='PASSED',independent_cash_days=24,independent_metrics=True,
        all_expected_methods_retained=True,no_new_training_after_evaluation=True))
    print('REPORT_READY',target,summary['profit'],flush=True)
