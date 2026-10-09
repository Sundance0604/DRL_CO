"""Independent stdlib scientific recomputation; optional existing Matplotlib plots."""
import argparse
import csv
import math
from pathlib import Path

from experiments.cfs_ppo_signal_action_diagnostic.analyse import (
    read, write, sha, identity, physical_hash, coefficient_checks)
from experiments.cfs_ppo_decision_coverage.analyse import canonical_hash


def summarize(rows):
    keys = [(r['scenario_seed'],r['period'],r['training_seed'],r['multiplier']) for r in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate logical probe")
    result=[]
    for scene in sorted({r['scenario_seed'] for r in rows})+[None]:
        for seed in sorted({r['training_seed'] for r in rows}):
            for scale in (1.,4.):
                for subset in ('all','axis_responsive','axis_nonresponsive'):
                    group=[r for r in rows if (scene is None or r['scenario_seed']==scene)
                           and r['training_seed']==seed and r['multiplier']==scale
                           and (subset=='all' or r['prior_axis_responsive']==(subset=='axis_responsive'))]
                    result.append(dict(scenario_seed=scene,training_seed=seed,multiplier=scale,subset=subset,
                        states=len(group),changed=sum(r['changed'] for r in group),
                        coefficients_changed=sum(r['coefficient_changed'] for r in group),
                        clipped_components=sum(r['clipped_components'] for r in group),
                        max_abs_residual_USD=max((r['max_abs_residual_USD'] for r in group),default=0.)))
    return result


def collect(target):
    m=read(target/'manifest.json')
    if m['status']!='NUMERICAL_COMPLETE_QA_PENDING':
        raise RuntimeError('numerical completion required')
    protocol=read(target/'protocol/preregistration.json')
    coverage=Path(protocol['coverage_source'])
    item=read(target/'data'/(protocol['condition']+'.json'))
    table=read(item['config']['baseline'])
    probes=[]; signals=[]; states=[]; calls=0; variation=[]
    for rel,digest in m['raw_hashes'].items():
        if sha(target/rel)!=digest:
            raise RuntimeError('raw hash changed')
        raw=read(target/rel)
        scene=raw['states'][0]['scenario_seed']
        frozen=read(coverage/'raw'/f's{scene}.json')
        if sha(coverage/'raw'/f's{scene}.json')!=m['history_hashes'][str(coverage/'raw'/f's{scene}.json')]:
            raise RuntimeError('frozen source raw changed')
        if [s['period'] for s in raw['states']]!=list(range(24)):
            raise RuntimeError('periods missing')
        for s,old in zip(raw['states'],frozen['states'],strict=True):
            axis=any(r['changed'] for r in old['logical'] if r['method']=='axis')
            if s['observation_hash']!=old['observation_hash'] or s['prior_axis_responsive']!=axis:
                raise RuntimeError('public context/subgroup changed')
            ref=old['probes'][old['reference_key']]
            if s['probes'][s['reference_key']]!=ref:
                raise RuntimeError('reference probe changed')
            if {a['training_seed'] for a in s['actors']} != set(protocol['training_seeds']):
                raise RuntimeError('actor coverage changed')
            actors={a['training_seed']:a['signal'] for a in s['actors']}
            if len(s['logical'])!=6 or len({(r['training_seed'],r['multiplier']) for r in s['logical']})!=6:
                raise RuntimeError('logical probe coverage changed')
            seen={s['reference_key']}
            for a in s['actors']:
                if len(a['signal'])!=10 or any(not math.isfinite(x) or abs(x)>1 for x in a['signal']):
                    raise RuntimeError('invalid actor signal')
                for i,x in enumerate(a['signal']):
                    signals.append(dict(scenario_seed=scene,period=s['period'],training_seed=a['training_seed'],
                        component=i,hub=item['config']['action']['hubs'][i//2],endpoint='near' if i%2==0 else 'far',signal=x))
            for r in s['logical']:
                if r['multiplier'] not in (1.,4.) or r['training_seed'] not in actors:
                    raise RuntimeError('unregistered intervention')
                rawsig=[r['multiplier']*x for x in actors[r['training_seed']]]
                expected=[max(-1.,min(1.,x)) for x in rawsig]
                key=identity(dict(observation=s['observation_hash'],revision=old['observation']['revision'],signal=expected))
                if r['signal']!=expected or r['probe_key']!=key or r['clipped_components']!=sum(abs(x)>1 for x in rawsig):
                    raise RuntimeError('scaled signal/context/clipping mismatch')
                cached=key in seen
                if r['cached']!=cached:
                    raise RuntimeError('cache accounting mismatch')
                calls+=not cached; seen.add(key)
                p=s['probes'][key]
                if physical_hash(p['plan'])!=p['action_hash'] or p['plan']['solver']['status']!='OPTIMAL':
                    raise RuntimeError('action identity/solver mismatch')
                changed=canonical_hash(p['plan'])!=canonical_hash(ref['plan'])
                if changed!=r['changed']:
                    raise RuntimeError('response flag mismatch')
                maxres=coefficient_checks(old,expected,p,item['config'],table)
                if any(q['time']>=24 for q in p['coefficients']):
                    raise RuntimeError('terminal gate leaked')
                probes.append(dict(scenario_seed=scene,period=s['period'],training_seed=r['training_seed'],
                    multiplier=r['multiplier'],prior_axis_responsive=axis,changed=changed,
                    raw_hash_changed=p['action_hash']!=ref['action_hash'],
                    coefficient_changed=any(abs(q['residual'])>1e-9 for q in p['coefficients']),
                    coefficient_count=len(p['coefficients']),max_abs_residual_USD=maxres,
                    clipped_components=r['clipped_components'],cached=cached,
                    current_operating_profit_USD=p['plan']['operating_profit'],
                    canonical_action_hash=canonical_hash(p['plan']),probe_key=key))
            if seen!=set(s['probes']):
                raise RuntimeError('orphan probe')
            states.append(dict(scenario_seed=scene,period=s['period'],prior_axis_responsive=axis))
    if (len(states),len(probes),len(signals))!=(48,288,1440) or len({(s['scenario_seed'],s['period']) for s in states})!=48:
        raise RuntimeError('registered state/signal/probe denominator mismatch')
    qa=read(target/'qa/numerical.json')
    if calls!=qa['actual_solver_calls'] or qa['baseline_solver_calls']!=0:
        raise RuntimeError('solver cost mismatch')
    for seed in protocol['training_seeds']:
        rs=[r for r in signals if r['training_seed']==seed]
        grouped={(r['scenario_seed'],r['period'],r['component']):r['signal'] for r in rs}
        cross=max(abs(grouped[23004,t,i]-grouped[23005,t,i]) for t in range(24) for i in range(10))
        within=max(max(grouped[s,t,i] for t in range(24))-min(grouped[s,t,i] for t in range(24))
                   for s in (23004,23005) for i in range(10))
        variation.append(dict(training_seed=seed,max_abs_signal=max(abs(r['signal']) for r in rs),
            same_period_cross_scene_Linf=cross,max_within_scene_component_range=within))
    return probes,signals,summarize(probes),variation,qa


def data(target):
    if (target/'qa/analysis.json').exists() or not (target/'qa/csv-marker.json').exists():
        raise RuntimeError('first authoring marker required; no repeated analysis')
    probes,signals,summary,variation,qa=collect(target)
    for name,rows in (('probes',probes),('signals',signals),('summary',summary)):
        with (target/'analysis'/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(target/'analysis/summary.json',dict(summary=summary,signal_variation=variation,numerical=qa))
    totals=[r for r in summary if r['scenario_seed'] is None]
    native=sum(r['changed'] for r in totals if r['subset']=='all' and r['multiplier']==1)
    amplified=sum(r['changed'] for r in totals if r['subset']=='all' and r['multiplier']==4)
    lines=['# PPO 已学方向告诉我：信号是否跨过决策边界','',
        '## 一个问题与固定边界','',
        '前批轴向控制在 48 个 W80/3 状态中改变了 24 个状态的整数动作。本批只读复用同两条 MC 轨迹，检查三个既有 PPO actor 原信号和固定 4 倍裁剪信号。全部状态保留，包括无轴向响应与日终；响应子组在本批之前冻结，不按利润选择。',
        '','三个训练 seeds 为 30001/30002/30003。原训练运行 8 updates，但验证选中的 checkpoint 均为 update 4；本批没有继续训练，也没有修改价格表、尺度、MILP、物理或奖励。',
        '','## 方向与整数动作','',
        '| Actor seed | 倍数 | 全部 48 状态改变 | 轴向响应 24 状态改变 | 其余 24 状态改变 | 非终端系数改变状态 |',
        '|---|---:|---:|---:|---:|---:|']
    for seed in (30001,30002,30003):
        for scale in (1.,4.):
            rows={r['subset']:r for r in totals if r['training_seed']==seed and r['multiplier']==scale}
            lines.append(f"| {seed} | {scale:g} | {rows['all']['changed']}/48 | {rows['axis_responsive']['changed']}/24 | {rows['axis_nonresponsive']['changed']}/24 | {rows['all']['coefficients_changed']}/48 |")
    lines += ['',f'原信号共 {native}/144 个 model-state 探针改变动作；4 倍信号为 {amplified}/144。144 不是 144 个独立实验样本，而是三个模型对两个场景内 48 个状态的重复观测。动作比较保留路线内次序，仅忽略顶层枚举排列。',
        '','## 输出是否随状态而变','',
        '| Actor seed | 原输出最大绝对分量 | 同 t 跨场景最大差 | 同场景跨 t 最大分量范围 |',
        '|---|---:|---:|---:|']
    for v in variation:
        lines.append(f"| {v['training_seed']} | {v['max_abs_signal']:.6f} | {v['same_period_cross_scene_Linf']:.6f} | {v['max_within_scene_component_range']:.6f} |")
    lines += ['', '这些差异只描述输出对观测的变化，不证明这种状态适应有利；没有固定信号对照，也没有探针政策整日闭环收益。4 倍仅为预登记方向放大/裁剪，不是另一个训练模型。',
        '', '## 本批判断及下一假设','']
    if native==0 and amplified==0:
        lines += ['在本批已确认轴向能响应的 24 个状态中，既有 actor 的原方向与 4 倍方向仍未跨过整数动作边界。系数变化不能当成控制有效；这支持检查已学方向/训练信号，而不是把前批固定时刻遗漏当作全部解释。',
            '下一批建议只检验一个假设：短训练时探索是否得到足够多的“信号改变动作并产生回报差”的样本。只读审计既有训练轨迹的探索幅度、整数动作与优势/梯度，先确认有效学习信号及饱和情况，再决定是否增加有限训练预算。不直接加大网络或开展长训练。']
    else:
        lines += ['至少部分既有 PPO 方向能跨过整数决策边界，不能归因于控制通道完全失效。但发生动作变化仍不等于改善系统收益，4 倍响应也不证明原 actor 足够有效。',
            '下一批建议只检验一个假设：改变动作的既有方向是否改善 full-cooldown realized profit。预登记少量新的独立场景，比较冻结 MC/原 actor/固定4倍 actor，并加入同尺度固定信号对照；三个模型全保留，不挑本批赢家。先做预算 pilot，再决定是否扩展或短训练。']
    lines += ['', '本批结束不自动运行下一批。', '', '## 核验、投入及解释限制','',
        f"新增求解 {qa['actual_solver_calls']} 次，基准新增求解 0 次；数值 {qa['wall_seconds']:.2f} 秒，峰值进程工作集 {qa['peak_working_set_bytes']/2**20:.1f} MiB。上限 288 次、10 分钟、2 GiB。",
        '48 个公开边界通过旧整数动作独立编译、现金与 before/after/observation hash 精确重建；两个 MC 日含完整 cooldown 独立回放。所有新探针 OPTIMAL，独立重算动作、现金和价值系数，探针不改变 live 状态。独立分析再重算全部信号/裁剪/cache/hash/系数/分母。',
        '最终相关测试 XML、真实两图视觉审阅及全部父实验/前批/本批 SHA 通过后才 COMPLETED；qa/ 保存证据。分析用 bundled 标准库 Python，绘图因 bundled 缺少 Matplotlib 使用仓库已有环境，不安装包。',
        '两个场景是已经看过轴向结果的开发机制集，不是新独立测试，不给总体显著性或利润胜出结论。当期 operating_profit 不代表探针动作的未来回报。轴向 +/-1 未穷举连续/联合信号域；本批非轴向信号可能在轴向不响应状态产生响应。表 support counts 缺失仍为 unknown。',
        '', '## 本地证据入口','',
        '- analysis/probes.csv：逐 t、逐模型、逐倍数动作/系数/裁剪/现金。',
        '- analysis/signals.csv：全部 1,440 个原 actor 信号分量。',
        '- analysis/summary.csv 与 summary.json：保留全部分母、场景与预定子组。',
        '- raw/：全部新增完整动作与系数；公共状态与 MC 日引用冻结前批。',
        '- figures/response.png、signals.png 及 SVG：动作响应与输出时序。']
    (target/'learned_direction_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    hashes={p.relative_to(target).as_posix():sha(p) for p in (target/'analysis').glob('*')}
    hashes['learned_direction_report.md']=sha(target/'learned_direction_report.md')
    write(target/'qa/analysis.json',dict(status='DATA_PASSED_FIGURES_PENDING',states=48,probes=288,
        signal_components=1440,independent_recomputation=True,output_hashes=hashes))
    print('Native changes',native,'/144; amplified',amplified,'/144; variations',variation,flush=True)


def figures(target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    qa=read(target/'qa/analysis.json')
    if qa['status']!='DATA_PASSED_FIGURES_PENDING':
        raise RuntimeError('no repeated figures')
    for rel,digest in qa['output_hashes'].items():
        if sha(target/rel)!=digest:
            raise RuntimeError('data/report changed')
    def rows(name):
        with (target/'analysis'/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f))
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':9})
    ps=rows('probes'); ss=rows('signals')
    fig,axes=plt.subplots(2,1,figsize=(8,4.8),sharex=True)
    for ax,scene in zip(axes,(23004,23005)):
        for offset,scale,color,hatch in ((-.17,1.,'#0072B2','//'),(.17,4.,'#D55E00','xx')):
            ax.bar([t+offset for t in range(24)],
                [sum(r['changed']=='True' for r in ps if int(r['scenario_seed'])==scene
                     and int(r['period'])==t and float(r['multiplier'])==scale) for t in range(24)],
                width=.32,label=f'{scale:g}x signal: actors changing action (of 3)',color=color,hatch=hatch)
        responsive=[t for t in range(24) if next(r for r in ps if int(r['scenario_seed'])==scene and int(r['period'])==t)['prior_axis_responsive']=='True']
        ax.scatter(responsive,[-.25]*len(responsive),marker='v',color='#009E73',label='Previously axis-responsive state')
        ax.set_ylim(-.5,3.5);ax.set_yticks(range(4));ax.set_ylabel('Actor count');ax.set_title(f'W80/3, scenario {scene}')
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xticks(range(24));axes[-1].set_xlabel('Decision period t (zero-based)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=1,fontsize=8)
    fig.tight_layout(rect=(0,.16,1,1))
    for ext in ('png','svg'):fig.savefig(target/'figures'/('response.'+ext),dpi=600,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(9,6),sharex=True,sharey=True)
    for row,seed in enumerate((30001,30002,30003)):
        for col,scene in enumerate((23004,23005)):
            ax=axes[row,col]
            for component in range(10):
                group=sorted([r for r in ss if int(r['training_seed'])==seed and int(r['scenario_seed'])==scene and int(r['component'])==component],key=lambda r:int(r['period']))
                ax.plot([int(r['period']) for r in group],[float(r['signal']) for r in group],
                    color=plt.cm.tab10(component//2),ls='-' if component%2==0 else '--',lw=1)
            ax.set_title(f'Actor {seed}, scenario {scene}',fontsize=9)
            ax.set_ylim(-1,1);ax.set_xticks((0,4,8,12,16,20,23));ax.spines[['top','right']].set_visible(False)
            if col==0:ax.set_ylabel('Raw signal')
    for ax in axes[-1]:ax.set_xlabel('Decision period t')
    fig.text(.5,.01,'Five hub colors; solid = near, dashed = far. Fixed [-1, 1] scale.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    for ext in ('png','svg'):fig.savefig(target/'figures'/('signals.'+ext),dpi=600,bbox_inches='tight')
    plt.close(fig)
    write(target/'figures/plot-config.json',dict(dpi=600,vector='SVG',response_source='analysis/probes.csv',
        signal_source='analysis/signals.csv',response_size=[8,4.8],signal_size=[9,6],signal_ylim=[-1,1],
        response_denominator=3,no_statistical_CI=True,black_white_distinction='hatches, markers, solid/dashed lines'))
    qa['output_hashes'].update({p.relative_to(target).as_posix():sha(p) for p in (target/'figures').glob('*')})
    qa['status']='PASSED'
    write(target/'qa/analysis.json',qa)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('data','figures'));p.add_argument('--attempt',required=True)
    args=p.parse_args();globals()[args.stage](Path(args.attempt).resolve())
