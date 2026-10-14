"""Audit frozen inputs and close a stopped attempt without fabricating outcomes."""
import csv
import datetime
import json
from pathlib import Path
import sys

from .preflight import ROOT, OUT, PARENT, save, sha


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def table(name, fields, rows=()):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def close():
    gpu=read(OUT/'environment/gpu_probe.json')
    assert gpu['status']=='STOPPED_PREFLIGHT' and gpu['error']['type']=='NotEnoughMemoryError'
    old=read(PARENT/'final_manifest.json')
    assert old['status']=='COMPLETED'
    protected={str(PARENT/k):v for k,v in old['artifact_hashes'].items()}
    protected[str(PARENT/'final_manifest.json')]=sha(PARENT/'final_manifest.json')
    for name in ['readonly_inputs','model_source_hashes','frozen_model_hashes','experiment_source_hashes']:
        protected.update(read(PARENT/f'qa/{name}.json'))
    checked={}
    for p,digest in protected.items():
        current=sha(p)
        if current!=digest: raise RuntimeError('Frozen asset changed: '+p)
        checked[p]=current
    save('qa/frozen_assets_final.json',dict(status='PASSED',checked_files=len(checked),sha256=checked))
    # Load the actual MC table in the original solver environment without solving.
    from model_families.single.value.providers import MCTableValueProvider
    config=read(PARENT/'data/runtime_config.json')
    provider=MCTableValueProvider(read(config['baseline']))
    import math
    val=provider.marginal_release_value(12,'04-429',2,dict(T=24))
    assert math.isfinite(val)
    save('qa/mc_table_load.json',dict(status='PASSED',path=config['baseline'],sha256=sha(config['baseline']),
         probe=dict(time=12,hub='04-429',q=2,value=val),new_solves=0))
    source={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    save('qa/new_source_hashes.json',source)
    resources=read(OUT/'environment/resource_estimates.json')
    weights=read(OUT/'environment/pretrained_weights.json')
    medium=read(OUT/'qa/medium_reproduction.json')
    planned=read(OUT/'dependency_manifest.json')
    planned.update(status='DEPENDENCIES_INSTALLED_FORMAL_EXPERIMENT_STOPPED_MEMORY',
        original_assets_final_audit_files=len(checked), actual_isolated_versions=gpu['versions'],
        actual_changes=['local torch 2.13.0+cu130','local torchvision 0.28.0+cu130','local numpy 2.2.6',
          'local numba 0.68.0 and llvmlite 0.50.0','tabarena optional packages listed in extreme_dependency_install.json'],
        numpy_change_reason='synthefy-nori 0.14.0 requires numpy>=2.0; original environment remains numpy1.26.4',
        dependency_health='pip check fails for inherited unrelated statsforecast/wechatauto packages; not a globally clean environment',
        formal_predictor_fits=0,mechanical_adapter_fit_attempts=1,mechanical_adapter_fit_successes=0,
        new_milp_solves=0,new_mc_labels=0)
    save('dependency_manifest.json',planned)
    protocol=dict(status='STOPPED_PREFLIGHT',condition='W40/8',features=['release_time','hub','q'],
        parent=str(PARENT),formal_fit_limit=1,formal_fits_used=0,time_limit=600,
        num_gpus=1,presets='extreme_quality',use_bag_holdout=True,
        groups='Eight predefined fold groups, six complete training scenarios each; column removed from features',
        original_train_scenarios=48,train_rows=10080,original_validation_scenarios=12,validation_rows=2520,
        reused_evaluation_scenarios=20,evaluation_rows=4200,planned_development_days=12,
        frozen_new_development_days=0,completed_policy_days=0,new_milp_solves=0,max_new_milp_solves=1200,
        preset=resources['extreme_preset'],portfolio=resources['actual_portfolio'],
        stop_reason='Nori adapter raises NotEnoughMemoryError in mechanical preflight; no memory guard override',
        github_push=False,automatic_next_experiment=False)
    save('protocol.json',protocol)
    logs=(OUT/'environment/gpu_preflight.log').read_text(encoding='utf-8-sig',errors='replace')
    (OUT/'training_log.txt').write_text('STATUS: STOPPED_PREFLIGHT\nFormal TabularPredictor.fit calls: 0\n'
        'This is a mechanical GPU preflight log, not an Extreme training log.\n'
        'No Extreme predictor, validation score, ensemble, or policy result exists.\n\n'+logs,encoding='utf-8')
    # Header-only files explicitly represent zero observations, never zero-valued results.
    empty={
        'autogluon_leaderboard.csv':['model','score_val','score_test','fit_time','pred_time_test','status'],
        'policy_profit_comparison.csv':['scenario_seed','method','profit_USD','completed_fare','realized_cost','delivered'],
        'same_state_decision_analysis.csv':['scenario_seed','period','method','before_state_hash','business_hash','changed'],
    }
    for name,fields in empty.items():table(name,fields)
    families={'NORI':'foundation','TABICL':'foundation','TABDPT-TURBO':'foundation',
              'GBM':'tree','CAT':'tree','REALMLP':'ordinary_neural_network'}
    rows=[]
    for kind,configs in resources['actual_portfolio'].items():
        for i,c in enumerate(configs):
            rows.append(dict(model_key=kind,configuration=i,family=families[kind],
                status='PREFLIGHT_MEMORY_BLOCKED' if kind=='NORI' else 'NOT_ATTEMPTED',
                min_rows=c.get('ag.min_rows',''),max_rows=c.get('ag.max_rows',''),
                validation_RMSE='',evaluation_RMSE='',inference_seconds='',actual_model_gpu_computation=False))
    table('foundation_model_comparison.csv',list(rows[0]),rows)
    historical=[]
    analysis=read(PARENT/'analysis.json')
    for r in analysis['prediction']:
        if r['method'] not in ['PureMC','Basic']:continue
        historical.append(dict(method='Medium' if r['method']=='Basic' else 'PureMC',
            scope=r['scope'],status='FROZEN_PREVIOUS_EXPERIMENT_ONLY',n=r['n'],
            RMSE=r['RMSE'],MAE=r['MAE'],R2=r['R2']))
    historical.append(dict(method='Extreme',scope='all nonterminal',status='NOT_TRAINED',n=0,RMSE='',MAE='',R2=''))
    table('supervised_prediction_comparison.csv',list(historical[0]),historical)
    cost=[dict(phase='frozen_medium_reproduction',model='Medium',rows=4200,
               wall_seconds=medium['seconds'],bytes='',source='old CPU environment',status='COMPLETED')]
    source_urls={'Nori-30M':'https://huggingface.co/Synthefy/Nori-30M',
                 'TabICLv2':'https://huggingface.co/jingang/TabICL',
                 'TabDPT-Turbo':'https://huggingface.co/Layer6/TabDPT'}
    for w in weights:
        w['source']=source_urls[w['model']]
        cost.append(dict(phase='initial_pretrained_download',model=w['model'],rows='',
          wall_seconds=w['download_or_cache_wall_seconds'],bytes=w['bytes'],source=w['source'],status='COMPLETED'))
    save('environment/pretrained_weights.json',weights)
    for phase in ['formal_extreme_fit','extreme_online_inference','policy_milp']:
        cost.append(dict(phase=phase,model='Extreme',rows=0,wall_seconds='',bytes='',source='',status='NOT_EXECUTED'))
    table('runtime_gpu_comparison.csv',list(cost[0]),cost)
    (OUT/'ensemble_analysis.md').write_text('''# 集成贡献分析：本轮未训练

本轮在 Nori 系统内存预检失败后停止，没有新的最佳模型、集成权重或基模型验证结果。
`autogluon_leaderboard.csv` 只有表头，不能当作训练成功的榜单。

上一轮冻结 Medium Basic 的最佳模型是 WeightedEnsemble_L2，CatBoost 权重 0.63636，NeuralNetFastAI 权重 0.36364。
这是历史事实，不能据此认定本轮 Extreme 的神经网络或基础模型改善了预测或实际利润。

实际 1.6.3 Extreme portfolio 包含 Nori-30M、TabICLv2、TabDPT-Turbo、两种 GBM 配置、CatBoost 和 RealMLP；没有 TabPFN。
后三个大样本配置的 `ag.min_rows=50000`，当前训练数据 10080 行；这是预设门槛，未执行训练，不能记录为实际训练中跳过。
Nori 的 `ag.max_rows=10000` 与全训练集 10080 行也需要在未来执行时核对折模型及默认 refit_folds 的行为，不能擅自删减 80 行。
''',encoding='utf-8')
    (OUT/'experiment_protocol.md').write_text('''# AutoGluon Extreme GPU 实验 II：执行协议与停止记录

本轮按 writing-block (8).md 执行，仅比较 W40/8 的 Pure MC、冻结 Medium Basic 和唯一新 Extreme Basic。
不新增 MC 标签、不增加状态特征、不启动 PPO、不推送 GitHub。全部新文件位于独立目录。

计划复用原 48 个训练场景（10080 行）、12 个独立验证场景（2520 行），仅以 release_time、hub、q 为预测输入。
20 个监督评价场景（4200 行）属于已使用过的描述性评价集。
计划正式 fit 最多一次，extreme_quality、600 秒、num_gpus=1，保留 preset 的 8 折、0 层 stacking、无 dynamic stacking、原商业模型组合。
采用 use_bag_holdout=True；将 48 个训练场景预定义为 8 个折组，每组 6 个完整场景。
groups 列只控制 LeaveOneGroupOut，不作为预测特征；独立验证数据不带 groups 列，不进入基础训练。
该方案尚未执行，不宣称已完成折隔离审计。

在模型就绪后，原计划从 21038 开始递增排除历史已用及已冻结场景，提前冻结 12 个新开发场景，运行 36 个完整冷却运营日。
主求解预算 864 次，含同状态诊断在内总计不超过 1200 次；不根据利润追加场景。
同状态诊断预选每个 Pure MC 运营日的 3、12、20 时刻，恢复完整相同状态；闭环分叉只记录为轨迹差异。
主要终点为完整冷却现金利润，主比较 Extreme−MC、次比较 Extreme−Medium；以运营日为单位报告配对差和探索性置信区间。

实际执行到 GPU/基础模型预检。Nori 32 行合成机械探针被默认系统内存检查拒绝，NotEnoughMemoryError。
这是适配器层 model.fit，未调用 TabularPredictor.fit，不使用冻结 MC 标签，不构成正式训练或模型选择。
没有提高内存保护阈值、降低训练行数、替换 preset 或退化为树模型对照。
正式训练 0 次、新开发场景冻结 0 个、完整运营日 0 次、新 MILP 求解 0 次、新 MC 标签 0 条。

预检初次存在适配器导入路径错误，已修正并保留 initial 日志；修正后实际失败原因为系统内存保护。
依赖解析先因锁定原 NumPy 1.26 无法满足 Nori 的 NumPy≥2 回溯，停止解析后在隔离环境记录变更计划并安装 NumPy 2.2.6；没有重复正式训练。
''',encoding='utf-8')
    estimated=resources['nori_estimates']['10080']
    gib=lambda n:n/1024**3
    environment=f'''# GPU 与环境核验

实际 GPU：RTX 5060 Laptop GPU，驱动 592.47，8151 MiB 显存，compute capability 12.0。
操作系统记录：{gpu['platform']}；Python {gpu['python']}。
原 AutoGluon 环境为 pavane：AutoGluon 1.6.3、Torch 2.13.0+cpu、NumPy 1.26.4，没有 Nori、TabICL、TabDPT。
新环境为 workspace/envs/autogluon_extreme_gpu，采用 system-site-packages 只读继承原包，本地覆盖并安装额外依赖。
新版本见 dependency_manifest.json：Torch 2.13.0+cu130，CUDA runtime 13.0，NumPy 2.2.6，Nori 0.14.0、TabICL 2.1.1、TabDPT 1.2.0。
CUDA 单卡可用，实际 16×16 矩阵运算与同步检查通过；此证据只能证明 CUDA 内核可执行。
GPU 基础模型 fit/predict 没有成功，因此不能声称 GPU 参与了 Extreme 训练或基础模型推理。

机械探针调用 Nori 适配器时，默认系统内存估计为 2.794 GiB，实际可用为 1.966 GiB，超出 90% 保护阈值并触发 NotEnoughMemoryError。
不是 CUDA OOM。系统物理内存约 {gib(resources['before_import_system_memory']['total']):.2f} GiB；可用内存会随其他应用与导入发生变化。
另一次只读静态审计，在导入前可用 {gib(resources['before_import_system_memory']['available']):.2f} GiB、导入后 {gib(resources['after_import_system_memory']['available']):.2f} GiB。
这不是重新成功运行的证据，也不能替代失败时记录。
官方适配器对 10080×3 的估计：系统内存 {gib(estimated['system_memory_bytes']):.3f} GiB，显存 {gib(estimated['gpu_memory_bytes']):.3f} GiB；均为静态估计，非实测峰值。
仅按 90% 系统内存保护计算，模型检查点至少需要约 {gib(estimated['required_available_system_memory_at_90_percent']):.3f} GiB 可用系统内存；这不包含预留给其他并存进程的裕量，不保证运行成功。
当前失败不能推导为这张 8 GiB 显卡永远无法运行 Extreme。

pip check 未全通过：继承的 statsforecast 对 numba/llvmlite 要求与 Nori 不兼容，另有原环境 wechatauto-replica 缺 opencv。
本轮不使用这些包；不把新环境称为全生态兼容环境，也不为修复无关依赖改动原环境。
原 CPU 环境中的 Medium 全部 4200 条预测复现最大差异 0，Gurobi 所在项目 .venv 未改动。
所有冻结输入与源码再次核对 SHA-256：{len(checked)} 个文件通过，详见 qa/frozen_assets_final.json。

首次下载三个基础权重共 {sum(w['bytes'] for w in weights):,} 字节，加载前下载与文件核验耗时合计 {sum(w['download_or_cache_wall_seconds'] for w in weights):.2f} 秒，详见 environment/pretrained_weights.json。
Torch CUDA wheel 的 1915.5 MB 下载属于环境准备，非模型权重、非正式 fit；缓存安装 41.22 秒。最初 dry-run 已下载 wheel，未保存精确下载计时，不虚构该时长。
'''
    (OUT/'gpu_environment_audit.md').write_text(environment,encoding='utf-8')
    report=f'''# AutoGluon Extreme GPU 资源价值实验 II：预检停止报告

## 结论

本轮没有完成 Extreme 正式数值对照。环境安装、CUDA 实际运算、旧模型复现和冻结资产核验已完成；Nori 的机械预检因当前系统内存不足被 AutoGluon 拒绝，按任务书停止，未绕过保护。
没有 Extreme 模型或新利润结果，不能回答 Extreme 是否比 Medium 或 Pure MC 更优。

## 已执行与未执行

| 项目 | 实际结果 |
|---|---|
| 原安装是否缺依赖 | 是，Torch 为 CPU 版，Nori/TabICL/TabDPT 未安装 |
| 隔离补齐依赖 | 完成，保留 AutoGluon 1.6.3，原环境不变 |
| CUDA 后端 | 实际矩阵运算通过，RTX 5060 单卡 |
| 旧 Medium 模型 | 4200 条预测逐条复现，最大差异 0 |
| 冻结交付与保护资产 | 原 154 个交付文件及总计 {len(checked)} 个保护文件哈希通过 |
| 基础模型权重 | Nori-30M、TabICLv2、TabDPT-Turbo 官方权重下载完成 |
| 新正式 AutoGluon fit | 0 次，600 秒预算未用 |
| 实际成功的新基础模型 | 0 个；Nori 内存预检失败，其余未进行计算探针 |
| 新开发场景与完整运营日 | 未冻结新场景，运营日 0 次 |
| 新 MILP / 新 MC 标签 | 0 / 0 |

## 停止证据

Nori 在 32 行合成机械探针中，官方适配器估计需要约 2.794 GiB 系统内存，进程运行当时可用约 1.966 GiB；需求占可用内存 142.1%，超过默认 90% 保护阈值，抛出 NotEnoughMemoryError。
原始日志在 training_log.txt 和 environment/gpu_preflight.log。错误发生在实际模型计算前，不能把通过 CUDA 内核测试写成“GPU 已训练成功”。

机器约有 15.73 GiB 系统物理内存、8 GiB 显存。失败来自当前进程与应用并存时的系统内存余量；尚未实测全数据 Nori 的显存峰值。
预检曾预先导入所有可选模型包，增加了运行时内存占用；该失败说明当前预检条件不可运行，不证明逐模型加载或释放其他应用内存之后仍必然失败。
未来可先释放应用占用或减少预检同时导入，再重新验证资源条件；本次不把静态余量变化当成成功，不绕过保护，不自动追加配置试验。

## 任务书十个问题的回答

1. 现有安装确实缺 Extreme 依赖；已在独立环境补齐。
2. GPU 只实际执行了 CUDA 机械运算，没有成功参与新基础模型训练/推理。
3. 本轮新基础模型成功数为 0；三种官方权重已具备，但 Nori 被系统内存检查阻止。
4. 没有本轮最佳模型。WeightedEnsemble_L2 是上一轮 Medium 的冻结结果。
5. 未测得本轮深度学习或表格基础模型的额外预测价值。
6. 未训练 Extreme，无法判断相对 Medium 的 MC 标签预测变化。
7. 未运行 12 个新场景，无法判断真实调度利润变化。
8. 未运行同状态 MILP 诊断，无法解释直接整数动作切换。
9. 没有 Extreme 在线推理延迟和每日成本，不能判断部署经济性。
10. 本轮证据不足以决定 Deep Residual、Deep Sets 或决策导向学习；未启动下一阶段。

## 文件与解读

foundation_model_comparison.csv 记录实际预检状态和官方配置门槛，未训练模型的 RMSE/延迟为空。
autogluon_leaderboard.csv、policy_profit_comparison.csv、same_state_decision_analysis.csv 只有表头，表示 0 个真实结果，绝非“利润为零”。
supervised_prediction_comparison.csv 的 Pure MC、Medium 指标来自上一轮冻结报告，并明确标注历史来源；Extreme 行为 NOT_TRAINED，不参与比较。
runtime_gpu_comparison.csv 仅包含实际完成的旧模型复现和首次权重下载；未执行阶段耗时为空。
模型权重、新环境和旧资产留在本地，没有 GitHub 提交或推送。

## 保留的研究限制

后续即便能训练，Extreme 仍是整个 preset 的比较，不能当作纯深度学习消融；进入集成不等于改善现金利润。
每条 MC 标签只有一个配对未来，Basic 不增加状态信息；MC 注入 q 与在线 MILP 槽位 q 的等价性仍未严格证明。
20 个监督评价场景已被使用，12 个新运营日只能提供探索性确认。历史标签构建成本约 1980.05 秒，并未因本轮复用而消失。

## 官方依据

[AutoGluon fit 文档](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html)和[安装说明](https://auto.gluon.ai/stable/install.html)用于核验 API 与依赖。
实际模型组合以本机 1.6.3 源码为准，完整配置保存在 environment/resource_estimates.json；不能用网站 stable 版本号替代实际安装版本。
[Nori 0.14.0 发布元数据](https://pypi.org/pypi/synthefy-nori/0.14.0/json)要求 NumPy≥2.0。
[PyTorch 官方 CUDA wheel 源](https://download.pytorch.org/whl/cu130/)提供保留 2.13.0 版本的 CUDA 构建，兼容性最终由实际 CUDA 内核验证。
权重来源为 [Nori-30M](https://huggingface.co/Synthefy/Nori-30M)、[TabICL](https://huggingface.co/jingang/TabICL)、[TabDPT](https://huggingface.co/Layer6/TabDPT)，本地文件 SHA-256 和 snapshot 版本均已记录。
'''
    (OUT/'autogluon_extreme_gpu_report.md').write_text(report,encoding='utf-8')
    (OUT/'.gitignore').write_text('environment/\nqa/\nmodels/\ndata/\nraw/\n*.csv\n*.json\n*.txt\n*.log\n',encoding='utf-8')
    save('qa/stop_summary.json',dict(status='STOPPED_PREFLIGHT',trigger='Nori NotEnoughMemoryError',
        formal_fits=0,new_milp_solves=0,new_mc_labels=0,full_days=0,model_compute_successes=0,
        cuda_kernel_passed=True,medium_reproduction_passed=True,frozen_assets_passed=True,
        unexecuted_result_files=list(empty),no_profit_or_supervision_claim=True))
    artifacts={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final_manifest.json'}
    save('final_manifest.json',dict(status='STOPPED_PREFLIGHT',formal_experiment_completed=False,
        stop_condition_handled=True,formal_fits=0,new_milp_solves=0,full_days=0,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifact_hashes=artifacts,
        source_code_hashes=source,parent_manifest_sha256=sha(PARENT/'final_manifest.json'),
        frozen_assets_checked=len(checked),parent_artifacts=154,github_push=False))
    print('STOPPED_PREFLIGHT_AUDITED',len(checked),'protected files;',len(artifacts),'new artifacts',flush=True)


if __name__=='__main__': close()
