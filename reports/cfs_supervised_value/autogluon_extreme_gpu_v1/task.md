# Codex 任务：AutoGluon Extreme GPU 资源价值实验 II

## 1. 研究背景与目标

项目：`Sundance0604/DRL_CO`

上一轮完成了 W40/8 工况的 AutoGluon 监督资源价值实验。

主要结果：

| 方法 | 六日平均利润 | 相对 Pure MC |
|---|---:|---:|
| Pure MC | 40,104.87 USD | — |
| AutoGluon Basic | 43,294.52 USD | +7.95% |
| AutoGluon State | 39,597.38 USD | −1.27% |

Basic 六个开发场景全部优于 Pure MC，但样本过少，尚未得到独立确认。

上轮使用的是：

- AutoGluon 1.6.3
- `medium_quality`
- CPU 训练
- 禁用 Bagging、Stacking 和 HPO
- 最佳模型为 `WeightedEnsemble_L2`
- Basic 集成：CatBoost 63.64%，NeuralNetFastAI 36.36%

因此，上轮没有充分检验 AutoGluon Extreme 中的表格基础模型。

**本轮主要问题：**

更强的 AutoGluon Extreme 表格学习模型，能否在完全相同的 MC 标签和资源价值接口下，提供比上一轮 Medium Basic 更好的预测和调度收益？

本轮不涉及 PPO，不重新设计强化学习控制器。

---

## 2. 严格控制实验规模

只研究：

**W40/8：Western Multicluster，40订单、8车辆。**

只新增训练一个模型：

`AutoGluon Extreme Basic`

保持上轮 Basic 特征：

$$
x=(\tau,h,q)
$$

其中：

- $\tau$：资源释放时刻；
- $h$：释放枢纽；
- $q$：资源槽位序号。

监督学习目标：

$$
\widehat V_\theta(\tau,h,q)
\approx
\mathbb E[\Delta J^\pi\mid\tau,h,q].
$$

不增加状态特征，不改变 MC 标签定义，也不重新生成标签。

本轮只比较三个 Value Provider：

| 编号 | 方法 | 来源 |
|---|---|---|
| M1 | Pure MC | 上轮冻结 MC Table |
| M2 | AutoGluon Medium Basic | 上轮冻结模型 |
| M3 | AutoGluon Extreme Basic | 本轮唯一新模型 |

不要训练 Extreme State、Deep Sets、GNN、Transformer 或新的 PPO。

### 硬预算

- 新 AutoGluon fit：最多1次正式训练
- Extreme fit `time_limit`：600秒
- 新开发场景：12个
- 完整运营日：最多36次
- 主 MILP 求解：预计864次
- 全部新增 MILP 求解原则上不超过1,200次
- 不新增 MC 标签生成
- 不进行大规模 HPO
- 不自动追加训练种子
- 不重复尝试多个 Extreme 配置

如果 GPU 显存或依赖不足，应报告限制，而不是自动扩充计算预算。

---

## 3. 第一阶段：检查现有 GPU 环境

用户的电脑确实配有 GPU。

AutoGluon 已通过：

```bash
pip install autogluon
```

安装。

但不能据此假设 PyTorch 已经安装 CUDA 版本，或者 Extreme 依赖已经齐全。

首先在实际执行实验的电脑和 Python 环境中检查：

```bash
python --version
python -m pip show autogluon
python -m pip show autogluon.tabular
python -m pip show torch
```

如果是 NVIDIA GPU，再运行：

```bash
nvidia-smi
```

以及：

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory)
```

保存：

- 操作系统；
- Python / Conda 环境；
- AutoGluon 版本；
- PyTorch 版本；
- GPU 型号；
- 显存容量；
- CUDA 可用状态；
- GPU 兼容性及驱动信息。

如果不是 NVIDIA GPU，检查是否存在 AutoGluon Extreme 支持的对应加速后端，不得将其他 GPU 类型直接当作 CUDA 使用。

**只有实际确认 Extreme 所需 GPU 后端可用，才进入训练。**

### 环境隔离

优先复用现有环境的依赖版本，但不要破坏已经验证过的 Gurobi 求解环境。

如果安装额外依赖可能修改 PyTorch、NumPy 或其他关键包，应先复制 Conda 环境，或者创建独立的 Extreme 推理环境。

可以继续沿用上一轮的子进程 RPC 方式，让 Gurobi 与 AutoGluon 使用不同环境。

禁止未经确认直接重装整个 AutoGluon 或覆盖现有可复现环境。

---

## 4. 第二阶段：检查 Extreme 依赖

AutoGluon 1.6 系列的 `extreme_quality` 使用新一代表格基础模型。

官方文档：

https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html

安装说明：

https://auto.gluon.ai/stable/install.html

官方建议的额外安装方式：

```bash
python -m pip install "autogluon.tabular[tabarena]"
```

注意：

- 这是在现有 AutoGluon 基础上补充可选模型依赖，不需要重新安装整个项目。
- 在隔离环境中执行。
- 先检查当前版本，优先保持上一轮的 AutoGluon 1.6.3。
- 如果依赖解析会升级或替换核心库，先记录变更计划。
- 不要盲目安装某个固定版本的 CUDA Toolkit 或 PyTorch。
- CUDA PyTorch 应与实际显卡、驱动、系统及依赖相匹配。

安装完成后核验 Extreme 所需模型的实际可加载情况，包括 Nori、TabICL、TabDPT 等当前版本支持的基础模型。

不要求为了加入 TabPFN-3 而启用 `noncommercial` preset。

本轮明确使用官方的：

```python
presets="extreme_quality"
```

不得悄悄替换为 `best_quality` 或 `medium_quality`。

如果部分模型安装失败，应记录具体原因；如果基础模型无法实际运行，则停止本轮正式 Extreme 对照，不得把普通模型组合冒充完整 Extreme。

---

## 5. 第三阶段：复用冻结的监督数据

读取上轮：

`autogluon_mc_value_pilot_report.md`

及对应实验目录、manifest 和数据审计。

原数据：

- 16,800条非终端配对边际记录；
- 80个独立场景；
- 48个训练场景；
- 12个验证场景；
- 20个监督评价场景。

本轮必须复用：

**原来的48个训练场景和12个验证场景。**

不改变：

- MC 标签；
- continuation policy；
- 数据样本；
- 特征定义；
- 样本筛选；
- 训练/验证划分；
- 标签单位；
- 资源槽位语义。

Extreme Basic 与 Medium Basic 必须使用相同输入。

注意：Pure MC 原来使用60个构建场景，这是既有基准设计。不要为了让三者看起来完全相同而重新训练或重建任何冻结模型。

20个监督评价场景已经在上轮报告中使用过，可以复用进行描述性模型比较，但不能宣称它们是本轮全新的最终确认集。

### 数据预检

确认以下内容：

1. 没有未来需求信息进入特征。
2. 没有场景ID进入预测器。
3. 标签仍是配对资源边际回报。
4. 监督特征仅为 $(\tau,h,q)$。
5. 上轮 Medium 模型可以成功加载，并复现冻结预测。
6. 上轮 MC Table 可以成功加载。
7. 已有数据、模型及源代码哈希一致。

如果冻结资产因没有同步而缺失，不得从报告中的统计结果反向构造伪模型或伪标签。

---

## 6. 第四阶段：训练 Extreme Basic

使用 AutoGluon 当前版本官方支持的 API。

参考：

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label="target",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    path=model_path,
).fit(
    train_data=train_df,
    tuning_data=val_df,
    presets="extreme_quality",
    time_limit=600,
    num_gpus=1,
)
```

上述代码为基础配置。

实际执行前必须核验：

- 当前 1.6.3 API 的参数；
- Extreme preset 的实际模型清单；
- GPU 分配方式；
- Extreme 默认 Bagging / Stacking 行为；
- `tuning_data` 与交叉验证配置是否兼容。

如果 Extreme 开启 Bagging，应按官方接口正确设置 holdout 机制，不得让原独立验证数据泄漏进入基础模型训练。

不要为了和 Medium 完全一致而强制覆盖 Extreme 的核心模型组合。

这次比较的是完整 Extreme preset 与既有 Medium preset，并非只改变神经网络深度的单因素实验。

### 训练预算

- 正式 Extreme fit 最多600秒；
- 使用一块已经核验可用的 GPU；
- 不追加 HPO；
- 不自动重复训练；
- 不增加特征；
- 不使用新场景的利润挑选模型。

如果预训练模型需要首次下载权重，应记录下载大小、耗时和来源，与 fit 时间分开。

保存完整训练日志和模型执行情况。

### 必须检查 GPU 是否真正参与训练

不要仅因为 `torch.cuda.is_available()` 为 True 就认定 Extreme 已在 GPU 上训练。

需要结合：

- AutoGluon 训练日志；
- 各模型实际资源配置；
- GPU 使用记录；
- 显存使用；
- 成功训练的模型列表。

记录哪些模型：

- 成功训练；
- 因依赖失败跳过；
- 因显存不足失败；
- 使用 CPU；
- 实际使用 GPU。

如果 Extreme 所需关键基础模型没有运行成功，则不能对外称完整 Extreme 已验证。

---

## 7. 第五阶段：分析深度学习的具体贡献

这部分是本轮新增的重要研究内容。

上轮 Medium Basic 的最佳集成：

$$
0.6364\,\widehat V_{\mathrm{CatBoost}}
+
0.3636\,\widehat V_{\mathrm{FastAI}}.
$$

因此，本轮不仅需要比较最终 AutoGluon 集成，还应检查基础模型的贡献。

在不追加训练的前提下，导出：

- Extreme 全部基础模型；
- 各模型验证 RMSE；
- 独立监督评价 RMSE；
- 最佳 Weighted Ensemble；
- 最佳集成的模型依赖及权重；
- 神经网络和表格基础模型是否进入最佳集成；
- 各模型推理延迟。

特别比较：

1. CatBoost 等传统树模型；
2. 普通神经网络；
3. Tabular Foundation Models；
4. Extreme 最佳集成。

如果某个深度学习模型进入最佳集成，只能说明它可能提供有用的预测互补性。

**不能仅凭集成权重宣称它改善了实际运营利润。**

本轮不额外训练手工深度网络，也不对所有基模型逐个进行大规模闭环利润测试。

---

## 8. 第六阶段：监督预测比较

使用冻结的监督评价数据。

比较：

$$
V_{\mathrm{MC}},\quad
V_{\mathrm{Medium}},\quad
V_{\mathrm{Extreme}}.
$$

指标包括：

- RMSE；
- MAE；
- $R^2$；
- 资源价值排序准确率；
- 符号错误率；
- q≥2 的预测误差；
- 同 q 跨枢纽排序；
- 不同释放时刻的误差；
- 资源价值极值与非单调现象；
- 推理时间。

需要特别关注：

**上轮 Medium Basic 的 RMSE 只比 Pure MC 略低，但利润提高较多。**

因此不能预设：

$$
\mathrm{RMSE}_{\mathrm{Extreme}}
<
\mathrm{RMSE}_{\mathrm{Medium}}
$$

就意味着：

$$
P_{\mathrm{Extreme}}
>
P_{\mathrm{Medium}}.
$$

要进一步检查价格差是否改变了 MILP 的整数决策。

---

## 9. 第七阶段：冻结模型后的独立利润验证

只运行 W40/8。

预先固定12个新的完整开发场景。

可以从 `21038` 开始递增扫描，但必须先检查历史训练、验证、开发与正式测试的场景注册记录。

跳过所有已使用或已冻结的场景。

选择与筛选过程必须在读取这些场景的策略利润之前完成。

本轮三个策略：

- Pure MC；
- Medium Basic（上一轮冻结）；
- Extreme Basic（本轮新训练）。

总计：

$$
12\times3=36
$$

个完整运营日。

如果每个运营日24次主动 MILP 决策，预计：

$$
36\times24=864
$$

次主求解。

所有策略使用：

- 同一订单实现；
- 同一初始车辆；
- 同一网络与成本；
- 同一 MILP 约束；
- 同一日终规则；
- 同一完整冷却结算。

只改变 Value Provider。

### 主要终点

$$
P_T=\text{Full-Cooldown Realized Cash Profit}.
$$

主要比较：

$$
\Delta_1
=
P_{\mathrm{Extreme}}-P_{\mathrm{PureMC}}.
$$

次要比较：

$$
\Delta_2
=
P_{\mathrm{Extreme}}-P_{\mathrm{Medium}}.
$$

必须报告：

- 每个场景完整利润；
- 平均与中位配对利润差；
- 正向收益场景数量；
- 配对置信区间及其小样本限制；
- 收入、空驶和运营成本；
- 完成订单数；
- 规范化整数业务决策变化；
- 实际 q 槽位占用；
- GPU 模型推理耗时；
- 总策略计算耗时。

12个场景只能构成小规模独立探索性确认，不能仅凭点估计宣布总体优势。

不因不利结果追加场景或训练轮次。

### 同状态决策诊断

特别避免重复过去的解释问题：

闭环运行中，一旦不同策略选择了不同动作，后续状态就会分叉。

因此，不能把不同后续状态下的动作差异全部解释为价值模型造成的直接决策切换。

应单独固定少量预选公开状态，分别输入三种冻结 Value Provider，比较：

- 资源价值；
- MILP 目标系数；
- 规范化整数动作；
- 动作切换时的价格差。

该诊断必须使用相同的完整状态，不得将闭环分叉误认为同状态对照。

---

## 10. 第八阶段：关注 GPU 模型的实际部署成本

Extreme 可能包含较大的预训练模型。

虽然训练预测精度可能更高，但 MILP 在同一个运营日可能查询数千次资源价值。

因此必须检查：

- 模型加载耗时；
- GPU 显存使用；
- 批量推理耗时；
- 单次推理延迟；
- 每个运营日的推理开销；
- GPU 和 CPU 数据传输；
- RPC 开销；
- 模型磁盘大小。

优先批量预测同一决策时刻的全部资源价值。

推理 worker 应尽量保持模型常驻，而不是每次查询重新加载模型权重。

训练速度快不代表在线推理一定快。

最终需要区分：

$$
T_{\mathrm{total}}
=
T_{\mathrm{data}}
+
T_{\mathrm{fit}}
+
T_{\mathrm{inference}}
+
T_{\mathrm{MILP}}.
$$

其中历史 MC 数据构建成本单独列示，不能因为复用而视为零。

如果 Extreme 推理成本明显高于其潜在利润收益，应明确报告。

---

## 11. 必须保留的研究限制

### 11.1 不能把 Extreme 视为纯深度学习消融

Extreme 是一个模型组合 preset，可能同时改变：

- 模型族；
- 基础模型结构；
- 集成配置；
- 预训练先验；
- 训练资源分配。

所以本次只能解释为 Extreme 配置的整体效果。

如果需要识别某个基础模型的独立因果贡献，后续必须另做冻结模型或单模型对照。

### 11.2 MC 标签仍有噪声

16,800条边际标签每条只有一个共同未来实现。

它们并不是无误差的最优价值。

### 11.3 Basic 没有增加状态信息

Extreme 即使利润更高，也不能证明状态条件化价值学习成功。

它可能仅仅是更有效地估计：

$$
\mathbb E[\Delta J\mid\tau,h,q].
$$

### 11.4 MC 标签与 MILP 决策仍存在语义近似

原 MC 资源注入实验中的 $q$，与在线 MILP 的释放槽位 $q$，尚未严格证明等价。

本轮不改变该映射，但必须继续明确记录。

### 11.5 不能自动进入下一轮

无论 Extreme 是否优于 Medium，都不要自动启动 Deep Sets、更多特征或新的强化学习实验。

---

## 12. 最终交付

建立新目录：

`reports/cfs_supervised_value/autogluon_extreme_gpu_v1/`

包含：

- `experiment_protocol.md`
- `gpu_environment_audit.md`
- `dependency_manifest.json`
- `training_log.txt`
- `autogluon_leaderboard.csv`
- `foundation_model_comparison.csv`
- `ensemble_analysis.md`
- `supervised_prediction_comparison.csv`
- `policy_profit_comparison.csv`
- `same_state_decision_analysis.csv`
- `runtime_gpu_comparison.csv`
- `qa/`
- `final_manifest.json`

最终中文报告：

`autogluon_extreme_gpu_report.md`

必须回答：

1. 用户现有 AutoGluon 安装是否缺少 Extreme 依赖？
2. GPU 是否真正参与了训练？
3. 本轮 Extreme 实际成功使用了哪些基础模型？
4. 最佳模型是否仍为 Weighted Ensemble？
5. 深度学习或表格基础模型是否提供了额外预测价值？
6. Extreme 相对 Medium 是否改善了 MC 标签预测？
7. Extreme 是否在12个新场景中改善真实调度利润？
8. 利润改善是否与同状态整数动作切换有关？
9. GPU 模型在线推理是否足够经济？
10. 下一阶段是否值得研究 MC + Deep Residual、Deep Sets 或决策导向学习？

## 13. 执行顺序

严格按照以下顺序：

1. 检查实际 GPU 与 PyTorch。
2. 检查并补齐 Extreme 可选依赖。
3. 复用并核验冻结数据与旧模型。
4. 冻结12个新开发场景。
5. 训练唯一一个 Extreme Basic 模型。
6. 比较独立监督预测表现。
7. 分析树模型、神经网络及基础模型的贡献。
8. 在12个新场景中运行三种策略。
9. 做少量同状态 MILP 决策诊断。
10. 核验账本、整理全部结果。

**本轮研究原则：**

不追求 Extreme 必须获胜，不根据开发利润反复挑选模型。

我们希望回答的是：

$$
\boxed{
\text{更强的表格学习模型}
\overset{?}{\Longrightarrow}
\text{更好的资源价值与调度决策}
}
$$

如果 Extreme 确实优于 Pure MC 和 Medium，就继续研究这种改善的统计及经济机制。

如果 Extreme 仅改善 RMSE，却没有提高利润，就优先研究价值预测与 MILP 决策边界的不匹配，而不是继续增加模型复杂度。

请直接执行预算内的工程核验和数值实验，生成真实报告，保留失败日志和所有负向结果。

未经明确要求，不覆盖旧模型、不修改冻结结果、不自动推送 GitHub、不开始下一轮实验。