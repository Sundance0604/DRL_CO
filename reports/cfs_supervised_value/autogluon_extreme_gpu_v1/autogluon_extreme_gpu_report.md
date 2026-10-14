# AutoGluon Extreme GPU 资源价值实验 II：预检停止报告

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
| 冻结交付与保护资产 | 原 154 个交付文件及总计 1278 个保护文件哈希通过 |
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
