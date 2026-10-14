# 集成贡献分析：本轮未训练

本轮在 Nori 系统内存预检失败后停止，没有新的最佳模型、集成权重或基模型验证结果。
`autogluon_leaderboard.csv` 只有表头，不能当作训练成功的榜单。

上一轮冻结 Medium Basic 的最佳模型是 WeightedEnsemble_L2，CatBoost 权重 0.63636，NeuralNetFastAI 权重 0.36364。
这是历史事实，不能据此认定本轮 Extreme 的神经网络或基础模型改善了预测或实际利润。

实际 1.6.3 Extreme portfolio 包含 Nori-30M、TabICLv2、TabDPT-Turbo、两种 GBM 配置、CatBoost 和 RealMLP；没有 TabPFN。
后三个大样本配置的 `ag.min_rows=50000`，当前训练数据 10080 行；这是预设门槛，未执行训练，不能记录为实际训练中跳过。
Nori 的 `ag.max_rows=10000` 与全训练集 10080 行也需要在未来执行时核对折模型及默认 refit_folds 的行为，不能擅自删减 80 行。
