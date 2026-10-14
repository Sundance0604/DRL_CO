# AutoGluon Extreme GPU 实验 II：执行协议与停止记录

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
