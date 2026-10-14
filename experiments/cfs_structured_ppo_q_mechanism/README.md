# PPO20 KL 与 q-active：只诊断，不重新训练

独立输出 structured_ppo_q_mechanism_v1。已完成 PPO20 light_v2、失败 light_v1 和 PPO10 ablation_v6 永久只读。不改平台/MILP/现金/概率/阈值/学习率/方差，不提交/推送。

KL：用18个既有模型的前8轮保存rollout，在内存中重建历史优化轨迹，不生成新rollout。仅在原backend实际检查处只读记录参数/解析高斯KL/原近似KL；最终逐tensor和日志/随机流一致才能称成功重建。保留接受与拒绝检查，不能拿update平均值冒充触发值。alpha/beta精确KL可加，分量近似KL不一定可加。使用配对alpha标准差计算beta均值项敏感性的离线算术，不改变实际策略。

W40：候选4个验证seed×5时刻=20，原PPO20/30001轨迹。对原MILP副本仅改可行性诊断目标为sum q2，不加/删约束，最优值0才证明当前macro-action接口没有q2可行方案。保留完整合法证人动作及约束矩阵摘要。先冻结前8个q-active与前2个q-only状态；不足则报告，不根据扰动利润补样本。各3冻结seed，A/B/D及8固定斜率，最多330probe。CF按固定顺序首2不同状态，后续共同PPO10/B8/30001，最多80续接调用。总上界450，小于500。

预算数值/离线重建累计600秒、500新MILP、512MiB输出；CPU串行Torch/solver/BLAS1线程。预检不算最终QA。阶段先prepare、kl、select、probe，数值完成仍待交付。delivery分析/核验/真实相关测试XML/三图真实视觉审阅之后freeze一次。源/阈值/状态规则不事后改，COMPLETED/FAILED/BUDGET_LIMITED不重复数值。

数值入口 `.venv/Scripts/python.exe -m experiments.cfs_structured_ppo_q_mechanism.run prepare|kl|select|probe|all`。交付入口 delivery analyze|verify|freeze。全部结果和条件性未执行项明确保存，不新增正式测试，不启动下一轮训练。CSV使用bundledPython，现有Matplotlib仅绘图，SVG/600dpiPNG，无PDF。
