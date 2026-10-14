# 结构化 q 价格 PPO：轻量、分阶段探索

独立包与结果 structured_ppo_q_light_v1。依据 writing-block (4).md，但优先遵守用户“轻量、不跑数小时”的最新要求。

先审计原 MILP 的 release-group prefix q，不修改原物理/约束/公开观测。每工况验证 seed21008/21009，t4/12/20，共18状态。固定旧B update8/seed30001 的 alpha，五单hub near+far斜率方向与一个全hub方向，beta=-.6/-.2/0/.2/.6，不扩大搜索。beta0完整日与原10维逐系数/目标/规范化动作/现金嵌套。每工况至多首个改变案例，共同未来，下一期共同旧B8/30001。全部退化状态保留。

结构秩：实际 slot objective map，扣除原模型 equality/固定变量行空间，容差1e-8。这是仿射松弛秩，不是精确 integer-hull rank；须同时存在真正 occupied q>=2 和 canonical action switch 才晋级。

若晋级：全部三工况×三seed各8updates×4完整日，reward/10000，validation只update8、首四旧验证场景，不选其他检查点。旧10D只读使用相同B8，而非B16 best。train采样随机流与前32日完全一致。九新模型先全部冻结，才生成并打开测试22068..22075，每工况8日。三个family各3比较，20k crossed bootstrap seed71000，Bonferroni3。小样本探索，不宣称确认性等价/优势。

总数值预算30分钟/20000 solver claims/2GiB，CPU串行Torch/BLAS/solver1线程。数值/统计/协议/测试先SHA冻结，历史只读，不自动扩到16/32update。全包源码及直接历史输入实际核验，已冻结历史大目录的全体回放不重复；引用上轮QA且明确覆盖。数值与交付分别计时。

采用任务书无界affine-q公式，最大价格2Sc；已有设计的 `(1-|alpha|)*beta` 会限制高alpha处斜率，本轮不采用。两者beta0都嵌套原10D。扩展既增加q形状也扩展可达价格包络，不能把所有效果归于q形状。

入口 `.venv/Scripts/python.exe -m experiments.cfs_structured_ppo_q.run prepare|pilot|train|evaluate|all`；交付 `python -m experiments.cfs_structured_ppo_q.delivery analyze|verify|freeze`。使用五标准环境变量。COMPLETED/FAILED/BUDGET_LIMITED不重跑，缓存仅真实通过阶段。测试只相关范围，不能冒称全平台测试。
