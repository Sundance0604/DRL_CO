# Direct PPO 训练机制消融与独立验证

独立源码包和独立结果目录 `reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v2`。不修改历史源码、权重、CFS、默认模型、MILP、10维near/far动作或训练价格Sc。A只读复用原v3九模型；新B奖励/10000、C完整episode return+历史时间基线且不训练Critic、D独立Actor/Critic编码器、Adam和0.5裁剪。D是共享表征/优化的组合干预，并非单因素因果分离。

训练20032–20159，验证21008–21023，三个训练seed30001–30003；27新模型32updates×4完整日，共3456训练日，验证8/16/24/32共1728日。同seed完全保留原A的有放回场景抽样序列；Actor初始化逐tensor一致，C/D在原完整初始化后移除value head，D Critic另用seed+2001。采样/minibatch/data generator与A保持同规则；on-policy轨迹不要求相同。当前批不能进入其自己的时间基线，C baseline在优化后更新。GAE仍lambda.95，clip.2，LR.0003，epochs4，batch64，entropy0，gamma1，KL.03。

新测试22044–22067，检查实际旧运行seed及场景内容，而不把此前已生成未使用准备池误算成旧测试。训练期间不解析测试；全部27选模和验证侧方法胜者冻结后才打开。正式864 PPO日＋72MC＋72M0=1008日，另144固定随机验证日，仅诊断不选模。配对同场景/对应训练seed，20000交叉bootstrap，seed61000，九项B/C/D−A主family Bonferroni区间，12项各PPO−MC次family另调。3×24不是72独立运营日，不按正式成绩选赢家。

12共同公开参考状态/工况，在训练前来自A30001前四验证场景t4/12/20，所有方案与检查点使用同状态。详细共享梯度只在每update首个admissible minibatch实际优化前测量，严格只含集合编码器/主干，不含Actor/Critic独立头，零范数cosine缺失而非伪造0。每batch保存裁剪前范数/系数、KL触发值、有效步数。A旧历史分项梯度不可恢复，标记missing，不重训A或用事后梯度冒充。

Critic逐update pre-update预测以USD与真实完整RTG、GAE lambda-return分别比较；分t误差完整保存。C指标为NA。额外成本D参数与训练/推理实测单列，梯度范数不直接解释为Adam有效更新贡献。

少量机制：九共同状态（每工况三个固定t/验证scene），4方案×3seed×每状态zero/det/两随机动作共432probe，不按利润筛选。每工况/方案至多首个动作改变案例（最多12例）双分支，原始当前状态/隐藏未来相同、t+1统一A30001继续，案例不是独立总体因果估计。保留无改变组。

先12模型smoke（每工况A/B/C/D，2update×1日、两次1日验证和一次重载回归，共60日），不进入正式样本。用实际成本保守1.5倍外推完整规模；超过180000calls/240数值分钟/15GiB即停止且不缩减方案/样本。数值时间逐阶段累计，编辑/审图等待不伪装数值耗时，最终回放QA额外披露。单进程CPU，Torch/BLAS/solver1thread，进程2GiB；不装新库、不为追收益改阈值。

## 操作与不可变门

使用仓库`.venv/Scripts/python.exe`，环境PYTHONHASHSEED=0、PYTHONIOENCODING=utf-8、PYTHONDONTWRITEBYTECODE=1、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。入口：

1. `-m experiments.cfs_direct_ppo_training_ablation_v2.run prepare`：完整历史/输入核验和登记，数值源码与数值测试首次冻结。
2. `...run smoke`：工程/数学/计时/预算门；不得重跑通过阶段。
3. `...run all`：未完成train/evaluate/baselines/mechanism顺序运行，已有完整模型只读复用；partial模型不自动覆盖。数值结束仅NUMERICAL_COMPLETE_QA_PENDING。
4. `...delivery analysis`：首次冻结分析源码，六主CSV与五图，SVG/600dpiPNG/config，不生成PDF。每期原始世界仍完整保存。CSV首次创建前单次marker，旧轮marker不复用。
5. `...delivery replay`：所有日、优势、基线、现金、状态/价格/动作/冷却、确定性Actor、checkpoint选择、数值文件索引与历史SHA实际重算。
6. 最终相关pytest XML必须真实运行：新测试、Direct v3、Direct、PPO、经典价值、rolling。`...delivery tests`读取XML，预检不替代最终。
7. 全部五张最新PNG实际查看，`...delivery visual --reviewed`才能记录视觉通过；SVG做XML检查，同源不冒称独立SVG渲染。
8. 报告逐项核对十问，非数值文字可补；不重跑分析覆盖人工解读。全部QA成功后`...delivery freeze`只执行一次，COMPLETED后目录也永久只读。

失败保留partial和stop.json，不静默改数学/混attempt；COMPLETED/FAILED/BUDGET_LIMITED不重跑。不提交/推送。工程预检输出只写workspace，不污染历史冻结目录。


## 工程修订披露

v1 smoke在C第二次更新被独立基线核验拦停：检查错误地写死4×此前更新数，而smoke每次仅1episode。算法实际基线按已完成历史计算，没有进入当前批；正式27模型尚未开始。v1源码/全部partial/stop保持只读，v2只将核验计数改为配置episodes_per_update×此前更新数，新增边界回归；完整重新smoke，不复用v1数值结果，不改变模型、种子、预算、训练和统计规则。
