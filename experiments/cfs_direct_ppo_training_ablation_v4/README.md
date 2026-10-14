# Direct PPO 训练机制消融：16次更新减半版

本轮独立源码包 `experiments/cfs_direct_ppo_training_ablation_v4`，唯一活动结果目录 `reports/cfs_ppo_strategic_control/direct_ppo_training_ablation_v4`。任务依据 writing-block (3).md，加上用户2026-10-13最新授权：保持240分钟预算，规模减半。先前三个attempt全部只读，不能误重启。

## 当前实际规模与公平对照

B/C/D各3工况×3训练seed=27新模型，每模型16updates×4episode=64训练日，共1728日。验证update8/16，每点16完整日，共864日。原任务的32updates/四验证点未执行，不能写成当前规模。减半的是训练/验证，保留全部工况、seed和独立24测试场景以维护统计结构。

A只读利用原Direct v3的前16update训练前缀，按相同验证规则从已冻结inference8/16选出九模型，first tie。它是A16，不是原32次预算下选出的A32；不用旧best-inference直接冒充预算公平基准。B/C/D沿用对应A前64日的场景抽样顺序与精确Actor初始化，不重训A。不同on-policy策略自然访问不同状态。

训练池20032–20159，验证21008–21023；seeds30001–30003；新测试22044–22067。72新测试场景检查历史实际使用seed/content，不能把早已生成但未用的准备池当历史已测试。全部27新模型/九A16选模及验证侧方法胜者冻结后才能打开测试。正式864PPO＋72MC＋72M0=1008完整日，另144固定随机验证日仅诊断。全数值3744正式日，此外300旧工程日只读复用且计成本。

## 唯一干预

共享平台、公开无MC观测、五hub/十维near-far动作、训练收入Q75×5价格尺度、H1 MILP、容量/路线/承诺、24决策期与完整冷却现金全部不改。

- A16：原共享PPO，奖励/1000，GAE lambda.95，value coefficient.5。
- B：仅奖励/Critic/GAE统一改为/10000；仍batch优势标准化，USD运营目标不变。
- C：PPO-Clip with episodic returns，reward/1000，真实完整RTG减历史时间均值；当前批优化后才更新基线，没有Critic/GAE/value loss。此处MC回报不是MC Table。
- D：独立Actor/Critic集合编码器和64宽主干、独立Adam、各0.5裁剪；奖励/1000、lambda.95、value.5不变。Actor初始化精确对齐A，Critic seed+2001。是参数共享/优化/参数量的组合干预，不能单独归因。

epochs4、minibatch64、LR.0003、clip.2、gamma1、entropy0、gradient.5、KL.03全保持。Direct训练/推理有实际no_mc_guard，MC只能在独立baseline阶段运行。

## 真实诊断与统计

共同参考每工况12公开状态，来自A16/30001前四验证场景t4/12/20，在新训练前冻结，全部方案/两个检查点同状态。保存归一化动作、USD near/far、Dpi、负载/时段及log_std。敏感性并不等于利润。

每update预先固定首个admissible minibatch，在真正更新前计算共享编码器/主干Actor和0.5Critic梯度，不混入独立head；零范数cosine缺失。各batch保存裁剪前范数、实际系数、有限性、KL停止触发和有效步数。A旧训练历史分项梯度缺失，绝不用事后checkpoint冒充。D两计算图不共享参数。

Critic以USD对真实RTG和GAE lambda-return分别评价，逐update/逐t保存，C为NA。随机/确定性比较同4验证场景、同固定噪声，绝不用于选模。训练利润与验证的分布区别明确披露。

主族九项B/C/D−A16；次族12项各PPO−Pure MC单独调整。20000交叉bootstrap，seed61000，训练seed与24场景两轴重抽，普通95%及Bonferroni区间；3×24不是72独立模型/日。最优新方法只由验证均值提前选，不以测试成绩挑赢家。

少量机制：九固定公共状态×四方案×三个seed×zero/det/两随机=432probe。规范化仅交换真实同质同状态车辆，不按利润筛选。至多每工况/方案首个改变案例双分支，原始状态/隐藏未来相同，t+1统一A16/30001续接，保留无改变组，不当总体因果均值。

## 工程记录、复用与预算

v1在C smoke第二次更新被基线核验硬编码4拦停，尚无正式训练；v2修正为配置计数后12工程模型全通过，但粗粒度每日外推保守372分钟，BUDGET_LIMITED。v3按正式4episode/update和16验证池记录分项耗时，完成300工程日，仍无正式训练。所有失败/预算与原始日永久保留。

v4只读复用v3工程结果，不再浪费300日重跑；network/backend/training.py必须与v3逐字节相同，全部历史raw/权重SHA入冻结集合，实际7200calls/工程秒数计入本轮。按16updates、两验证点和完整测试规模重新保守外推。240分钟/180000calls/15GiB/进程2GiB上限不扩大；超限停止并通知，不能只完成好方案。数值时间阶段累计，不含编辑等待，最终回放QA另列。CPU串行Torch/BLAS/solver1线程，不安装库。

## 入口与交付

数值/测试用仓库.venv/Scripts/python.exe，五环境变量：PYTHONHASHSEED=0、PYTHONIOENCODING=utf-8、PYTHONDONTWRITEBYTECODE=1、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。

1. run prepare：旧实验/CFS/失败/工程SHA核验、新数据隔离、A16/共同参考和数值/分析/测试源码冻结。
2. run smoke：工程只读复用与半规模预算门，不再次求解。预算通过才run all。
3. run all：train→evaluate→baselines→mechanism，已完成模型只读复用，partial模型不静默覆盖；所有数值结束仅NUMERICAL_COMPLETE_QA_PENDING。
4. delivery analysis：bundled Python纯NumPy统计，仓库现有Matplotlib仅绘图，不装依赖。八科学CSV、五组SVG/600dpiPNG/config及十问中文报告；不生成PDF。首次CSV前只执行一次本轮CSV8marker，旧轮marker不算本轮。analysis.py随prepare已冻结，不能改统计/选模。
5. delivery replay：3744日加300工程日独立现金/动作/价格/GAE/C基线/冷却回放，Actor及噪声复算、432probe/反事实、选模、历史SHA和完整numeric索引、实际MILP计数。
6. 真实最终pytest输出qa/pytest-final.xml，范围为test_direct_ppo_training_ablation_v4.py、test_direct_ppo_training_ablation_v4_delivery.py、test_direct_ppo_v3.py、test_direct_ppo.py、test_single_strategic_ppo.py、test_classical_value.py、test_single_rolling.py；delivery tests读实际XML。预检不能替代最终，不冒称全平台矩阵。
7. 全部五张最新PNG实际逐张显示查看，delivery visual --reviewed；SVG同源并检查XML，不冒称独立SVG渲染。布局变化必须重审。
8. 中文报告实际核对十问，补非数值文字/成本/条件性解释，不再次analysis覆盖人工段落。所有gate成功才delivery freeze一次，COMPLETED后本目录也永远只读。

外部非数值delivery/figures可修审计/布局错误，不能修改数值/冻结analysis、放宽容差、删异常样本或追结果。正式数值错误保留证据并通知。不提交/推送/PR、不改平台默认模型、不启动q方案、别的实验或另一聊天。
