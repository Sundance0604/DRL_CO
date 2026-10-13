# Direct PPO 无 MC 独立模型变体

代码局限在本包，不修改 model_families/single、原 PPO 残差模型或历史实验。复用原 H1 MILP、物理执行、现金账本、集合编码器结构、tanh 高斯采样及 PPO backend。DirectObserver 完全不查询 baseline，删除每个 hub 四个 MC 价值/钳制特征（5 hub 主干输入301→281）；DirectProvider 完全不持有 baseline，价格仅为训练经济尺度乘 Actor near/far 插值，释放 >=T 归零。Critic 保留，只用于 GAE。新版本观测/动作和 checkpoint 显式区分旧残差模型。

run prepare/engineering/smoke/train/mc/evaluate/replay/freeze 以独立目录 reports/cfs_ppo_strategic_control/direct_ppo_no_mc_v3 保存结果。mc 是单独基准进程；train/evaluate/replay 使用 no_mc_guard 阻止 MC baseline 构造/查询、表文件与旧 actor 读取。Direct 配置中不存在 baseline 字段；价格尺度 max(100,5×训练池订单收入Q75)，采用线性分位数，各hub同尺度。初始化全新随机网络，无标签监督、无模仿、无 MC 奖励。

场景只读引用原冻结准备池的新区段：train20032–20159（128）、validation21008–21023（16）、test22020–22043（24），每工况；训练seeds30001–30003。与已实际使用数据的内容/seed核验；未执行准备池不误当已训练数据。测试密封至全部九模型验证选模冻结完成，所有训练seed保留。32 updates×4完整日，epochs4,minibatch64,LR.0003,clip.2,gamma1,lambda.95,value coefficient.5,entropy0,gradient.5,KL.03,reward/1000,hidden64；验证8/16/24/32，最大均值选模并列保留较早者。

预算：正式2064完整episode（训练1152、验证576、MC验证48、测试288）；smoke6日及工程零动作回归另计。串行CPU Torch/BLAS/solver1线程，Windows低优先级。数值软上限150分钟、55000求解、8GiB新增文件、进程2GiB；首次smoke后按耗时外推报告预算，超过安全上限不降低超参数混入主结果。源首次prepare后冻结，不因测试收益改尺度/训练预算。程序故障保留原attempt，修订需另建attempt。

输出 paired_results.csv、business_metrics.csv、learning_curves.csv、summary.csv、三组SVG/600dpiPNG/source/config与中文报告；CSV4首次authoring前一次marker。分析采用bundled科学Python，若其无Matplotlib则仅图形使用仓库现有库并披露，不安装包。主要比较为训练seed×同场景配对差；交叉bootstrap20000次，固定seed42000，三主比较Bonferroni98.333%与普通95%区间，不把72差值视为独立运营日。三个训练seed的尾部CI精度有限，不宣称未显著即等效。

最终完整轨迹/价格/GAE/现金/容量/就绪/截止/full-cooldown/望远镜独立回放、选模复算、相关真实pytest XML、统计复算、全部图真实视觉、历史/CFS/输入/源/产物SHA通过才COMPLETED冻结。异常非OPTIMAL或非有限值停止，不删样本。测试和工作文件只写新目录，不提交/推送、不更改平台默认模型。

准备修订：首次 direct_ppo_no_mc_v1 因旧机制数据路径误指sealed/而INVALID，尚未冻结数值源码、创建训练数据或运行任何模型；失败目录保留。本次正确来源为旧机制data/<condition>.json，正式准备修正版输出目录 direct_ppo_no_mc_v2。修订发生于正式数值冻结前，只改历史数据读取路径，不改模型/经济尺度/统计。协议和报告必须披露这次零数值准备失败。

正式预算修订：v2工程及六日smoke真实通过，但保守耗时134分钟超过其预设120分钟门槛，状态BUDGET_LIMITED，九个正式模型尚未开始，全部输出/源码保持只读。v3在任何正式结果前将墙钟门槛固定为150分钟，数值模型、尺度、随机范围、32updates、统计规则保持完全相同。v3完整重新执行，不把两个smoke当作不同正式样本，额外工程与smoke成本另计；旧v2不续写。

