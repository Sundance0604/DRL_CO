# 冻结 PPO 已学方向诊断

本批承接 full_period_v1 的 48 个已冻结 W80/3 公共状态（场景 23004/23005，t=0…23），不是新增独立测试集。问题：既有三个验证选定 actor 的原信号及固定 4 倍裁剪信号能否跨过整数动作边界？全部状态保留，前批轴向响应/不响应为预先确定的描述性子组，不按本批利润或动作变化筛状态。

三个 training seeds 30001/30002/30003，验证选定 checkpoint 均 update 4；训练原运行 8 updates，不将最终训练步数冒充选定 checkpoint。不训练、不改尺度、价格表、物理、decoder、终端 gate、奖励或平台默认值。没有新基准求解：逐步独立编译旧 MC 动作、提交并发布下一公共状态，核验所有 before/after/observation hash 和 full-cooldown 现金。每个状态缓存只读的旧零探针，新探针原信号/4 倍共至多 288 次求解、600 秒、2 GiB；CPU 串行、solver 一线程。10 分钟为数值阶段的软墙钟上限，每次求解的原 30 秒 solver limit 不改，单次调用可能跨过软截止。

保存所有 actor 原始输出、裁剪数、完整探针动作/系数、现金、真实调用数和缓存情况。动作身份忽略顶层列表枚举顺序但保留路线内次序；独立分析重新计算 signal、hash、表插值/系数和子组计数。同状态不同模型/倍数不是独立统计样本，只作描述性机制分析；当期 operating profit 不等于 full-cooldown 政策优势。轴向能改而 actor 不能改不证明整域不可控，actor 能改也不证明学到了有利的状态适应。

输出 3 CSV（probes/signals/summary）、JSON、两组 600 dpi PNG/SVG 及配置、中文报告。首次 CSV 前执行一次 create marker（3 CSV）。bundled Python 进行标准库独立数据重算；绘图使用仓库已有 Matplotlib（bundled 缺少该库），不安装依赖。最终相关测试 XML、实际视觉审阅与所有历史/源码/结果 SHA 通过才 COMPLETED 冻结。

## 入口

仓库 .venv/Scripts/python.exe：`-m experiments.cfs_ppo_learned_direction_v2.run prepare|run|freeze --attempt experiments/cfs_ppo_learned_direction_v2/runs/direction_v1`。环境 PYTHONHASHSEED=0、PYTHONIOENCODING=utf-8、PYTHONDONTWRITEBYTECODE=1、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。分析入口 `-m experiments.cfs_ppo_learned_direction_v2.analyse data|figures --attempt ...`。prepare 后数值/分析/测试源码不再改动；终态不可重跑。

相关测试：test_single_strategic_ppo.py、test_ppo_signal_action_diagnostic.py、test_ppo_decision_coverage.py、test_ppo_learned_direction_v2.py、test_classical_value.py、test_single_rolling.py。使用新 basetemp、关闭 pytest cache，输出真实 XML 到本批 qa，不写历史目录。

本批完成只建议一个下一假设，不自动启动训练或扩展搜索。

## 修正披露

原 cfs_ppo_learned_direction/runs/direction_v1 在第一状态、新增求解 0 次时 INVALID：Python tuple 与 JSON list 直接比较触发 public state reconstruction mismatch，规范化 observation/state hash 均完全一致。原 attempt 及全部源码只读保留。本 v2 仅改用相同 canonical JSON identity 核验序列化状态，不更改数值、actor、倍数、预算、分析或选择规则；重新完整运行，不复用任何新探针缓存。

