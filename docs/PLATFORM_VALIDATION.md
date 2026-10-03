# 实验平台实施与验证报告

验证日期：2026-10-03。实施分支：`react-experiment-platform`，基于 `single-level-dispatch-model`。未修改远程 `main`。本报告区分实测、实现边界与未验证事项，不把实施方案当成既有功能证明。

## 已交付内容

- React + TypeScript 本地页面：总览、数据管理、实验编辑、队列、运行详情、比较、回放、BHH 分析、复现和诊断。
- 公共实验核心与 FastAPI/CLI，四维注册表、严格 JSON/Pydantic 校验、生成的 OpenAPI/JSON Schema、冻结版本数据与 train/test 分离。
- 单层匹配 myopic、流体价值、真实 rollout、新特征监督训练与评估、完全信息松弛上界。
- 独立 SAC 模型族，真实 actor/critic/replay 更新及 held-out checkpoint 评估。
- BHH 稳态、有限时域共享车队 MILP、真实滚动承诺、小规模空间校准、扫描/取整/分解/非对称/异质示例。
- 单协调器持久化队列、独立进程执行、取消/超时、事件恢复、事实索引重建、提交时源代码快照、复现 ZIP 与脱敏诊断。
- 项目独立环境、锁文件、跨平台薄脚本与本地受控启动器。

## 原代码的必要调整

| 文件 | 修改与原因 | 回归依据 |
|---|---|---|
| `model/mt_prototype.py` | 注入实际物理/求解参数；纯 solve 与 commit 分离；状态 hash/revision 拒绝陈旧/重复提交；支持 string hub ID | 零价值基线收益与指派量等价，重复 commit 被拒绝 |
| `model/online_lookahead.py` | 参数与求解信息透传，状态克隆保留 revision | rollout 真实完成，克隆不污染真实状态 |
| `model/bounds_check.py` | 参数透传，区分状态/incumbent/方向与边界范围 | oracle 上界输出，不能记作运营收益 |
| `experiments/training/train_fixed_id.py` | 下层优化限时/gap/线程参数及无 incumbent 错误协议 | 原 18 项回归与新 SAC worker |
| `drl_co/simulation/transitions.py` | 在显式允许时应用限时求解的可行 incumbent；默认仍只接收最优解 | 原 18 项回归与新 SAC worker |

本次没有把 SAC 动作头套到新的枢纽价值函数，也没有悄悄把原指派收益改成送达收益。新增模型和口径均显式分族。

## 实际验证

Windows x86_64；原有基线环境通过 18 项测试。新平台隔离环境为 Python 3.11.14、Node 24.18.0、CPU、Gurobi 10.0.3（实际优化验证许可证）。

```powershell
.venv\Scripts\python.exe -m pytest -q
# frontend/ 内
npm run build
```

新测试覆盖严格非有限 JSON/未知字段、不可变 hash 与导入、参数展开、纯单步与基线等价、工作区锁/幂等/取消、复现路径穿越、API token/Origin 防护与 SSE 断点、训练评估防泄漏、BHH 公式/分解恒等式/共享车队/滚动承诺、索引重建、精确代码版本拒绝、运行超时、SAC 训练后独立评估、指标改善方向，以及诊断脱敏保持有效 JSON。全量回归结果：33 项通过，25.25 秒；静态检查通过。

已实际完成的 tiny smoke：myopic、fluid、rollout、value training、旧 supply、SAC training、oracle LP、BHH steady/finite/rolling/spatial/asymmetric/heterogeneous，以及 τ×Λ 的 9 个真实运行。SAC 训练记录包含非零 actor/critic loss、entropy 和 replay 样本；不是仅保存一个随机权重文件。网页在本地真实服务上检查过数据选择、实验校验和真实提交，网页提交的匹配任务已 COMPLETED；最终截图作为可视核验依据。

前端 TypeScript 编译与生产构建通过，分拆了图表依赖。完整 npm 依赖审计（含开发依赖）结果为 0 已知漏洞；这不等于完整安全审计。自动测试有一条 Starlette/AnyIO 兼容别名弃用警告，不影响测试结果。

## 检查中发现并修复的问题

1. 新 Gurobi 大版本与本机许可证不兼容：依赖锁回到已实测的 10.0.3，避免“安装成功但无法优化”。其他机器需独立运行 doctor。
2. 稳态结果中的 NumPy boolean 无法按严格 JSON 序列化：转换为原生 bool 后重跑完成。
3. 直送容量不能沿用局部容量公式：按源数学表达式单独求根，并为零载荷车队回流保留弧。
4. 滚动窗口不能每期重置库存或只采用最后一个窗口成本：携带已启动弧/货物流，从累计承诺计算全局结果。
5. 同名数据和代码 hash 不足以保证排队任务复现：提交时物化源代码快照，worker 在快照目录执行；代码变化后的“同版本重跑”拒绝执行。
6. 快照内包含测试代码，导致 pytest 重复递归收集：限制测试根目录并排除 workspace/环境/前端依赖。
7. SAC 学习率和网页服务并发原先未接入实际路径：已连接实际优化器与 Coordinator，避免配置仅在页面显示。
8. 完整开发依赖审计发现初选 Vite 的已知文件访问问题：升级至兼容的 7.3.6 后重建和复查，审计清零。问题说明参见 [维护方安全公告](https://github.com/vitejs/vite/security/advisories/GHSA-fx2h-pf6j-xcff)。
9. 比较不能把最小化成本的增量称为改善，也不能混用不同源快照：按 metric sense 计算改善率，校验源代码一致性，并加入定向测试。
10. 脱敏不能在序列化字符串上破坏 JSON 引号：先递归脱敏结构再序列化，测试保证 ZIP 内 JSON 可解析且测试凭据被隐藏。

## 研究与安全限制

没有证据证明新平台或 tiny learned value 提升算法收益。测试证明的是小规模计算与契约可运行，不是大规模收敛、运力标定或数学定理成立。

稳态 BHH 的均衡需求、非绑定车队、单位成本，与有限时域的已知订单、有限车队、总成本不能直接排名。有限 BHH 是 divisible stops 近似，不是完整订单路径模拟。rollout 条件采样仍为近似；旧 SAC 的 delivered 没有经验证计数，返回 null。计划距离不是实际里程，局部 gap 不是全局 gap。

说明文档的高级验收仍有缺口：warm-up 稳态长跑桥接、原生 macOS、断点续跑、行业数据匿名化、完整行业 profile、专门资源甘特/多阶段 Sankey 与前端 SSE 自动重连。当前实现明确拒绝不支持的配置，而非悄悄忽略参数。功能边界和可执行命令见 [平台说明](EXPERIMENT_PLATFORM.md)。

服务仅供可信本机用户，workspace token 不是互联网多用户认证。复现包包含真实冻结数据；诊断默认脱敏，但复现导出没有自动匿名化，分享前需检查。ZIP 中的代码和外来 pickle 从不自动执行。需要部署到公网、接入敏感行业数据或升级 Gurobi 时，应重新设计和验证安全/许可证/数学契约。
