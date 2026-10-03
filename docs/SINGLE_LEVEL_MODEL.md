# DRL_CO 单层匹配新模型（2026-10-02）

> 本文由原始交付代码包的根说明迁入。原代码包把 GitHub 仓库放在外层目录的 `DRL_CO/` 子目录中；当前分支已经把新增文件直接合入仓库根目录，并把运行命令改为模块形式。审阅结论见 [SINGLE_LEVEL_MODEL_REVIEW.md](SINGLE_LEVEL_MODEL_REVIEW.md)。

三段式（HV–AV–HV）城际拼车里，自动驾驶干线车调度的单层匹配模型及其上界、在线前瞻策略的原型实现。对应的模型说明是 `doc/main_v2.tex`。

## 目录

| 路径 | 内容 |
|---|---|
| `model/` | 新模型的全部代码（下表） |
| `checks_on_original/` | 对原仓库代码做核验用的两个脚本 |
| `doc/main_v2.tex` | 模型说明（Overleaf 项目 672a124acca194695eeb2ce6 的提交 `6d4f470`），用 XeLaTeX 编译 |
| `drl_co/`、`experiments/`、`tests/` 等 | 原仓库提交 `59935d3` 的代码；交付包中的副本经逐文件校验与该提交一致 |

`model/` 里的文件：

| 文件 | 作用 | 对应 `main_v2.tex` |
|---|---|---|
| `demand.py` | 城内需求与首段接驳的生成器：无提前量、各自直接到枢纽、成批到达三种设定 | 第 1.3 节 |
| `mt_prototype.py` | 单层匹配模型 $(\mathrm{M}_t)$ 和价值增广 $(\mathrm{M}^V_t)$，滚动仿真，四条不变量检查 | 第 2–5 节 |
| `bounds_check.py` | 完全信息上界 (H)（MILP 与 LP 松弛）和流体上界 (F)，以及数值验证 | 第 6.2、6.3 节 |
| `online_lookahead.py` | 在线前瞻策略：流体对偶价值、每时段重解、rollout；已知未来的 rollout 诊断 | 第 4 节 |
| `penalized_bound.py` | 带惩罚的信息松弛上界 | 第 6.4 节 |
| `pervehicle_bound.py` | 按车建模的完全信息上界（不拆单、不换车） | — |
| `online_lookahead_chosen.json` | 训练场景上选出的规则，`rollout` 和 `penalized_bound.py` 会读 |  |

## 环境

Python 3.11，依赖见 `requirements.txt`。需要可用的 Gurobi 许可证；学术许可证即可，pip 自带的受限许可证规模不够。

```bash
python -m pip install -r requirements.txt
```

## 运行

所有命令都从仓库根目录运行。`model/` 已是 Python 包，不再需要手工设置 `PYTHONPATH`。

然后：

| 命令 | 做什么 | 大致用时 |
|---|---|---|
| `python -m model.mt_prototype` | 三种首段设定下的短视策略，60 个场景，带不变量检查 | 3–4 分钟 |
| `python -m model.bounds_check` | 策略利润不超过 (H)；Jensen 检查 | 约 25 分钟 |
| `python -m model.online_lookahead phase1` | 固定规则与流体价值策略，训练 10 个加测试 30 个场景；写出 `online_lookahead_chosen.json` | 约 3 分钟（最多 24 进程） |
| `python -m model.online_lookahead rollout 10` | rollout（每期抽 10 条未来）和已知未来的 rollout | 约 1 小时（最多 24 进程） |
| `python -m model.penalized_bound` | 带惩罚的信息松弛上界，80 个场景 | 约 8 分钟（最多 24 进程） |
| `python -m model.pervehicle_bound high 10000 180` | 按车建模的上界，单个场景，限时 180 秒 | 3 分钟 |
| `python -m pytest -q` | 原仓库自带的 15 个测试 | 10 秒 |

并行任务默认使用 `min(24, CPU 核数)` 个进程；也可调用 `execute(..., workers=N)` 显式调小。

`checks_on_original/` 的两个脚本针对原仓库：`reloc_check.py` 是纯 Python，不需要 Gurobi；`trace_check.py` 的运行方式写在文件开头。

## 已验证的

- 四条不变量在三种首段设定、60 个场景上全部成立：每个路段上车内人数不超过 7；送达不晚于截止时间；上车不早于就绪时间；价值函数取常数时目标值只差一个常数。
- 所有策略的日利润都不超过完全信息上界；Jensen 不等式在经验均值处成立。
- 合入审阅分支后已重新跑过导入、完整原型批次、单场景流体价值/rollout/惩罚上界/按车上界和 18 个自动化测试，均通过；详细证据见审阅报告。

## 要注意的

- **成本参数是占位值**（`mt_prototype.py` 开头：载客每单位距离 10、等待 0、取消罚 300；空驶成本在各脚本里取 10 或 50），没有标定。
- **城市几何是占位的**（`demand.py` 开头）：城市是圆盘，需求均匀分布，枢纽在圆周上，半径除以城内车速在 0.5–1.5 个时段之间；收单周期 2 个时段，接驳车 7 座，巡回常数 0.9，每单停靠 0.1 个时段。
- 需求的 OD、人数、下单时段和截止宽限沿用原仓库生成器（均匀 OD、恒定到达）。
- 时域末端没有处理：票价在指派时入账，最后几个时段指派的订单不一定在时域内送达。
- 带惩罚的信息松弛上界在这组设定下没有收紧上界（最优缩放为 0）。
- `load_case(..., lead=False)` 复现最早的无提前量设定，`load_case(..., batches=False)` 是各自直接到枢纽，默认是成批到达。

## 这次量出来的数字

成批到达设定，30 个测试场景，相对短视基准：在线 rollout 在 normal 档（11 车，每时段 5 单）+9.6%，在 high 档（5 车，每时段 8 单）+36.3%；流体价值策略 −0.6% 和 +25.3%；完全信息上界（LP 松弛）+55% 和 +199%。原交付说明称三个设定的完整对比位于 `DRL_CO_可赢空间初测_20260930.md`，但该笔记没有随代码包提供，本分支也不包含该文件。
