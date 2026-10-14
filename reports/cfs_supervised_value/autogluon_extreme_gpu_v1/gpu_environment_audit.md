# GPU 与环境核验

实际 GPU：RTX 5060 Laptop GPU，驱动 592.47，8151 MiB 显存，compute capability 12.0。
操作系统记录：Windows-10-10.0.26200-SP0；Python 3.11.15。
原 AutoGluon 环境为 pavane：AutoGluon 1.6.3、Torch 2.13.0+cpu、NumPy 1.26.4，没有 Nori、TabICL、TabDPT。
新环境为 workspace/envs/autogluon_extreme_gpu，采用 system-site-packages 只读继承原包，本地覆盖并安装额外依赖。
新版本见 dependency_manifest.json：Torch 2.13.0+cu130，CUDA runtime 13.0，NumPy 2.2.6，Nori 0.14.0、TabICL 2.1.1、TabDPT 1.2.0。
CUDA 单卡可用，实际 16×16 矩阵运算与同步检查通过；此证据只能证明 CUDA 内核可执行。
GPU 基础模型 fit/predict 没有成功，因此不能声称 GPU 参与了 Extreme 训练或基础模型推理。

机械探针调用 Nori 适配器时，默认系统内存估计为 2.794 GiB，实际可用为 1.966 GiB，超出 90% 保护阈值并触发 NotEnoughMemoryError。
不是 CUDA OOM。系统物理内存约 15.73 GiB；可用内存会随其他应用与导入发生变化。
另一次只读静态审计，在导入前可用 3.48 GiB、导入后 3.29 GiB。
这不是重新成功运行的证据，也不能替代失败时记录。
官方适配器对 10080×3 的估计：系统内存 2.795 GiB，显存 6.489 GiB；均为静态估计，非实测峰值。
仅按 90% 系统内存保护计算，模型检查点至少需要约 3.105 GiB 可用系统内存；这不包含预留给其他并存进程的裕量，不保证运行成功。
当前失败不能推导为这张 8 GiB 显卡永远无法运行 Extreme。

pip check 未全通过：继承的 statsforecast 对 numba/llvmlite 要求与 Nori 不兼容，另有原环境 wechatauto-replica 缺 opencv。
本轮不使用这些包；不把新环境称为全生态兼容环境，也不为修复无关依赖改动原环境。
原 CPU 环境中的 Medium 全部 4200 条预测复现最大差异 0，Gurobi 所在项目 .venv 未改动。
所有冻结输入与源码再次核对 SHA-256：1278 个文件通过，详见 qa/frozen_assets_final.json。

首次下载三个基础权重共 485,516,618 字节，加载前下载与文件核验耗时合计 39.07 秒，详见 environment/pretrained_weights.json。
Torch CUDA wheel 的 1915.5 MB 下载属于环境准备，非模型权重、非正式 fit；缓存安装 41.22 秒。最初 dry-run 已下载 wheel，未保存精确下载计时，不虚构该时长。
