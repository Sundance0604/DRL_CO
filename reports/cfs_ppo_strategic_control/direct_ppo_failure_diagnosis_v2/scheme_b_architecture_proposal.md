# 方案 B：无 MC 的供给槽位条件化战略价格

2026-10-13。本文件是技术设计与可行性判断，不是已训练模型，也不声称方案B比A更好。正式训练不在本轮授权范围内。

## 1. 接入语义：先承认 q 的真实含义

实际入口为 `model_families/single/value/contracts.py:add_release_slots`。每个候选路径最终流入一个release group `(when, hub)`，binary slot 的总和等于该组流入的车辆数，`z_q <= z_(q−1)`保证前缀占用；q是**当前候选联合方案中该释放时刻、该枢纽组的第q个free车辆槽位**，不是任意第q辆车的固定编号，也不是已经存在于未来全系统的全部供给量。约束允许正、负或非单调系数，`when>=T`不产生战略项。

早期MC的q来自在特定原始状态上注入q/q−1资源的评估干预。它与在线release-group prefix的背景状态、资源来源及组外供给不完全一致。原任务书将两者写成同一个V(t,h,q)只是记号，并没有消除语义差别。本轮尚未量化这部分误差的因果贡献。

新无MC策略不使用注入标签，直接按实际decoder的q语义输出系数并以完整现金利润训练。它可以绕开标签错位，却仍有group-wise可加表示限制。不能因此把Actor系数叫作已经正确估计的Bellman边际价值；Critic估计完整收益，Actor只产生可学习的战略控制价格。

## 2. 最小20维原型：先可执行，不先追求通用性

保留现有公开集合编码器 `e(o)`、宽度64、共享Actor–Critic和PPO-Clip。Actor均值头由10改20：每hub输出near/far各一个基价格a和一个供给斜率b，合计5×2×2。20维latent Gaussian经tanh得到a,b∈[−1,1]。仍不读MC表、MC特征、MC尺度或MC权重。

令 `ell=(tau−t)/(T−t)`，`u=(q−1)/(K−1)`，K=1时u=0。定义：

\[
F_{h,j}(q)=a_{h,j}+(1-|a_{h,j}|)u\,b_{h,j},\qquad j\in\{n,f\},
\]

\[
V_\theta(o,h,\tau,q)=S_c[(1-\ell)F_{h,n}(q)+\ell F_{h,f}(q)],\quad\tau<T.
\]

这个构造保持价格在[−Sc,Sc]，K=1边界明确；b=0时**精确包含原10维价格函数**，而非通过额外缩小价格使A/B不公平。u也可写成 `(q/K−1/K)/(1−1/K)`，所以是状态编码、hub、相对释放时间与q/K的结构化函数。near/far斜率可正可负，不强制边际价值递减。

它只是最小有限基展开：在固定状态/释放时刻下对q仍为仿射，不能表达任意非单调槽位曲线，不应冒称通用 `f_theta(e(o),e_h,ell,q/K)` MLP。若最小版本有证据，后续才增加供给basis或一般query网络。纯q维扩展不能表达跨hub、跨车联合动作交互。

## 3. PPO动作概率：随机的是20维latent，不是每个query独立抽样

一个决策期只采样一次20维latent z，完整保存z、old_logp和版本。所有MILP query用同一个z及公开状态确定性展开，不能在provider每次查询时重新采样，也不能把`5×24×K`个查询当独立PPO动作。

对角高斯与tanh概率沿用现有稳定实现：

\[
\log\pi(a|o)=\sum_{i=1}^{20}\{\log\mathcal N(z_i;\mu_i,\sigma_i)-\log(1-\tanh^2z_i)\}.
\]

PPO ratio是联合log probability之差的指数。不能除以20来“维持旧KL”而仍称同一联合概率比；新增维度会改变KL与探索的实际量级，需训练侧pilot检查，不按正式利润调阈值。query价格是20维动作的确定性推前，训练不需要对MILP求导，也不计算不存在的高维价格密度。PPO的代理目标与minibatch更新依据[原始PPO论文](https://arxiv.org/abs/1707.06347)，此处20维供给基是本项目的新设计，不是论文保证其有效。

## 4. 模块隔离与可执行接口

在独立模型变体包中新增：

- `QPriceSchema`：版本、hubs顺序、20维布局、K、训练Sc、terminal规则；不能加载原10维checkpoint。
- `QPricePolicy`：一次推理/采样，返回signal、latent和joint logp；Critic仍为单值收益估计。
- `QPriceProvider`：缓存 `(tau,hub,q)`，只做上述有界展开；q整数1..K、tau>=t、tau>=T归零。`marginal_release_value`签名保持兼容。
- `QPriceDecoder`：复用原H1 planner和compile_action；在解MILP之前把每个query的数字系数算完。不能把神经网络或非线性函数作为solver变量引入。
- 独立runner/config/checkpoint/测试：观测仍为无MC版本；episode/reward/full-cooldown不修改。价格表和所有执行系数逐t保存供审计。

最小provider计算可直接按以下伪代码落地，无需改MILP主体：

```python
def price(tau, hub, q):
    if tau >= T:
        return 0.0
    assert tau >= t and 1 <= q <= K and int(q) == q
    ell = (tau - t) / (T - t)
    u = (q - 1) / (K - 1) if K > 1 else 0.0
    near = a_near[hub] + (1 - abs(a_near[hub])) * u * b_near[hub]
    far = a_far[hub] + (1 - abs(a_far[hub])) * u * b_far[hub]
    return S_c * ((1 - ell) * near + ell * far)
```

## 5. 是否需要单调性

不需要硬性单调递减。离散路线、容量、订单组合与不同释放时间会破坏通常的凹资源价值假设，当前MILP前缀机制本身支持非单调系数。若未来使用单调性regularizer，应是单独声明的经济先验／独立模型，不偷换为合法性条件。最小原型的斜率符号不限制，不添加MC-derived prior。

## 6. 尺度、预算与公平比较

Sc继续只用训练收入Q75及冻结系数5，五hub同尺度；不使用MC或22020–22043的利润重新校准。最小B比A多10个均值输出和10个log_std参数，隐藏表示不扩大。solver同样H1、单线程、每期一次，query数不增加，仅系数展开增加少量运算。

先做6个完整训练episode的工程pilot：zero与M0、b=0与A逐query/整数动作/账本、K=1与K=3/8、terminal、logp数值与hash、两期状态过期检查、随机sample不随query重抽。之后才按实测可承受上限冻结正式预算；不在本文件承诺未测算的大批训练。

正式A/B必须从新随机初始化、同训练数据及seed、同updates/episode/验证选择预算、同MC对照和完整现金奖励、同Sc起步，不把B的更大预算与旧A混比。新确认性测试场景不能复用22020–22043。20维探索不同于10维是新模型的一部分，应同时报告实际价格方差/动作切换率/成本，不能将改log_std的隐性调参当q本身的作用。

若本轮诊断提示Critic质量或共享梯度存在问题，训练修订应先形成单独可检验的改变（如收益target尺度、value loss权重或共享结构），再固定共同训练设置比较A/B。不要同时改q、骨干、loss和训练时长后宣布q导致增益。q结构的价值尚未由本轮证明。

## 7. 当前判断

接入是可行的：provider协议和前缀资源槽位已经存在，不需要修改物理或整数调度主体。进入B的合理理由是验证一个已知表达限制，而不是既有负向利润“证明必须加q”。是否优先实施B，应结合本轮M0、Critic、探索及受控利润诊断；本轮只完成评估和设计，不自动进入新训练。
