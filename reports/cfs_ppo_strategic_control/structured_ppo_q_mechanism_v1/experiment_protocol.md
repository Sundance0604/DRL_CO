# 实际执行协议

```json
{
  "KL_capture": "Python frame at actual backend stopping check; no parameter/RNG mutation",
  "KL_models": "both10/20,3conditions,3seeds,8updates; saved rollout reconstruction",
  "caps": {
    "bytes": 536870912,
    "calls": 500,
    "seconds": 600
  },
  "continuation": "frozenPPO10/B8/30001 from next period",
  "controls": [
    "learned20",
    "beta0",
    "old10",
    "fixed single witness-hub slope",
    "fixed all-hub slope"
  ],
  "counterfactual": "first2 distinct states with canonical switch, state→seed→fixed control order; no profit selection",
  "kind": "mechanism diagnosis, NOT new training or formal test",
  "minimum_elapsed_claim": "offline includes update reconstruction; does not fit new data",
  "no_hyperparameter_change": true,
  "no_new_formal_days": true,
  "qactive_certificate": "copy original MILP constraints; maximize sum ordered q2 slots; OPTIMAL0 proves q-only",
  "seeds": [
    30001,
    30002,
    30003
  ],
  "selection": "first8 provablyq-active + first2 q-only ordered byseed,time; retain shortages, no profit selection",
  "slopes": [
    -0.6,
    -0.2,
    0.2,
    0.6
  ],
  "state_pool": {
    "periods": [
      4,
      8,
      12,
      16,
      20
    ],
    "scenario_seeds": [
      21008,
      21009,
      21010,
      21011
    ],
    "source": "frozenPPO20/30001 validation"
  }
}
```

原始库/模型只读；数值源码与本协议在执行前SHA冻结。PPO优化仅重建保存的历史轨迹，不产生新训练数据或新模型；正式测试未访问。状态冻结发生于q-feasibility检查后、扰动前，未使用隐藏未来利润。
