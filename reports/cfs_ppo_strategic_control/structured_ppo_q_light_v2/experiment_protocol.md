# 轻量实验协议

{
  "adjustment": "Bonferroni3 within family",
  "bootstrap": 20000,
  "bootstrap_seed": 71000,
  "cap_bytes": 2147483648,
  "cap_calls": 20000,
  "cap_seconds": 1800,
  "directions": "each hub near+far; all hubs near+far",
  "episodes_per_update": 4,
  "formula": "alpha+beta*psi; no clipping; max2Sc",
  "gate": "all correctness; equality-quotient rank added; occupiedq>=2; canonical switch",
  "kind": "lightweight exploratory; user overrides16update24test",
  "periods": [
    4,
    12,
    20
  ],
  "primary": "PPO20-PPO10/B8",
  "reference_seeds": [
    21008,
    21009
  ],
  "secondary": "PPO20-PureMC; separate family",
  "selection": "only update8; no checkpoint search",
  "slopes": [
    -0.6,
    -0.2,
    0.0,
    0.2,
    0.6
  ],
  "test_seeds": [
    22068,
    22069,
    22070,
    22071,
    22072,
    22073,
    22074,
    22075
  ],
  "training_seeds": [
    30001,
    30002,
    30003
  ],
  "updates": 8,
  "validation_seeds": [
    21008,
    21009,
    21010,
    21011
  ]
}

用户的轻量要求优先于附件16-update/24-test建议，改动在数值结果前登记。保留三工况三seed，训练缩为8update，验证仅update8，正式新测试缩为8场景。旧10D只用B8原权重，不用事后B16 best。旧训练成本没有消失，推理复用不宣称总成本对等。历史完整QA只读引用，当前实际核验所用输入SHA和所有新增轨迹。
