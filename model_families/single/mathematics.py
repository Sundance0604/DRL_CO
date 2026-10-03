"""Implementation-aligned summaries; not a replacement for the full research model."""
BASE = {
"title":"Single-level route / order dispatch",
"equations":[r"\max\ \Pi_t(x,y,z,w)+\sum_{v,r}V(h_r,t_r)y_{vr}+\sum_{v,h}V(h,t+\tau_{vh})z_{vh}+\sum_vV(h_v,t+1)w_v-\epsilon\sum_{v,r}d_r y_{vr}",
r"\sum_v x_{ov}+s_o=1,\quad x_{ov}\leq\sum_{r:o\in r}y_{vr}",
r"\sum_{o:e\in P_o}q_o x_{ov}\leq Q_v-L_{ve},\quad \forall(v,e)",
r"t_{\rm pickup}\geq a_o^{\rm visible}(t),\quad t_{\rm destination}\leq b_o"],
"notes":["Capacity is enforced on each route arc, including already committed load.",
         "New route/extension endpoints require assigned anchor orders; existing commitments are preserved.",
         "Profit accounts for assignment revenue, planned route/reposition cost and period penalties; it is not actual cash flow."]}
PROFILES = {
"myopic":{**BASE,"title":"Myopic single-level dispatch","equations":[r"V(h,t)=0",*BASE["equations"]]},
"fluid":{**BASE,"title":"Analytical / fluid dual value","equations":[r"V(h,t)=\lambda^{\rm fluid}_{h,t}\cdot m_V",*BASE["equations"]],"notes":BASE["notes"]+["The coarse fluid LP is a heuristic value estimator, not an operational-profit bound."]},
"rollout":{**BASE,"title":"Sampled look-ahead rollout","equations":[r"a_t=\arg\max_{a\in{\cal A}_{\rm rules}}\frac1K\sum_{k=1}^K \widehat J(s_t,a,\omega_k)",*BASE["equations"]],"notes":BASE["notes"]+["Uses sampled futures and a zero-value base rule; no fitting. Conditional first-mile batch simulation remains approximate."]},
"learned":{**BASE,"title":"Optional supervised hub-time value","equations":[r"V(h,t)=w^\top\phi(s,h,t),\quad \min_w\sum_i(w^\top\phi_i-\lambda_i^{\rm fluid})^2+\lambda_{\rm reg}\|w\|_2^2",*BASE["equations"]],"notes":BASE["notes"]+["Supervised fluid-dual regression, not reinforcement learning. Held-out data and a verified checkpoint are required for evaluation."]},
"oracle_lp":{"title":"Perfect-information pooled LP bound","equations":[r"J_{\rm online}\leq U_{\rm LP}"],"notes":["Full-horizon time-expanded pooled-capacity relaxation. Bound feasibility relaxes vehicle indivisibility and route anchors; it is not an executed policy."]},
"oracle_mip":{"title":"Perfect-information pooled MIP bound","equations":[r"J_{\rm online}\leq U_{\rm MIP}\leq U_{\rm LP}"],"notes":["Pooled network MIP still relaxes some vehicle-level route restrictions. No operational trace or realized profit is claimed."]}}
