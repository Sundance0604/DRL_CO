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
PROFILES["rolling"]={
    "title":"Single rolling-horizon route-macro-action MILP",
    "equations":[r"[t,\min(t+H,T))\ {\rm plan};\quad[t,\min(t+k,T))\ {\rm execute};\quad k\le H",
                  r"\max\ \sum_{o,v,a}(r_o+B_o-p_o(s_a-t))x_{ova}-\sum_{v,a}c_{va}y_{va}-\sum_oB_o,\quad V=0",
                  r"\sum_{a\in\delta^+(n)}y_{va}-\sum_{a\in\delta^-(n)}y_{va}=b_{vn}",
                  r"\sum_{v,a}x_{ova}\le1,\quad x_{ova}\le y_{va}",
                  r"\sum_{o:e\in P_o}q_o x_{ova}+L^{\rm committed}_{vae}\le Q_v",
                  r"\mathcal{N}_{\rho}(u)=\{v\ne u:d_G(u,v)\le\rho\}"],
    "notes":["One joint window MILP schedules waiting, repositioning, anchored trips and vehicle reuse over several periods.",
             "Uses only currently booked orders; unrevealed readiness stays at the current conservative upper bound. No forecast or oracle future orders.",
             "Past/active commitments are immutable. Only the first k periods are executed; unexecuted suffixes are discarded at replanning.",
             "Trips are route macro-actions within a window. Intermediate continuation is fixed; prefix-compatible extensions are reconsidered at the next replan.",
             "B_o is unassigned backlog/expiry cost over this window; s_a is assignment time. c_va includes full newly committed route cost, waiting/reposition and delay tie-break, even if completion is beyond the window.",
             "Completion extends automatically beyond the planning boundary when necessary; no new dispatch after T. Window gaps are not full-run bounds.",
             "Optimization uses booked stage accounting; evaluation can use the independent full-cooldown realized ledger. No training."]}
PROFILES['oracle_preview']={**PROFILES['rolling'],'title':'Oracle demand preview (NOT deployable)',
    'notes':PROFILES['rolling']['notes'][:1]+[
        'Information-relaxation diagnostic: additionally reveals frozen bookings in (t,t+H]. No assignment before booking and no pickup before ready. Only the current action is committed.',
        'H retains the platform half-open decision-period convention. A booking exactly at t+H is visible but has no decision slot in that window.',
        'Unrevealed batch readiness is still frozen at its current bound. Route-macro-action restriction remains; this is not a global optimal bound or proof of RL benefit.']+PROFILES['rolling']['notes'][2:]}
