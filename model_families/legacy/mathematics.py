"""Legacy-specific semantics: virtual pickup actions and lower-layer dispatch."""
PROFILES = {
"supply":{"title":"Supply-city heuristic + lower-layer dispatch","equations":[r"a_o={\rm supply\_city}(s,o)",r"\max_x\Pi_t(x\mid a)\quad{\rm subject\ to\ legacy\ dispatch\ constraints}"],
"notes":["Non-learning supply heuristic. Order, vehicle, state and reward definitions are the legacy family definitions."]},
"sac":{"title":"Candidate-city SAC + lower-layer dispatch","equations":[r"\pi_\theta(a_o\mid s_o),\quad a_o={\rm virtual\ departure\ city}",
r"\max_x\Pi_t(x\mid a)\quad{\rm subject\ to\ legacy\ dispatch\ constraints}",
r"J_\pi=\mathbb E[\alpha\log\pi_\theta(a\mid s)-Q_\phi(s,a)]"],
"notes":["Discrete candidate-city action and legacy assignment rewards.",
"Training and held-out evaluation are separate phases; checkpoint features and physical parameters must match."]}}
