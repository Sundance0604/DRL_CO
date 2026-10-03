"""BHH-specific summaries: divisible flows, tours, shared inventories."""
CAPACITY=[r"f_c(q)=a_cq+b_c\sqrt q",r"C_c(v,d)=\min\{vM,f_c^{-1}(v(d\,t_0-2\rho_c))\}",
r"2\rho_o+\frac{f_o(q)}v+\tau t_0+f_d(q/v)\leq d\,t_0"]
PROFILES={
"steady":{"title":"BHH stationary balanced-demand unit-cost study","equations":[r"s(W)=a+\frac b{\sqrt W},\quad W\in[1,W_{\max}]",r"(W^*,n_1^*,n_2^*)=\arg\min g_{\rm hub}(W,n_1,n_2)",r"(W_d^*,n_d^*)=\arg\min g_{\rm direct}(W,n_d)"],
"notes":["Two symmetric cities, divisible demand and stationary sizing. Continuous and integer-reoptimized policies are distinct.",
"Outputs are theoretical unit costs. Demand parameter configurations and repeated identical theory calculations are not statistical samples."]},
"finite":{"title":"BHH finite-horizon divisible-flow oracle","equations":[r"\min\ C_{\rm fleet}+C_{\rm waiting}+C_{\rm resort}+C_{\rm unserved}",*CAPACITY],
"notes":["Full-horizon oracle with shared integer HV/AV inventories, divisible commodities and cumulative transfer causality.",
"Discrete finite total cost is not stationary unit cost."]},
"rolling":{"title":"BHH rolling-horizon commitments","equations":[r"[t,t+H]\ {\rm plan};\quad[t,t+k)\ {\rm commit};\quad k\leq H",*CAPACITY],
"notes":["Only committed departures consume fleet inventory; later windows preserve committed trips and transfers.",
"Online information; completion extension is explicit."]},
"spatial":{"title":"BHH spatial tour calibration","equations":[r"\widehat T(q)=b\sqrt q",r"b^*=\frac{\sum_i\sqrt{q_i}\,\overline T_i}{\sum_iq_i}"],
"notes":["Independent seeded spatial sampling calibrates routing coefficients; it is not a dispatch policy.",
"Zero-stem fit of b only; no fitted linear stop coefficient or operational-policy claim. Calibration randomness is owned by this component."]}}
