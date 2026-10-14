# 单工况 beta 探索尺度干预

独立输出 reports/cfs_ppo_strategic_control/structured_ppo_beta_std_pilot_v1。
只改变 W40/8、30001 的20维 Actor初始 beta log_std=-1.5→-.5，alpha=-.5不变；所有其他初始tensor和数学设置与旧PPO20相同，从初始化重训，不续接旧权重。

32个训练episode、8updates，每update4完整24期episode。读取历史相同128场景池与抽样顺序；sample generator31002，minibatch31003，data随机流31004。共同标准高斯噪声不意味着同动作或同轨迹，不强制离开on-policy。

原backend真实KL检查只读插桩；固定update8，无验证/测试赢家选择。动作诊断复用上轮已查看的7active+2only；旧模型30001的完整/归零记录只读复用，新增18次probe。开发场景从21024开始递增取前4个未执行、内容不重叠的既有CFS准备池场景；生成与实际使用分开披露，池若不足必须停止准备，不静默替换。只四开发配对日、无正式测试、无MC、无显著性检验。

数值硬上限600秒/1000solver claims，预定978；无新增smoke求解，单元测试不调MILP。准备前用历史实测32日耗时作保守外推预算门，数值各阶段累计真实时间。存储512MiB。CPU Torch/solver/BLAS各1线程，规定PYTHONHASHSEED=0、PYTHONIOENCODING=utf-8、PYTHONDONTWRITEBYTECODE=1、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。

入口仓库.venv Python：run prepare|train|probes|evaluate|all。COMPLETED/FAILED/BUDGET_LIMITED只读，不自动重试或追加样本。完整网络/Adam/三个RNG按更新保存；partial不混正式结果，所有失败保留。数值源、analysis、README、数值测试在prepare冻结，qa/delivery非数值修复独立。

数值完成仅NUMERICAL_COMPLETE_QA_PENDING。qa独立重建新/旧网络采样、共同epsilon、GAE、RTG、更新/优化器/RNG、实际触发参数、全部现金/价格/物理/冷却和18probe；CSV用bundled NumPy重算，5CSV单次marker。4SVG/600dpiPNG由现有Matplotlib绘制，真实看图，无PDF。最终相关非求解测试：tests/test_structured_beta_std.py及tests/test_structured_q_mechanism.py -k 'not certificate'；XML仅当前qa，非平台全矩阵。所有QA/实际视觉/输入与raw哈希通过后delivery freeze一次。

不提交/推送/PR，不改平台默认，不写历史。已清理的测试临时副本不是研究依赖，不重建大测试临时树。
