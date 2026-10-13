# Direct PPO 失败机制诊断

独立输出 reports/cfs_ppo_strategic_control/direct_ppo_failure_diagnosis_v1。
历史 Direct v3、MC和所有早期结果只读；不训练、不变更平台默认。

入口 numerics prepare/m0/pilot/diagnose，随后独立 analysis data/figures 与 delivery qa/freeze。
prepare 冻结源码、任务书、状态选择和上限。新增72个M0完整日与72个Direct-zero独立回归日；复用原216个Direct及72个MC正式日，属于已用测试集的事后诊断。

机制仅验证侧前四场景21008–21011。每模型在 t4/12/20 每阶段依公开待处理订单数选最低/最高（相同状态只取一次，按scenario seed破同分），最多54状态。与收益和隐藏未来无关，完整保留不可行动状态。每状态15控制：zero/deterministic/4随机/3方向×3幅度。9模型都有覆盖。
pilot另取三工况seed30001的21008,t12，读4个训练日和12公开状态；先计时外推，再执行已冻结正式上限，不能以收益改变规则。
全局6000 MILP/1200秒/2GiB进程/2GiB新增文件，上限包含M0/zero和pilot，CPU串行各1线程。
受控长期诊断最多每模型一例：按冻结状态与控制顺序取第一个不同业务动作，原确定性分支及该变体使用相同原始状态/未来订单，t+1以后共同该冻结Direct模型；不选利润赢家、不冒充总体独立样本。

车辆规范化只允许当前公开物理状态和承诺完全等价的同质车辆互换。业务多重集保留具体订单与每条路线内部顺序；不任意删除vehicle ID。

Actor：全部九模型×4checkpoint在相同96验证公开状态（前四场景×24t，来源各模型best轨迹）重算确定性信号；每checkpoint本侧均值作Dpi参照。保存动作维度/时段/负载汇总与实际查询美元系数。不是动态性现金因果实验。
Critic：原128训练episode/模型的pre-update value，独立完整实际收益return-to-go /1000，与GAE lambda-return分别计算RMSE和explained variance；按update汇总、统计单位仍scenario/model，不把27648时段当独立样本。只有总裁剪前梯度日志，不虚构历史Actor/Critic分项；recovery仅保存最终Critic，新增冻结梯度诊断明确标为事后，不执行optimizer。

M0对比保留每训练seed逐场景差，20k crossed bootstrap seed53000、九个描述性比较Bonferroni99.444%区间和95%；MC−M0只重抽24场景，不能伪装3独立MC模型。旧Direct−MC必须逐值一致。本轮全为事后/机制探索，不是新确认性family。
CSV四份主文件+两补充（business、counterfactual），SVG/600dpi PNG四图，不生成PDF。CSV creation marker一次6文件。相关pytest/XML、M0/zero/全部反事实独立回放、探针动作编译/状态/现金/身份、统计独立实现、全历史/源/输入/SHA、真实图审阅通过才能COMPLETED。
