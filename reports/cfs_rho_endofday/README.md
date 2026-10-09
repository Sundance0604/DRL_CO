# 本次成果入口

1. [修改前审计](implementation_audit.md)：原邻接接载、随机边长、assignment-time计账、旧drain再次优化的事实。
2. [正式仿真报告](simulation_report.md)：40配对场景、210运行、每条件10seed，全部结果、CI、机制、边界和go/no-go。
3. [测试记录](test-results.json)：73通过、0失败；另对210正式运行独立复核账簿、容量、事件、配对hash、源码归档。
4. [复现与模块索引](../../experiments/cfs_rho_endofday/README.md)：数据路径、命令、参数和结果文件含义。
5. [原始与汇总结果](../../results/cfs_rho_endofday/)：manifest、source snapshot、真实抽样字段、冻结场景、每t trace、双ledger、CSV/JSON统计与plotting config。

图在 `figures/`，7种图各有PNG/PDF/SVG。本次没有提交或推送，也没有修改原始CFS/Intercity仓库。中间smoke、早期核算核对版本及测试临时文件可在ignored `workspace/cfs-task-qa/` 找到；它们不是正式结果，未删除源数据或用户已有文件。

解释时请保持三条边界：真实CFS货运需求≠真实时间/运费；几何网络≠观测美国公路；所有承诺完成≠所有需求已服务（未分配需求仍记录）。
