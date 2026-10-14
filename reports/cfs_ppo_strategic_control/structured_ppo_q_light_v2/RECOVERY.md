# 文件系统恢复披露

原 light_v1 在保存新测试场景时缺少 evaluation/scenarios 目录，以 FileNotFoundError 停止；没有运行正式测试日。原目录永久只读。

light_v2 逐字节复制全部九个完成模型、原始训练/验证/pilot、预算与协议，并仅创建缺少的输出子目录。冻结数值源码、参数、统计和随机流未改变，没有重新训练。

copied qa/stop.json 属于原目录历史故障；本目录活动状态以 manifest 和 qa/recovery.json 为准。原始 source 保持旧身份，另冻结恢复入口 SHA，不能混称全部源码来自新包。
