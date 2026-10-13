# 仅完整性检查位置修复的诊断恢复

原诊断v1已真实完成72M0+72zero并独立核验；随后离线Actor pilot尚未产生任何诊断结果时，verify_parent在no_mc_guard内部尝试读取密封文件哈希而触发保护。不是PPO读入MC、未来订单泄漏或数值失效，保护不放宽。

v1源码和输出永久只读。v2使用原pilot/diagnose/恢复状态/收益/控制/选择函数，唯一修复是全历史/输入哈希预先在guard之外检查，guard内只用已核验父metadata，不读密封文件、MC表或额外未来信息。144已成功运营日只读复用，全部复用文件哈希保存；原v1失败证据保存。v2最大6000calls/1200数值秒仍包含v1 M0实际成本，作者编辑暂停不冒称数值耗时。没有改变种子、状态、扰动、预算、checkpoint或统计。

入口 run prepare、run run；后续delivery qa/freeze。analysis使用原独立纯numpy分析源码，仅路径适配并读取原v1M0，不重跑M0。图形/报告非数值错误可披露修复，数值与选择规则不得改变。
