# 第六轮证据

- `checkpoint_audit.json`：官方ZIP逐成员哈希、现用权重/测试集身份、完整张量形状与训练ID端窗口核对。训练内部成员关系未检查。
- `supplement_inventory.json`：CVF原始补充包来源、哈希和文件类型清单；不重新分发其内容。
- `compatibility_controls.json`：11项CPU控制结果。
- `compatibility.json`：8次小批前向、96格/384次配对带噪评估的完整记录；这不是384个独立布局或新生成的排布候选。
- `compatibility_verification.json`：原环境CPU重建噪声、基线和汇总的严格验证，包含重复格/篡改误差拒绝控制；不重新计算模型score。
- `local_portability_check.json`：跨环境噪声字节哈希不一致的首次拒绝记录；后续在原环境验证通过，未放宽要求。
- `audit_controls.json`：基础列表窗口读取、禁止对象/引用操作码和补充包完整CRC控制。
- `upstream_sources.json`：公开历史与作者回复的检查范围、链接和简述。
- `local_binding.json`：本地与服务器的5个共同源码、当前辅助脚本、资源和验证输出哈希绑定；列出的证据文件按原字节记录哈希。

本目录不含账号、服务器地址、权重、数据或完整补充材料。实验实际只保留本地，没有推送或创建PR。完整性/公式验证不等于确认checkpoint训练历史或论文成绩。
