# 第七轮证据索引

- `plan.json`：冻结16例、每例1个raw代表+GT、三路、两种精度及失败预算，非512候选评价。
- `build.json`、`strip_repair.patch`：独立条带API构建、官方源码与头文件绑定；旧有限容器API保留。
- `strip_controls.json`：8项条带/原压缩副作用可见性控制及12组指标控制。
- `legacy_controls.json`、`failure_controls.json`：原11+34项控制回归。
- `metric_controls_extended.json`：192份全失败输出仍保留全部输入及失败分母的检查。
- `verification.json`：全部96次调用、192终态、修复中间态、float32转换、独立几何和原生计量复核。只表示证据一致，不表示全部排布有效。
- `representatives.png`：样本25、46的三路float64布局和利用率/交叠变化。
- `local_binding.json`：本地/服务器源码、构建、验证器、图与证据逐字节哈希及取回包哈希。

完整动作和日志保存在本地私有目录；本目录不含权重、账号、服务器信息或完整数据集。所有失败及恶化均保留，不以外部回滚掩盖压缩副作用。
