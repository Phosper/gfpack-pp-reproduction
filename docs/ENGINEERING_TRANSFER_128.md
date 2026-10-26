# 第九轮：128×1固定候选工程迁移

2026-10-26启动，16:11（北京时间）独立复核完成。沿用第八轮冻结的工程后处理，对已有官方默认模式基线的128个样本各处理一个既定候选，另运行128份原始GT对照。候选选择先于新算法执行冻结；原16例开发输入逐字节保留，其余112例单独报告为迁移回归。

这轮没有新采样或训练，不能作为128×512候选率或论文完整复现结果。新输出在Windows的NumPy 2.4.3、Shapely 2.1.2、GEOS 3.13.1环境运行；Linux旧修复起点与新参考几何评价分别保留，不将新输出称为官方calutil重放。

## 固定协议

- 实验编号：`engineering_transfer_128x1_20261026_01`。
- 预测/GT分母各128，开发16与迁移112分别汇总；执行失败、超时和几何无效保留。
- 高度1200，无水平墙；面积50参考判据、严格面积与边界1e-9分别记录。
- 两起点来自同一输入：raw及第七轮`overlap_repair_strip`的1.0/2000修复结果；原16预测与GT的新生成旧起点全部与历史动作一致。
- 每个新尝试45秒、4遍压缩，真正写出的float32端点通过严格判据后按最小水平跨度选择，等宽选raw。旋转保持不变。
- profile按旧raw交叠量和轮廓顶点数的最小/最大值预选，先串行完成预测和GT并独立重放，再以固定2进程运行其余输入。
- 记录旧起点、新求解、输入校验/输出I/O、内存及独立复核成本。计算时间之和与各阶段墙钟分别记录，冻结和网络传输时间不计入求解时间。

冻结manifest SHA256为`1f1a1e2662c4e11eea2ea0d3a6ad0c491478e339748b7d6aaa3f00d8850c1b5a`，评价合同SHA256为`e30f6230d6d424c5365a710b96afbc6acaaa78198f84bf04e43884df07596993`。下载包SHA256为`085ac789822ecc98e5754d2e26db8d703c7156b2ae18e895e2149b2f3d249fc4`；本地已复核128份案例和391个旧起点输出/来源快照。

## 运行与复核

完整动作、原生二进制、输入和日志保存在被Git忽略的`results/engineering_transfer_128x1_20261026_01/`，不加入代码仓库。`run_v1/summary.json`随输入完成更新；`profile.json`在小规模成本检查后生成。开始时的记录不是最终效果。

新增入口：

```text
python freeze_engineering_transfer.py --results VERIFIED_BASELINE --diagnosis MATCHING_DIAGNOSIS --development-frozen ORIGINAL_FROZEN16 --output NEW_FROZEN128
python prepare_transfer_seeds.py --frozen NEW_FROZEN128 --extension ROUND7_LINUX_BUILD --output NEW_SEEDS
python run_engineering_transfer.py --frozen NEW_FROZEN128 --seeds NEW_SEEDS --output NEW_RESULTS --workers 2
```

输出目录必须不存在。运行结束自动调用`verify_engineering_transfer.py`，独立重建全部新旧几何、存储精度、同候选来源与双起点移动记录，并核对哈希、选择和分母。独立复核报告与完成状态保存在结果目录旁的`run_v1_verification.json`和`run_v1_completion.json`；验证成功表示证据一致，效果与GT退化另行报告。

## 完成结果

128例预测全部完成并通过严格参考几何检查，执行失败和超时均为0；平均条带利用率75.5520%，最小61.4117%、最大79.0458%。开发16例严格16/16、平均72.6870%，其余112例严格112/112、平均75.9612%。这些是每例一个既定候选的结果。

GT正式记录为127/128完成且严格有效，1例执行失败、0超时。样本47求解已经完成，第二次保存结果时`os.replace`遇到Windows拒绝访问，按既定规则仍保留为执行失败，不改写原分母。其保留的两份工件动作一致，另行只读独立重放通过，零交叠、零越界；该诊断旁证没有替换正式记录。其余127份GT未出现宽度增加或严格有效性丢失。

主运行648.224秒（约10分48秒），独立校验908.608秒（约15分9秒），旧起点生成阶段60.963秒；三个阶段墙钟之和1617.795秒（约26分58秒），不含冻结、网络传输及准备时间。独立复核状态`verified`，核对518个结果文件、391个旧起点输出/来源快照，重放337条独立求解流程及50,828条移动记录，包含2,388,916次移动件配对复测。验证通过表示证据一致，GT保存失败仍保留。

公开证据包括[独立校验](engineering_transfer_128/verification.json)、[成本profile](engineering_transfer_128/profile.json)、[分项耗时](engineering_transfer_128/cost.json)与[源码及证据哈希](engineering_transfer_128/binding.json)。校验报告为原始报告的逐字节副本，仅含指标和哈希，不含完整动作、轮廓、日志或账户路径；Git属性保留证据字节。

后续需单独修复Windows重复写入的可靠性并做I/O回归；本轮原结果不覆盖。128×512全候选与论文协议核验继续按原路线推进。
