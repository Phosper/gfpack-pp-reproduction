# 第六轮：权重来源、测试集身份与尺度兼容性

本轮继续补查第五轮列出的公开材料缺口，并做一次短的受控去噪检查。没有训练、生成新的排布候选或改变既有基线；工作保留在本地分支 `checkpoint-alignment-round6`。

## 已排除的数据与版本误解

现用权重与官方资源包 `data/ckpt/dental.pth` 逐字节相同，SHA256为 `e3585572100e61c5f3112a8925e7add420b70c8c7a8f23e29be97788ad3c5643`。包中只有这一份权重，没有训练配置、优化器状态、训练日志或另一份快照。

现用 `dataset_dental_vali_128.pkl` 与资源包的 **`data/test/dental_test.pkl` 完全相同**，SHA256为 `dffdd191803ddfeb7701d9012b1f38dd88f01ee9333747a19ba86e18870848e0`。文件名中的vali来自本项目准备脚本的重命名，不能据此认定用了训练尾部验证数据。作者也明确区分训练末128条validation与已有test set，见 [作者回复](https://github.com/TimHsue/GFPack-pp/issues/7#issuecomment-4441702715)。

另外只读扫描训练PKL的ID字段：共196,113条ID记录。测试集既不是训练前128条，也不是后128条的原样切片；两个端窗口与测试集相同的有序ID记录各为0。这不证明测试集与训练内部完全不相交，训练actions/height未展开，也不能证明它就是论文统计表的确切测试划分。证据见 [资源审计](checkpoint_alignment/checkpoint_audit.json)。

公开主线共8次提交，首次实际源码发布到当前版本之间的6次变更没有修改 `src/` 或原生几何实现。因此没有找到遗漏的另一版40M模型、sigma_min0.1或旋转1000训练代码。见 [官方版本比较](https://github.com/TimHsue/GFPack-pp/compare/3d70b232860ec89f2e2e508b11e138f63c5f61ea...ff75815e24ec50cd7fa8b2abdec1acfe0b55be95)。

## 补充材料和模型能证明什么

按作者 [issue4指针](https://github.com/TimHsue/GFPack-pp/issues/4#issuecomment-3803533180) 下载了 [CVF原始补充包](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Xue_GFPack_Attention-Driven_Gradient_ICCV_2025_supplemental.zip)：171,436,303字节、2,324项，SHA256为 `f4d9662fa7ff0950485111cc08673bde3bfc79fb5796d28745456e6f9f7e2a4d`。其中有图、视频、Puzzle边界与生成脚本及5页补充PDF，没有另一份模型、Dental测试PKL或训练/碰撞求解器配置。原始PDF第2页仍写GFPack++约40M，排除了该表述仅来自后来arXiv改版的可能。原文件只保存在私有证据目录，入库的只是 [补充包清单](checkpoint_alignment/supplement_inventory.json)。

权重包含298个张量、10,621,276个元素；形状显示主Transformer宽128，编码器8层、解码器8层，几何Transformer编码器2层。按当前源码实例化时，总参数10,621,276，其中固定Fourier参数32个，可训练参数10,621,244。权重本身不保存requires_grad，不能把所有元素都称为可训练参数。

形状不能确定训练噪声、动作尺度、注意力头数、dropout、学习率、EMA、训练轮数或样本划分。当前源码默认值只是源码事实，不能反推为这份checkpoint的真实训练设置。作者另说明任意边界拟通过额外多边形输入网络，并表示考虑释放相关代码，见 [issue9回复](https://github.com/TimHsue/GFPack-pp/issues/9#issuecomment-6150885031)。这解释了公开边界实现的缺口，不能用随机扩充网络参数来补齐论文结果。

## 受控去噪检查

`check_checkpoint_compatibility.py` 保持同一权重，分别检查噪声下限1或0.1、旋转输入比例500或1000的四种组合。开发样本仍为0、45；每点使用4组固定高斯噪声，整个前向（包括几何编码器dropout）配对随机状态。没有优化器、梯度更新、扩散采样或后处理。

两组检查分开报告：固定t使用7点；匹配实际sigma使用 `[1.1,3,10,100,1000]`，按各噪声下限反算t，使同旋转编码下的带噪输入完全相同，缩小“噪声任务难度不同”的混杂。共96格、384次配对带噪评估全部完成，耗时5.11秒；它们不是384个独立布局。11项CPU算术/随机数/失败分母控制通过。

指标为官方DSM残差 `score*(sigma+1e-5)+epsilon` 的均方，相对同一epsilon的零score基线作比；小于1表示此检查中比零score好。合并两例时累计误差与基线再相除，不平均各格比值。匹配实际sigma组的平移/旋转/全部通道比值分别为：

- sigma_min1、旋转500：**0.5424 / 0.3930 / 0.4652**。
- sigma_min0.1、旋转500：**0.6659 / 0.6030 / 0.6334**。
- sigma_min1、旋转1000：**1.4370 / 19.0705 / 10.5525**。
- sigma_min0.1、旋转1000：**1.1066 / 16.9234 / 9.2830**。

当前权重在这组开发输入上更兼容发布版1/500设置；1000编码的较大损失主要集中在旋转通道。旋转编码也改变输入分布及相对角噪声，所以不能将这个结论推广成每个噪声点或所有排布任务上的优劣。模型输出除以std，DSM残差又乘回，显式尺度会相消；这不是识别真实训练sigma的方法。径向旋转向量误差也不等于最终角度或排布失败率。

四个sigma1000/t1端点的配对控制通过，最大score差约3.81e-9；固定噪声、dropout及匹配sigma输入哈希一致，冻结输入未改变。原始 [去噪记录](checkpoint_alignment/compatibility.json) 包含全部格子与分通道指标；[独立CPU复核](checkpoint_alignment/compatibility_verification.json) 已在原运行环境严格通过：重新生成噪声哈希、核对逐格基线与汇总，并拒绝重复格和被篡改的误差。没有再次计算模型score或重新生成dropout掩码。

本地Windows/PyTorch2.12与原Linux/PyTorch2.6生成的正态数存在字节差异，首次跨环境严格哈希复核如实拒绝，见 [可移植性记录](checkpoint_alignment/local_portability_check.json)。没有放宽哈希要求或更改原报告；随后改用原环境仅CPU复核通过。统一seed不保证跨版本、跨平台逐位复现。

## 本轮决定与下一步

数据准备中的vali命名歧义已解释；公开历史和原始补充包没有找到可直接替换的论文模型。保留发布尺度作为工程基线，完整论文数值配置继续作为单独失败探针，不放宽阈值或把本轮去噪误差当作候选成功率。

严格论文复现仍需要：对应约40M表述的模型构造及匹配权重、真实训练噪声/动作尺度配置、论文测试划分，以及非凸碰撞位移和后处理步长/顺序细节。公开来源检索覆盖全10个issue、1个PR和15条评论；[来源清单](checkpoint_alignment/upstream_sources.json) 固定检查范围，不能据此否定私有材料的存在。未联系作者或发布issue。

在这些材料未补齐时，可继续的工程路线是回到已冻结16例，完成原计划的三路后处理消融，先处理“部分候选经后处理反而增加交叠”的问题，并独立核查float32回写。它不等于恢复论文未公开细节；本轮没有执行该消融或启动训练。

后续第七轮已实际完成该消融，结果和新增精度问题见 [三路后处理记录](POSTPROCESS_ABLATION.md)。

## 可重复入口

```bash
python audit_checkpoint_provenance.py --checkpoint WEIGHT --archive OFFICIAL_ZIP --dataset USED_TEST --provenance MANIFEST --upstream PINNED_SOURCE --train-windows --output NEW_AUDIT_JSON
python check_checkpoint_compatibility.py --self-test --output NEW_CONTROL_JSON
python check_checkpoint_compatibility.py --root ROOT --output NEW_DIAGNOSTIC_JSON --max-seconds 60
python verify_checkpoint_compatibility.py --report DIAGNOSTIC_JSON --output NEW_VERIFICATION_JSON
```

`--train-windows`只接受基础数值列表pickle操作码，不支持的引用/构造格式会明确失败，不会执行pickle里的可调用对象。输出必须不存在。权重、原数据与补充包不入Git，完整资料保留在本地私有目录。
