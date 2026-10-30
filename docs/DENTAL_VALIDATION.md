# Dental 复现：固定配置、分阶段核验

目标是复核论文 Dental 条带排布：高度1200、每例48个零件、128例测试，每例512个候选。每个候选都执行后处理，只从合法候选中选利用率最高者，然后对实例求平均。论文表1报告均值77.53%；不能把该数字当作任何一份公开权重在任意配置下必然达到的保证。

依据：[v2第4.4节](https://arxiv.org/html/2406.07579v2#S4.SS4)、[第5.2节](https://arxiv.org/html/2406.07579v2#S5.SS2)、[第6.1节](https://arxiv.org/html/2406.07579v2#S6.SS1)、[补充S2](https://arxiv.org/html/2406.07579v2#A2)。原始源码固定为`ff75815e24ec50cd7fa8b2abdec1acfe0b55be95`，`src/`与原始C++均未修改。

## 两条路径分别回答什么问题

| 入口 | 设置 | 能回答的问题 |
| --- | --- | --- |
| `--sampling-profile paper`（默认） | 高斯初始化；σ下限0.1、上限1000；128步EM；第97—128步加入面积加权碰撞场 | 依据v2文字重建的生成过程效果如何 |
| `--sampling-profile public-source` | 直接调用原`src.sde.pc_sampler_state`；均匀初始化；σ下限1；原漂移倍率14；原函数没有后期几何回调 | 作者公开函数配合公开权重实际生成什么，避免把重建采样器误当作原函数 |

两条路径都用原网络和同一份已核对的Dental权重。`public-source`指生成函数保持原样，**不代表本项目后处理也是原生C++，也不代表它完全符合论文文字**。元数据、调用轨迹与验证器明确区分两条路径；不允许把公开路径的σ改成0.1后仍称其为原函数配置。

默认保留公开入口的`train()`状态，始终`no_grad()`，不训练、不更新参数。`eval`需显式指定并分别评测。论文未明确该状态。公开训练代码中旋转编码为500倍，v2写1000倍；不在推理时强行把预测旋转分量乘2。权重文件本身不记录完整训练配置，不能据此确认训练历史。

论文版本也有差异：v1第5.2节描述生成后增强，v2明确加入最后32步碰撞约束。主线以v2为准，不能把不同版本和公开代码的细节拼成一套“已确认的作者参数”。

## 本分支提供的改动

1. 增加直接调用公开采样器的独立对照入口；观察钩子不消耗额外随机数，测试逐位核对输出、轨迹和最终RNG状态。
2. 默认论文重建仍用单次EM更新。可显式用`--constraint-integrator split-flow`测试稳定数值积分：保留神经EM提议，再对同一面积加权碰撞场作限步长、能量回溯的子步积分。**该分裂次序和回溯是数值扩展，不是作者确认的算法**；未验证前不作为默认论文结论。
3. 独立候选后处理使用进程池。工作进程只导入NumPy/几何库，不各自载入Torch或CUDA；候选间互不共享可变布局。单进程、双进程与缓存重放测试逐位一致。
4. 保存每个候选的原始指标、后处理坐标及轨迹；从JSON重新读入后，用独立仿射变换测量。串并行与断点续跑不得改变候选ID或选择标准。
5. 断点续跑核对输入、配置、求解器源码哈希；任一变化都拒绝混用历史候选。失败结果和错误信息保留，运行错误停止批评测。

现有后处理保留：面积加权SAT分离位移、上下边界修正、负x/负y二分压缩、最多100轮、固定生成后的旋转。凹形分离计算、逐对更新次序、松弛1.75、接触余量和接受规则仍是明确披露的重建细节，见[完整实现说明](REPRODUCTION.md)。存在这些未确认项时，不能声称“所有算法参数都与作者一致”。

零件轮廓不改变，面积不变，不用GT替换结果。正式验收始终是：保存坐标的最大两两交叠面积=0、两两交叠面积之和=0、上下边界越界=0。指标接近时允许的比较误差不能把非零交叠改判为零。

## 队友GPU机器运行

先准备环境与官方资源，见[环境说明](REPRODUCTION.md#环境与资源)。所有输出目录必须是新目录。不要覆盖已有运行。

本轮首跑两条路径各测前2例（样本0、1），每例512候选，两组共2048个候选。批量入口默认`--count 2`、`--candidates 512`；完整Dental评测需显式指定`--count 128`。

```bash
python -m unittest discover -s tests -v

# 首先各跑前2例，每例512候选，使用同一种后处理以分离生成器影响。
python -m reproduce.benchmark --sampling-profile paper --start 0 --count 2 --candidates 512 --device cuda --constraint-threads 4 --post-workers 16 --output runs/dental_paper_first2
python -m reproduce.benchmark --sampling-profile public-source --start 0 --count 2 --candidates 512 --device cuda --post-workers 16 --output runs/dental_public_first2

# 在相同旧raw上复核后处理，不重新生成，不挑选少数候选代替512。
python -m reproduce.replay --input runs/dental_paper_first2/sample_000 --output runs/dental_paper_first2_unit/sample_000 --post-policy pairwise --relaxation 1 --workers 16

# 仅用于稳定积分消融，独立保存；不得悄悄改名为原论文配置。
python -m reproduce.benchmark --sampling-profile paper --constraint-integrator split-flow --constraint-threads 4 --start 0 --count 2 --candidates 512 --device cuda --post-workers 16 --output runs/dental_split_first2

# 选定、冻结配置后运行完整128例；以下为论文文字重建默认路径。
python -m reproduce.benchmark --sampling-profile paper --start 0 --count 128 --device cuda --constraint-threads 4 --post-workers 16 --output runs/dental_paper_full128

# 仅后处理中断且源码/输入未改变时恢复；不重新生成原始候选。
python -m reproduce.resume runs/dental_paper_first2/sample_000 --workers 16

# 独立测量每个原始/最终候选，输出跨度变化和交叠变化；不运行模型。
python -m reproduce.diagnose runs/dental_paper_first2 --output runs/dental_paper_first2_diagnosis.json
```

`--post-workers`是候选后处理CPU进程数；神经网络仍按候选批量在GPU运行。`constraint-threads`是论文重建路径最后32步独立候选几何计算的线程数，EM与split-flow均支持。不能据此声称已经复现作者C++并行后端或论文运行时间。

## 结果怎么判断

- 先看每例原始候选的最高**名义**利用率，以及交叠分布；非法候选的名义利用率只用于定位问题。
- 再看每个候选后处理后的合法性和跨度变化，判断是修复扩宽还是压缩不足。
- 最终只比较每例最优合法候选。分别报告实例成功率、合法候选数、成功样本均值和失败计零均值，不能隐藏失败实例。
- 前2例用于初步排查，单候选或8候选只算入口诊断；不能用来判断哪种σ总体最好。正式Dental复现结论需要冻结配置的128×512运行和全部保存坐标核验。

本分支不把自动测试通过、局部碰撞下降或少量实例有合法解称为“复现77.53%”。如果完整跑分仍有差距，下一步应对照作者实际训练/采样配置和原生增强实现；不能只放宽交叠判据以获得相近分数。
