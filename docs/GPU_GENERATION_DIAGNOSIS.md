# 队友代跑：生成阶段GPU对照与核验

这是定位第五轮失败的生成阶段实验，不是完整论文评测。当前本机CPU重放样本0候选0：100轮后最大两两交叠面积2923.89、总两两面积3566.43、名义利用率68.14%、高度越界3.64，仍未严格通过。它只重放既有坐标，没有重新运行模型。

## 直接执行

在第五轮已经能运行的GPU/Python环境中，进入本仓库。ROOT是包含upstream、resources、extensions的既有实验资源根目录；BUILD是第五轮带paper_constraints入口的原生构建目录。不要新建或覆盖官方源码。OUT及其同名ZIP必须尚不存在。

```bash
git fetch origin
git switch codex/paper-reproduction
git pull --ff-only

ROOT=/实际资源根目录
BUILD="$ROOT/paper_build"
OUT="$ROOT/results/generation_diagnosis_v1"

python -u run_generation_diagnosis.py \
  --root "$ROOT" --extension "$BUILD" --output "$OUT"
```

脚本默认按顺序运行8次：样本0、45各4种设置，每次128个候选、128步生成、关闭生成后的100轮增强。它自动复制运行代码、保存全部坐标/噪声/轨迹、进行CPU核验，最后生成`$OUT.zip`。失败日志也保留；核验失败时命令退出码非0，仍会打包可用证据。默认单次超时900秒，超时未测候选不能删除分母。

不需要重下模型或数据。脚本检查当前已审计的官方权重和测试PKL哈希，要求CUDA版PyTorch和原有干净的固定版本upstream。若路径不符或检查失败，保留报错，不要通过修改校验值“绕过”。

## 四种设置回答什么

| 名称 | 采样配置 | 模型模式 | 末32步约束 | 用途 |
| --- | --- | --- | --- | --- |
| paper_train_off | 论文数值配置 | train | 关闭 | 生成基准探针 |
| paper_eval_off | 同上 | eval | 关闭 | 检查推理运行模式的影响 |
| sigma1_train_off | 同一Gaussian/EM框架，噪声下限1 | train | 关闭 | 检查与权重配套噪声配置的局部兼容性 |
| paper_train_native_on | 论文数值配置 | train | 历史原生场开启 | 与第一行比较生成阶段碰撞修正的影响 |

每次仍用相同官方权重，不做训练。第三行不是“完整官方推理配置”，仍保持Gaussian初始化、128等间隔EM和单位漂移倍率；旋转500/1000的字段仅改变记录用GT，不能用它声称这轮验证了模型训练旋转尺度。旋转尺度的受控去噪证据见[第六轮检查](CHECKPOINT_ALIGNMENT.md)。

第四行使用legacy_native，是为了隔离第五轮原生碰撞场的影响；它不是修正后paper_sat非凸求解器的全候选验收。关闭约束/增强的行属于消融，也不能拿其交叠作为完整论文算法的结论。

## 为什么额外保存随机噪声

只给train/eval相同seed不够：运行模式会影响Dropout等随机操作，可能连后续采样噪声都改变。新探针在独立CPU生成器中预先生成初始Gaussian随机数和128步Wiener随机数，并保存`sampling_noise.pt`。同一样本的4种设置使用相同随机数；噪声幅度仍按各自SDE配置计算。

这样train/eval的比较不被随机数消耗顺序混淆；碰撞约束开关两行的前96步轨迹必须逐元素相同。默认论文复现入口没有强制启用此模式，旧采样随机顺序保留。不能声称新探针与第五轮历史输出逐位相同。

## 看结果与回传

优先看`verification.json`：verified表示文件哈希、配对随机数、步骤覆盖、前96步一致性及CPU几何复算通过，**不是排样成功**。每个子运行的raw_metrics.json仍独立报告严格有效候选、交叠、越界和利用率；没有有效候选时不得把名义利用率当作合格成绩。

核验器不重跑GPU网络前向，不能证明GPU分数、权重训练来源或论文结果已复现；近阈值的可行性标志按记录的1e-9规则核对，几何数值复算另有浮点比较容差。测试只有2个开发样本，每次128候选不等于128个测试实例。

**请回传完整`generation_diagnosis_v1.zip`。** 包内有代码快照和每个子运行的metadata、sampler、sampling_noise.pt、predictions_raw.pt、trajectory.pt、input_sample.json、input_geometry.json、raw_metrics.json及日志。权重和完整数据集不会被打包。

需要在同一环境手动复核时，用新输出文件名：

```bash
python "$OUT/code/verify_generation_diagnosis.py" \
  --run "$OUT" --output "$OUT/verification_again.json"
```

本机完成新增5项CPU自动检查：随机数消耗隔离、前96步配对、错误噪声拒绝、8格证据覆盖、改写几何即使重算哈希仍被拒绝。这些使用合成模型/数据，不是实际GPU运行。
