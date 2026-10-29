# 第五轮论文重建历史基线

更新时间：2026-10-29。本页记录提交5814876的 **paper_settings + legacy_unit + legacy_native** 基线。后续几何错误修正见[排查记录](PAPER_OVERLAP_FIX_20261029.md)，当前入口默认后端为paper_sat；要复跑本页历史方法必须显式指定legacy_native。

这里的“回到论文原版”具体指回到第五轮依据论文描述实现的基线，不表示已经找回作者原始求解器，或已验证整篇论文复现成功。

## 本次切换

- `run_paper_alignment.py` 默认采样profile改成 `paper_settings`。
- 默认后处理仍为 `legacy_unit`：执行固定系数1的原重建碰撞位移，随后调用原负x/负y压缩。
- 主线不启用 `guarded_backtracking` 的碰撞回溯、三项非增接受和停滞提前终止。
- 原 `legacy_unit` 已有的压缩回退和停止约定保留，它们属于第五轮实现，不能称作者已确认的细节。
- 正式候选数允许128和512，不需要标记 `--smoke`；其他缩减规模仍须显式标记。
- 模型、权重、采样器计算、碰撞场实现和二分压缩函数本次均未修改。

`guarded_backtracking` 及固定候选回放工具保留为可选对照，不作为论文复现主线。启用它仍需显式指定参数。

## 当前证据

已收到队友的第五轮固定候选回放 `report.md` / `summary.csv`，范围是样本0、45，各三配置、512候选。它们没有重新运行模型。`paper_settings` 下，float32保存后的原重建策略与guarded策略均为0/1024严格有效候选；guarded策略的平均交叠率更高。不能把其他配置的3→20有效候选计数当作论文设置的改善。

因此回到基线是为了沿同一主线排查问题，不是声称基线已经成功。现有0、45的原策略结果可继续使用，不必为了这次默认值切换重复生成。

## 队友运行方式

先激活第五轮使用的Python/GPU环境，进入本分支代码目录。`ROOT`为已有官方资源根目录，`paper_build`为第五轮带论文约束入口的构建目录。输出子目录必须尚不存在。

```bash
git fetch origin
git switch codex/paper-reproduction
git pull --ff-only

ROOT=/实际资源根目录
OUT="$ROOT/results/paper_baseline_5cases_v1"

for i in 0 1 45 46 83; do
  python -u run_paper_alignment.py \
    --root "$ROOT" \
    --extension "$ROOT/paper_build" \
    --output "$OUT/sample_$i" \
    --profile paper_settings \
    --sample-index "$i" \
    --candidates 512 \
    --geometry-backend legacy_native \
    --result-dtype fp32 \
    --post-iterations 100 \
    --post-policy legacy_unit
done
```

若直接复用已有0、45结果，只需对新增样本运行，把循环列表改为 `1 46 83`；不要把跨配置或不同生成来源的结果合并为同一受控实验。5例属于开发诊断，不代表128例主评测。

不加 `--no-constraints`，生成阶段第97—128步的碰撞约束会执行。这里调用的是论文重建的 `paper_repulsion_all`，不是简单取消官方 `rm_overlap` 注释。

`run_paper_suite.py` 仍是历史三配置、两样本探针，不能代替这条主线命令。正式b128可把候选数改成128；旧第五轮校验器有固定512协议，不能直接拿它给新的128候选运行验收。

## 结果阅读

完整保留 `predictions_raw.pt`、两种保存精度的输出、`input_geometry.json`、`input_sample.json`、`sampler.json`、`constraint_callbacks.json`、`post.json`、各阶段metrics和 `metadata.json`。

重点比较原始输出、增强后float64、保存后float32的交叠、越界和各判据有效候选数。用 `enhanced_fp32_metrics.json` 的 `reference_strict_fixed_strip` 查看float32严格诊断。当前入口已修正选优，`selected_result.json`只从所指定保存精度的严格通过者中选择；本页命令显式指定fp32，修正后的主线默认保留float64，两者均单独报告。没有有效候选时仅产生明确标记失败的诊断图。5814876及此前的预览采用float64面积50判据，旧图不能当作严格成功证据。

目前仍需核实：公开模型/训练配置与论文的对应、凹形位移算法、生成时间网格及后处理数值约定。主线切换没有自动消除这些未知项。
