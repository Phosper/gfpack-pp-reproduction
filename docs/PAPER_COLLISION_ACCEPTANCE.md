# 第五轮碰撞移动与接受规则修订

日期：2026-10-28。分支：`codex/paper-reproduction`。

## 这次解决什么

第五轮 `paper_enhancement.py` 原来直接执行 `位置 += 碰撞位移`。如果同时推开多个零件导致新的交叠或越界，这一步仍会保留。后面的 rollback 只撤销压缩，不能撤销已经变差的碰撞修正。

新增显式策略 `guarded_backtracking`：**先试走，检查变好才接受；变差就缩步，仍无改善就保持原状态。** `legacy_unit` 继续保留第五轮原逻辑和默认行为，方便使用同一批已生成候选对照。

## 与论文的对应关系

依据：[论文 §4.4](https://arxiv.org/html/2406.07579v2#S4.SS4)、[§5.2](https://arxiv.org/html/2406.07579v2#S5.SS2) 和补充材料 S2。

| 项目 | 本次处理 |
| --- | --- |
| 碰撞位移方向、按面积分配位移 | 保留现有 Eq.9 / SAT 重建实现，不改 `paper_constraints.hpp` |
| 凹形轮廓 | 仍是现有边轴投影代理，不声称等于作者的凹形实现 |
| 间隙压缩 | 保留现有沿负 x、负 y 的二分搜索调用 |
| 迭代预算 | 最多100轮，不能把到达预算当作修复成功 |
| 生成模型、权重、采样器 | 本次不改，只隔离检查生成后的增强 |
| 修正步长与接受判定 | 论文未明确，下面的回溯规则属于**我们的数值实现约定** |

这不是“找回了作者原版后处理”。保留论文的机制，不意味着未公开的细节已完全对齐。

## 新规则

1. 每个候选单独取当前碰撞场方向，试步长 `1, 1/2, 1/4, …, 1/1024`。只缩放同一个方向，不增加随机搜索、旋转或额外强制投影。
2. 用 GEOS/float64 重算几何。**两两交叠面积之和、最大两两交叠面积、最大竖向越界量三项都不能增加，且至少一项有实际下降**，才接受碰撞移动。仅总面积下降不够。
3. 所有步长均不接受时，保持修正前的位置。随后仍可尝试原有压缩，避免提前放弃压缩可能带来的改善。
4. 压缩及其全局 x 平移作为独立事务检查：上述三项不能增加，横向跨度也不能增加；失败则只撤销该次压缩，保留已通过检查的修正。
5. 仍有交叠或越界且无法移动，记 `stalled_infeasible`；耗尽预算记 `iteration_limit`。只有满足参考几何判据且运动停止才记 `reference_converged`。
6. 转成 float32 后再测一次，结果单列 `fp32_reference_feasible`，不把 float64 通过等同于保存后通过。

非增比较不添加增长容差，防止每轮累积小交叠。实际下降采用 `1e-12 * max(1, 基线值)` 的噪声阈值；可行性仍按本项目的交叠面积 `<=1e-9`、越界 `<=1e-9` 判定。这些阈值不是论文常数，也不表示数学上的严格零。GEOS 使用浮点计算，不是精确算术。

这些限制可能拒绝一些有利于后续逃离局部状态的移动，因此会停滞，**不保证所有候选最终都能无交叠**。只缩步也不能解决错误/受限的方向、凹形代理误差、原生量化漏检或生成阶段已有的异常。

## 如何观察变化

新 `post.json` 的每轮 `trace` 包含逐候选的：

- `before_repair_reference_geometry`：碰撞修正前；
- `full_step_trial_reference_geometry`：试走完整步长后的状态，即使被拒绝也记录；
- `after_repair_reference_geometry`：接受/回退后的修正状态；
- `gap_trial_reference_geometry`、`after_gap_reference_geometry`：压缩提议与实际保留状态；
- `repair_step_per_candidate`、`repair_trials_per_candidate`、接受标记和停滞计数。

由此可区分“提出了变坏的移动”和“最终真的保留了变坏的移动”，也能确认异常发生在修正还是压缩。

## 使用已保存候选对照，不重跑模型

在保存了第五轮完整产物和原生扩展的运行环境执行，路径替换为实际目录：

```bash
python replay_paper_enhancement.py --source-run /path/to/round5/sample_000_paper_settings --extension /path/to/paper_native_build --output /path/to/new_replay_000
```

脚本默认对同一份512候选依次运行 `legacy_unit` 和 `guarded_backtracking`，分别保存 float64/float32 坐标、逐轮记录及 `comparison.json`。不重新生成随机候选，不需要 CUDA，不修改原结果。输出目录必须是新目录。

所需原始运行文件：`metadata.json`、`input_sample.json`、`input_geometry.json`、`predictions_raw.pt`、`raw_metrics.json`；另需与原运行哈希一致的原生扩展目录及 `build.json`。脚本核对输入哈希、候选顺序和原先使用的 x 平移基准。验证的是所提供清单的一致性，不是外部真实性认证。

此次收到的两个 ZIP 只有 `sampler.json`、`constraint_callbacks.json`、`post.json` 等日志，缺少上述坐标和几何输入，因此不能用它们回放并宣称第五轮成功率已经提高。

若需要新生成候选，原入口显式增加：

```text
--post-policy guarded_backtracking
```

`run_paper_suite.py` 和历史 `verify_paper_alignment.py` 仍服务于原第五轮协议；不要把新策略输出套入旧的状态/源码校验规则。优先使用上述固定输入回放，只比较后处理规则。

## 本地验证范围

```bash
python -m unittest -v test_paper_guarded_enhancement test_replay_paper_enhancement
```

测试覆盖真实几何小案例的面积权重、边界破坏、总交叠下降但最大单对变差、逐候选缩步、停滞、压缩回退、原生零面积漏检、float32 残余、旋转保持，以及保存候选回放和哈希不一致拒绝。

2026-10-28本地运行：18项测试全部通过；两个运行入口的帮助命令和 Python 语法检查通过。未运行实际第五轮512候选回放。

控制案例中，五个正方形的全步移动让总交叠从 `1.344459` 降到 `0.428342`，但最大两两交叠从 `0.344846` 升到 `0.410025`；新规则拒绝全步、接受半步，得到总交叠 `0.788805`、最大两两交叠 `0.319405`。这解释了为什么不能只看总交叠有没有下降。

测试使用实际 Shapely 几何和一个遵循现有 SAT/面积权重的 Python 小适配器；压缩接口以恒等或故意不安全的提议测试接受/回退。它们不替代原生 Clipper/二分搜索的端到端验证，更不代表 Dental 的512候选或128样本结果。
