# 位移落点检查与压缩终点验收

代码分支：`codex/validated-displacement-code`，基于`a7044c7`。此分支只包含代码、合成几何测试和通用运行说明，不附带新增的用户运行明细。

1. 按原来的面积权重分配SAT最小分离向量。先试松弛位移，检查两个零件移动后的完整轮廓；如果放大后仍交叠，回退到1倍最小分离位移并复查。复查失败则记录几何错误，不写入失败落点。
2. 允许仍有交叠的零件尝试负x、负y二分压缩。每个被接受的终点必须与其他所有零件交叠面积为0，并且高度合法；不要求起点已经有效，也不要求移动路径无碰撞。
3. 每轮仍先修正碰撞和高度，再压缩，最多100轮；保留本候选迭代中最紧凑的严格合法结果。最终保存后重新测量，微小正交叠仍算失败。

一对分离不代表整个布局已经有效：移动可能影响第三个零件，仍须全局迭代和最终检查。显式选择小于1的松弛系数时，保留阻尼更新语义，不声称单步就能分离。

[论文第4.4节](https://arxiv.org/html/2406.07579v2#S4.SS4)给出面积权重与SAT最小分离位移；[第5.2节](https://arxiv.org/html/2406.07579v2#S5.SS2)给出生成后增强及负x/负y二分压缩。此处1.75、逐对顺序、回退和终点接受规则仍属于声明的重建选择，不声称与作者实现逐行一致，也不保证任意候选在100轮内成功。

```bash
python -m unittest discover -s tests -p test_reproduction.py -v

# 直接重放已保存的raw，不重新运行模型；输出目录必须是新的
python -m reproduce.replay --input /path/to/sample_directory --output runs/displacement_subset --post-policy pairwise --candidate-indices 0 1

# 正式比较须重放全部候选，不能只重跑旧胜出候选
python -m reproduce.replay --input /path/to/sample_directory --output runs/displacement_all --post-policy pairwise
```

`diagnostic.json`记录实际重放的候选编号、独立核验指标、最高有效候选和是否重放了全部保存候选。`candidate_####.json`记录逐轮指标、回退次数及被拒绝放大位移的交叠面积。新结果不可覆盖旧实验。

当前重放入口串行执行；已有的多进程后处理可以继续调用更新后的`enhance_pairwise`，函数参数和返回主体兼容。此修复没有实现或声称实现作者的C++并行后端。
