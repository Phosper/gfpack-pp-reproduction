# GFPack++ NLP 课程作业

基于 [TimHsue/GFPack-pp](https://github.com/TimHsue/GFPack-pp) 的官方权重推理复现。这里保存课程作业的准备、构建和推理脚本，不包含官方源码副本、权重、数据和运行环境。论文见 [arXiv:2406.07579](https://arxiv.org/abs/2406.07579)。

## 获取源码与资源

```bash
git clone https://github.com/TimHsue/GFPack-pp.git upstream
git -C upstream checkout ff75815e24ec50cd7fa8b2abdec1acfe0b55be95
python3 prepare_assets.py
```

资源从官方提供的下载地址获取，准备脚本校验文件并生成本地来源清单。请先查看官方仓库的使用条款；本仓库不重新分发官方权重或数据。

## Linux GPU 环境与运行

需要可用的 NVIDIA GPU、CUDA 版 PyTorch、Python 和支持 OpenMP 的 C++ 编译器。已运行的环境为 Python 3.11、PyTorch 2.6.0+cu124、PyG 2.6.1，与作者环境不同。

`BASE_PYTHON` 应指向已安装 CUDA 版 PyTorch 及官方环境所需依赖的 Python。脚本创建继承该环境依赖的独立 venv，额外安装脚本列出的依赖；它不是从空环境一键安装全部依赖。

```bash
BASE_PYTHON=/path/to/python bash prepare_server.sh
bash run_server.sh smoke
bash run_server.sh full
```

`smoke` 生成 1 个候选，`full` 生成 512 个候选；二者都只评估固定样本 0、128 步、seed 3407。`full` 不代表完整测试集评测。输出写入新的 `results/` 子目录。

## 当前进度

2026-10-21 已在 A800 单卡完成官方权重推理，未开始训练。固定样本 0、48 个零件、512 个候选：原始官方有效 0/512，增强后 10/512，最佳利用率 75.4713%，入口耗时约 17.52 秒。PyTorch 最大已分配显存约 0.553 GiB。

以上为此前运行记录，本仓库整理期间未重新运行 GPU 推理。最佳结果按官方交叠面积阈值判为有效，但未通过 1e-9 面积容差下的严格零交叠检查。完整 128 例评测、论文指标和训练复现待完成。

## 队友协作

克隆本仓库后，按上述步骤分别获取官方源码和资源。每项修改使用独立分支，通过 Pull Request 合并到 `main`。不要提交账号、服务器地址、私钥、虚拟环境、权重或完整结果包；共享实验时记录源码版本、环境、随机种子、命令和指标口径。
