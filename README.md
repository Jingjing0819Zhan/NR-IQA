# NR-IQA 实验工作区

当前目标：亲手跑通 RALI 预训练模型的单图质量评分（E000），然后开展固定测试子集评估（E001）。当前阶段不训练模型。

候选研究问题：快速模型在哪些图片上不可靠？推理模型能否以有限额外计算纠正其中部分错误？该问题尚待实验验证。

## 开始前读什么

1. `notes/handoff.md`：当前状态和下一步。
2. `experiments/E000/plan.md`：第一次运行的目标和成功标准。
3. `docs/RALI实验方案与逐步复现指南.md`：完整参考步骤；其中示例命令需按实际环境调整。
4. `docs/E000操作清单.md`：以本项目脚本为准的操作入口。

## 项目结构

```text
configs/              实验配置，不放权重
src/                  后续公共代码
scripts/              自己的运行、评估、绘图入口
environment/          环境说明和依赖清单，不放实际虚拟环境
manifests/            样本名单和版本记录
experiments/E000/     单图实验计划、结论、轻量运行记录
experiments/E001/     小规模评估计划、结论、轻量运行记录
notes/                想法、决策、交接、论文笔记
reports/weekly/       周报和组会反馈
experiment_index.csv  实验总表
```

实际环境、图片、权重及大输出在运行主机另外准备，不通过 Git 同步。现有 `.gitignore` 忽略所有层级的 `runs/`，因此可同步的轻量记录使用 `records/`。

## E000 入口

由使用者先确认终端所在目录是本仓库，并选择已安装 RALI 依赖的解释器。下面仅以 GPU 主机的独立 RALI 环境为例：

```powershell
.\.venv-rali\Scripts\python.exe scripts\run_rali_single.py --check-inputs
.\.venv-rali\Scripts\python.exe scripts\run_rali_single.py
```

第一条只检查路径，不导入深度学习库，也不运行模型；第二条才执行推理。输入未准备好时检查失败是预期行为。

`configs/e000_rali_single.json` 中的相对路径以本仓库根目录解析。修改配置即可适配作者代码、权重和图片的位置。

## 已知环境记录

独显主机曾验证 Python 3.12.14、PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130 和 RTX 5070 Laptop GPU 的 CUDA 张量运算。记录见 `environment/gpu-host-20261025.md`，完整旧环境清单为根目录 `requirements-gpu-lock.txt`。

这些历史记录不代表每台拉取仓库的电脑都已安装同样环境，也不代表 RALI 已运行成功。RALI 独立环境尚需在执行主机上确认。

## 版本与结果

作者代码建议放在本仓库外的 `../NR-IQA-baselines/RALI/`，记录作者 commit。自己修改脚本或配置后先提交代码，再运行；运行结果包含执行时自己的 commit、作者 commit、配置、预测和时间。

每次实际运行在 `experiments/E000/records/` 新建唯一编号。里面只有小型 JSON 和日志，可检查后提交。大规模逐图结果及权重另行备份，记录位置。

仓库地址：https://github.com/Jingjing0819Zhan/NR-IQA.git
