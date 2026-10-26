# NR-IQA 实验工作区

此目录用于通用无参考图像质量评价实验。第一阶段使用 PyTorch、pyiqa，先做 TOPIQ-NR 推理，再比较 NIQE 和 MUSIQ。模型权重、图片和实验输出尚未下载或生成。

## 本机工具链（2026-10-25）

- 项目解释器：`.venv\Scripts\python.exe`，Python 3.12.14。
- PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130；已在 RTX 5070 Laptop GPU 上验证 CUDA 张量运算。
- pyiqa 0.1.16；`topiq_nr`、`niqe`、`musiq` 均在模型列表中。
- SciPy、pandas、Matplotlib 已安装。完整版本见 `requirements-gpu-lock.txt`。
- VS Code 的工作区解释器设为 `.venv`。

在此目录下运行 Python 时，明确使用：

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

GPU 版 PyTorch 来自官方 CUDA 13.0 包索引。本机安装的 CUDA Toolkit 12.4 与 PyTorch 包内的 CUDA 13.0 运行组件是不同的版本；运行这些预编译包不需要把 Toolkit 改成 13.0。

## 两台电脑同步

GitHub 仓库地址为 `https://github.com/Jingjing0819Zhan/NR-IQA.git`。代码、配置、说明和小型结果摘要适合同步；`.venv`、本机 Python、缓存、原始图片、模型权重及大型实验输出由 `.gitignore` 排除。另一台电脑应建立自己的环境；不要复制本机 `.venv`，也不要将本机 GPU 锁定文件直接作为无 GPU 机器的安装命令。

本机使用 GitHub no-reply 邮箱记录初始提交；以后可在 Git 中配置你希望显示的姓名和邮箱。
