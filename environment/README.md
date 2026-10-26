# 环境记录，不是虚拟环境

这里保存环境说明、依赖清单和安装命令的记录。实际 Python 和第三方库不放在这个文件夹中。

- `gpu-host-20261025.md`：已有独显主机的历史环境记录。
- `requirements-rali-extra.txt`：RALI 推理的附加依赖起点，不包含 PyTorch 安装方案，不是完整锁定环境。
- RALI 独立环境实际验证后，再导出 `requirements-rali-lock.txt`；同时记录 Python、GPU、驱动和 PyTorch 包来源。

每台机器独立建立环境。不要把 GPU 主机的完整锁定清单直接用于 CPU 机器。
