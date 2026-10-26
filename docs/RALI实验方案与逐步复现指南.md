# RALI 实验方案与逐步复现指南

更新日期：2026-10-26

当前阶段：第一次深度学习实验，从预训练模型推理开始。

本文包含研究方案、操作步骤和完成标准。由你亲手执行；遇到问题时记录输出，再逐项排查。本文中的命令尚未在你的 GPU 电脑上实际运行验证，不代表已经完成复现。

## 一、研究目标与边界

### 1. 当前研究范围

通用无参考图像质量评价（NR-IQA）：输入一张图片，在没有高质量参考图的情况下预测人的感知质量。

当前研究对象是清晰度、噪声、压缩、曝光等因素对质量判断的影响，不评价图片是否符合生成提示词。

### 2. 研究方向与第一步

候选方向：在有限计算预算下，判断哪些图片值得调用推理式质量评价模型。

初步问题：RALI 已能快速评分，它在哪些图片上与人的判断存在稳定分歧？Q-Insight 等推理模型是否有能力纠正其中一部分错误？

第一步只复现 RALI 的评分与评估。后续方向是待验证假设，不应先假定大模型一定更准确，也不预先承诺形成论文。

### 3. 几个术语

| 术语 | 本项目中的含义 |
|---|---|
| baseline | 用来建立起点、比较后续方法的已有模型 |
| 预训练权重 | 作者已训练好的模型参数 |
| 推理 | 加载权重后给图片评分，参数不更新 |
| 推理式 IQA | 生成质量分析文字并输出评分的质量评价方法 |
| 快速模型 | 本方案中的 RALI；运行时不需要生成分析文字 |
| 推理专家 | 后续可能接入的 Q-Insight 等模型，是角色称呼，不代表绝对正确 |
| MOS | 多位观察者的主观质量评分经过规定处理后的平均分 |
| SRCC | 衡量预测与人类评分的排序一致性 |
| PLCC | 衡量预测与人类评分的线性相关性 |

## 二、为什么选择 RALI

RALI 对应《Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment》，作者提供代码、权重与评估入口。其模型将图像编码与原型评分结合，运行时无需加载语言大模型生成解释。

研究起点：先建立可靠的 RALI 结果，再讨论它与推理模型的互补性。

| 方法 | 在当前方案中的角色 | 什么时候运行 |
|---|---|---|
| RALI | 主 baseline | 现在 |
| TOPIQ-NR、MUSIQ | 其他快速模型对照 | RALI 流程跑通后 |
| Q-Insight | 候选推理专家 | 老师认可方向且具备运行条件后 |
| VisualQuality-R1 | 补充推理方法 | 后续检验结论的普适性时 |

TOPIQ 不再是当前选题的主研究对象。之前准备的代码管理、数据与指标知识仍然有用。

## 三、阶段安排与交付物

| 阶段 | 你要完成什么 | 完成标准 |
|---|---|---|
| E000 单图检查 | 加载 RALI，对作者示例图片评分 | 权重完整加载、输出有限数值、保存日志 |
| E001 小规模评估 | 固定约 100 张 KonIQ 测试样本，保存预测与指标 | 样本无遗漏、路径正确、评分字段清楚 |
| E002 完整协议复核 | 按所用 checkpoint 对应的完整测试协议评估 | 记录与论文设置的差异，并解释结果差距 |
| E003 错误诊断 | 查看模型与人类判断分歧较大的图片 | 形成可重复观察，不靠挑选个例下结论 |
| E004 互补性预研 | 对同批开发图片运行 RALI 与 Q-Insight | 判断是否有纠错收益，决定是否继续 |

E000—E001 是近期目标。E002 之后再逐步推进，不必在第一次实验时完成全部任务。

## 四、准备代码、权重与输出位置

以下路径是操作示例。若你的实际路径不同，统一替换后再执行。

```text
D:/NR-IQA/                              自己的 Git 项目
    docs/                              方案和阶段总结
    scripts/                           自己的运行、抽样、分析脚本
    manifests/                         固定样本名单及运行信息
    reports/                           可提交到 Git 的小型结果摘要
    data/Data-DeQA-Score/               图片与作者元数据
    checkpoints/RALI/                  作者预训练权重
    outputs/E000/run01/                第一次单图输出
    outputs/E001/run01/                第一次小规模评估输出

D:/NR-IQA-baselines/RALI/               作者代码，独立于自己的仓库
```

将作者仓库放在自己的 Git 项目外面，避免一开始就处理嵌套 Git 仓库。记录它的远程地址和 commit，即可追溯来源。

### 第 1 步：获取作者代码

如果尚未克隆：

```powershell
New-Item -ItemType Directory -Force -Path 'D:\NR-IQA-baselines' | Out-Null
git clone https://github.com/xuanyuzhang21/RALI.git D:\NR-IQA-baselines\RALI
git -C D:\NR-IQA-baselines\RALI rev-parse HEAD
```

最后一条输出是代码版本号。记录到本次实验信息中。若仓库已存在，先查看状态，不重复克隆或覆盖。

### 第 2 步：获取正确的预训练权重

进入作者权重页面：

[ByteDance/Q-Insight 的 RALI 子目录](https://huggingface.co/ByteDance/Q-Insight/tree/main/RALI)

需要整个 RALI 子目录中的配套文件，而不是只下载一个 `ckpt.pt`，也不要误下载 Q-Insight 的其他模型。按下面结构放好：

```text
D:/NR-IQA/checkpoints/RALI/
    ckpt.pt
    pca.pkl
    basis.npz
    best/
        config.json
        model.safetensors
        preprocessor_config.json
        tokenizer_config.json
        以及该目录其余配套文件
```

检查时确认 `best` 文件夹中是实际权重文件，不是 Git LFS 指针文本。模型文件有数 GB 规模，下载进度与文件大小应合理。

记录：下载页面、日期、版本或 revision、文件大小。可以用 PowerShell `Get-FileHash -Algorithm SHA256` 保存关键文件的校验值。

### 第 3 步：确认运行入口

后面的 PowerShell 命令使用如下变量，须在每个新终端中重新设置。`$raliPython` 必须替换为已经配置好 RALI 依赖的解释器；下面是独立 RALI 环境的示例位置。

```powershell
$raliPython = 'D:\NR-IQA\.venv-rali\Scripts\python.exe'
$raliRepo = 'D:\NR-IQA-baselines\RALI'
```

这些变量只用于明确运行位置，不会创建或安装环境。

## 五、E000：亲手跑通单张图片

### 第 4 步：阅读最少量代码

先打开作者的 `demo_rali_score.py`，只找以下几个位置：

1. `ClipScorer`：如何加载模型和 checkpoint。
2. `score_image`：输入图片后怎样返回分数。
3. 文件末尾的示例：权重、图片路径在哪里。

第一轮理解“图片从哪里进入、权重从哪里加载、分数从哪里返回”即可，暂时不要求读懂所有矩阵运算。

### 第 5 步：新建自己的单图入口

作者示例末尾使用 `../../checkpoints/...`，与 README 的根目录摆放方式不一致。先不改网络结构，自己建立一个仅处理路径与调用的入口。

在自己的项目中新建 `scripts/run_rali_single.py`，亲手输入以下代码：

```python
from pathlib import Path
import json
import math
import sys
import time

import torch
from PIL import Image

project = Path(__file__).resolve().parents[1]
upstream = Path("D:/NR-IQA-baselines/RALI")
sys.path.insert(0, str(upstream))
from demo_rali_score import ClipScorer

weights = project / "checkpoints" / "RALI"
image_path = upstream / "assets" / "demo_image.png"
required = [weights / "ckpt.pt", weights / "pca.pkl", weights / "basis.npz",
            weights / "best" / "config.json", image_path]
for path in required:
    if not path.is_file():
        raise FileNotFoundError(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scorer = ClipScorer(
    ckpt_path=str(weights / "ckpt.pt"),
    clip_model=str(weights / "best"),
    pca_path=str(weights / "pca.pkl"),
    basis_path=str(weights / "basis.npz"),
    device=device,
)
with Image.open(image_path) as raw:
    image = raw.convert("RGB")

# 预热；不把第一次运行计入稳定推理时间。
scorer.score_image(image)
if device.type == "cuda":
    torch.cuda.synchronize()
started = time.perf_counter()
score = scorer.score_image(image)
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - started
if not math.isfinite(score):
    raise ValueError("模型输出不是有限数值")

result = {"image": str(image_path), "score": score,
          "device": str(device), "seconds_after_warmup": elapsed}
destination = project / "outputs" / "E000" / "run01"
destination.mkdir(parents=True, exist_ok=True)
output = destination / "prediction.json"
if output.exists():
    raise FileExistsError("已有 run01 结果，请更换运行编号后再运行")
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
```

这段入口只调用作者模型，不重新实现网络。预热后的单图时间只是初步检查，不能当成最终性能基准；也不包含模型加载和读图开销。

运行：

```powershell
Set-Location D:\NR-IQA
& $raliPython scripts\run_rali_single.py
```

### 第 6 步：判断是否真正跑通

检查：

- 打印的设备是否符合预期。
- checkpoint 加载没有缺失或维度不匹配。
- PCA、原型文件读取没有失败后退回默认值的警告。
- 输出一个有限分数，并产生 `prediction.json`。
- 同一模型、同一输入重复执行时，分数在合理数值误差内一致；重复运行需更换输出编号。

不能把“打印出一个数字”单独作为成功标准。作者代码存在部分加载失败后继续运行的分支，必须检查日志。

**本阶段交付：** 单图输入、预测 JSON、完整终端输出、代码 commit、权重来源。

## 六、E001：固定约 100 张图片进行定量评估

### 第 7 步：准备与作者评估入口一致的数据

优先使用 RALI README 指向的 [Data-DeQA-Score 数据](https://huggingface.co/datasets/zhiyuanyou/Data-DeQA-Score/tree/main)。先看 [KONIQ 元数据目录](https://huggingface.co/datasets/zhiyuanyou/Data-DeQA-Score/tree/main/KONIQ/metas)，其中有 `test_koniq_2k.json`、`train_koniq_7k.json`、`split.json` 等。

图片下载按该数据页面的实际文件组织进行，只取需要的数据集，不下载整个合集。第一次为保持协议清楚，先沿用作者数据与预处理，暂不混用其他分辨率包。

预计结构：

```text
D:/NR-IQA/data/Data-DeQA-Score/
    KONIQ/
        images/
        metas/
            test_koniq_2k.json
            ...
```

核对 checkpoint 的训练信息与该测试名单；不能只因文件名含 `test` 就认定它一定未被模型见过。

### 第 8 步：先检查元数据，不急着评分

作者评估脚本使用每条记录的 `image` 和 `gt_score_norm` 字段，实际图片路径为 `image_root / image`。先打开元数据，查看第一条记录和字段含义，再决定 `image_root`。

例如：若 `image` 已包含 `KONIQ/images/文件名`，根路径应是 `Data-DeQA-Score`；不要再把根路径设成 `KONIQ/images`，否则会重复拼接。

注意：

- `gt_score_norm` 的实际范围与转换方式要从数据说明、记录中核实，不凭字段名猜测。
- MAE 只有在预测值与真值处于可比较的尺度时才有意义。
- MOS、DMOS、归一化 MOS 不是可以随意互换的字段。
- 不能自行把每次抽样的最小最大值映射到固定范围，之后再宣称与原论文协议一致。

### 第 9 步：固定抽样名单

在 `scripts/make_rali_subset.py` 中亲手输入：

```python
import json
import random
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "data/Data-DeQA-Score/KONIQ/metas/test_koniq_2k.json"
records = json.loads(source.read_text(encoding="utf-8"))
if not isinstance(records, list):
    raise TypeError("先核对作者元数据结构，再调整读取方式")
if not all("image" in r and "gt_score_norm" in r for r in records):
    raise ValueError("存在缺失字段，先检查，不要静默跳过")
if len({r["image"] for r in records}) != len(records):
    raise ValueError("存在重复图片，请先核对")

chosen = random.Random(42).sample(records, min(100, len(records)))
destination = root / "manifests/E001_koniq100_seed42.json"
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    raise FileExistsError("名单已存在；直接复用它，或新建版本")
destination.write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved samples:", len(chosen))
print(destination)
```

运行：

```powershell
& $raliPython scripts\make_rali_subset.py
```

在运行模型前，逐条确认抽样图片存在且可解码；缺图应处理原因，不能悄悄减少样本数。

该 100 张子集用于复现流程和探索问题。以后若据此调参，它就成为开发数据，应另留未参与选择的测试样本作最终验证。

### 第 10 步：运行作者的批量评估脚本

下面命令假设第 8 步已确认 `image_root` 应为 `Data-DeQA-Score`。若实际记录不同，先改根路径。

```powershell
Set-Location D:\NR-IQA
New-Item -ItemType Directory -Force -Path outputs\E001\run01 | Out-Null
& $raliPython "$raliRepo\eval_json.py" `
  --ckpt D:\NR-IQA\checkpoints\RALI\ckpt.pt `
  --clip_model D:\NR-IQA\checkpoints\RALI\best `
  --pca_path D:\NR-IQA\checkpoints\RALI\pca.pkl `
  --basis_path D:\NR-IQA\checkpoints\RALI\basis.npz `
  --test_json D:\NR-IQA\manifests\E001_koniq100_seed42.json `
  --image_root D:\NR-IQA\data\Data-DeQA-Score `
  --out_json D:\NR-IQA\outputs\E001\run01\predictions.json `
  --batch_size 1 `
  --num_workers 0 `
  --prec fp32 2>&1 | Tee-Object -FilePath outputs\E001\run01\console.log
```

说明：

- PowerShell 的换行符是反引号，后面不要有空格；也可以把整条命令写在同一行。
- 从 batch size 1 开始，先保证正确，再讨论加速。
- `num_workers=0` 先减少多进程读取带来的排错负担。
- 第一次保持 fp32，不同时更改多个运行因素。
- 复跑使用 `run02` 等新目录，避免覆盖已有记录。

### 第 11 步：核对指标和逐图结果

作者脚本输出预测 JSON，并打印 MAE、PLCC、SRCC。当前代码中的 PLCC 直接由预测与真值计算，是原始 PLCC；不要把它写成经过非线性映射的 PLCC。

核对以下项目：

1. 输入与输出的图片名单完全一致，数量一致，没有重复或遗漏。
2. 所有预测为有限数值；若预测几乎恒定，要回查模型加载。
3. 预测与真值通过图片标识关联，不能仅假定两个文件行序一致。
4. 使用 SciPy 正式计算相关性，确认没有因为缺库等问题进入备用计算。
5. 独立用保存的逐图预测复算指标，与终端结果核对。

作者脚本会先预测保存结果，再进行一次评估计算，相当于执行两遍模型。第一次可保留这个行为以减少改动；不能把整个脚本耗时简单除以样本数，宣称是单次推理延迟。

**本阶段交付：** 固定名单、预测 JSON/CSV、指标摘要、终端日志、运行信息。

## 七、E002—E003：从跑通到分析问题

### 完整协议复核

小子集结果不应要求和论文全测试集数值完全一致。下一步对齐：代码版本、权重版本、测试名单、图片分辨率、预处理、指标定义，然后再扩大评估。

预训练推理复现不等于从头训练复现。当前只声明自己实际完成的部分。

### 错误诊断

先建立逐图表：图片标识、人类评分、预测、分歧大小、图片路径。若分数尺度尚未对齐，可以先分析排序分歧，不直接按原始分数差排列。

查看代表性图片，区分：

- 路径、通道、缩放或评分方向等实现问题。
- 主观评分本身存在较大分歧的图片。
- 模糊、噪声、曝光、压缩等可能稳定重复的失败模式。

保留表现正常的对照图片。单张反例用于提出假设，不能证明机制。

自制局部退化图片只能用作额外诊断。原图 MOS 不适用于修改后的图片；需要主观标注才能声称与人类质量判断一致。

## 八、后续研究：什么时候值得调用推理模型

本节是预研设计，不是已经成立的结论。

1. 在固定的开发数据上获得 RALI、Q-Insight 的逐图预测。
2. 在开发数据确定分数转换方法，统一比较尺度。
3. 分析大模型纠正小模型和反而改坏的情况。
4. 以真实评分事后选择更合适模型，估计理想选择的收益；这只是诊断参考，不能作为可部署系统性能。
5. 若有足够互补性，再训练小型选择器，预测调用专家的收益。

选择器部署时只能用图片和快速模型可产生的信息，不能读取真实 MOS，也不能先运行大模型再声称省去了该次调用。

必须比较：全 RALI、全 Q-Insight、随机调用、按分歧或置信度调用、固定融合、候选选择方法。所有额外模型和多视图的开销都要计入。

最终关心：在相同预算下能否改善质量相关性或错误率；平均延迟、调用比例、峰值显存怎样变化；结论是否能跨数据集成立。

若理想选择收益也很小，或简单规则已达到相同效果，就不急着做复杂方法。先向老师汇报结果，决定调整问题。

## 九、实验记录与组会材料

每次运行保存：

```text
实验编号：
本次问题：
作者代码地址与 commit：
自己代码 commit / 未提交差异：
权重来源、revision、校验值：
数据来源、版本、划分、名单：
完整运行命令：
实际软件版本与硬件：
开始/结束时间、状态：
指标、样本数、缺失数：
逐图预测与日志位置：
观察：
解释假设：
尚未排除的问题：
下一步：
```

关键摘要放在 `reports/`，例如 `reports/E001_run01.md`。若当前项目忽略了整个输出目录，Git 不会自动保存那里生成的摘要，需要显式安排另一个受版本管理的位置。大文件另行备份，记录位置与校验值。

第一次组会有以下材料就足够具体：

- 一句话说明复现对象与问题。
- 一张小规模结果表，明确样本数和评估协议。
- 3—5 张代表性图片，同时包含正常与异常案例。
- 当前能支持什么结论、不能支持什么结论。
- 下周准备验证的问题，以及需要老师指导的选择。

## 十、常见问题的处理顺序

| 现象 | 先检查什么 |
|---|---|
| 找不到模块 | 实际使用的解释器路径和该环境的依赖 |
| 找不到权重 | 绝对路径、文件是否下载完整、是否多嵌套一级 RALI 目录 |
| PCA 加载警告 | joblib/scikit-learn/NumPy 版本和文件完整性，先停下核对 |
| strict checkpoint 加载失败 | 代码、模型配置和权重版本是否匹配，不直接改成非严格加载 |
| 找不到图片 | `image_root` 与 JSON 的 `image` 字段怎样拼接 |
| 显存不足 | 先用 batch size 1，再判断其他措施 |
| 输出几乎恒定 | 权重、PCA、原型是否正确加载 |
| 相关性异常或为负 | 标签方向、图片对应关系、字段与输入预处理 |
| 比论文数值低 | 先对齐协议，再判断模型问题 |

## 十一、资料入口

- [RALI 论文](https://arxiv.org/abs/2510.11369)
- [RALI 作者仓库与 README](https://github.com/xuanyuzhang21/RALI)
- [单图实现](https://github.com/xuanyuzhang21/RALI/blob/main/demo_rali_score.py)
- [批量评估实现](https://github.com/xuanyuzhang21/RALI/blob/main/eval_json.py)
- [作者依赖脚本](https://github.com/xuanyuzhang21/RALI/blob/main/setup.sh)
- [RALI 权重](https://huggingface.co/ByteDance/Q-Insight/tree/main/RALI)
- [作者指向的数据与元数据](https://huggingface.co/datasets/zhiyuanyou/Data-DeQA-Score/tree/main)
- [Q-Insight 作者仓库](https://github.com/bytedance/Q-Insight)

核对状态：已检查作者代码的入口、参数、读取字段和指标计算逻辑；尚未下载权重、运行 RALI 或验证你的 GPU 环境与作者依赖的兼容性。
