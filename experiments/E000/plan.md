# E000：RALI 预训练模型单图检查

状态：准备完成，尚未执行模型。

## 目的

确认作者代码、配套权重、图片读取和模型调用可以正确衔接。该阶段不计算 SRCC/PLCC，不训练模型。

## 输入

- 配置：`configs/e000_rali_single.json`。
- 作者代码：`https://github.com/xuanyuzhang21/RALI`。
- 权重：`https://huggingface.co/ByteDance/Q-Insight/tree/main/RALI`。
- 图片：作者 `assets/demo_image.png`，实际存在性需检查。

## 操作

先按 `docs/E000操作清单.md` 检查输入，再执行 `scripts/run_rali_single.py`。

## 成功标准

1. 作者代码、PCA、原型和 checkpoint 正确加载，没有退回默认初始化。
2. 实际运行设备符合配置，CUDA 不可用时不能静默当成 GPU 结果。
3. 同一输入的重复输出为有限数值，记录重复差异。
4. 保存自己的代码版本、作者版本、配置、日志、预测和初步运行时间。

## 不得据此得出的结论

- 单图分数不能证明和人类相关。
- 单图计时不能代表正式吞吐量。
- demo 成功不等于论文完整复现。

## 下一步

E000 通过后再准备 E001 的数据与固定测试名单。
