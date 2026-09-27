# CCLAS 实验方案（待验证）

## 研究假设与代码边界

旧方案 `sampled_geometry` 使用左密右疏的固定坐标。已有单种子结果表明它在 ISIC2017 源域提高 Dice，却降低 PH² Dice；ISIC2018 上还出现两张空预测，且 T4 FPS 低于 baseline。因此不能把旧方案的源域收益直接解释为稳定的跨域创新。

新主实验 `--main` 对应 `model.variant=cclas`，只修改 UltraLight VM-UNet 的 `encoder4` PVM 层。一个 depthwise 3×3 + pointwise 1×1 评分头从特征预测采样密度；固定每行 24/32 个 token；每行逆 CDF 取样并强制包含左右端点；覆盖约束把相邻间距限制在平均间距的 0.25–2.5 倍；SSM 的几何 delta 归一化后限制在 0.5–1.5。训练与推理都不读取测试 mask。该组合是待检验假设，不以现有代码测试宣称精度、速度或文献新颖性。

实现延续 [UltraLight VM-UNet 作者代码](https://github.com/wurenkai/UltraLight-VM-UNet)的拓扑，以及 [Mamba 官方实现](https://github.com/state-spaces/mamba)的参数与 selective scan。固定非均匀坐标实验仍保留为 `--ablation geometry`，便于与先前结果比较。代码来源和具体差异见 [REFERENCES_MODEL.md](REFERENCES_MODEL.md)。

## 公平比较矩阵

所有实验使用同一来源、官方 train/val/test、256×256、batch 8、FP32、AdamW、BCE+Dice、验证集最小 loss 选 best、固定阈值 0.5、相同 epoch 与 seed。不要用 test 或 PH² 选超参数。每个模型默认只输出对应 ISIC 来源 test 与 PH² test；`--include-other-isic` 仅用于补充分析，不能进入主表或跨三域平均选模。

| 名称 | 开关 | 机制目的 |
|---|---|---|
| baseline | `--baseline` | 作者网络，完整 PVM 序列 |
| uniform_sampling | `--ablation uniform_sampling` | 固定 75% token、均匀坐标、delta=1 |
| adaptive_sampling | `--ablation adaptive_sampling` | 加学习式逆 CDF 坐标，delta=1 |
| adaptive_geometry | `--ablation adaptive_geometry` | 加原始局部几何 delta |
| adaptive_coverage | `--ablation adaptive_coverage` | 加覆盖约束，几何 delta 不限幅 |
| CCLAS | `--main` | 覆盖约束 + 有界几何 delta |
| old_geometry | `--ablation geometry` | 旧固定左密右疏对照，单独列出 |

`adaptive_coverage` 与 CCLAS 用相同采样坐标，二者只差 delta 限幅；`adaptive_geometry` 与 `adaptive_coverage` 只差空间覆盖约束。`adaptive_sampling` 与 `adaptive_geometry` 只差 delta 几何校准。`uniform_sampling` 与 `adaptive_sampling` 只差评分头/动态坐标，但前者没有额外评分头参数，报告参数与延迟时必须说明这一点。

## Kaggle 命令

先在单张 T4 上运行 `python scripts/install_kaggle.py` 和 `python scripts/verify_cuda.py --precision fp32`。若选 AMP，整组实验一致地改为 `--set training.precision=amp_fp16`，并重新跑 GPU 核验；不要把 FP32 与 AMP 的 FPS 混表。

```python
!python -m skinmamba train --main --epoch 1 --seed 2026 --config configs/baseline_isic2018.yaml --data-root /kaggle/input/datasets/nero20260505/data-20260925 --run-dir /kaggle/working/runs/cclas_smoke_s2026
```

1 epoch 只检查安装、前向、反向和保存，不纳入论文表格。

阶段 1：ISIC2018、seed 42、统一 80 epoch，**只用验证集**筛选是否值得继续；由于默认学习率调度 `T_max=50`，保留同一调度配置比较。命令默认仅打印计划，确认后加 `--execute`。

```bash
python scripts/run_matrix.py --data-root /kaggle/input/datasets/nero20260505/data-20260925 --output-root /kaggle/working/runs/screen80 --sources isic2018 --seeds 42 --epoch 80 --variants baseline uniform_sampling adaptive_sampling adaptive_geometry adaptive_coverage main --execute
```

阶段 2：只有在验证集 Dice/loss 呈一致优势、训练稳定且 T4 峰值显存可接受时，冻结超参数与模型选择规则。正式 250 epoch、seed 42/43/44，先做 ISIC2018 六组，再做 ISIC2017 baseline、CCLAS 和关键消融。正式测试一次性由 `--evaluate` 执行；不要因 PH² 分数回头改模型。若研究资源允许，ISIC2017 也补全六组消融。

```bash
python scripts/run_matrix.py --data-root /kaggle/input/datasets/nero20260505/data-20260925 --output-root /kaggle/working/runs/final250 --sources isic2018 --seeds 42 43 44 --epoch 250 --variants baseline uniform_sampling adaptive_sampling adaptive_geometry adaptive_coverage main --execute --evaluate
python scripts/run_matrix.py --data-root /kaggle/input/datasets/nero20260505/data-20260925 --output-root /kaggle/working/runs/final250 --sources isic2017 --seeds 42 43 44 --epoch 250 --variants baseline adaptive_sampling adaptive_coverage main --execute --evaluate
python -m skinmamba aggregate --root /kaggle/working/runs/final250 --output /kaggle/working/runs/final250/aggregate.csv
```

单独运行的等价示例：

```bash
python -m skinmamba run --main --epoch 250 --seed 42 --config configs/baseline_isic2018.yaml --data-root /kaggle/input/datasets/nero20260505/data-20260925 --run-dir /kaggle/working/runs/isic2018_cclas_s42
python -m skinmamba run --ablation adaptive_coverage --epoch 250 --seed 42 --config configs/baseline_isic2017.yaml --data-root /kaggle/input/datasets/nero20260505/data-20260925 --run-dir /kaggle/working/runs/isic2017_coverage_s42
```

## 报告与判定

主表按训练来源分开：ISIC2017→ISIC2017/PH²，ISIC2018→ISIC2018/PH²。每个 test 域单独列 Dice、IoU、mIoU、HD95、HD95 失败数和空预测数；补充 accuracy、sensitivity、specificity、F1。计算成本列 T4 实测 FPS、p50/p95 延迟、峰值显存、参数、FP32 参数 MiB、受限定义的 FLOPs。`summary.csv` 为逐图宏平均；`*_per_image.csv` 附带目标面积比例和是否接触图像边界，可做预先定义的小病灶/边界病例分析。HD95 有空预测时，有限值平均必须与失败数及 `hd95_strict=inf` 同时报。不可只报告有限值平均。

优先看源域 Dice 是否稳定提高，PH² 是否维持或提高，空预测是否消失，再看 T4 FPS 和参数。单种子优势只是探索信号；三种子均值与标准差也不能代替独立外部验证。若 CCLAS 仅在 ISIC2018 胜出而 ISIC2017/PH² 下降，就把结论收窄到受支持的来源和场景。若 T4 FPS 低于 baseline，明确报告代价，不把 token/FLOPs 减少写成加速。
