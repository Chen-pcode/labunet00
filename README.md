# Skin Mamba Lab

以官方 **UltraLight VM-UNet** 为 baseline 的独立训练、消融和跨数据集评估项目。原下载仓库和 `data` 文件夹均不修改。默认参数量 49,457；正式实验使用 Kaggle 单张 T4、256×256 输入。

**先读数据审计：ISIC2018 训练/验证与 ISIC2017 测试重合 532/600，去重后仅 68 张。全量测试不能当作严格跨域结果。** 项目仍按要求输出三个完整测试集，另外输出排除已见图像后的子集结果。PH² 全部 200 张只用于测试。

## 大体框架

`固定 train/val/test 文件夹 → 配对和重复审计 → 仅源域 train 训练 → 源域 val 选 best.pt → 三个 test 集公共评估 → CSV/JSON/预测掩码 → 多种子汇总`

- `skinmamba/models/`：作者网络适配；官方 CUDA Mamba 和仅用于小规模测试的真实递推 reference backend。
- `skinmamba/data.py`：图像/掩码配对、成对增强、ID+解码 RGB 哈希审计。
- `skinmamba/engine.py`：训练、验证、完整断点续训，优化器/调度器/AMP/RNG/历史均保存。
- `skinmamba/metrics.py`、`profiling.py`：所有模型共用的分割指标与复杂度/速度统计。
- `skinmamba/evaluation.py`：三个测试域、全量/排重子集、逐图结果和掩码导出。
- `configs/ablations/`：机制、执行路径、宽度、bridge、损失的独立开关。
- `scripts/`、`Kaggle_Quickstart.ipynb`：Kaggle 环境安装、GPU 核验和实验矩阵。

## 数据与划分

```text
data/
  isic2017/{train,val,test}/{images,masks}/
  isic2018/{train,val,test}/{images,masks}/
  ph2/test/{images,masks}/
```

ISIC 使用 `ISIC_xxx.jpg → ISIC_xxx_segmentation.png`；PH² 图像和掩码同名。掩码使用最近邻缩放并二值化。图片已经为256²，HD95 单位是该分辨率的像素。

| 训练来源 | 训练 | 原始验证→实际验证 | ISIC2017全量/排重测试 | ISIC2018测试 | PH²测试 |
|---|---:|---:|---:|---:|---:|
| ISIC2017 | 2000 | 150→149 | 600 / 597 | 1000 | 200 |
| ISIC2018 | 2594 | 100→100 | 600 / 68 | 1000 | 200 |

ISIC2017 排重测试 597 张仍有 2 组测试内重复，共595个不同 RGB；当前不擅自改变用户提供测试集的权重。排重指排除与源域训练/验证相同 ID 或 RGB 的图像，不保证患者级独立或近重复排除。完整报告见 [数据评估协议](reports/DATA_EVALUATION_PROTOCOL.md)。验证重复默认只在加载时排除，设置 `data.validation_overlap_policy=error` 可要求遇到重复就终止。

## Kaggle 运行

可直接从 GitHub 获取本项目：

```bash
git clone https://github.com/Chen-pcode/labunet00.git
cd labunet00
```

仓库包含代码、配置、测试与说明。数据集、checkpoint及生成的运行报告由本地或Kaggle实验产生。普通CPU测试环境可先安装PyTorch，再执行 `python -m pip install -r requirements.txt`；Kaggle使用下面的专用安装脚本，保留已有CUDA版PyTorch。

1. 将本仓库作为代码 Dataset 上传 Kaggle，将原 `data` 作为数据 Dataset；Notebook 打开 **GPU T4** 和安装时所需 Internet。也可在Kaggle中将仓库克隆到 `/kaggle/working/skin_mamba_lab`，再从该目录运行以下命令。
2. 打开 `Kaggle_Quickstart.ipynb`，修改两个输入路径；它会复制代码到 `/kaggle/working/skin_mamba_lab`。
3. 安装并核验 CUDA 扩展后再训练。CUDA 不可用时会报错，不会偷偷改用 CPU。

以下命令均在本项目根目录执行，`/kaggle/input/your-data/data` 换成实际目录：

```bash
python scripts/install_kaggle.py
python scripts/verify_cuda.py --precision fp32
python -m skinmamba audit --data-root /kaggle/input/your-data/data
python -m skinmamba train --config configs/baseline_isic2018.yaml --data-root /kaggle/input/your-data/data --run-dir /kaggle/working/runs/isic2018_baseline_s42
python -m skinmamba evaluate --checkpoint /kaggle/working/runs/isic2018_baseline_s42/best.pt --data-root /kaggle/input/your-data/data --save-predictions
```

`python -m skinmamba run ...` 合并完整训练与最终三域测试；调参筛选阶段使用 `train`，避免反复查看测试结果选方案。

切换 ISIC2017 使用 `configs/baseline_isic2017.yaml`。切换候选机制使用 `configs/ablations/geometry.yaml`。任意字段可以 `--set key=value` 覆盖，例如 `--set seed=43`、`--set model.sample_ratio=0.5`。默认沿用作者 batch=8、FP32；T4 若显存不足，可将**同一比较组全部**改为 batch=4，不自动改变批量。

断点续训（保持原配置、250 epoch 计划及数据内容不变）：

```bash
python -m skinmamba train --config configs/baseline_isic2018.yaml --data-root /kaggle/input/your-data/data --run-dir /kaggle/working/runs/isic2018_baseline_s42 --resume /kaggle/working/runs/isic2018_baseline_s42/last.pt
```

每个完整 epoch 保存一次。Kaggle 会话结束前将 `runs` 保存为 Notebook Output；新会话可从 `/kaggle/input/.../last.pt` 续训到新的可写 `--run-dir`。`last.pt` 内同时保留历史最佳权重，换目录也不会丢失 best。只加载自己信任的 checkpoint。

## 实验设计

详细设计见 [EXPERIMENTS.md](EXPERIMENTS.md)。先对 baseline 跑通，再做验证集筛选，最后冻结方案，使用 42/43/44 三个种子做正式实验。默认250 epoch、AdamW(lr=0.001, wd=0.01)、CosineAnnealingLR(T_max=50, eta_min=1e-5)、BCE+Dice，与下载的官方配置对应。T_max=50 在250 epoch中会再次上升，**不是单次250 epoch衰减**，这里保留原设置。

图像采用作者等价的逐图 min-max 到0–255（全局z-score后再逐图min-max会抵消）；增强保持两个独立50%概率的 rot90+flip、20–79度最近邻旋转。验证损失改用样本加权平均，二值掩码读取方式、固定用户划分和排重协议也与作者npy准备流程有差别，因此不能直接声称复现了论文表格数值。

```bash
# 默认只打印命令；加 --execute 真正串行训练，不同时占用T4。
python scripts/run_matrix.py --data-root /kaggle/input/your-data/data --output-root /kaggle/working/runs --sources isic2018 --seeds 42
# 最终冻结配置后，两个来源、三个种子、指定方法训练+最终测试：
python scripts/run_matrix.py --data-root /kaggle/input/your-data/data --output-root /kaggle/working/runs --sources isic2017 isic2018 --seeds 42 43 44 --variants baseline geometry --execute --evaluate
python -m skinmamba aggregate --root /kaggle/working/runs --output /kaggle/working/runs/aggregate.csv
```

短筛选可统一加 `--set training.epochs=50 --set training.t_max=50`；它是探索协议，不能和250 epoch正式结果混在同一比较表。不要先跑全部组合，先验证主机制是否有稳定收益。

## 公共指标口径

所有比例范围0–1，主表为逐图平均，同时输出 `global_*` 像素汇总指标。

| 字段 | 定义 |
|---|---|
| params | 唯一参数元素总数，包含不可训练参数 |
| flops | 一次 batch=1 前向的**核心算术估算**；2 FLOPs/MAC，含 Mamba projections、depthwise conv、selective scan |
| size_mb | FP32参数张量占用 / 2²⁰，即MiB；不含优化器、buffer、序列化开销 |
| fps | 同步实测、batch=1、30次预热/100次计时；仅模型前向，不含加载、传输、外部sigmoid；记录硬件及精度 |
| dice、f1 | 前景 Dice，相同定义下二者数值相同 |
| iou | 前景交并比 |
| miou | 前景IoU和背景IoU的平均；与原仓库名为miou的前景IoU分开 |
| accuracy | (TP+TN)/总像素 |
| sensitivity / specificity | TP/(TP+FN)、TN/(TN+FP)，类别在GT不存在时定义为1 |
| hd95 | 两个方向表面距离合并后的95分位数；4邻域内表面；像素单位 |

FLOPs 不是硬件指令数，不包括norm/通用激活/采样/插值等；`flops_scope`、`flops_complete=false` 与估计值一同输出，不能当作完整模型总FLOPs。未识别计算模块返回空值及诊断，不填虚假总量。采样方法的实际开销用 T4 延迟/FPS 判断。

HD95：两张空mask→0，只有一张为空→∞；主字段 `hd95` 是有限样本均值，**必须同时报告** `hd95_finite_count`、`hd95_failed_count`；`hd95_strict` 保留一例失败即∞的严格均值，不能隐藏失败。IoU/Dice类别双方为空视作完全匹配。测试集存在1张全前景标注，specificity不存在背景时的约定已明确。

额外提供 p50/p95 延迟、CUDA峰值显存、逐图结果及三种子均值/样本标准差。分辨率、精度、threshold=0.5和batch=1在速度比较中保持一致；全量/排重子集不混合。

外部模型也可使用：

```python
from skinmamba.evaluation import evaluate_model
from skinmamba.profiling import profile_model
rows, scores = evaluate_model(model, test_loader, device="cuda", output_kind="logits")
cost = profile_model(model, input_shape=(1, 3, 256, 256), device="cuda", precision="fp32")
```

模型须返回 B×1×H×W 张量；概率输出用 `output_kind="probabilities"`；tuple/dict输出先写显式adapter。外部SSM算子需要注册/暴露计数规则，不能只用Conv钩子漏掉scan。

## 结果文件与验证

每次运行保存 `config.yaml / environment.json / manifest.json / data_audit.json / validation_protocol.json / history.csv / last.pt / best.pt / weights.pt`。评估产生 `evaluation/summary.csv / results.json / profile.json / *_per_image.csv`，可选 `predictions/`。

本地验证命令：`python -m pytest -q`。CUDA reference 数值对照在无GPU环境跳过；Kaggle 用 `scripts/verify_cuda.py` 强制执行实际GPU检查。CPU reference只用于小尺寸正确性测试，不适合256²完整训练，也不能代表T4速度。已执行验证结果见 [VERIFICATION.md](VERIFICATION.md)。

来源及机制边界见 [REFERENCES_MODEL.md](REFERENCES_MODEL.md)、[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)。新增采样/间距校准是待验证原型，不是Serp-Mamba复现，不宣称严格旋转/尺度等变或已确认论文创新。
