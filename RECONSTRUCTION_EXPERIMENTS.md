# 状态读回与空间重建：完整实验说明

## 当前版本与研究范围

当前 `--main` 是 **reconstruction**：替换六个 PVM 层以及 SAB/CAB 的功能。此前 CCLAS 保留为 `--ablation cclas`，旧 checkpoint 按其保存的配置恢复，不会被新 `--main` 改写。`--baseline` 仍是原 UltraLight VM-UNet。此版本没有模型精度提升或首创性结论。

研究假说：全局状态读回和跨层空间重建产生的差异，能否比完整特征或固定高通特征更有效地指导皮肤病灶的细节补偿？三组机制服务于同一假说，不能把借来的基础算子分别宣称为原创。

代码位置：`skinmamba/models/reconstruction.py`。训练、数据、损失、checkpoint、评估沿用原项目。新架构使用原生 PyTorch 运算，CPU/CUDA 均可运行；不以 T4 适配作为设计条件。原 baseline 仍需要官方 CUDA Mamba 扩展，reference backend 只供小规模软件检查。

## 从 GitHub 获取的原始实现

| 来源 | 固定提交 | 实际复用 |
|---|---|---|
| [EfficientViM](https://github.com/mlvlab/EfficientViM) | `304340cb9c339b61669250d058525c9cdadd5e93` | `HSMSSD` 的投影、空间 softmax、状态写入、隐藏状态混合、状态读回 |
| [UPLiFT](https://github.com/mwalmer-umd/UPLiFT) | `e58d213d79c125d5cceaa7af0fefb6a94677bf55` | `LocalAttender` 的 3×3 固定偏移、复制填充、邻域 softmax 和加权聚合 |

这两份仓库实际通过 Git 克隆。未修改的作者文件和 MIT 许可证放在 `third_party/EfficientViM/`、`third_party/UPLiFT/`；SHA-256 与提交记录见 `third_party/RECONSTRUCTION_SOURCES.json`。生产代码包含归属说明，完全不依赖运行时联网下载。

```bash
python scripts/fetch_upstream.py                 # 离线验证原始文件哈希
python scripts/fetch_upstream.py --download      # 可选：重新获取相同提交，不下载权重
```

通道交互参考 [Dual Cross-Attention](https://github.com/gorkemcanates/Dual-Cross-Attention) 的思想，代码是本项目的 decoder 条件化适配，**不是 DCA 官方模块的完整复现**。VSSD、SegMAN、RS-SSM 提供理论与对照依据，本次没有将其整个模型拼入网络。

## 已落实的计算与方案细化

### PVM 替换

设 `U` 为通道归一化特征，形状 B×C×N：

```text
A = softmax_spatial(dt + author_A)
H0 = U @ (A * B).transpose(-1,-2)
G = HSM(H0) @ C_read
U_hat = ChannelNorm(H0 @ C_read)
R = U - U_hat
L = Pointwise(GELU(Depthwise3x3(R)) * sigmoid(Pointwise(U_hat)))
Y = Project(ChannelNorm(U + G + beta * L))
```

`HSM(H0)` 完整保留作者 `h,z` 投影、`h*SiLU(z)+h*D` 与输出投影。`hsm_only` 不计算 U_hat/L；`hsm_local` 与 `readback` 使用完全相同参数和上下文，只将 L 的输入切换为 U 或 R。`hsm_highpass` 则用 `U - AvgPool3x3(U)`，边界采用复制填充。

具体适配：保留原 U-Net 六阶段和通道数；HSM-SSD 在每层全部通道上工作，替代原四组共享 Mamba；增加输出通道映射以匹配网络。它是 HSM-SSD **模块迁移**，不是整个 EfficientViM 分类网络复现。保留原项目初始化策略；作者核心的数值一致性测试在加载同一组权重后进行。

细节：作者 A 为每状态一个常数，加在沿空间 softmax 的 logits 上，理论上会被 softmax 平移不变性抵消。为了准确迁移，本实现保留这一参数化；不能把 A 的数值解释为有效的空间遗忘强度。H0 的读回没有重建监督，也不是正交投影，因此称“读回差异”，不把它预先认定为丢失的边界。

### SAB 替换

每次 decoder 融合时，用同层 encoder 与双线性上采样的 decoder 生成 guide，再用 LocalAttender 重建 decoder。核心采用作者 `num_connected=9, conv_res=False`。实现改为逐偏移累加，保留同样数学运算，支持矩形尺寸；没有重新训练作者的完整 foundation-model upsampler。

同层 encoder 与 decoder 此时已有相同通道数，使用**共享 ChannelNorm** 对齐再相减，得到 `R_l`。新增局部分支根据 `R_l` 补偿。相对研究草案，实际实现保留原有 encoder skip 加法，以减少初始训练中细节通路丢失的风险；这是明确的实现选择。完整特征对照保留相同 skip、归一化和分支容量，仅替换 R_l 的输入。

### CAB 替换与消融外壳

Q 来自归一化的 decoder，K/V 来自 R_l；计算 d×d 通道关系，再混合 R_l 的值。softmax 沿 key 通道，缩放为 `sqrt(N)*positive_temperature`。默认 d=C，温度初始为 1，beta/gamma 初始为 0.1；这些是首版实验设置，不是经测试得到的最优超参数。

全部新版本都在相同的逐层解码外壳中运行：

- 保留 SAB 时，仍用作者的 `s = SAB(E)*E, r = E+s`。
- 保留 CAB 时，仍用五级 r 的原 CAB 得到通道权重 c。
- 基础 skip 为 `c*r+s`；替换 SAB 时 s=0，替换 CAB 时 c=1。
- 两者都保留时，桥接计算与原 SC_Att_Bridge 完全一致，并有数值测试。
- 空间重建和通道补偿独立开关；替换某个模块不会改变 skip 接入层数或输出分辨率。

第一轮**没有新增损失**，继续 BCE+Dice，避免把架构效果与新监督混在一起。模型 forward 只接受图片，测试真值不参与门控、路由或特征补偿。

## 固定训练协议

默认新配置：256×256，batch=8，FP32，AdamW，lr=0.001，weight decay=0.01，300 epoch，seed=42，BCE+Dice=1:1，验证集 val_loss 最小的 `best.pt`。

**`training.t_max=50` 沿用旧实验**，不会跟随 `--epoch` 自动改变。这意味着 300 epoch 下余弦调度不是一次从头到尾的单周期。若要改成单周期，可给所有对照同时增加 `--set training.t_max=300` 并重新训练 baseline，不能与旧学习率曲线的结果混作受控对照。

数据目录保持：

```text
data/isic2017/{train,val,test}/{images,masks}
data/isic2018/{train,val,test}/{images,masks}
data/ph2/test/{images,masks}
```

训练只用所选源域 train，选 checkpoint 只看该源域 val；数据审计与原验证去重规则保留。最终 ISIC2018 模型只评估 ISIC2018 test、PH² 全部 200 张；ISIC2017 模型只评估 ISIC2017 test、PH²。不做三域平均，不生成 clean 子集成绩。另一 ISIC 测试域仍只能显式 `--include-other-isic` 打开。

## 实验矩阵

| 选择开关 | 编号/作用 | 核心局部输入 | 空间桥 | 通道桥 |
|---|---|---|---|---|
| `--baseline` | B0，原始基线 | 原 PVM | 原 SAB | 原 CAB |
| `--ablation hsm_only` | B1，核心直接迁移 | 无 | 原 SAB | 原 CAB |
| `--ablation hsm_local` | B2，普通局部分支 | 完整特征 | 原 SAB | 原 CAB |
| `--ablation readback` | A，核心创新候选 | 读回差异 | 原 SAB | 原 CAB |
| `--ablation readback_spatial` | A+S | 读回差异 | 引导重建+差异局部补偿 | 原 CAB |
| `--ablation readback_channel` | A+C | 读回差异 | 原 SAB | 条件通道交互 |
| `--main` / `--ablation reconstruction` | A+S+C | 读回差异 | 引导重建+差异局部补偿 | 条件通道交互 |
| `--ablation hsm_highpass` | 固定高通对照 | 固定高通 | 原 SAB | 原 CAB |
| `--ablation bridge_full` | 桥接差异必要性 | 读回差异 | 新桥，值改成完整 E | 新桥，K/V 改成完整 E |
| `--ablation all_full` | 同容量完整特征对照 | 完整特征 | 新桥，完整 E | 新桥，完整 E |
| `--ablation direct_transfer` | 直接移植组合 | 无 | LocalAttender，无差异局部分支 | 普通完整特征交互 |
| `--ablation no_core_compensation` | 去核心补偿 | 无 | 完整新空间桥 | 完整新通道桥 |
| `--ablation no_bridge_local` | 去桥接局部补偿 | 读回差异 | LocalAttender，无局部分支 | 差异通道交互 |

消融开关集中在 `skinmamba/experiments.py`，单独 YAML 在 `configs/ablations/`。高级设置示例：`--set model.state_dim=8`、`--set model.residual_init=0.05`、`--set model.channel_reduction=2`。所有覆盖值写入 checkpoint，不会在评估时从当前 YAML 猜测模型。

建议按顺序运行：先 B0/B1/B2/A；A 值得保留后运行 A+S/A+C/main；最后运行机制对照与多 seed。若 A 不优于 B2，读回差异的必要性没有被支持。若完整模型不优于单一模块，则按结果缩小最终方法。不要仅看更大的参数量带来的增益；`hsm_local`/`readback`、`all_full`/`main` 为同容量关键对照，另外可配置加宽 baseline 做容量参照。

## Kaggle 直接执行

从已更新的本仓库根目录执行；Notebook 中在 shell 命令前加 `!`。已有安装成功的环境可跳过安装。

```bash
python -m pip install -r requirements.txt
python scripts/fetch_upstream.py
python scripts/verify_reconstruction.py --device cuda --precision fp32
```

新架构不需要额外编译 CUDA scan。跑原 baseline 时继续沿用已安装的 Mamba；若为新环境，用 `python scripts/install_kaggle.py` 与 `python scripts/verify_cuda.py --precision fp32` 安装并验证。不要为了新模块替换已有 CUDA PyTorch。

一轮完整训练示例，`--epoch 1` 仅用于跑通：

```bash
python -m skinmamba train --main --epoch 300 --seed 42 --config configs/main_isic2018.yaml --data-root /kaggle/input/datasets/nero20260505/data-20260925 --run-dir /kaggle/working/runs/recon_v1/isic2018_main_s42

python -m skinmamba train --ablation readback --epoch 300 --seed 42 --config configs/main_isic2018.yaml --data-root /kaggle/input/datasets/nero20260505/data-20260925 --run-dir /kaggle/working/runs/recon_v1/isic2018_readback_s42

python -m skinmamba train --baseline --epoch 300 --seed 42 --config configs/main_isic2018.yaml --data-root /kaggle/input/datasets/nero20260505/data-20260925 --run-dir /kaggle/working/runs/recon_v1/isic2018_baseline_s42
```

更推荐矩阵脚本，一次串行运行所选实验；不加 `--execute` 时只打印命令：

```bash
# 阶段1：baseline / hsm_only / hsm_local / readback
python scripts/run_matrix.py --stage core --sources isic2018 --seeds 42 --epoch 300 --data-root /kaggle/input/datasets/nero20260505/data-20260925 --output-root /kaggle/working/runs/recon_v1 --execute

# 阶段2：已有 readback 不重复训练，运行三个新增组合
python scripts/run_matrix.py --variants readback_spatial readback_channel main --sources isic2018 --seeds 42 --epoch 300 --data-root /kaggle/input/datasets/nero20260505/data-20260925 --output-root /kaggle/working/runs/recon_v1 --execute

# 阶段3：机制对照；自行选择需要的 variants，勿与已存在 run 目录重复
python scripts/run_matrix.py --stage controls --sources isic2018 --seeds 42 --epoch 300 --data-root /kaggle/input/datasets/nero20260505/data-20260925 --output-root /kaggle/working/runs/recon_v1 --execute
```

`--sources isic2017` 切换训练集。最终使用至少 `42 2026 3407` 三个 seed；为已有 seed=42 的实验补 `--seeds 2026 3407` 即可。`--stage final` 默认 baseline/main；关键消融也需要多 seed，不能只给主模型重复实验。矩阵支持 `--device`、重复 `--set`，不会自动覆盖或重跑失败目录。

筛选阶段用 `train`。定稿后的最终训练可用 `run`，或矩阵加 `--evaluate`，结束后自动测试。单独评估、续训与汇总：

```bash
python -m skinmamba evaluate --checkpoint /kaggle/working/runs/recon_v1/isic2018_main_s42/best.pt --data-root /kaggle/input/datasets/nero20260505/data-20260925 --save-predictions

# 续训的 epoch 为原计划总数，而不是再增加多少 epoch；保留相同配置和 seed。
python -m skinmamba train --main --epoch 300 --seed 42 --config configs/main_isic2018.yaml --data-root /kaggle/input/datasets/nero20260505/data-20260925 --run-dir /kaggle/working/runs/recon_v1/isic2018_main_s42 --resume /kaggle/working/runs/recon_v1/isic2018_main_s42/last.pt

python -m skinmamba aggregate --root /kaggle/working/runs/recon_v1 --output /kaggle/working/runs/recon_v1/aggregate.csv
```

矩阵脚本目录格式为 `output-root/source/variant/seed_N/`，与手写示例的扁平目录不同。评估时使用实际生成的 best.pt 路径。

## 指标、机制诊断与产物

公共指标沿用 `skinmamba/metrics.py`、`profiling.py`、`evaluation.py`：params、flops、size_mb、fps、dice、iou、miou、accuracy、sensitivity、specificity、f1、hd95。二分类相同口径下 F1=Dice。HD95 空预测失败次数另行报告，不能只比较有限值均值。

新增 FLOPs hook 计入 HSM 的写/读矩阵乘、通道注意力两次矩阵乘、LocalAttender 加权聚合，子 Conv/Linear 只计一次。仍沿用 1 MAC=2 FLOPs 的**核心运算估计**，不伪称完整算子 FLOPs。size_mb 为 FP32 参数存储 MiB，不等于 checkpoint 文件大小。FPS 是指定硬件上同步计时的 batch=1 模型 forward，包含实际模块开销；CPU 检查不能代表 GPU 速度。

每次训练保存 resolved config、环境、数据审计、history.csv、last.pt、best.pt、weights.pt；评估保存 summary.csv、results.json、profile.json、逐图 CSV、可选预测图。汇总按 source→target 分开，并按模型和测量协议分组。

训练后可在源域验证集上检查假说：

```bash
python scripts/inspect_reconstruction.py --checkpoint /kaggle/working/runs/recon_v1/isic2018_main_s42/best.pt --data-root /kaggle/input/datasets/nero20260505/data-20260925 --output-dir /kaggle/working/runs/recon_v1/diagnostics --max-images 8
```

保存各层差异、补偿幅值、预测错误图、真值边界带，以及各区域的特征均值。PNG 独立缩放只用于显示，定量值在 maps.npz。差异图不应直接命名为“边界概率”或“置信度”；它是否与有效补偿相关需要对照验证。该命令只选源域 val，真值不传入模型。

## 软件验证与限制

```bash
python -m pytest -q
python scripts/verify_reconstruction.py --device cpu --precision fp32
python scripts/smoke_real_data.py --data-root ../data --image-size 256 --variants hsm_only hsm_local readback reconstruction
```

测试覆盖作者核心与 LocalAttender 的输出/梯度一致性、所有预设的矩形输入与反向传播、同容量控制、checkpoint 恢复、源域+PH² 指标输出、计数与诊断开关。GPU FP16 需另跑 `--device cuda --precision all`；原作者 snapshot parity 通过不等于新架构的训练精度验证。具体本机结果见 `RECONSTRUCTION_VERIFICATION.md`。
