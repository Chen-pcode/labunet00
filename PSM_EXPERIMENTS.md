# PSM v1：跨尺度持久潜记忆与源域状态一致性实验

实验代码与方案版本：2026-09-28。当前状态：软件原型，尚无新方案的真实数据训练结果；不是已证实有效、已完成查重或可保证发表的方法。baseline 为本项目对作者 UltraLight-VM-UNet 的实现。

## 1. 针对什么问题

皮肤病灶分割需要同时识别病灶整体范围和局部边缘。病灶大小、形状及内部纹理不同，背景皮肤与病灶可能低对比；不同采集设备、照明和肤色分布也可能改变模型使用的外观线索。我们的**待检验假设**是：跨尺度保存压缩的病灶相关信息，并约束其对适度外观变化的敏感性，可能改善分割和外部数据泛化。

原模型已经通过 U 形跳连、SAB 和 CAB 交换多尺度信息，不能声称它没有跨尺度语义。新方案检验的是：**额外的显式跨尺度记忆是否有增量价值，以及约束这个记忆是否比约束普通特征更有用。**

这里的“状态校准”落实为训练时的状态一致性正则。**没有单独的推理校准映射，也没有 PH2 统计自适应。**“校准前后”的主要实验是相同结构、相同双视图监督、同 seed 的 `psm_memory_aug` 与 `psm_main` 两个训练模型。`queries` 和 `states` 分别是写入候选与更新后记忆，不能把它们直接当成校准前后。

本版保留原 PVM、SAB、CAB，新增记忆写入与读取。它不把 Mamba 内部 selective-scan 的隐藏状态跨层传递；不能写成“改进 Mamba 内核”或“替换 PVM/SAB/CAB”。若论文最终需要该主张，须另行设计和验证。

## 2. 方案与理论边界

编码器第 4、5 层归一化/池化后的输出，以及第 6 层激活后的输出，记为 `f4,f5,f6`。每层先对空间维度平均池化，再投影到默认 16 维的共同潜空间：

```text
q_l = tanh(Linear_l(GAP(f_l)))
g_l = sigmoid(Gate_l(concat(q_l, m_(l-1))))
m_l = g_l * m_(l-1) + (1-g_l) * q_l
m_3 = 0
```

`m4 → m5 → m6` 为单张图像的一次前向中的持久传递，不跨图片、batch、伪视图或测试域保存。各层投影可学习，但维度相同并不证明语义已经对齐。门控更新逐坐标是凸组合，且 `q_l` 经过 tanh，因此从零初态可推出 `|m_l[j]|≤1`。这个有界性不等于对输入扰动严格收缩，也不保证泛化。

解码器 `decoder1,2,3` 分别读取 `m6,m5,m4`，在对应跳连相加之前进行 FiLM 式调制：

```text
(gamma_l, beta_l) = 0.1 * tanh(Read_l(m_l))
decoder_feature = decoder_feature * (1 + gamma_l) + beta_l
```

读取初始化较小，不做额外归一化来强制所有图片具有相同状态均值/方差。旧模型的所有基础权重初始化不变。`independent` 消融保留完全相同的参数和读取路径，只在每个尺度写入前将前一状态置零。

源域训练样本经过原有成对几何增强后，产生原视图 `x` 和保持几何不变的外观视图 `x'`；两者用同一个掩码监督。外观扰动是亮度、对比度、RGB 通道增益和 gamma，不使用 PH2 样本或 PH2 风格模板，也不添加模糊或形变。默认幅度分别为 `0.10,0.20,0.15,0.20`；这是初始超参数，不是已验证最优值。图像原输入范围为 `[0,255]` 时，扰动内部转换到 `[0,1]` 再转换回去，不额外做独立 min-max。

损失对三个记忆阶段分别计算，再平均：

```text
L_seg = (BCE_Dice(logits(x), y) + BCE_Dice(logits(x'), y)) / 2
L_pair = mean((m(x) - m(x'))²)                         # 保留图片配对信息
L_mean = mean((mean_batch(m(x)) - mean_batch(m(x')))²)
L_cov  = ||Cov_batch(m(x)) - Cov_batch(m(x'))||_F² / (4*d²)
L_var  = 两视图 mean(relu(0.05 - sqrt(diag(Cov) + 1e-4))) 的平均
L_dom  = L_pair + L_mean + L_cov + 0.1 * L_var
L      = L_seg + lambda(epoch) * L_dom
```

默认最终 `lambda=0.05`，前 5 个 epoch 线性升温：第一个 epoch 为 0.01，第五个为 0.05。协方差沿 **batch 图片维度** 计算，使用 `N-1` 分母；尾部 batch 仅一张图时不估计协方差/方差项，但保留配对项和均值项。小 batch 的协方差秩低、噪声较大，所以主方案同时使用配对一致性。`L_var` 只是减轻坍塌的辅助项，不能保证不会坍塌。`history.csv` 单独记录各项和实际 `dom_weight`，便于识别某个损失过小、过大或不起作用。

本版是单源域泛化，不是访问目标域训练数据的域适应。均值/协方差下降不保证分类分布差异界下降，更不保证 Dice 上升。状态也可能包含有用的病灶颜色信息，因此必须保留分割监督并检查源域性能。

## 3. 实验组与消融

所有组共用原来的数据划分、BCE+soft Dice、优化器、checkpoint 选择和评估器。表中的“单视图”仍包含 baseline 原有几何增强。

| CLI 名称 | 记忆 | 外观双视图 | 状态约束 | 检验目的 |
|---|---|---|---|---|
| `psm_baseline` | 无 | 否 | 无 | 原架构参考 |
| `psm_baseline_aug` | 无 | 是 | 无 | 单独增加外观增强/双视图监督 |
| `psm_memory` | 跨尺度 | 否 | 无 | 单视图下记忆增量，与 baseline 比 |
| `psm_memory_aug` | 跨尺度 | 是 | 无 | 与主实验同结构同增强的零 `L_dom` 对照 |
| `psm_independent_aug` | 各层独立 | 是 | 无 | 与 memory_aug 同参数量，去掉跨尺度传递 |
| `psm_independent_dom` | 各层独立 | 是 | 有 | 没有持久传递时，状态约束是否仍有效 |
| `psm_main` | 跨尺度 | 是 | 完整 | 主实验 |
| `psm_feature_dom` | 跨尺度 | 是 | 约束 `q_l` | 相同损失作用于当前尺度摘要，而非持久状态 |
| `psm_moments_only` | 跨尺度 | 是 | 去掉 `L_pair` | 均值/协方差约束是否足够；保留方差防退化项 |
| `psm_no_variance` | 跨尺度 | 是 | 去掉 `L_var` | 防退化项的作用 |

其中四组 `independent_aug / independent_dom / memory_aug / main` 构成“跨尺度传递 × 状态约束”的 2×2 对照。可按同一域、同一 seed 计算 Dice 交互量：

```text
interaction = (Dice(main) - Dice(memory_aug))
            - (Dice(independent_dom) - Dice(independent_aug))
```

正交互量支持二者具有额外协同，但仍需多个 seed 和不确定性分析。不能只挑最好的一次运行。原 baseline 与主实验训练计算量不同：双视图会增加训练时间；正式报告训练时长，推理均为单视图。主表最低建议前述 2×2 四组 + baseline + baseline_aug；其余消融用于解释机制。

## 4. 数据与选择协议

目录结构沿用现有工程：

```text
data/
  isic2017/{train,val,test}/{images,masks}/
  isic2018/{train,val,test}/{images,masks}/
  ph2/test/{images,masks}/                 # 全部 200 张
```

- ISIC2018 训练、验证 → ISIC2018 test 和 PH2 test，分别报告。
- ISIC2017 训练、验证 → ISIC2017 test 和 PH2 test，分别报告。
- 不计算三域平均 Dice，不输出 `clean` 子集性能表。原有数据完整性/精确重复审计保留；审计读取文件不参与梯度、风格估计或模型选择。
- 默认 256²、batch 8、AdamW、lr=0.001、FP32、BCE+soft Dice，按源域 **val_loss 最小**选 checkpoint，阈值固定 0.5。不要为主实验单独改用 val_dice 选择。
- 本套配置默认 100 epochs、T_max=100。对本套实验指定 `--epoch N` 时，T_max 默认同步成 N；显式 `--set training.t_max=...` 优先。旧实验配置的日程语义不变。
- PH2 不参与 lambda、增强强度、记忆维度、epoch 或 checkpoint 选择。之前已经看过的数据集测试结果不能重新称为从未接触的盲测；如计划不断根据 PH2 修改设计，应增加独立外部验证。

## 5. Kaggle 安装与运行

先在 Kaggle 下载仓库，已有仓库时更新到最新版本：

```bash
git clone https://github.com/Chen-pcode/labunet00.git /kaggle/working/labunet00
# 已有仓库时使用：
git -C /kaggle/working/labunet00 pull --ff-only
```

以下 shell 命令均在 `/kaggle/working/labunet00` 目录运行。Notebook 中先 `%cd /kaggle/working/labunet00`；单行 shell 命令前加 `!`，多行命令可放在 `%%bash` 单元格。

已有可用环境可跳过安装。新环境使用已有安装脚本，保留其版本报告：

```bash
python scripts/install_kaggle.py
python scripts/verify_cuda.py --precision fp32
python scripts/verify_psm.py --device cuda --precision fp32
# 如果计划 AMP，再单独检查；检查成功不等于已复现准确率
python scripts/verify_psm.py --device cuda --precision amp_fp16
```

`verify_psm.py` 用合成数据检查前向、梯度、优化器更新和复杂度计数，未使用任何测试集。当前本机验证的是 CPU reference backend；CUDA/AMP 需要在实际 Kaggle 环境完成。

### 单独运行 baseline、主实验和消融

```bash
python -m skinmamba train --baseline --epoch 100 --seed 42 \
  --config configs/psm/isic2018.yaml \
  --data-root /kaggle/input/datasets/nero20260505/data-20260925 \
  --run-dir /kaggle/working/runs/psm/isic2018/psm_baseline/seed_42

python -m skinmamba train --psm-main --epoch 100 --seed 42 \
  --config configs/psm/isic2018.yaml \
  --data-root /kaggle/input/datasets/nero20260505/data-20260925 \
  --run-dir /kaggle/working/runs/psm/isic2018/psm_main/seed_42

python -m skinmamba train --ablation psm_memory_aug --epoch 100 --seed 42 \
  --config configs/psm/isic2018.yaml \
  --data-root /kaggle/input/datasets/nero20260505/data-20260925 \
  --run-dir /kaggle/working/runs/psm/isic2018/psm_memory_aug/seed_42
```

其他消融只改 `--ablation` 和独立输出目录。换 ISIC2017 时使用 `configs/psm/isic2017.yaml`。**新主实验使用 `--psm-main`，历史 `--main` 仍然表示 reconstruction。**

需要调整参数时在命令末尾加，例如 `--set model.memory_dim=32 --set domain.weight=0.02`。这些不是已经推荐的最优值；控制变量对照也要对应更新。preset 会重置它管理的消融开关，显式 `--set` 在 preset 后生效；运行目录内的 `config.yaml` 是最终实际配置。

### 先跑最小筛选

默认只训练并用源域验证，禁止 `screen --evaluate`：

```bash
python scripts/run_psm_matrix.py --stage screen --sources isic2018 \
  --seeds 42 --epoch 50 \
  --data-root /kaggle/input/datasets/nero20260505/data-20260925 \
  --output-root /kaggle/working/runs/psm_screen --execute

python scripts/summarize_psm.py --root /kaggle/working/runs/psm_screen \
  --output /kaggle/working/runs/psm_screen_summary
```

会跑 `baseline / baseline_aug / memory_aug / main` 四组。`validation.csv` 报告**实际选中 checkpoint 的** val_loss/val_dice，不混用最低 loss 和最高 Dice 所在的不同 epoch。50 epoch 只是早期信号；若模型仍在改善，应在预先统一的更长预算下复核，不能断言已经收敛。筛选预算与正式预算不同，使用新目录从头训练；不要把 50-epoch cosine 运行强行续成 100-epoch 正式运行。

窗口会实时显示 `[Experiment 1/4]`、`[Epoch 1/50]`、训练/验证 batch 进度、loss、验证 Dice 和最佳 epoch。数据审计、构建模型、保存 checkpoint 也有阶段提示；数据审计完成前尚未进入 epoch。默认首个 batch、每 20 个 batch、最后一个 batch 打印一次；若距离上次日志超过 30 秒，也在下一个 batch 完成时打印。需要更频繁输出可加 `--set training.log_interval=5`。

矩阵脚本用 Python 无缓冲子进程并逐行转发日志，适配 Kaggle `!python` 单元格。没有 `--execute` 时会明确显示 `DRY RUN`，此时只列出命令，不会训练，也不会出现 epoch 进度。已启动的旧进程不会因 `git pull` 自动加载新日志代码；本次运行仍可从各 run 目录的 `history.csv` 查看已完成 epoch。

### 固定方案后的正式实验

下面命令跑建议的六组、两个源域、三个 seed，并对每组做 source+PH2 测试。只有冻结配置后才运行：

```bash
python scripts/run_psm_matrix.py --stage formal \
  --variants psm_baseline psm_baseline_aug psm_independent_aug psm_independent_dom psm_memory_aug psm_main \
  --sources isic2018 isic2017 --seeds 42 2026 3407 --epoch 100 \
  --data-root /kaggle/input/datasets/nero20260505/data-20260925 \
  --output-root /kaggle/working/runs/psm_formal --execute --evaluate
```

如需表中全部十组，去掉 `--variants ...`。额外四组也可单独指定执行，避免重复已完成运行。`--stage ablation` 默认列出补充六组；没有 `--execute` 时所有矩阵命令只打印命令，不启动训练。已有 `last.pt` 的目录不会被无提示覆盖，矩阵遇到失败会停止。

暂停续训要用相同总 epoch、seed、配置，只增加：

```bash
python -m skinmamba train --psm-main --epoch 100 --seed 42 \
  --config configs/psm/isic2018.yaml \
  --data-root /kaggle/input/datasets/nero20260505/data-20260925 \
  --run-dir /kaggle/working/runs/psm/isic2018/psm_main/seed_42 \
  --resume /kaggle/working/runs/psm/isic2018/psm_main/seed_42/last.pt
```

## 6. 统一评估指标

所有模型复用 `skinmamba/evaluation.py`、`skinmamba/metrics.py`、`skinmamba/profiling.py`，不为主实验单独定义有利指标。

| 指标 | 定义/注意事项 |
|---|---|
| params | 模型参数总数 |
| flops | 每张图核心算术估计；1 MAC=2 FLOPs，含 Mamba scan、记忆线性层与显式读写算术；不是完整算子图总 FLOPs |
| size_mb | FP32 参数字节/2²⁰，实际单位 MiB；不含优化器、buffer 与序列化开销 |
| fps | 真实设备 batch=1、预热后同步测量模型前向；不含读盘、预处理、传输与模型外阈值操作 |
| dice / f1 | 前景逐图 Dice，二值定义下二者相等 |
| iou | 前景逐图 IoU |
| miou | 每张图前景与背景 IoU 的平均，再跨图平均 |
| accuracy | 像素准确率，逐图再平均 |
| sensitivity / specificity | 前景召回率/背景真负率，逐图再平均 |
| hd95 | 双向表面最近距离合并后的 95% 分位数，单位为缩放到评估分辨率后的像素 |

另外保存 latency p50/p95、GPU peak memory、空预测数、HD95 失败数与 strict HD95。两掩码都空时 HD95=0，只有一边空时为无穷；汇总 `hd95` 是有限样本均值，必须连同失败数量报告。没有真实物理间距时不写毫米。`profile.json` 给出算术遗漏范围，不能把估算 FLOPs 当成实测速度。

```bash
python -m skinmamba evaluate \
  --checkpoint /kaggle/working/runs/psm/isic2018/psm_main/seed_42/best.pt \
  --data-root /kaggle/input/datasets/nero20260505/data-20260925 \
  --save-predictions

python -m skinmamba aggregate --root /kaggle/working/runs/psm_formal \
  --output /kaggle/working/runs/psm_formal/aggregate.csv
```

默认测试本源 ISIC 与 PH2，分开输出 `summary.csv` 和各域逐图结果；多 seed 汇总分组包含训练、数据、domain-loss、阈值和设备计时协议。不要把 CPU reference 的 FPS 写成 GPU FPS。

## 7. 机制分析：校准前后状态距离

每张图片在每个阶段产生一个 d 维状态。诊断统计沿完整图片集合计算，**不把像素、尺度或 batch 混成独立样本**。

输出包括：

- `mu_l2 = ||mu_S-mu_T||_2` 与 `cov_fro = ||Sigma_S-Sigma_T||_F`。
- 同一模型每个尺度使用源域 **train 原图** 的协方差迹 `v` 固定尺度：`mu_l2/sqrt(v)`、`cov_fro/v`。源域、伪域、PH2 共用该尺度，不分别标准化。
- RBF-MMD²：在同样的尺度下，用源域 train 原图最多 512 张的距离中位数固定带宽，不用 PH2 拟合。使用 U-statistic；有限样本可能为负，保留原值，不截断到零。源—伪域使用配对修正，排除同一图片的跨视图对角项。
- 各域状态方差迹、平均通道标准差、平均向量范数、协方差有效秩，检查退化。
- 均值/协方差距离的 500 次图片级 bootstrap 区间；有对照时对两模型复用相同抽样，计算 `main-control` 差值区间。MMD 当前只有点估计，不虚构其置信区间。
- source、pseudo、PH2 的分割指标；可选冻结权重的 `no_carry` / `no_read` 干预。

源—伪域 bootstrap 保留同图配对；源—PH2 分别对两个集合抽样，两模型之间保持相同抽样。区间条件于已经训练好的模型与训练域参考统计，不包含训练 seed、患者相关性或参考统计估计的不确定性。没有患者 ID 时不能宣称患者级置信区间。多个训练 seed 需要分别运行、分别报告。

### 筛选期间：只有 source val 与其伪视图

```bash
python -m skinmamba diagnose-states \
  --checkpoint /kaggle/working/runs/psm_screen/isic2018/psm_main/seed_42/best.pt \
  --control /kaggle/working/runs/psm_screen/isic2018/psm_memory_aug/seed_42/best.pt \
  --data-root /kaggle/input/datasets/nero20260505/data-20260925 \
  --bootstrap 500 --interventions
```

没有 `--final` 就不提取 PH2 状态或计算 PH2 指标。数据清单仍会执行原有完整性审计。

### 冻结方案后的最终分析：source test、其伪视图与全部 PH2

```bash
python -m skinmamba diagnose-states --final \
  --checkpoint /kaggle/working/runs/psm_formal/isic2018/psm_main/seed_42/best.pt \
  --control /kaggle/working/runs/psm_formal/isic2018/psm_memory_aug/seed_42/best.pt \
  --data-root /kaggle/input/datasets/nero20260505/data-20260925 \
  --bootstrap 500 --interventions
```

每个 seed、源域都运行一次。诊断时图像 ID 决定伪视图随机数，因此 batch size 改变不会改变伪域内容；训练时使用独立的逐 epoch 风格随机数生成器，避免不同模型的参数初始化消耗随机数后改变增强序列。程序要求主实验和对照具有相同结构、seed、数据、训练预算、双视图增强和阈值；对照必须为零 `L_dom`，主实验为正 `L_dom`。两者使用同一模型选择规则，最佳 epoch 可以不同。最终命令检查 PH2 恰有 200 张，防止误用小子集。诊断输出目录若已有文件，使用新的 `--output-dir`，防止重复分析混入过期结果。

最终目录下有：

```text
state_distances.csv   # 各模型、各尺度、各目标的绝对距离和区间
state_changes.csv    # main-control 的距离差；负数表示下降
state_health.csv     # 方差、范数、有效秩
segmentation.csv     # 相应域和冻结干预下的分割表现
protocol.json        # checkpoint hash、实际配置、样本协议和统计定义
main/*.npz           # 原始状态与图片 ID，可离线重新核查
control/*.npz
```

`summarize_psm.py` 也会整理 `state_changes_all.csv`、`state_health_all.csv`，保留 source、target、seed、phase，不跨域平均。

这不是独立的因果证明：不同训练模型的潜坐标可以变化，状态距离也包含病灶组成差异。原始/归一化距离、状态健康度和任务表现要联合判断。`no_carry/no_read` 是冻结干预，可能引入分布偏移，不能替代重新训练的消融。

## 8. 如何判断是否继续

先看源域验证：主实验应相对同结构 `memory_aug` 显示稳定收益，不能只比训练预算不同的原 baseline。检查是否只是 `baseline_aug` 就达到了相同水平，再检查跨尺度传递与独立摘要的对照。多 seed 阶段按各测试域报告均值/标准差和逐 seed 配对差异。

源—伪域与源—PH2 的状态距离均下降、未出现明显退化、PH2 分割改善且源域保持，才能较强支持所提机制。若距离下降但分割下降，应优先考虑有用信息被压缩；若精度改善但距离没有下降，可以报告精度结果，但不能继续声称这个状态距离机制得到支持。不能因为某个尺度好看就事后只报告该尺度。

本项目没有证明这项组合的独创性。简单“GAP+门控+FiLM+CORAL”容易被解释为已有组件组合；真正的论文贡献需要由必要性对照、跨尺度×约束交互和真实泛化结果支撑，并另做有针对性的最新文献查重。

## 9. 代码位置与参考来源

- `skinmamba/models/persistent.py`：跨尺度写入、读取、独立状态对照和冻结干预。
- `skinmamba/persistent_experiments.py`：所有新实验开关，避免混淆历史 `main`。
- `skinmamba/domain.py`：外观扰动、两视图损失和配对/矩/方差项。
- `skinmamba/state_diagnostics.py`：源域统计参考、距离、MMD、bootstrap、干预分析。
- `configs/psm/`：两个数据源的基础配置；`scripts/run_psm_matrix.py`：批量实验。
- `tests/test_persistent.py`：新方案正确性与端到端小数据验证。

来源与改动边界：

1. [UltraLight-VM-UNet 作者项目](https://github.com/wurenkai/UltraLight-VM-UNet)：实际骨干来自已保存的作者源码；出处与 MIT 许可见 `REFERENCES_MODEL.md`、`SOURCE_PROVENANCE.json`、`third_party/UltraLight_VM_UNet.original.py`。未重新实现一套同名替代骨干。
2. [Mamba](https://arxiv.org/abs/2312.00752)、[官方实现](https://github.com/state-spaces/mamba)：CUDA 使用官方 `mamba_ssm.Mamba`。新记忆位于其外部，不假设官方 last-state 返回值支持跨层反传。
3. [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)：提供特征仿射条件化的设计依据。本版有界读取为自行适配的公式实现，不是论文模型复现。
4. [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/abs/1607.01719)：参考协方差匹配及 `4*d²` 归一化。本版在源域与伪视图间训练，不能冒称使用了无监督目标域适应。
5. [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906)、[作者代码](https://github.com/facebookresearch/vicreg)：参考方差下限防退化的思路。本版阈值、权重与任务均不同，不是 VICReg 复现。

新记忆与组合损失是本项目实验实现，没有把自行设计的部分标成他人官方代码。以上理论来源支持设计动机，不构成该组合新颖性或有效性的证明。

## 10. 本地验证范围

2026-09-28：完整 pytest 为 **111 passed, 1 skipped**（新方案 17 项，原工程 94 项通过，CUDA 检查因本机无 CUDA 跳过）。新检查包括原骨干初始化与 no_read 前向一致、所有开关的梯度和 FLOPs、实际跨尺度梯度依赖、跨样本/跨调用状态重置、外观变换输入范围和逐图可复现、矩损失数值、归一化距离对整体缩放的不变性、完整训练/续训/评估/诊断/汇总小数据闭环。

另运行 `verify_psm.py --device cpu --image-size 32 --variants psm_baseline psm_main`：两组前向/反向/优化器更新均通过。默认原骨干参数量为 **49,457**，主模型及相同结构记忆消融为 **56,929**，增加 7,472 个参数。32² 合成检查的 FLOPs/FPS 不用于正式 256² GPU 比较。

这些是软件检查，不是真实分割性能、创新性或 GPU 加速证据。正式运行请保留 `config.yaml`、`environment.json`、`history.csv`、`best.pt`、`training_status.json`、`evaluation/` 与诊断目录。
