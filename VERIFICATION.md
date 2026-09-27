# 已执行的验证

日期：2026-09-27。本地环境为Windows、Python3.11、PyTorch2.14.0+cpu；测试环境独立放在项目外的 `.venv-skin-mamba`，不需要上传该环境到Kaggle。

## 自动化测试

命令：`python -m pytest -q`

当前版本结果：**94 passed, 1 skipped**。跳过的是需要真实 CUDA 及官方扩展的数值对照，并非失败项目被忽略。

覆盖：

- 原作者源文件与适配baseline的结构/state_dict键、初始化逐张量相等、原sigmoid输出与新logits.sigmoid对照。两边注入相同reference backend，不能替代官方CUDA验证。
- Mamba手算递推、Δ加bias/softplus/间距的顺序、实际非均匀坐标、插值复原、未采样残差、各控制组及有效梯度。
- 已知掩码的Dice/IoU/mIoU/HD95、空掩码/全前景与HD95失败统计。
- 参数与Mamba核心FLOPs计数、profiling状态恢复、设备错误处理及precision别名。
- 配对失败检测、不同文件ID但相同RGB的跨split重复识别、增强可复现性。
- 真实训练引擎在小型合成数据上跑2 epoch；连续运行和1+1 epoch续训最终模型逐张量完全相同。
- 最佳checkpoint对本来源与PH²完整测试集的公共评估、可选另一ISIC测试域、CSV输出和全部请求字段。
- 聚合隔离不同阈值、数据fingerprint及GPU；同checkpoint同协议重复评估不算额外种子。
- 新命令参数实际传入训练：baseline/main/消融、epoch和seed；互斥与非法输入检查。
- 实际执行了 baseline、旧 CCLAS（`--ablation cclas`）和 reconstruction（`--main`）的 CLI 单轮检查，保存的 checkpoint 确认 epoch/seed 配置，并完成来源与 PH² full 评估；新架构还检查了 HSM-SSD、LocalAttender、状态读回和条件通道交互的梯度及 FLOPs 报告。
- 即使测试图与源训练图重复，仍完整评估所有测试样本且不产生clean行；旧报告中的clean行不进入汇总。

## 用户数据检查

使用当前项目的 `python -m skinmamba audit --data-root ../data` 实际运行。

6644对文件一一配对、全部可解码、256×256、mask为0/255。数据重复和排重计数见 [审计摘要](reports/initial_data_audit.md)。原数据未被移动或修改。完整本地JSON清单和运行日志不纳入Git；运行上述audit命令可为自己的数据生成 `reports/data_audit/`。

实际运行 `scripts/smoke_real_data.py --data-root ../data --image-size 256 --variants hsm_only hsm_local readback reconstruction`：两个训练来源 × 四个新方案，共**8组通过**。每组使用真实训练图2张，完成一次前向、BCE+Dice反向、梯度检查、优化器更新，再检查源域验证及 PH² 测试加载和公共评估指标 schema。运行产物 `reports/real_data_smoke.json` 不纳入 Git。

这是软件贯通检查，不是学习收敛或测试集精度实验；未把这些未训练模型分数放入论文结果表。

## 仍需在Kaggle执行

- 官方Mamba/causal-conv1d安装及256² CUDA数值、梯度、FP32/可选AMP核验。
- 完整 reconstruction 训练、三种子与最终测试。
- T4真实FPS、延迟和显存；当前没有T4性能测量，也没有承诺训练时长。
- 新方案是否在源域与 PH² 上稳定提高 Dice、是否减少空预测，以及 T4 真实 FPS 能否接受。

`scripts/verify_cuda.py` 在缺CUDA或数值检查失败时返回非零，不会把CPU测试或跳过当作GPU通过。GPU实际结果请保存 `reports/cuda_verification.json` 和各run的 `evaluation/profile.json`。
