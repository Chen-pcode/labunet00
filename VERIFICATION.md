# 已执行的验证

日期：2026-09-26。本地环境为Windows、Python3.11、PyTorch2.14.0+cpu；测试环境独立放在项目外的 `.venv-skin-mamba`，不需要上传该环境到Kaggle。

## 自动化测试

命令：`python -m pytest -q --junitxml=reports/pytest_results.xml`

结果：**48 passed, 1 skipped**。跳过的是需要真实CUDA及官方扩展的数值对照，并非失败项目被忽略。

覆盖：

- 原作者源文件与适配baseline的结构/state_dict键、初始化逐张量相等、原sigmoid输出与新logits.sigmoid对照。两边注入相同reference backend，不能替代官方CUDA验证。
- Mamba手算递推、Δ加bias/softplus/间距的顺序、实际非均匀坐标、插值复原、未采样残差、各控制组及有效梯度。
- 已知掩码的Dice/IoU/mIoU/HD95、空掩码/全前景与HD95失败统计。
- 参数与Mamba核心FLOPs计数、profiling状态恢复、设备错误处理及precision别名。
- 配对失败检测、不同文件ID但相同RGB的跨split重复识别、增强可复现性。
- 真实训练引擎在小型合成数据上跑2 epoch；连续运行和1+1 epoch续训最终模型逐张量完全相同。
- 最佳checkpoint对三域的完整公共评估、CSV输出和全部请求字段。
- 聚合隔离不同阈值、数据fingerprint及GPU；同checkpoint同协议重复评估不算额外种子。

## 用户数据检查

使用当前项目的 `python -m skinmamba audit --data-root ../data` 实际运行。

6644对文件一一配对、全部可解码、256×256、mask为0/255。数据重复和排重计数见 [审计摘要](reports/initial_data_audit.md)。原数据未被移动或修改。完整本地JSON清单和运行日志不纳入Git；运行上述audit命令可为自己的数据生成 `reports/data_audit/`。

额外实际运行 `scripts/smoke_real_data.py`：两个训练来源 × 五个核心模型，共**10组通过**。每组使用真实训练图2张，64×64，做一次前向、BCE+Dice反向、梯度检查、优化器更新；再检查源域验证及三个测试域各2张的公共评估。脚本生成 `reports/real_data_smoke.json`，该运行产物不纳入Git。

这是软件贯通检查，不是学习收敛或测试集精度实验；未把这些未训练模型分数放入论文结果表。

## 仍需在Kaggle执行

- 官方Mamba/causal-conv1d安装及256² CUDA数值、梯度、FP32/可选AMP核验。
- 完整250 epoch训练、三种子与最终测试。
- T4真实FPS、延迟和显存；当前没有T4性能测量，也没有承诺训练时长。
- 新增几何校准是否带来收益、是否值得继续作为论文方法。

`scripts/verify_cuda.py` 在缺CUDA或数值检查失败时返回非零，不会把CPU测试或跳过当作GPU通过。GPU实际结果请保存 `reports/cuda_verification.json` 和各run的 `evaluation/profile.json`。
