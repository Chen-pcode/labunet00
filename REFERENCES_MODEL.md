# Model provenance and interpretation

## Baseline

The baseline adapts **UltraLight VM-UNet: Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation**, Renkai Wu et al., Patterns (2025), [DOI 10.1016/j.patter.2025.101298](https://doi.org/10.1016/j.patter.2025.101298). Author project: [wurenkai/UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet).

The specific local source was `UltraLight-VM-UNet-main/UltraLight-VM-UNet-main/models/UltraLight_VM_UNet.py`, supplied in the adjacent original project. An unchanged copy is included as `third_party/UltraLight_VM_UNet.original.py` for comparison, with source hashes in `SOURCE_PROVENANCE.json`. The MIT notice is retained in `third_party/UltraLight-VM-UNet.LICENSE`.

With default channels `[8,16,24,32,48,64]`, groups 4 and bridge enabled, the baseline retains the source's topology, module/state-dict names, shared PVM core, normalization, skips, bilinear interpolation settings and outer initialization. It replaces `timm.trunc_normal_` with `torch.nn.init.trunc_normal_`, removes a constructor print, validates input dimensions, and returns **logits** instead of sigmoid probabilities. Apply sigmoid for probabilities. This project's training loss explicitly computes **FP32 sigmoid + probability BCELoss + per-image soft Dice**, matching the original source; it does not replace that loss with BCEWithLogitsLoss. Inputs must have spatial dimensions divisible by 32. PVM groups are configurable; GroupNorm retains the original four groups.

The author's outer `self.apply(_init_weights)` reinitializes **all** Linear layers, including Mamba's dt projection and its bias. The official Mamba bias `_no_reinit` flag is not consulted by that source initializer. This behavior is deliberately preserved for baseline fidelity; do not silently call a changed initializer a reproduction. Optimizer choices, augmentation and dataset split still need separate fidelity checks: architecture compatibility does not establish reproduced paper accuracy.

## Mamba execution

- Source equations and official implementation: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752), [mamba_simple.py](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py), [selective_scan_interface.py](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py).
- `backend: cuda` constructs official `mamba_ssm.Mamba`. The unmodified baseline uses its normal forward path. The sampled controls reuse the **same official parameters** and standard PyTorch projections/causal convolution, then call the official `selective_scan_fn`. No custom CUDA kernel is introduced.
- `variant: unfused_control` changes only the selected stage to the manual projection/convolution/scan path on the full grid, with no sampling or delta scaling. Compare sampled variants against this control as well as baseline to separate sampling effects from the official fused versus manual execution path. CUDA fused/unfused numerical agreement must be measured with tolerance; it is not assumed bit-exact.
- `backend: reference` explicitly selects a small-input, pure PyTorch Mamba-1 recurrence with matching parameter names and shapes. It includes input-dependent B/C/delta, state decay, approximate delta-B input injection, D skip and SiLU gate. It is not a convolution surrogate and is not suitable for runtime comparisons or full-resolution production training.
- Requesting CUDA when CUDA or the official extension is unavailable raises an error; there is no automatic reference fallback.
- Mixed precision: raw delta and the bias/softplus/factor operation are evaluated in float32, then delta is cast to the scan input dtype at the official CUDA kernel boundary. The reference recurrence accumulates in float32. GPU/AMP parity and practical T4 latency require actual GPU validation.
- `PVMLayer.profiling_scan_spec(inputs, output)` exposes analytical core dimensions. A profiler must count each PVM's complete Mamba core `groups` times and exclude its `.mamba` subtree from Conv/Linear hooks; official fused execution and functional dt projection otherwise lead to missed or duplicate arithmetic.

## Unvalidated geometry prototype

The variants `sampled_index`, `sampled_constant` and `sampled_geometry` modify only `geometry_stage` (default encoder4). They share exactly the same learnable parameters, deterministic coordinates, sampling, reconstruction, and Mamba implementation. This is **not a Serp-Mamba reproduction**, a learned deformable sampler, or an established novel method.

For a stage of height H and width W, sample M columns at `x_j=(W-1)*(j/(M-1))**sampling_power`, with `M=min(W,max(2,round(W*sample_ratio)))`. W=1 is a one-sample degenerate case. Coordinates span both endpoints and are monotone. `grid_sample` samples each row; rows remain in their original order and are concatenated. Outputs are interpolated back using those actual nonuniform x coordinates. The residual uses the **original unsampled normalized features**.

For token t the calibrated scan uses

`tau_t = softplus(raw_delta_t + learned_delta_bias) * factor_t`.

The CUDA scan receives `delta_bias=None, delta_softplus=False`, so neither bias nor softplus is applied twice. Its recurrence is

`h_t = exp(tau_t * A) * h_(t-1) + tau_t * B_t * u_t`.

Thus tau changes both memory decay and input injection. This follows Mamba-1's approximate input discretization, **not exact ZOH**.

| Variant | Factor within each row |
|---|---|
| sampled_index | 1 |
| sampled_constant | mean original-grid interval `(W-1)/(M-1)` |
| sampled_geometry | actual interval `x_j-x_(j-1)` |

One unit is one pixel of the **original stage feature grid**, independent of M. We do **not** divide intervals by their sample mean. Every row's first token has factor 1, including the first token of the sequence. The H-1 row-transition positions and their factor are explicitly reported by `model.sampling_report()` after a forward pass. They are not claimed to represent true Euclidean or arc-length distances between consecutive raster rows.

`sample_ratio=1, sampling_power=1` gives the full uniform control: all factors are 1 and no spatial samples are discarded. `sample_ratio=1, sampling_power=1.5` is still nonuniform sampling and is not an identity baseline. There is no predicted lesion-scale normalization or use of masks for sampling.

If M is only 2, the samples are the two endpoints, so the within-row constant and geometry factors coincide regardless of power. For example, a 32² input gives encoder4 width 4 and M=2 at ratio 0.5. That smoke test verifies execution only. Use at least 64² at the default stage/ratio (and 256² for the planned experiment) to distinguish constant versus nonuniform local intervals.

Sampling changes what a fixed index-window causal convolution sees. Geometric delta calibration alone therefore does not guarantee density invariance, rotational equivariance, continuous-scale equivariance, or equivalence to a continuous spatial system. Fixed power-law sampling favors one side of each row. This is a controlled mechanism probe, not evidence of a clinically meaningful directional prior. Parameter reduction or nominal scan-token reduction must not be presented as measured acceleration.

## CCLAS adaptive prototype

The new `cclas` variant is a testable research hypothesis, not an established
novelty claim. It retains the author network and official Mamba-1 scan. Only
the selected PVM stage gains a depthwise 3x3 plus pointwise 1x1 score head.
The head predicts a positive feature-importance map, without using the ground
truth mask at inference. Per-row scores are mixed with uniform density and a
discrete CDF is inverted at a fixed number of quantiles. This produces `K`
monotone coordinates per row, including both image borders, for every image.
`grid_sample` reads features and differentiable linear interpolation restores
the original grid; the unsampled residual stays intact.

The coverage option shrinks each row's intervals toward their uniform mean by
the smallest common factor that places every interval within 0.25-2.5 times
the mean. It preserves both endpoints and the fixed token count. The final
`cclas` variant also scales the Mamba delta by each within-row interval divided
by the mean and clamps that factor to 0.5-1.5; each row's first token has factor
1. The score head, coverage rule, and bounded delta are separately ablated.
These dimensionless factors are a design choice for this prototype and do not
claim exact continuous-time discretization, lesion supervision, or rotation
equivariance. The score head and CDF operations add latency that must be
measured on T4. Similar adaptive/deformable scan and token selection work may
limit novelty; literature collision review remains a separate requirement.

## Verification scope

`tests/test_models.py` checks a hand-calculated recurrence, delta-bias order, fixed spatial units and row seams, nonuniform interpolation, full-uniform equivalence, meaningful gradients, controlled-variant parameter compatibility and original-grid residual preservation. A comparison against the bundled unchanged source checks state keys, exact initialized tensors and original probability outputs against sigmoid of new logits. This comparison injects the **reference backend into both versions**, so it is an architecture/initialization check, not verification of the external CUDA package.

The adaptive tests additionally check fixed token count, border coverage,
interval bounds, distinct coverage/delta ablations, score-head gradients and
an end-to-end one-epoch CPU checkpoint/evaluation path. The CUDA/reference
parity test is skipped without CUDA/the official package and must never be
reported as passed when skipped. No segmentation accuracy, novelty, hardware
speed or paper reproduction is established by these software checks.

## Kaggle installation and mandatory GPU verification

`scripts/install_kaggle.py` prints the existing Python, PyTorch, CUDA and `nvcc` environment. It preserves the existing PyTorch installation, installs the project's explicit requirements, then installs `causal-conv1d` and `mamba-ssm` from their official PyPI packages with `--no-build-isolation`. It does not guess wheel URLs. The source projects are [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) and [state-spaces/mamba](https://github.com/state-spaces/mamba). Their installers may use an available compatible prebuilt wheel or compile locally; local compilation needs a compatible CUDA toolkit, compiler and `nvcc`, and can take time.

The install script defaults to the versions resolved by pip at run time, **not an asserted universally compatible version pair**. Use its `--mamba-version` and `--causal-conv1d-version` options to pin versions after a successful environment has been verified. The script records resolved versions. Mamba-1 API/kernel compatibility is checked by actual execution, not inferred from package names or successful import. Keep the install report and GPU verification report with experiment results.

`scripts/verify_cuda.py` requires CUDA and the official extensions and exits nonzero on missing GPU or failed checks; it never treats a skip as verification. It checks official versus reference short-sequence forward/input-gradient/parameter-gradient agreement, including nonuniform and batched delta factors and the unfused control. It checks baseline, old sampled controls and all four adaptive network variants with 256², batch 1 synthetic input and backward. `--precision` selects `fp32`, `amp_fp16`, `amp_bf16`, or `all`; the default `all` runs fp32 and amp_fp16 (the relevant T4 modes). BF16 requires explicit support and is not a T4 default. Reports include tolerances, device, versions, allocated/reserved peak memory and failure details. These are numerical/software checks; they do not train on a dataset or establish segmentation improvement.
