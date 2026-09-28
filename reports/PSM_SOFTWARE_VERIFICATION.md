# PSM v1 software verification — 2026-09-28

Environment: Windows, Python 3.11.9, PyTorch 2.14.0+cpu. No usable CUDA runtime or official Mamba CUDA package in this environment.

`python -m pytest -q`: **111 passed, 1 skipped**. The skipped item is the existing CUDA/reference parity check, not a successful GPU test. The new `tests/test_persistent.py` accounts for 17 passing tests, including the full synthetic-data training/resume/evaluation/state-diagnostic path. Exact epoch-boundary CPU resume preserves all model tensors.

`python scripts/verify_psm.py --device cpu --image-size 32 --variants psm_baseline psm_main`: baseline and main passed forward, finite gradients, optimizer update and core FLOP accounting checks. Default parameter counts: baseline 49,457; PSM main 56,929. No reference-backend timing is a GPU performance claim.

Experiment matrix commands were dry-run checked. Diagnostic CLI exposes a source-validation default and explicit final source-test/PH2 mode. The final diagnostic checks the full 200-image PH2 count. Synthetic tests intentionally use a two-image fixture through the internal Python test argument; the public CLI has no subset bypass.

No actual ISIC or PH2 model training, target-statistics fitting, CUDA/AMP verification, accuracy benchmark or novelty verification has been performed for this new method. Run `scripts/verify_psm.py` on Kaggle before production training and retain environment/configuration logs.

Existing checkpoint formats and historical `--main` semantics are preserved. New commands, source provenance, protocol definitions and limitations are documented in `PSM_EXPERIMENTS.md`.
