"""Frozen-model state distribution diagnostics, never a training/selection input.

One row per image and one vector per scale. Training-source statistics define
the normalization and RBF bandwidth; the target never fits either. Bootstrap
intervals describe image sampling conditional on a fitted model, not seed or
patient variability. Train a separate set of seeds for that second uncertainty.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import SkinDataset, make_loader, make_manifest, audit_manifest
from .domain import fixed_style_view
from .engine import amp_context, get_device
from .evaluation import evaluate_model
from .metrics import segmentation_metrics, aggregate_metrics
from .models import build_model
from .utils import load_checkpoint, file_sha256, write_csv, write_json

STAGES = ("encoder4", "encoder5", "encoder6")
DISTANCES = ("mu_l2", "cov_fro", "mu_l2_normalized", "cov_fro_normalized")


def moments(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Statistics require at least two finite image vectors")
    center = x - x.mean(0)
    return x.mean(0), center.T @ center / (len(x) - 1)


def distance_values(source, target, reference_trace):
    mu_s, cov_s = moments(source)
    mu_t, cov_t = moments(target)
    dm, dc = np.linalg.norm(mu_s - mu_t), np.linalg.norm(cov_s - cov_t, ord="fro")
    trace = max(float(reference_trace), 1e-12)
    return np.array([dm, dc, dm / np.sqrt(trace), dc / trace])


def squared_distances(x, y):
    return np.maximum((x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T, 0.)


def source_bandwidth(reference, seed=2026):
    rng = np.random.default_rng(seed)
    x = reference[rng.choice(len(reference), min(512, len(reference)), replace=False)]
    d = squared_distances(x, x)
    distances = d[np.triu_indices(len(x), 1)]
    return max(float(np.median(distances)), 1e-12)


def mmd2(source, target, sigma2, paired=False):
    """RBF U-statistic; negative finite-sample estimates are retained.

    Paired pseudo-views exclude corresponding cross-view diagonals as well,
    so the estimator uses independent image pairs despite paired augmentations.
    """
    n, m = len(source), len(target)
    if min(n, m) < 2 or sigma2 <= 0:
        raise ValueError("MMD needs >=2 vectors/domain and a positive bandwidth")
    kss = np.exp(-squared_distances(source, source) / (2 * sigma2))
    ktt = np.exp(-squared_distances(target, target) / (2 * sigma2))
    kst = np.exp(-squared_distances(source, target) / (2 * sigma2))
    a = (kss.sum() - np.trace(kss)) / (n * (n - 1))
    b = (ktt.sum() - np.trace(ktt)) / (m * (m - 1))
    if paired:
        if n != m:
            raise ValueError("Paired MMD requires matching image counts")
        c = (kst.sum() - np.trace(kst)) / (n * (n - 1))
    else:
        c = kst.mean()
    return float(a + b - 2 * c)


def collapse_stats(values):
    _, cov = moments(values)
    eig = np.maximum(np.linalg.eigvalsh(cov), 0.)
    trace = eig.sum()
    p = eig / trace if trace > 1e-12 else np.zeros_like(eig)
    positive = p[p > 0]
    return dict(variance_trace=float(trace), mean_channel_std=float(np.sqrt(np.diag(cov)).mean()),
                mean_vector_norm=float(np.linalg.norm(values, axis=1).mean()),
                effective_rank=float(np.exp(-(positive * np.log(positive)).sum())) if len(positive) else 0.)


def compare_distributions(source, target, reference, paired=False, bootstrap=500, seed=2026):
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    _, cov = moments(reference)
    trace = float(np.trace(cov))
    scale = np.sqrt(max(trace, 1e-12))
    sigma2 = source_bandwidth(np.asarray(reference, dtype=np.float64) / scale, seed)
    result = dict(zip(DISTANCES, distance_values(source, target, trace).tolist()))
    result.update(reference_variance_trace=trace, reference_near_collapse=trace < 1e-10,
                  mmd2=mmd2(source / scale, target / scale, sigma2, paired), rbf_sigma2=sigma2,
                  n_source=len(source), n_target=len(target), n_reference=len(reference))
    rng, samples = np.random.default_rng(seed), []
    for _ in range(bootstrap):
        si = rng.integers(len(source), size=len(source))
        ti = si if paired else rng.integers(len(target), size=len(target))
        samples.append(distance_values(source[si], target[ti], trace))
    samples = np.asarray(samples, dtype=np.float64).reshape(-1, len(DISTANCES))
    if bootstrap:
        for i, key in enumerate(DISTANCES):
            result[f"{key}_ci_low"], result[f"{key}_ci_high"] = np.quantile(samples[:, i], [.025, .975]).tolist()
    return result, samples


@torch.inference_mode()
def extract_states(model, loader, device, config, styled=False, analysis_seed=2026, score=True):
    model.eval()
    all_states, ids, rows = [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        if styled:
            image = fixed_style_view(image, batch["id"], config["data"]["normalization"],
                                     config["domain"], analysis_seed)
        with amp_context(device, config["training"]["precision"]):
            logits, aux = model(image, return_states=True)
        states = torch.stack(aux["states"], dim=1).float().cpu().numpy()
        if not np.isfinite(states).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite states/logits during frozen diagnosis")
        all_states.append(states)
        ids.extend(batch["id"])
        if score:
            pred = (logits.float().sigmoid() >= config["evaluation"]["threshold"]).cpu().numpy()[:, 0]
            truth = batch["mask"].numpy()[:, 0] >= .5
            for key, p, y in zip(batch["id"], pred, truth):
                rows.append(dict(id=key, **segmentation_metrics(p, y), empty_prediction=int(not p.any()),
                                 empty_target=int(not y.any()), target_fraction=float(y.mean())))
    if not all_states:
        raise ValueError("No images available for state extraction")
    return np.concatenate(all_states).astype(np.float64), ids, rows


class Intervention(nn.Module):
    def __init__(self, model, mode):
        super().__init__()
        self.model, self.mode = model, mode

    def forward(self, image):
        return self.model(image, intervention=self.mode)


def _matched_control(main, control):
    for key in ("data", "training", "loss", "seed"):
        a, b = copy.deepcopy(main[key]), copy.deepcopy(control[key])
        if key == "data":
            a.pop("root", None)
            b.pop("root", None)
        if key == "training":
            a.pop("workers", None)
            b.pop("workers", None)
        if a != b:
            raise ValueError(f"Unmatched control {key}; use paired seeds/data/training budgets")
    a, b = dict(main["model"]), dict(control["model"])
    a.pop("variant", None)
    b.pop("variant", None)
    if a != b:
        raise ValueError("Control must use the same memory architecture (only L_dom changes)")
    if main["domain"]["weight"] <= 0 or control["domain"]["weight"] != 0:
        raise ValueError("Use a positive-L_dom checkpoint and its zero-L_dom control")
    keys = ("two_view", "brightness", "contrast", "color", "gamma")
    if not main["domain"]["two_view"] or any(main["domain"][k] != control["domain"][k] for k in keys):
        raise ValueError("Control must have identical two-view augmentation")
    if main["evaluation"]["threshold"] != control["evaluation"]["threshold"]:
        raise ValueError("Control evaluation threshold differs")


def diagnose_states(checkpoint_path, control_path=None, data_root=None, output_dir=None,
                    device="cuda", backend=None, final=False, bootstrap=500, analysis_seed=2026,
                    interventions=False, expected_ph2=200):
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    checkpoints = {"main": (checkpoint_path, load_checkpoint(checkpoint_path))}
    if control_path:
        checkpoints["control"] = (control_path, load_checkpoint(control_path))
        _matched_control(checkpoints["main"][1]["config"], checkpoints["control"][1]["config"])
    main_config = checkpoints["main"][1]["config"]
    root = data_root or main_config["data"]["root"]
    manifest = make_manifest(root)
    source = main_config["data"]["source"]
    audit = audit_manifest(manifest, source)
    split = "test" if final else "val"
    excluded = set(audit["source_train_val_overlap"]) if not final else set()
    records = manifest["records"]
    groups = {"reference": [r for r in records if r["domain"] == source and r["split"] == "train"],
              "source": [r for r in records if r["domain"] == source and r["split"] == split and r["id"] not in excluded]}
    if final:
        groups["ph2"] = [r for r in records if r["domain"] == "ph2" and r["split"] == "test"]
        if len(groups["ph2"]) != expected_ph2:
            raise ValueError(f"Final PH2 protocol requires all {expected_ph2} images, found {len(groups['ph2'])}")
    if any(len(v) < 2 for v in groups.values()):
        raise ValueError("Each diagnostic distribution needs >=2 images")
    output = Path(output_dir or Path(checkpoint_path).parent / ("state_diagnostics_final" if final else "state_diagnostics_val"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Diagnostic output already contains files: {output}; choose a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    device = get_device(device)
    distance_rows, health_rows, score_rows, boot_samples = [], [], [], {}
    protocol = dict(version="persistent_diagnostics_v1", phase="final" if final else "validation",
                    source=source, source_split=split, fingerprint=manifest["fingerprint"],
                    analysis_seed=analysis_seed, bootstrap=bootstrap,
                    normalization="raw plus scalar trace normalization fitted on source TRAIN per model/scale",
                    state="pooled latent memory; not selective-scan hidden state", stages=STAGES,
                    reference="all source training images without augmentation",
                    target_fitting=False, stateful_test_adaptation=False,
                    mmd="RBF unbiased U-statistic, source-train median bandwidth, paired correction for pseudo views; CI not computed",
                    ci="image bootstrap conditional on fitted checkpoints and source-train reference; not seed/patient CI",
                    comparison="separate models trained with/without L_dom, NOT pre/post inference calibration",
                    checkpoints={})
    for label, (path, checkpoint) in checkpoints.items():
        config = copy.deepcopy(checkpoint["config"])
        config["data"]["root"] = str(root)
        if backend:
            config["model"]["backend"] = backend
        if config["model"].get("family") != "persistent":
            raise ValueError("Baseline has no persistent states; use evaluate for baseline metrics")
        if checkpoint["manifest"]["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Diagnostic data differ from checkpoint manifest")
        model = build_model(config).to(device).eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        protocol["checkpoints"][label] = dict(path=str(Path(path).resolve()), sha256=file_sha256(path),
            epoch=checkpoint["epoch"] + 1, config=config)
        loaders = {key: make_loader(SkinDataset(root, value, config["data"]["image_size"],
                    config["data"]["normalization"]), config["evaluation"]["batch_size"],
                    config["training"]["workers"]) for key, value in groups.items()}
        loaders["pseudo"] = loaders["source"]
        arrays = {}
        for group, loader in loaders.items():
            print(f"diagnose {label}/{group}: {len(loader.dataset)} images", flush=True)
            values, ids, rows = extract_states(model, loader, device, config, group == "pseudo", analysis_seed,
                                               score=group != "reference")
            arrays[group] = values
            (output / label).mkdir(exist_ok=True)
            np.savez_compressed(output / label / f"{group}_states.npz", states=values, ids=np.asarray(ids))
            for index, stage in enumerate(STAGES):
                health_rows.append(dict(model=label, domain=group, stage=stage, n=len(values),
                                        **collapse_stats(values[:, index])))
            if rows:
                write_csv(output / label / f"{group}_per_image.csv", rows)
                score_rows.append(dict(model=label, domain=group, intervention="none", n=len(rows),
                                       **aggregate_metrics(rows), empty_prediction_count=sum(r["empty_prediction"] for r in rows)))
        for target in ("pseudo", "ph2") if final else ("pseudo",):
            for index, stage in enumerate(STAGES):
                # Same bootstrap seed => paired model comparisons on identical image draws.
                result, samples = compare_distributions(arrays["source"][:, index], arrays[target][:, index],
                    arrays["reference"][:, index], target == "pseudo", bootstrap, analysis_seed + index)
                distance_rows.append(dict(model=label, source=source, target=target, stage=stage,
                                          seed=config["seed"], **result))
                boot_samples[(label, target, stage)] = samples
        if interventions:
            for mode in ("no_carry", "no_read"):
                for group in ("source", "ph2") if final else ("source",):
                    rows, scores = evaluate_model(Intervention(model, mode), loaders[group], str(device),
                        config["evaluation"]["threshold"], precision=config["training"]["precision"])
                    write_csv(output / label / f"{group}_{mode}_per_image.csv", rows)
                    score_rows.append(dict(model=label, domain=group, intervention=mode, n=len(rows), **scores))
        del model
    changes = []
    if control_path:
        indexed = {(r["model"], r["target"], r["stage"]): r for r in distance_rows}
        for target in ("pseudo", "ph2") if final else ("pseudo",):
            for stage in STAGES:
                a, b = indexed[("main", target, stage)], indexed[("control", target, stage)]
                change = dict(target=target, stage=stage, seed=main_config["seed"], sign="main minus control; negative means decrease")
                samples = boot_samples[("main", target, stage)] - boot_samples[("control", target, stage)]
                for i, key in enumerate(DISTANCES):
                    change[f"delta_{key}"] = a[key] - b[key]
                    if bootstrap:
                        change[f"delta_{key}_ci_low"], change[f"delta_{key}_ci_high"] = np.quantile(samples[:, i], [.025, .975]).tolist()
                change["delta_mmd2"] = a["mmd2"] - b["mmd2"]
                changes.append(change)
    write_csv(output / "state_distances.csv", distance_rows)
    write_csv(output / "state_changes.csv", changes)
    write_csv(output / "state_health.csv", health_rows)
    write_csv(output / "segmentation.csv", score_rows)
    write_json(output / "protocol.json", protocol)
    print(f"State diagnostics saved to {output}", flush=True)
    return dict(distances=distance_rows, changes=changes, health=health_rows, segmentation=score_rows, protocol=protocol)
