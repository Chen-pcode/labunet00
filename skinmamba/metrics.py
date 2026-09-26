"""Per-image binary segmentation metrics with explicit empty-mask policies.

Inputs are two-dimensional binary masks (bool or numeric 0/1), never logits or
probabilities. Dice/F1 is foreground Dice; ``miou`` averages foreground and
background IoU. A class absent from both masks has IoU=1. Sensitivity is 1 when
the target has no positive pixels; specificity is 1 when it has no negatives.
These conventions keep perfect empty/full predictions perfect and avoid NaNs.

HD95 uses 4-connected inner surfaces, including the image border, and the 95th
percentile of the CONCATENATED two directed nearest-surface-distance arrays.
Both masks empty gives 0; exactly one empty gives +inf. Spacing=(1,1) means
pixels. Nonunit spacing changes the distance unit to the supplied spacing unit.
Aggregation reports finite-only HD95 plus a strict mean and failure counts, so
failed empty/nonempty predictions cannot silently disappear from a report.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from scipy import ndimage


_MAIN_METRICS = (
    "dice", "f1", "iou", "miou", "accuracy", "sensitivity", "specificity"
)


def _binary_mask(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a nonempty 2-D binary mask.")
    if array.dtype.kind not in "bifu":
        raise ValueError(f"{name} must contain bool or numeric 0/1 values.")
    if not np.all(np.isfinite(array)) or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only 0 and 1; threshold explicitly first.")
    return array.astype(bool, copy=False)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 1.0


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(2, 1)
    return mask & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)


def segmentation_metrics(
    pred_binary: object,
    target_binary: object,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> dict[str, float]:
    """Evaluate one pair of equal-size 2-D masks; all returned values are floats.

    ``tp/tn/fp/fn`` are pixel counts, allowing optional pooled (micro) scores to
    be computed without confusing them with the default per-image macro mean.
    HD95 failure (+inf) is intentional, not a numerical error. No smoothing
    epsilon is added, so exact matches and disjoint masks retain exact scores.
    """
    pred = _binary_mask(pred_binary, "pred_binary")
    target = _binary_mask(target_binary, "target_binary")
    if pred.shape != target.shape:
        raise ValueError("Prediction and target masks must have the same shape.")
    spacing_array = np.asarray(spacing, dtype=float)
    if spacing_array.shape != (2,) or not np.all(np.isfinite(spacing_array)) or np.any(spacing_array <= 0):
        raise ValueError("spacing must contain two finite positive values (row, column).")

    tp = float(np.count_nonzero(pred & target))
    fp = float(np.count_nonzero(pred & ~target))
    fn = float(np.count_nonzero(~pred & target))
    tn = float(pred.size) - tp - fp - fn
    dice = _ratio(2 * tp, 2 * tp + fp + fn)
    iou = _ratio(tp, tp + fp + fn)
    background_iou = _ratio(tn, tn + fp + fn)

    if not pred.any() and not target.any():
        hd95 = 0.0
    elif not pred.any() or not target.any():
        hd95 = float("inf")
    else:
        pred_surface, target_surface = _surface(pred), _surface(target)
        to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing_array)
        to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing_array)
        distances = np.concatenate((to_target[pred_surface], to_pred[target_surface]))
        hd95 = float(np.percentile(distances, 95))

    return {
        "dice": dice,
        "f1": dice,
        "iou": iou,
        "miou": float((iou + background_iou) / 2),
        "accuracy": float((tp + tn) / pred.size),
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "hd95": hd95,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def aggregate_metrics(records: Iterable[Mapping[str, float]]) -> dict[str, float | int]:
    """Macro-average per-image scores and expose all nonfinite HD95 outcomes.

    ``hd95`` is the finite-case mean (NaN if no finite cases); ``hd95_strict`` is
    +inf if ANY image failed. ``hd95_finite_count`` and ``hd95_failed_count`` must
    accompany reported HD95. Nonfinite overlap scores and negative distances
    are rejected. NaN/+inf HD95 from another evaluator count as failures.
    If every row includes confusion counts, additional ``global_*`` pooled
    scores are returned, explicitly separate from the per-image macro scores.
    An empty record list raises ValueError instead of producing a fake result.
    """
    rows = list(records)
    if not rows:
        raise ValueError("Cannot aggregate an empty set of images.")
    result: dict[str, float | int] = {"image_count": len(rows)}
    for key in _MAIN_METRICS:
        values = np.asarray([row[key] for row in rows], dtype=float)
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{key} must be finite and in [0,1] for every image.")
        result[key] = float(values.mean())

    distances = np.asarray([row["hd95"] for row in rows], dtype=float)
    if np.any(distances < 0):
        raise ValueError("hd95 must be nonnegative (or NaN/+inf for failure).")
    finite = np.isfinite(distances)
    finite_count = int(finite.sum())
    result["hd95"] = float(distances[finite].mean()) if finite_count else float("nan")
    result["hd95_finite_count"] = finite_count
    result["hd95_failed_count"] = int(len(rows) - finite_count)
    result["hd95_strict"] = result["hd95"] if finite.all() else float("inf")

    count_keys = ("tp", "tn", "fp", "fn")
    if all(all(key in row for key in count_keys) for row in rows):
        counts = np.asarray([[row[key] for key in count_keys] for row in rows], dtype=float)
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("Confusion counts must be finite and nonnegative.")
        tp, tn, fp, fn = counts.sum(axis=0)
        if tp + tn + fp + fn == 0:
            raise ValueError("Confusion counts must include at least one pixel.")
        result.update({
            "global_dice": _ratio(2 * tp, 2 * tp + fp + fn),
            "global_f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "global_iou": _ratio(tp, tp + fp + fn),
            "global_miou": float((_ratio(tp, tp + fp + fn) + _ratio(tn, tn + fp + fn)) / 2),
            "global_accuracy": float((tp + tn) / (tp + tn + fp + fn)),
            "global_sensitivity": _ratio(tp, tp + fn),
            "global_specificity": _ratio(tn, tn + fp),
        })
    return result
