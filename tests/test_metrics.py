import math

import numpy as np
import pytest

from skinmamba.metrics import aggregate_metrics, segmentation_metrics


def test_confusion_scores_are_exact_and_f1_is_dice():
    pred = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 0], [1, 0]])
    result = segmentation_metrics(pred, target)
    assert result["dice"] == result["f1"] == 0.5
    assert result["iou"] == pytest.approx(1 / 3)
    assert result["miou"] == pytest.approx(1 / 3)
    assert result["accuracy"] == result["sensitivity"] == result["specificity"] == 0.5
    assert all(result[k] == 1 for k in ("tp", "tn", "fp", "fn"))
    assert all(isinstance(value, float) for value in result.values())


@pytest.mark.parametrize("value", [False, True])
def test_perfect_empty_or_full_has_perfect_scores(value):
    mask = np.full((4, 5), value)
    result = segmentation_metrics(mask, mask)
    for key in ("dice", "f1", "iou", "miou", "accuracy", "sensitivity", "specificity"):
        assert result[key] == 1.0
    assert result["hd95"] == 0.0


def test_one_empty_mask_is_a_distance_failure_not_zero():
    empty = np.zeros((4, 4), dtype=bool)
    one = empty.copy()
    one[1, 1] = True
    missing = segmentation_metrics(empty, one)
    extra = segmentation_metrics(one, empty)
    assert missing["dice"] == extra["iou"] == 0.0
    assert missing["sensitivity"] == 0.0
    assert extra["sensitivity"] == 1.0  # Explicit no-positive-target convention.
    assert math.isinf(missing["hd95"]) and math.isinf(extra["hd95"])


def test_hd95_is_symmetric_euclidean_surface_distance_with_spacing():
    pred = np.zeros((8, 9), dtype=bool)
    target = pred.copy()
    pred[1, 1] = True
    target[4, 5] = True
    assert segmentation_metrics(pred, target)["hd95"] == 5.0
    expected = math.sqrt((3 * 2) ** 2 + (4 * 3) ** 2)
    assert segmentation_metrics(pred, target, (2, 3))["hd95"] == pytest.approx(expected)
    assert segmentation_metrics(target, pred, (2, 3))["hd95"] == pytest.approx(expected)


def test_hd95_uses_concatenation_not_max_of_two_percentiles():
    pred = np.ones((1, 101), dtype=bool)
    target = np.zeros_like(pred)
    target[0, 0] = True
    assert segmentation_metrics(pred, target)["hd95"] == pytest.approx(94.95)


def test_aggregation_retains_hd95_failures_and_separates_macro_from_micro():
    perfect = segmentation_metrics(np.ones((1, 1)), np.ones((1, 1)))
    failed = segmentation_metrics(np.zeros((3, 3)), np.ones((3, 3)))
    result = aggregate_metrics([{"id": "perfect", **perfect}, {"id": "failed", **failed}])
    assert result["dice"] == 0.5
    assert result["global_dice"] == pytest.approx(2 / 11)
    assert result["hd95"] == 0.0
    assert math.isinf(result["hd95_strict"])
    assert result["hd95_finite_count"] == result["hd95_failed_count"] == 1
    assert result["image_count"] == 2
    all_failed = aggregate_metrics([failed])
    assert math.isnan(all_failed["hd95"])
    assert all_failed["hd95_failed_count"] == 1


@pytest.mark.parametrize("bad", [np.array([[0.2]]), np.array([[255]]), np.array([[np.nan]]), np.zeros((2, 2, 1)), np.zeros((0, 2))])
def test_rejects_nonbinary_or_wrong_rank_masks(bad):
    with pytest.raises(ValueError):
        segmentation_metrics(bad, bad)


@pytest.mark.parametrize("spacing", [(0, 1), (1, -1), (1, float("inf")), (1,), (1, 2, 3)])
def test_rejects_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        segmentation_metrics(np.ones((2, 2)), np.ones((2, 2)), spacing)


def test_rejects_shape_mismatch_and_empty_or_invalid_aggregation():
    with pytest.raises(ValueError):
        segmentation_metrics(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        aggregate_metrics([])
    row = segmentation_metrics(np.ones((2, 2)), np.ones((2, 2)))
    row["dice"] = float("nan")
    with pytest.raises(ValueError):
        aggregate_metrics([row])
