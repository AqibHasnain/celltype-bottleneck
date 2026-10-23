import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from ctb.eval.metrics import (
    causal_auroc,
    centered_profile_correlation,
    effective_rank,
    per_column_auroc,
    sign_concordance,
    specificity_auroc,
    specificity_fraction,
)

NAN = np.nan


def test_per_column_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(300, 7)).astype(np.float32)
    labels = (rng.random((300, 7)) < 0.2).astype(np.uint8)
    got = per_column_auroc(torch.from_numpy(scores), torch.from_numpy(labels)).numpy()
    want = [roc_auc_score(labels[:, j], scores[:, j]) for j in range(7)]
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize(
    ("m", "expected"),
    [
        (np.tile([[2.0], [-1.0], [0.5]], (1, 6)), 0.0),  # rank-1 row-constant
        (np.array([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]), 1.0),  # zero row mean
        (np.array([[1.0, 3.0]]), 2.0 / 10.0),  # centered (-1, 1) over total 1 + 9
        (np.array([[1.0, 3.0, NAN]]), 2.0 / 10.0),  # NaN entries are masked out
    ],
)
def test_specificity_fraction(m, expected):
    assert specificity_fraction(m) == pytest.approx(expected)


def test_centered_profile_correlation_uses_top_half_by_centered_norm():
    b = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0 + 1e-3], [3.0, 1.0, 2.0], [0.0, 0.0, 0.1]])
    e = np.array([[0.0, 1.0, 2.0], [9.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 0.0, 0.0]])
    mean, per_row = centered_profile_correlation(e, b)
    # Rows 0 and 2 have the largest centered norms; r = 1 and r = -0.5.
    assert np.isnan(per_row[[1, 3]]).all()
    assert per_row[0] == pytest.approx(1.0)
    assert per_row[2] == pytest.approx(-0.5)
    assert mean == pytest.approx(0.25)


def test_centered_profile_correlation_masks_missing_types_in_both_matrices():
    b = np.array([[1.0, 2.0, 3.0, NAN], [0.0, 0.0, 0.0, 0.0]])
    e = np.array([[1.0, 2.0, 3.0, 100.0], [1.0, 1.0, 1.0, 1.0]])
    mean, _ = centered_profile_correlation(e, b)
    assert mean == pytest.approx(1.0)


def test_specificity_auroc_scores_centered_absolute_effect():
    e = np.array([[1.0, -3.0, 1.0], [2.0, 2.0, -0.5]])
    mask = np.ones_like(e, dtype=bool)
    # |E| centered: row 0 -> (-2/3, 4/3, -2/3); row 1 -> (0.5, 0.5, -1).
    label = np.array([[0.0, 1.0, NAN], [1.0, NAN, 0.0]])
    assert specificity_auroc(e, mask, label) == pytest.approx(1.0)
    flipped = np.where(np.isnan(label), NAN, 1 - label)
    assert specificity_auroc(e, mask, flipped) == pytest.approx(0.0)


def test_aggregate_metrics_use_the_row_mean():
    causal = np.array([[1.0, 1.0], [-2.0, 0.0], [0.1, -0.1]])
    control = np.array([[0.5, -0.5], [0.2, 0.2]])
    # |row means|: causal 1, 1, 0; control 0, 0.2 -> 4.5 of 6 pairs ordered (ties count half).
    assert causal_auroc(causal, control) == pytest.approx(4.5 / 6)
    assert sign_concordance(causal[:2], np.array([0.3, 0.7])) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("m", "expected"),
    [
        (np.eye(4), 4.0),
        (np.outer([1.0, 2.0, 3.0], [1.0, -1.0]), 1.0),
        (np.diag([1.0, 1.0, 0]), 2.0),
    ],
)
def test_effective_rank(m, expected):
    assert effective_rank(m) == pytest.approx(expected)
