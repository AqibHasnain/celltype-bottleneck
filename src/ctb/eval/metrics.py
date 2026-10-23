"""Metrics from PROJECT_PLAN.md section 5.

Effect matrices are variants x cell types. Rows may carry NaN for cell types where the observed
profile is missing (peak not tested there); every specific-component metric uses, per row, the
cell types observed in that row, and the same mask is applied to predictions.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def per_column_auroc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """AUROC of each column (cell) over rows (peaks), via the rank-sum identity.

    Ties in scores are broken by position; scores are continuous logits, so ties are negligible.
    Columns with no positives or no negatives return NaN.
    """
    # float64 rank sums need the CPU: MPS has no float64.
    scores, labels = scores.float().cpu(), labels.cpu()
    n = scores.shape[0]
    order = scores.argsort(dim=0)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks.scatter_(
        0,
        order,
        torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1).expand_as(order).contiguous(),
    )
    pos = labels.to(torch.float64)
    n_pos = pos.sum(dim=0)
    n_neg = n - n_pos
    u = (ranks * pos).sum(dim=0) - n_pos * (n_pos + 1) / 2
    auc = u / (n_pos * n_neg)
    auc[(n_pos == 0) | (n_neg == 0)] = float("nan")
    return auc


def pseudobulk_correlation(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Pearson r per cell type between predicted and observed pseudobulk (types x peaks)."""
    return np.array([np.corrcoef(p, o)[0, 1] for p, o in zip(pred, obs, strict=True)])


def _masked_center(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mean = np.nansum(np.where(mask, m, 0.0), axis=1, keepdims=True) / mask.sum(1, keepdims=True)
    return np.where(mask, m - mean, 0.0)


def causal_auroc(e_causal: np.ndarray, e_control: np.ndarray) -> float:
    """Causal vs matched controls, scored by |row mean of E| (all six types, no mask)."""
    s = np.concatenate([np.abs(e_causal.mean(1)), np.abs(e_control.mean(1))])
    y = np.concatenate([np.ones(len(e_causal)), np.zeros(len(e_control))])
    return float(roc_auc_score(y, s))


def sign_concordance(e: np.ndarray, lead_slope: np.ndarray) -> float:
    """Fraction of variants where sign(row mean of E) equals the sign of the lead-type slope."""
    return float(np.mean(np.sign(e.mean(1)) == np.sign(lead_slope)))


def centered_profile_correlation(
    e: np.ndarray, b: np.ndarray, quantile: float = 0.5
) -> tuple[float, np.ndarray]:
    """Mean Pearson r between centered rows of E and B, over rows whose centered-B norm exceeds
    the given quantile. Returns the mean and the per-row r (NaN for rows not averaged)."""
    mask = ~np.isnan(b)
    ec = _masked_center(e, mask)
    bc = _masked_center(np.nan_to_num(b), mask)
    bnorm = np.linalg.norm(bc, axis=1)
    enorm = np.linalg.norm(ec, axis=1)
    r = (ec * bc).sum(1) / np.maximum(enorm * bnorm, 1e-30)
    r[enorm == 0] = 0.0
    keep = bnorm > np.quantile(bnorm, quantile)
    r_kept = np.where(keep, r, np.nan)
    return float(np.nanmean(r_kept)), r_kept


def specificity_scores(e: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Entry of |E| minus its row mean over the observed cell types."""
    return _masked_center(np.abs(e), mask)


def specificity_auroc(e: np.ndarray, mask: np.ndarray, label: np.ndarray) -> float:
    """label: +1 specific-positive, 0 specific-negative, NaN unlabelled, per variant x type."""
    s = specificity_scores(e, mask)
    use = ~np.isnan(label) & mask
    return float(roc_auc_score(label[use], s[use]))


def specificity_fraction(m: np.ndarray, mask: np.ndarray | None = None) -> float:
    """||M - rowmean(M)||_F^2 / ||M||_F^2 over observed entries."""
    if mask is None:
        mask = ~np.isnan(m)
    m0 = np.where(mask, np.nan_to_num(m), 0.0)
    total = (m0**2).sum()
    return float((_masked_center(m0, mask) ** 2).sum() / total) if total > 0 else float("nan")


def effective_rank(m: np.ndarray) -> float:
    """Entropy effective rank (Roy and Vetterli 2007): exp of the entropy of normalized
    singular values."""
    s = np.linalg.svd(np.asarray(m, dtype=np.float64), compute_uv=False)
    s = s[s > 0]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))
