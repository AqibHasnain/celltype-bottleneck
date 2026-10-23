"""Metrics from PROJECT_PLAN.md section 5.

Effect matrices are variants x cell types. Rows of the observed matrix B may carry NaN for
types where the peak was not tested; every specific-component metric uses, per row, the observed
types only, and applies the same mask to the predictions. Weighted forms take one non-negative
weight per row, which is how a locus bootstrap resample is applied.
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


def masked_center(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract each row's mean over its observed entries; unobserved entries become 0."""
    m0 = np.where(mask, m, 0.0)
    mean = m0.sum(axis=1, keepdims=True) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    return np.where(mask, m0 - mean, 0.0)


def _weights(w: np.ndarray | None, n: int) -> np.ndarray:
    return np.ones(n) if w is None else np.asarray(w, dtype=float)


def causal_auroc(e_causal, e_control, w_causal=None, w_control=None) -> float:
    """Causal vs matched controls, scored by |row mean of E|."""
    s = np.concatenate([np.abs(e_causal.mean(1)), np.abs(e_control.mean(1))])
    y = np.concatenate([np.ones(len(e_causal)), np.zeros(len(e_control))])
    w = np.concatenate([_weights(w_causal, len(e_causal)), _weights(w_control, len(e_control))])
    return float(roc_auc_score(y, s, sample_weight=w))


def sign_concordance(e: np.ndarray, lead_slope: np.ndarray, w=None) -> float:
    """Fraction of variants where sign(row mean of E) equals the sign of the lead-type slope."""
    hit = np.sign(e.mean(1)) == np.sign(lead_slope)
    return float(np.average(hit, weights=_weights(w, len(e))))


def profile_correlations(e: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Pearson r between E and B over the types observed in B."""
    mask = ~np.isnan(b)
    ec = masked_center(e, mask)
    bc = masked_center(np.nan_to_num(b), mask)
    en, bn = np.linalg.norm(ec, axis=1), np.linalg.norm(bc, axis=1)
    r = (ec * bc).sum(1) / np.maximum(en * bn, 1e-30)
    r[en == 0] = 0.0
    return r


def cpc_rows(b: np.ndarray, eligible: np.ndarray, quantile: float = 0.5) -> np.ndarray:
    """Eligible rows whose centered-B norm exceeds the quantile of eligible rows' norms."""
    mask = ~np.isnan(b)
    bn = np.linalg.norm(masked_center(np.nan_to_num(b), mask), axis=1)
    return eligible & (bn > np.quantile(bn[eligible], quantile))


def centered_profile_correlation(e, b, rows: np.ndarray, w=None) -> float:
    """Mean per-row r over the selected rows (section 5.2 CPC)."""
    r = profile_correlations(e, b)
    return float(np.average(r[rows], weights=_weights(w, len(e))[rows]))


def specificity_scores(e: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Entry of |E| minus its row mean over the observed cell types."""
    return masked_center(np.abs(e), mask)


def specificity_auroc(e: np.ndarray, mask: np.ndarray, label: np.ndarray, w=None) -> float:
    """label: 1 specific-positive, 0 specific-negative, NaN unlabelled, per variant x type."""
    s = specificity_scores(e, mask)
    use = ~np.isnan(label) & mask
    ww = np.broadcast_to(_weights(w, len(e))[:, None], label.shape)
    return float(roc_auc_score(label[use], s[use], sample_weight=ww[use]))


def specificity_fraction(m: np.ndarray, mask: np.ndarray | None = None, w=None) -> float:
    """||M - rowmean(M)||_F^2 / ||M||_F^2 over observed entries."""
    if mask is None:
        mask = ~np.isnan(m)
    m0 = np.where(mask, np.nan_to_num(m), 0.0)
    num = (masked_center(m0, mask) ** 2).sum(1)
    den = (m0**2).sum(1)
    ww = _weights(w, len(m))
    total = (ww * den).sum()
    return float((ww * num).sum() / total) if total > 0 else float("nan")


def effective_rank(m: np.ndarray) -> float:
    """Entropy effective rank (Roy and Vetterli 2007): exp of the entropy of normalized
    singular values."""
    s = np.linalg.svd(np.asarray(m, dtype=np.float64), compute_uv=False)
    s = s[s > 0]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))
