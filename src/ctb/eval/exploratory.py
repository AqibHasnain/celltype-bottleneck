"""Exploratory analysis added after the pre-registered results were seen (docs/DECISIONS.md).

Pooled specificity AUROC ranks every (variant, type) entry together, so a model that scales one
cell type's effects up for every variant moves it through the types' different positive-label
rates. Within-type specificity AUROC computes the AUROC separately in each cell type and averages
over types, which removes any per-type offset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ctb.eval.aggregate import BOOT_SEED, N_BOOT, Truth, bootstrap_weights, run_dirs
from ctb.eval.metrics import specificity_scores
from ctb.paths import RESULTS

COMPARISONS = [
    ("W512", "W32"),
    ("W512", "W8"),
    ("W32", "W8"),
    ("COND", "W32"),
    ("COND-SHUF", "W32"),
]


def within_type_auroc(score: np.ndarray, label: np.ndarray, w: np.ndarray) -> float:
    vals = []
    for j in range(label.shape[1]):
        use = ~np.isnan(label[:, j])
        y = label[use, j]
        if 0 < y.sum() < len(y):
            vals.append(roc_auc_score(y, score[use, j], sample_weight=w[use]))
    return float(np.mean(vals))


def main() -> None:
    t = Truth()
    arms = run_dirs()
    scores = {}
    for arm, dirs in arms.items():
        for d in dirs:
            e = np.load(d / "scores.npz")["e_causal_all"][t.test_rows]
            if np.isnan(e).any():
                continue
            scores[(arm, d.name)] = specificity_scores(e, t.mask)
    rng = np.random.default_rng(BOOT_SEED)
    ones = np.ones(len(t.test))
    point = {k: within_type_auroc(s, t.label, ones) for k, s in scores.items()}
    boot = {k: np.empty(N_BOOT) for k in scores}
    for i in range(N_BOOT):
        w = bootstrap_weights(t, rng)[0]
        for k, s in scores.items():
            boot[k][i] = within_type_auroc(s, t.label, w)

    def arm_mean(arm, src):
        return np.mean([src[k] for k in src if k[0] == arm], axis=0)

    rows = []
    for arm in sorted({k[0] for k in scores}):
        b = arm_mean(arm, boot)
        rows.append(
            {
                "arm": arm,
                "a": arm,
                "b": "",
                "value": arm_mean(arm, point),
                "ci_lo": np.percentile(b, 2.5),
                "ci_hi": np.percentile(b, 97.5),
            }
        )
    for a, b in COMPARISONS:
        seeds = sorted({k[1] for k in scores if k[0] == a} & {k[1] for k in scores if k[0] == b})
        d = np.mean([boot[(a, s)] - boot[(b, s)] for s in seeds], axis=0)
        diff = np.mean([point[(a, s)] - point[(b, s)] for s in seeds])
        rows.append(
            {
                "arm": f"{a} - {b}",
                "a": a,
                "b": b,
                "value": diff,
                "ci_lo": np.percentile(d, 2.5),
                "ci_hi": np.percentile(d, 97.5),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(
        RESULTS / "exploratory_within_type_specificity_auroc.tsv",
        sep="\t",
        index=False,
        float_format="%.5g",
    )
    print(out.to_string())


if __name__ == "__main__":
    main()
