"""External references scored on the same variant-peak pairs as the arms.

MOTIF: for each JASPAR matrix, the change in best match score over placements covering the
variant (alt minus ref, both strands), counted only when the better allele reaches a relative
score of 0.8; each cell type sums these changes weighted by the factor's mean log-normalized
expression in that type (the mean over components for dimers). Matrices whose genes are not all
measured are skipped.

CHROMBPNET: the TenK10K per-cell-type ChromBPNet models (ctb.eval.chrombpnet), test pairs only.

Both write runs/<NAME>/seed0/scores.npz with the arms' layout.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ctb.data.annotate import LABELS, mean_expression
from ctb.data.genome import fetch_codes
from ctb.data.variants import VARIANTS_DIR
from ctb.eval import chrombpnet
from ctb.eval.motifs import REL_SCORE_MIN, best_scores_over, load_jaspar, score_range
from ctb.eval.score import BASE_CODE
from ctb.paths import RUNS
from ctb.train.loop import device
from ctb.utils.logging import get_logger
from ctb.utils.manifest import write_json

log = get_logger(__name__)


def motif_effects(pairs: pd.DataFrame, chunk: int = 2000) -> np.ndarray:
    motifs = load_jaspar()
    expr = mean_expression([g for m in motifs for g in m.genes])
    motifs = [m for m in motifs if all(g in expr.index for g in m.genes)]
    weight = np.stack([expr.loc[m.genes].mean(axis=0).to_numpy() for m in motifs])  # (M, 6)
    lo, hi = score_range(motifs)
    threshold = lo + REL_SCORE_MIN * (hi - lo)
    wmax = max(m.probs.shape[0] for m in motifs)
    pos0 = pairs["pos"].to_numpy() - 1
    ref = np.stack(
        [fetch_codes(c, p - wmax + 1, p + wmax) for c, p in zip(pairs["chrom"], pos0, strict=True)]
    )
    alt = ref.copy()
    alt[:, wmax - 1] = [BASE_CODE[b] for b in pairs["alt"]]
    out = []
    for i in range(0, len(pairs), chunk):
        r = best_scores_over(ref[i : i + chunk], motifs, wmax - 1)
        a = best_scores_over(alt[i : i + chunk], motifs, wmax - 1)
        delta = np.where(np.maximum(r, a) >= threshold, a - r, 0.0)
        out.append(delta @ weight)
    log.info("motif baseline: %d matrices with measured genes", len(motifs))
    return np.concatenate(out)


def save(name: str, e_causal_all: np.ndarray, e_controls: np.ndarray, note: str) -> None:
    out = RUNS / name / "seed0"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scores.npz", e_causal_all=e_causal_all, e_controls=e_controls)
    write_json({"arm": name, "seed": 0, "note": note, "columns": LABELS}, out / "baseline.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["MOTIF", "CHROMBPNET"])
    args = ap.parse_args()
    causal = pd.read_parquet(VARIANTS_DIR / "causal_all.parquet")
    controls = pd.read_parquet(VARIANTS_DIR / "controls_test.parquet")
    if "MOTIF" in args.which:
        save("MOTIF", motif_effects(causal), motif_effects(controls), "JASPAR x RNA expression")
    if "CHROMBPNET" in args.which:
        dev = device()
        test = (causal["split"] == "test").to_numpy()
        e_causal = np.full((len(causal), len(LABELS)), np.nan)
        e_causal[test] = chrombpnet.score_pairs(causal[test], dev)
        save(
            "CHROMBPNET",
            e_causal,
            chrombpnet.score_pairs(controls, dev),
            "TenK10K ChromBPNet bias-free, 5-fold mean; test pairs only",
        )


if __name__ == "__main__":
    main()
