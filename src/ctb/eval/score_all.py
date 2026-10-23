"""Score every finished run: variant effects, motif probes, ISM spectrum, representation ranks.

Writes runs/<arm>/seed<k>/scores.npz and probes.tsv. Idempotent per run.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ctb.data.annotate import load_labels
from ctb.data.peaks import PEAKS_DIR
from ctb.data.variants import VARIANTS_DIR
from ctb.eval.metrics import effective_rank
from ctb.eval.score import (
    N_ISM_PEAKS,
    ism_matrix,
    load_run,
    motif_probes,
    probe_backgrounds,
    type_average_matrix,
    variant_effects,
)
from ctb.paths import RUNS
from ctb.train.loop import device
from ctb.utils.logging import get_logger

log = get_logger(__name__)
ISM_SEED = 0


def finished_runs() -> list:
    return sorted(
        p.parent for p in RUNS.glob("*/seed*/summary.json") if (p.parent / "model.pt").exists()
    )


def score_run(run_dir, inputs: dict, force: bool = False) -> None:
    out = run_dir / "scores.npz"
    if out.exists() and not force:
        return
    labels = inputs["labels"]
    model = load_run(run_dir, labels, device())
    avg = type_average_matrix(labels)
    e_causal_all = variant_effects(model, inputs["causal_all"], avg)
    e_controls = variant_effects(model, inputs["controls"], avg)
    probes = motif_probes(model, inputs["backgrounds"], avg)
    ism = ism_matrix(model, inputs["ism_peaks"], avg)
    ism_sv = np.linalg.svd(ism, compute_uv=False)
    pred = np.load(run_dir / "test_predictions.npz")
    np.savez_compressed(
        out,
        e_causal_all=e_causal_all,
        e_controls=e_controls,
        ism_singular_values=ism_sv,
        ism_effective_rank=effective_rank(ism),
        pred_pseudobulk_effective_rank=effective_rank(pred["pred_pseudobulk"]),
        obs_pseudobulk_effective_rank=effective_rank(pred["obs_pseudobulk"]),
    )
    probes.to_csv(run_dir / "probes.tsv", sep="\t", index=False)
    log.info("scored %s: probes passed %d/8", run_dir, int(probes["pass"].sum()))


def load_inputs() -> dict:
    peaks = pd.read_csv(PEAKS_DIR / "peaks.tsv", sep="\t")
    codes = np.load(PEAKS_DIR / "codes.npy", mmap_mode="r")
    test_idx = np.flatnonzero(peaks["split"].to_numpy() == "test")
    test_codes = np.asarray(codes[test_idx])
    rng = np.random.default_rng(ISM_SEED)
    return {
        "labels": load_labels()["label"].to_numpy(),
        "causal_all": pd.read_parquet(VARIANTS_DIR / "causal_all.parquet"),
        "controls": pd.read_parquet(VARIANTS_DIR / "controls_test.parquet"),
        "backgrounds": probe_backgrounds(test_codes),
        "ism_peaks": test_codes[rng.choice(len(test_codes), N_ISM_PEAKS, replace=False)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run directories; default every finished run")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    inputs = load_inputs()
    runs = [RUNS / r for r in args.runs] if args.runs else finished_runs()
    for run_dir in runs:
        score_run(run_dir, inputs, args.force)


if __name__ == "__main__":
    main()
