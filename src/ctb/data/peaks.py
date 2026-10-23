"""Binarized peak x cell matrix, per-peak split, and 1,344 bp sequence codes."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml

from ctb.data.annotate import load_labels
from ctb.data.genome import windows
from ctb.data.splits import split_array
from ctb.paths import CONFIGS, PROCESSED, RAW, RESULTS
from ctb.utils.logging import get_logger

log = get_logger(__name__)

PEAKS_DIR = PROCESSED / "peaks"


def build() -> None:
    base = yaml.safe_load((CONFIGS / "base.yaml").read_text())
    min_frac = base["data"]["min_cell_fraction"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.read_10x_h5(RAW / "10x" / "filtered_feature_bc_matrix.h5", gex_only=False)
    adata = adata[:, adata.var["feature_types"] == "Peaks"]
    labels = load_labels()
    if not np.array_equal(labels["barcode"].to_numpy(), adata.obs_names.to_numpy()):
        raise ValueError("cell order differs between the label table and the peak matrix")

    coords = adata.var_names.to_series().str.extract(r"^([^:]+):(\d+)-(\d+)$")
    peaks = pd.DataFrame(
        {
            "chrom": coords[0].to_numpy(),
            "start": coords[1].astype(np.int64).to_numpy(),
            "end": coords[2].astype(np.int64).to_numpy(),
        }
    )
    peaks["center"] = (peaks["start"] + peaks["end"]) // 2
    peaks["split"] = split_array(peaks["chrom"].to_numpy())

    x = sp.csc_matrix(adata.X)
    x.data[:] = 1
    frac = np.asarray(x.sum(axis=0)).ravel() / x.shape[0]
    keep = (peaks["split"] != "none").to_numpy() & (frac >= min_frac)
    peaks = peaks[keep].reset_index(drop=True)
    peaks["frac_cells"] = frac[keep]
    y = sp.csr_matrix(x[:, keep].T).astype(np.uint8)  # peaks x cells

    codes = windows(peaks["chrom"].to_numpy(), peaks["center"].to_numpy())
    PEAKS_DIR.mkdir(parents=True, exist_ok=True)
    peaks.to_csv(PEAKS_DIR / "peaks.tsv", sep="\t", index=False)
    sp.save_npz(PEAKS_DIR / "y.npz", y)
    np.save(PEAKS_DIR / "codes.npy", codes)

    counts = peaks["split"].value_counts().reindex(["train", "validation", "test"])
    summary = {
        "n_cells": int(y.shape[1]),
        "n_peaks_raw": int(len(keep)),
        "n_peaks_kept": int(keep.sum()),
        "min_cell_fraction": min_frac,
        "peaks_per_split": {k: int(v) for k, v in counts.items()},
        "density": float(y.nnz / (y.shape[0] * y.shape[1])),
        "n_windows_with_N": int((codes == 4).any(axis=1).sum()),
    }
    (RESULTS / "phase0").mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase0" / "peak_matrix.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    log.info("peak matrix: %s", summary)


if __name__ == "__main__":
    build()
