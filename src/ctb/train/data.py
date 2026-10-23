"""Training tensors: sequence codes, binarized targets and per-split peak indices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from ctb.data.annotate import load_labels
from ctb.data.peaks import PEAKS_DIR


@dataclass
class PeakData:
    peaks: pd.DataFrame
    codes: torch.Tensor  # (P, L) uint8
    y: torch.Tensor  # (P, C) uint8
    labels: np.ndarray  # (C,) cell-type label strings
    split_idx: dict[str, np.ndarray]


def load_peak_data(device: torch.device) -> PeakData:
    peaks = pd.read_csv(PEAKS_DIR / "peaks.tsv", sep="\t")
    codes = torch.from_numpy(np.load(PEAKS_DIR / "codes.npy")).to(device)
    y = torch.from_numpy(sp.load_npz(PEAKS_DIR / "y.npz").toarray()).to(device)
    split_idx = {
        s: np.flatnonzero(peaks["split"].to_numpy() == s) for s in ("train", "validation", "test")
    }
    return PeakData(peaks, codes, y, load_labels()["label"].to_numpy(), split_idx)
