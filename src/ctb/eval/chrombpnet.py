"""TenK10K ChromBPNet (bias-free) count heads loaded into PyTorch from the Keras HDF5 weights.

Only the log-count head is needed for a variant score. The layer graph read from the HDF5
config is: a 21-wide valid convolution, eight residual blocks of valid dilated convolutions
(dilation 2..256) added to the centrally cropped input, global average pooling over the
remaining 1,074 positions, and a dense layer. A variant is scored in the 2,114 bp window centered
on the variant, as in the ChromBPNet variant scorer, as the mean over the five folds of
logcount(alt) - logcount(ref), averaged over strands.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn

from ctb.data.genome import fetch_codes
from ctb.data.tenk10k import CELL_TYPES
from ctb.eval.score import BASE_CODE
from ctb.model.scbasset import one_hot, reverse_complement
from ctb.paths import RAW

INPUT_LEN = 2114
MODELS = RAW / "tenk10k" / "chrombpnet"
N_FOLDS = 5


class CountHead(nn.Module):
    def __init__(self, path):
        super().__init__()
        w = h5py.File(path, "r")["model_weights"]

        def conv(name: str, dilation: int) -> nn.Conv1d:
            g = w[name][name]
            kernel = np.asarray(g["kernel:0"])  # (k, in, out)
            layer = nn.Conv1d(kernel.shape[1], kernel.shape[2], kernel.shape[0], dilation=dilation)
            layer.weight.data = torch.from_numpy(kernel.transpose(2, 1, 0).copy())
            layer.bias.data = torch.from_numpy(np.asarray(g["bias:0"]))
            return layer

        self.first = conv("wo_bias_bpnet_1st_conv", 1)
        self.blocks = nn.ModuleList(conv(f"wo_bias_bpnet_{i}conv", 2**i) for i in range(1, 9))
        g = w["wo_bias_bpnet_logcount_predictions"]["wo_bias_bpnet_logcount_predictions"]
        self.dense = nn.Linear(512, 1)
        self.dense.weight.data = torch.from_numpy(np.asarray(g["kernel:0"]).T.copy())
        self.dense.bias.data = torch.from_numpy(np.asarray(g["bias:0"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.first(x))
        for i, block in enumerate(self.blocks, start=1):
            crop = 2**i
            h = torch.relu(block(h)) + h[..., crop:-crop]
        return self.dense(h.mean(dim=-1)).squeeze(-1)


def variant_windows(pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    half = INPUT_LEN // 2
    pos0 = pairs["pos"].to_numpy() - 1
    ref = np.stack(
        [
            fetch_codes(c, p - half, p - half + INPUT_LEN)
            for c, p in zip(pairs["chrom"], pos0, strict=True)
        ]
    )
    expected = np.array([BASE_CODE[b] for b in pairs["ref"]], dtype=np.uint8)
    if not np.array_equal(ref[:, half], expected):
        raise ValueError("reference allele does not match hg38")
    alt = ref.copy()
    alt[:, half] = np.array([BASE_CODE[b] for b in pairs["alt"]], dtype=np.uint8)
    return ref, alt


@torch.no_grad()
def _logcounts(model: CountHead, codes: np.ndarray, dev: torch.device, batch: int = 256):
    out = []
    for i in range(0, len(codes), batch):
        x = one_hot(torch.as_tensor(codes[i : i + batch], device=dev))
        out.append((0.5 * (model(x) + model(reverse_complement(x)))).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def score_pairs(pairs: pd.DataFrame, dev: torch.device) -> np.ndarray:
    """(n_pairs, 6) fold-averaged log-count differences, columns in CELL_TYPES order."""
    ref, alt = variant_windows(pairs)
    out = np.zeros((len(pairs), len(CELL_TYPES)))
    for j, ct in enumerate(CELL_TYPES):
        for fold in range(N_FOLDS):
            model = CountHead(MODELS / ct / f"fold{fold}" / "chrombpnet_nobias.h5").to(dev).eval()
            out[:, j] += (_logcounts(model, alt, dev) - _logcounts(model, ref, dev)) / N_FOLDS
    return out
