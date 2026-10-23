"""PyTorch reimplementation of scBasset with a configurable bottleneck and readout.

The convolutional tower follows scvi.external.SCBASSET (itself a port of the Keras original):
GELU-before-conv blocks, BatchNorm eps 1e-3, a 1x1 pre-bottleneck convolution, a dense
bottleneck with dropout 0.2 and a final GELU. Readouts:

- linear: logit[c] = w_c . z + b_c (published scBasset).
- film:   logit[c] = w_c . (z * (1 + gamma_t) + beta_t) + b_c, t the cell's type. Because w_c is
          free per cell this has the same function class as `linear`; it is the pre-registered
          COND arm.
- film_pre: FiLM on the pre-bottleneck activations, per channel and type, so the bottleneck
          vector itself depends on the cell type: z_t = bottleneck(h * (1 + gamma_t) + beta_t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

SEQ_LEN = 1344
_TOWER_OUT_LEN = 7  # 1344 / 3 / 2**6


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(1.702 * x) * x


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, pool: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding="same", bias=False)
        self.bn = nn.BatchNorm1d(c_out, eps=1e-3)
        self.pool = nn.MaxPool1d(pool, padding=(pool - 1) // 2) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.bn(self.conv(_gelu(x))))


@dataclass(frozen=True)
class ModelConfig:
    n_cells: int
    bottleneck: int = 32
    readout: str = "linear"
    n_types: int = 1
    n_filters_init: int = 288
    n_repeat_blocks_tower: int = 6
    filters_mult: float = 1.122
    n_filters_pre_bottleneck: int = 256


def one_hot(codes: torch.Tensor) -> torch.Tensor:
    """(B, L) int codes with N=4 -> (B, 4, L) float; N becomes all zeros."""
    return nn.functional.one_hot(codes.long(), num_classes=5)[..., :4].permute(0, 2, 1).float()


def reverse_complement(x: torch.Tensor) -> torch.Tensor:
    """RC of a (B, 4, L) one-hot in ACGT order: flipping both axes maps A<->T and C<->G."""
    return torch.flip(x, dims=(-2, -1))


def shift(x: torch.Tensor, s: int, pad: float = 0.25) -> torch.Tensor:
    if s == 0:
        return x
    out = torch.roll(x, s, dims=-1)
    if s > 0:
        out[..., :s] = pad
    else:
        out[..., s:] = pad
    return out


class ScBasset(nn.Module):
    def __init__(self, cfg: ModelConfig, cell_types: np.ndarray | None = None):
        super().__init__()
        self.cfg = cfg
        if cfg.readout not in ("linear", "film", "film_pre"):
            raise ValueError(f"unknown readout {cfg.readout}")
        self.stem = _ConvBlock(4, cfg.n_filters_init, 17, 3)
        blocks, c = [], cfg.n_filters_init
        for i in range(cfg.n_repeat_blocks_tower):
            c_new = int(np.round(c * cfg.filters_mult)) if i > 0 else c
            blocks.append(_ConvBlock(c, c_new, 5, 2))
            c = c_new
        self.tower = nn.Sequential(*blocks)
        self.pre = _ConvBlock(c, cfg.n_filters_pre_bottleneck, 1, 1)
        self.dense = nn.Linear(
            cfg.n_filters_pre_bottleneck * _TOWER_OUT_LEN, cfg.bottleneck, bias=False
        )
        nn.init.kaiming_normal_(self.dense.weight)
        self.dense_bn = nn.BatchNorm1d(cfg.bottleneck, eps=1e-3)
        self.dropout = nn.Dropout(0.2)
        self.cell_weight = nn.Parameter(torch.randn(cfg.bottleneck, cfg.n_cells))
        self.cell_bias = nn.Parameter(torch.randn(cfg.n_cells))
        if cfg.readout != "linear":
            if cell_types is None or len(cell_types) != cfg.n_cells:
                raise ValueError("FiLM readouts need one type index per cell")
            self.register_buffer("cell_types", torch.as_tensor(cell_types, dtype=torch.long))
            width = cfg.bottleneck if cfg.readout == "film" else cfg.n_filters_pre_bottleneck
            # Zero init makes every FiLM arm start as the plain linear readout.
            self.gamma = nn.Parameter(torch.zeros(cfg.n_types, width))
            self.beta = nn.Parameter(torch.zeros(cfg.n_types, width))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pre(self.tower(self.stem(x)))

    def _bottleneck(self, h: torch.Tensor) -> torch.Tensor:
        return _gelu(self.dropout(self.dense_bn(self.dense(h.flatten(1)))))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, L) -> (B, d) for linear/film, (B, T, d) for film_pre."""
        h = self._features(x)
        if self.cfg.readout != "film_pre":
            return self._bottleneck(h)
        b, t = h.shape[0], self.cfg.n_types
        ht = h.unsqueeze(1) * (1 + self.gamma)[None, :, :, None] + self.beta[None, :, :, None]
        return self._bottleneck(ht.reshape(b * t, *h.shape[1:])).reshape(b, t, -1)

    def readout(self, z: torch.Tensor) -> torch.Tensor:
        """Bottleneck output -> (B, n_cells) logits."""
        if self.cfg.readout == "linear":
            return z @ self.cell_weight + self.cell_bias
        if self.cfg.readout == "film":
            zt = z.unsqueeze(1) * (1 + self.gamma)[None] + self.beta[None]  # (B, T, d)
        else:
            zt = z
        per_type = torch.einsum("btd,dc->btc", zt, self.cell_weight)
        idx = self.cell_types[None, None, :].expand(z.shape[0], 1, -1)
        return per_type.gather(1, idx).squeeze(1) + self.cell_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.embed(x))


def n_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
