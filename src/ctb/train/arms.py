"""Arm configuration, cell-type indices and parameter matching."""

from __future__ import annotations

import copy
from functools import lru_cache

import numpy as np
import yaml

from ctb.data.annotate import LABELS, OTHER
from ctb.model.scbasset import ModelConfig, ScBasset, n_trainable
from ctb.paths import CONFIGS

TYPE_NAMES = [*LABELS, OTHER]
ARMS = ("W2", "W8", "W32", "W128", "W512", "COND", "COND-SHUF", "COND-PRE", "COND-PRE-SHUF")


def load_arm(arm: str) -> dict:
    base = yaml.safe_load((CONFIGS / "base.yaml").read_text())
    override = yaml.safe_load((CONFIGS / "arms" / f"{arm}.yaml").read_text())
    cfg = copy.deepcopy(base)
    for section, values in override.items():
        cfg[section].update(values)
    cfg["arm"] = arm
    return cfg


def type_indices(labels: np.ndarray, shuffle_seed: int | None) -> np.ndarray:
    idx = np.array([TYPE_NAMES.index(lab) for lab in labels], dtype=np.int64)
    if shuffle_seed is not None:
        idx = np.random.default_rng(shuffle_seed).permutation(idx)
    return idx


def _model_config(model_cfg: dict, n_cells: int, pre_width: int) -> ModelConfig:
    return ModelConfig(
        n_cells=n_cells,
        bottleneck=model_cfg["bottleneck"],
        readout=model_cfg["readout"],
        n_types=len(TYPE_NAMES),
        n_filters_init=model_cfg["n_filters_init"],
        n_repeat_blocks_tower=model_cfg["n_repeat_blocks_tower"],
        filters_mult=model_cfg["filters_mult"],
        n_filters_pre_bottleneck=pre_width,
    )


@lru_cache(maxsize=64)
def _count(model_cfg_items: tuple, n_cells: int, pre_width: int) -> int:
    mc = _model_config(dict(model_cfg_items), n_cells, pre_width)
    types = np.zeros(n_cells, dtype=np.int64)
    return n_trainable(ScBasset(mc, types if mc.readout != "linear" else None))


def _hashable(model_cfg: dict) -> tuple:
    return tuple(sorted((k, v) for k, v in model_cfg.items() if not isinstance(v, list)))


def matched_pre_width(model_cfg: dict, target_cfg: dict, n_cells: int) -> int:
    """Largest pre-bottleneck width whose parameter count does not exceed the target arm's.

    Exact equality is impossible in general: one pre-bottleneck channel costs
    (tower channels + 2 + 7 * bottleneck) parameters, more than the FiLM vectors add.
    """
    target = _count(_hashable(target_cfg), n_cells, target_cfg["n_filters_pre_bottleneck"])
    width = model_cfg["n_filters_pre_bottleneck"]
    while _count(_hashable(model_cfg), n_cells, width) > target:
        width -= 1
    return width


def build_model_config(cfg: dict, n_cells: int) -> ModelConfig:
    model_cfg = cfg["model"]
    width = model_cfg["n_filters_pre_bottleneck"]
    if model_cfg.get("match_params_to"):
        target = load_arm(model_cfg["match_params_to"])["model"]
        width = matched_pre_width(model_cfg, target, n_cells)
    return _model_config(model_cfg, n_cells, width)
