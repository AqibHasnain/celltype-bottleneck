import numpy as np
import pytest

from ctb.model.scbasset import ScBasset, n_trainable
from ctb.train.arms import build_model_config, load_arm

N_CELLS = 500


def _count(arm: str, width: int | None = None) -> int:
    mc = build_model_config(load_arm(arm), N_CELLS)
    if width is not None:
        mc = type(mc)(**{**mc.__dict__, "n_filters_pre_bottleneck": width})
    types = np.zeros(N_CELLS, dtype=int) if mc.readout != "linear" else None
    return n_trainable(ScBasset(mc, types))


@pytest.mark.parametrize("arm", ["COND", "COND-SHUF", "COND-PRE", "COND-PRE-SHUF"])
def test_conditioned_arm_is_the_widest_that_does_not_exceed_w32(arm):
    w32 = _count("W32")
    width = build_model_config(load_arm(arm), N_CELLS).n_filters_pre_bottleneck
    assert _count(arm) <= w32
    assert _count(arm, width + 1) > w32
    assert w32 - _count(arm) < 738  # less than one pre-bottleneck channel at d = 32
