import numpy as np
import torch

from ctb.eval.score import type_logits
from ctb.model.scbasset import ModelConfig, ScBasset


def _tiny(readout: str) -> ScBasset:
    torch.manual_seed(0)
    cfg = ModelConfig(
        n_cells=12,
        bottleneck=4,
        readout=readout,
        n_types=3,
        n_filters_init=8,
        filters_mult=1.0,
        n_filters_pre_bottleneck=4,
    )
    types = np.arange(12) % 3 if readout != "linear" else None
    model = ScBasset(cfg, types).eval()
    for bn in [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]:
        bn.running_mean.uniform_(-0.5, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
    if readout != "linear":
        torch.nn.init.normal_(model.gamma, std=0.5)
        torch.nn.init.normal_(model.beta, std=0.5)
    return model


def _rc_codes(codes: np.ndarray) -> np.ndarray:
    comp = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
    return comp[codes][:, ::-1].copy()


def test_strand_averaged_effect_is_identical_on_either_strand():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 4, size=(6, 1344)).astype(np.uint8)
    alt = ref.copy()
    alt[:, 700] = (alt[:, 700] + 1) % 4
    avg = torch.as_tensor(np.eye(12)[:, :6] + np.eye(12)[:, 6:], dtype=torch.float32) / 2
    for readout in ("linear", "film", "film_pre"):
        model = _tiny(readout)
        fwd = type_logits(model, alt, avg) - type_logits(model, ref, avg)
        rev = type_logits(model, _rc_codes(alt), avg) - type_logits(model, _rc_codes(ref), avg)
        single = type_logits(model, alt, avg, rc_average=False) - type_logits(
            model, ref, avg, rc_average=False
        )
        np.testing.assert_allclose(fwd, rev, rtol=1e-5, atol=1e-6)
        # The model itself is not strand-symmetric, so the averaging is doing real work.
        assert not np.allclose(single, fwd, atol=1e-6)


def test_film_readout_routes_each_cell_through_its_own_type():
    model = _tiny("film")
    x = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 1344)), 4).permute(0, 2, 1).float()
    with torch.no_grad():
        z = model.embed(x)
        logits = model(x)
        for c in range(12):
            t = model.cell_types[c]
            expected = (z * (1 + model.gamma[t]) + model.beta[t]) @ model.cell_weight[:, c]
            torch.testing.assert_close(logits[:, c], expected + model.cell_bias[c])
