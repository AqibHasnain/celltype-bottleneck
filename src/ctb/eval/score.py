"""Model-side scoring: variant effects, motif-insertion probes and in-silico mutagenesis.

A cell-type effect is the mean over that type's cells of logit(alt) - logit(ref), averaged over
the forward and reverse-complement strands.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from ctb.data.annotate import LABELS
from ctb.data.genome import SEQ_LEN, window_start, windows
from ctb.eval.motifs import dinucleotide_shuffle, find_motif, load_jaspar
from ctb.model.scbasset import ScBasset, one_hot, reverse_complement
from ctb.train.arms import build_model_config, type_indices

BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
# PROJECT_PLAN.md section 5.4: motif -> cell types that carry the factor.
PROBES = {
    "PAX5": ["B"],
    "EBF1": ["B"],
    "TBX21": ["NK"],
    "EOMES": ["NK"],
    "SPI1": ["CD14 Mono", "CD16 Mono"],
    "CEBPB": ["CD14 Mono", "CD16 Mono"],
    "TCF7": ["CD4 T", "CD8 T"],
    "LEF1": ["CD4 T", "CD8 T"],
}
N_BACKGROUNDS = 1000
N_ISM_PEAKS = 500


def load_run(run_dir, labels: np.ndarray, device: torch.device) -> ScBasset:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["config"]
    mc = build_model_config(cfg, n_cells=len(labels))
    types = None
    if mc.readout != "linear":
        shuffle = 1000 + cfg["seed"] if cfg["model"]["shuffle_types"] else None
        types = type_indices(labels, shuffle_seed=shuffle)
    model = ScBasset(mc, types).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    return model.eval()


def type_average_matrix(labels: np.ndarray) -> torch.Tensor:
    """(n_cells, 6): column t averages the cells labelled LABELS[t]."""
    m = np.stack([(labels == t) / max((labels == t).sum(), 1) for t in LABELS], axis=1)
    return torch.as_tensor(m, dtype=torch.float32)


@torch.no_grad()
def type_logits(
    model: ScBasset, codes: np.ndarray, avg: torch.Tensor, batch: int = 512, rc_average: bool = True
) -> np.ndarray:
    """(N, L) codes -> (N, 6) cell-type mean logits."""
    dev = next(model.parameters()).device
    avg = avg.to(dev)
    out = []
    for i in range(0, len(codes), batch):
        x = one_hot(torch.as_tensor(codes[i : i + batch], device=dev))
        logit = model(x)
        if rc_average:
            logit = 0.5 * (logit + model(reverse_complement(x)))
        out.append((logit @ avg).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def allele_windows(pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Reference and alternate windows centered on each pair's TenK10K peak center."""
    ref = windows(pairs["chrom"].to_numpy(), pairs["center"].to_numpy())
    idx = (pairs["pos"].to_numpy() - 1) - window_start(pairs["center"].to_numpy())
    if (idx < 0).any() or (idx >= SEQ_LEN).any():
        raise ValueError("a variant falls outside its window")
    rows = np.arange(len(pairs))
    expected = np.array([BASE_CODE[b] for b in pairs["ref"]], dtype=np.uint8)
    if not np.array_equal(ref[rows, idx], expected):
        raise ValueError("reference allele does not match hg38 inside a window")
    alt = ref.copy()
    alt[rows, idx] = np.array([BASE_CODE[b] for b in pairs["alt"]], dtype=np.uint8)
    return ref, alt


def variant_effects(model: ScBasset, pairs: pd.DataFrame, avg: torch.Tensor) -> np.ndarray:
    ref, alt = allele_windows(pairs)
    return type_logits(model, alt, avg) - type_logits(model, ref, avg)


def probe_backgrounds(test_codes: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(test_codes), size=N_BACKGROUNDS, replace=False)
    return np.stack([dinucleotide_shuffle(test_codes[i], rng) for i in pick])


def insert_center(backgrounds: np.ndarray, motif_codes: np.ndarray) -> np.ndarray:
    out = backgrounds.copy()
    start = SEQ_LEN // 2 - len(motif_codes) // 2
    out[:, start : start + len(motif_codes)] = motif_codes
    return out


def motif_probes(model: ScBasset, backgrounds: np.ndarray, avg: torch.Tensor) -> pd.DataFrame:
    motifs = load_jaspar()
    bg = type_logits(model, backgrounds, avg)
    rows = []
    for tf, targets in PROBES.items():
        m = find_motif(motifs, tf)
        eff = type_logits(model, insert_center(backgrounds, m.consensus_codes()), avg) - bg
        mean = eff.mean(axis=0)
        top = LABELS[int(mean.argmax())]
        per_bg_pass = np.isin(np.array(LABELS)[eff.argmax(axis=1)], targets).mean()
        rows.append(
            {
                "tf": tf,
                "matrix_id": m.matrix_id,
                "targets": "+".join(targets),
                "top_type": top,
                "pass": top in targets,
                "per_background_pass": per_bg_pass,
                **{f"effect:{t}": v for t, v in zip(LABELS, mean, strict=True)},
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def ism_matrix(
    model: ScBasset, peak_codes: np.ndarray, avg: torch.Tensor, batch: int = 1344
) -> np.ndarray:
    """Rows: every (peak, position, alternative base); columns: cell-type logit change.
    Forward strand only."""
    out = []
    for codes in peak_codes:
        ref = type_logits(model, codes[None], avg, rc_average=False)[0]
        muts = np.repeat(codes[None], SEQ_LEN * 3, axis=0)
        pos = np.repeat(np.arange(SEQ_LEN), 3)
        alts = np.array([[b for b in range(4) if b != c] if c < 4 else [0, 1, 2] for c in codes])
        muts[np.arange(len(muts)), pos] = alts.reshape(-1)
        out.append(type_logits(model, muts, avg, batch=batch, rc_average=False) - ref)
    return np.concatenate(out)
