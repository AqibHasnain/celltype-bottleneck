"""JASPAR parsing, consensus sequences, dinucleotide shuffling and PWM scanning."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

from ctb.paths import RAW

JASPAR = RAW / "jaspar" / "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
PSEUDO = 0.01
REL_SCORE_MIN = 0.8


@dataclass(frozen=True)
class Motif:
    matrix_id: str
    name: str
    probs: np.ndarray  # (w, 4) in ACGT order

    @property
    def genes(self) -> list[str]:
        return [g.upper() for g in self.name.split("::")]

    def consensus_codes(self) -> np.ndarray:
        return self.probs.argmax(axis=1).astype(np.uint8)

    def log_odds(self) -> np.ndarray:
        p = (self.probs + PSEUDO) / (1 + 4 * PSEUDO)
        return np.log2(p / 0.25)


def load_jaspar() -> list[Motif]:
    text = JASPAR.read_text()
    motifs = []
    for block in text.split("MOTIF ")[1:]:
        lines = block.strip().splitlines()
        matrix_id, name = lines[0].split()[:2]
        rows = [r for r in lines[2:] if re.match(r"^\s*[0-9.]", r)]
        probs = np.array([[float(x) for x in r.split()] for r in rows])
        motifs.append(Motif(matrix_id, name, probs))
    return motifs


def find_motif(motifs: list[Motif], name: str) -> Motif:
    hits = [m for m in motifs if m.name.upper() == name.upper()]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} JASPAR matrices named {name}")
    return hits[0]


def dinucleotide_shuffle(codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform random sequence with the same dinucleotide counts (Altschul and Erickson 1985),
    via a random Eulerian path that ends on the original last symbol."""
    seq = np.asarray(codes)
    if len(seq) < 3:
        return seq.copy()
    edges: dict[int, list[int]] = {}
    for a, b in zip(seq[:-1], seq[1:], strict=True):
        edges.setdefault(int(a), []).append(int(b))
    last = int(seq[-1])
    # Wilson's algorithm: a random arborescence of last-exit edges rooted at `last`.
    in_tree = {last}
    nxt: dict[int, int] = {}
    for v in edges:
        u = v
        while u not in in_tree:
            nxt[u] = int(rng.choice(edges[u]))
            u = nxt[u]
        u = v
        while u not in in_tree:
            in_tree.add(u)
            u = nxt[u]
    order: dict[int, list[int]] = {}
    for v, succ in edges.items():
        succ = list(succ)
        if v != last:
            succ.remove(nxt[v])
        rng.shuffle(succ)
        if v != last:
            succ.append(nxt[v])
        order[v] = succ
    out = [int(seq[0])]
    pos = {v: 0 for v in order}
    for _ in range(len(seq) - 1):
        v = out[-1]
        out.append(order[v][pos[v]])
        pos[v] += 1
    return np.array(out, dtype=seq.dtype)


def best_scores_over(codes: np.ndarray, motifs: list[Motif], center: int) -> np.ndarray:
    """Best log-odds score, over both strands, of each motif among placements covering `center`.

    codes: (N, L) uint8 with N=4. Returns (N, n_motifs). Every placement covering `center` must
    fit in the sequence: with wmax the widest motif, pass L = 2 * wmax - 1 and center = wmax - 1.
    """
    wmax = max(m.probs.shape[0] for m in motifs)
    weights = np.zeros((2 * len(motifs), 4, wmax), dtype=np.float32)
    widths = np.zeros(2 * len(motifs), dtype=np.int64)
    for i, m in enumerate(motifs):
        lo = m.log_odds().T  # (4, w)
        w = lo.shape[1]
        weights[2 * i, :, :w] = lo
        weights[2 * i + 1, :, :w] = lo[::-1, ::-1]  # reverse complement
        widths[2 * i : 2 * i + 2] = w
    x = torch.nn.functional.one_hot(torch.as_tensor(codes, dtype=torch.long), 5)[..., :4]
    x = x.permute(0, 2, 1).float()
    out = torch.nn.functional.conv1d(x, torch.from_numpy(weights))  # (N, 2M, L - wmax + 1)
    starts = torch.arange(out.shape[-1])
    w = torch.from_numpy(widths)[:, None]
    covers = (starts[None, :] <= center) & (starts[None, :] + w > center)
    out = out.masked_fill(~covers[None], float("-inf"))
    best = out.amax(dim=-1).reshape(codes.shape[0], len(motifs), 2).amax(dim=-1)
    return best.numpy()


def score_range(motifs: list[Motif]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([m.log_odds().min(axis=1).sum() for m in motifs])
    hi = np.array([m.log_odds().max(axis=1).sum() for m in motifs])
    return lo, hi
