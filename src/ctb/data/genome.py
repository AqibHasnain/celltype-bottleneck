"""hg38 sequence access and DNA encoding.

Codes are A=0, C=1, G=2, T=3 (scBasset and scvi order) and N=4, which one-hots to all zeros.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pyfaidx

from ctb.paths import RAW

SEQ_LEN = 1344
HALF = SEQ_LEN // 2
FASTA = RAW / "genome" / "hg38.fa"

_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i
BASES = np.frombuffer(b"ACGTN", dtype=np.uint8)


@lru_cache(maxsize=1)
def genome() -> pyfaidx.Fasta:
    return pyfaidx.Fasta(str(FASTA), as_raw=True, sequence_always_upper=True)


def encode(seq: str) -> np.ndarray:
    return _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def decode(codes: np.ndarray) -> str:
    return BASES[np.asarray(codes, dtype=np.int64)].tobytes().decode("ascii")


def window_start(center: int) -> int:
    """0-based start of the SEQ_LEN window whose middle base is `center` (0-based)."""
    return center - HALF


def fetch_codes(chrom: str, start: int, end: int) -> np.ndarray:
    """Codes for the 0-based half-open interval, N-padded past either chromosome end."""
    fa = genome()[chrom]
    n = len(fa)
    lo, hi = max(start, 0), min(end, n)
    out = np.full(end - start, 4, dtype=np.uint8)
    if hi > lo:
        out[lo - start : hi - start] = encode(fa[lo:hi])
    return out


def windows(chroms: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            fetch_codes(c, window_start(int(m)), window_start(int(m)) + SEQ_LEN)
            for c, m in zip(chroms, centers, strict=True)
        ]
    )


def gc_fraction(codes: np.ndarray) -> np.ndarray:
    """GC fraction per row over called bases (N excluded)."""
    called = codes < 4
    gc = (codes == 1) | (codes == 2)
    return gc.sum(axis=-1) / np.maximum(called.sum(axis=-1), 1)
