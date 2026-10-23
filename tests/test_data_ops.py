import numpy as np
import pandas as pd

from ctb.data.variants import MATCH_AF, MATCH_DIST, MATCH_GC, assign_loci, match_controls
from ctb.eval.motifs import dinucleotide_shuffle


def _dinucs(s: np.ndarray) -> dict:
    return pd.Series(list(zip(s[:-1], s[1:], strict=True))).value_counts().to_dict()


def test_dinucleotide_shuffle_preserves_dinucleotides_and_ends():
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 5, size=1344).astype(np.uint8)
    out = dinucleotide_shuffle(seq, rng)
    assert _dinucs(out) == _dinucs(seq)
    assert out[0] == seq[0] and out[-1] == seq[-1]
    assert not np.array_equal(out, seq)


def test_loci_link_positions_within_100kb_on_the_same_chromosome():
    chrom = np.array(["chr2", "chr2", "chr2", "chr8", "chr2"])
    pos = np.array([1_000, 90_000, 400_000, 1_000, 180_000])
    ids = assign_loci(chrom, pos)
    assert ids[0] == ids[1] == ids[4]  # chained through 90,000
    assert len({ids[0], ids[2], ids[3]}) == 3


def test_controls_respect_every_tolerance_and_are_never_reused():
    rng = np.random.default_rng(0)
    causal = pd.DataFrame(
        {"pair_id": ["a", "b"], "dist": [100, -100], "gc": [0.5, 0.5], "af": [0.3, 0.3]}
    )
    pool = pd.DataFrame(
        {
            "dist": rng.integers(-300, 300, 400),
            "gc": rng.uniform(0.4, 0.6, 400),
            "af": rng.uniform(0.1, 0.5, 400),
        }
    )
    ctrl = match_controls(causal, pool, rng)
    c = causal.set_index("pair_id").loc[ctrl["matched_to"]]
    assert (np.abs(ctrl["dist"].abs().to_numpy() - c["dist"].abs().to_numpy()) <= MATCH_DIST).all()
    assert (np.abs(ctrl["gc"].to_numpy() - c["gc"].to_numpy()) <= MATCH_GC).all()
    assert (np.abs(ctrl["af"].to_numpy() - c["af"].to_numpy()) <= MATCH_AF).all()
    assert ctrl.groupby("matched_to").size().max() <= 5
    assert not ctrl.duplicated(subset=["dist", "gc", "af"]).any()
