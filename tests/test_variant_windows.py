import numpy as np
import pandas as pd
import pytest

from ctb.data.genome import FASTA, bases_at, decode
from ctb.data.variants import VARIANTS_DIR
from ctb.eval.score import allele_windows

needs_data = pytest.mark.skipif(
    not (FASTA.exists() and (VARIANTS_DIR / "causal_test.parquet").exists()),
    reason="genome or variant sets not built",
)


@needs_data
@pytest.mark.parametrize("name", ["causal_test", "controls_test"])
def test_reference_matches_hg38_and_alt_differs_at_one_base(name):
    pairs = pd.read_parquet(VARIANTS_DIR / f"{name}.parquet").sample(200, random_state=0)
    assert (bases_at(pairs["chrom"].to_numpy(), pairs["pos"].to_numpy() - 1) == pairs["ref"]).all()
    ref, alt = allele_windows(pairs)
    diff = ref != alt
    assert (diff.sum(axis=1) == 1).all()
    idx = diff.argmax(axis=1)
    assert [decode(alt[i, j : j + 1]) for i, j in enumerate(idx)] == list(pairs["alt"])
    # The variant sits at its signed distance from the window middle.
    assert np.array_equal(idx - 672, pairs["dist"].to_numpy())


@needs_data
def test_mismatched_reference_is_rejected():
    pairs = pd.read_parquet(VARIANTS_DIR / "causal_test.parquet").head(3).copy()
    pairs["ref"] = pairs["alt"]
    with pytest.raises(ValueError, match="reference allele"):
        allele_windows(pairs)
