import pandas as pd
import pytest

from ctb.data.peaks import PEAKS_DIR
from ctb.data.splits import load_splits, split_of
from ctb.data.variants import VARIANTS_DIR


def test_config_assigns_every_autosome_once_and_excludes_sex_and_mito():
    splits = load_splits()
    assigned = [c for name in ("train", "validation", "test") for c in splits[name]]
    assert sorted(assigned) == sorted(f"chr{i}" for i in range(1, 23))
    assert {split_of(c) for c in ("chrX", "chrY", "chrM", "chr1_KI270706v1_random")} == {None}
    assert split_of("2") == "test"


@pytest.mark.skipif(not (PEAKS_DIR / "peaks.tsv").exists(), reason="processed data not built")
def test_no_held_out_peak_or_variant_in_training():
    splits = load_splits()
    peaks = pd.read_csv(PEAKS_DIR / "peaks.tsv", sep="\t")
    # load_peak_data takes training indices from this column.
    train_chroms = set(peaks.loc[peaks["split"] == "train", "chrom"])
    assert train_chroms <= splits["train"]
    assert set(peaks.loc[peaks["split"] == "test", "chrom"]) == splits["test"]
    for name in ("causal_test", "controls_test"):
        path = VARIANTS_DIR / f"{name}.parquet"
        if path.exists():
            v = pd.read_parquet(path)
            assert v["chrom"].isin(splits["test"]).all()
            assert not v["chrom"].isin(train_chroms).any()
