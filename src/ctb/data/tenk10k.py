"""TenK10K caQTL access: pinned URLs and the near-center extraction of nominal pairs.

The full nominal-pair parquet files are 1 to 6 GB per chromosome per cell type. Only SNVs
within `NEAR_CENTER_BP` of their peak center are ever used (causal pairs, matched controls and
effect profiles all sit inside the 1,344 bp window), so each file is streamed one row group at
a time over HTTP with column projection and filtered in memory. Nothing large touches disk.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ctb.paths import PROCESSED
from ctb.utils.logging import get_logger

log = get_logger(__name__)

HF_REPO = "anglixue/TenK10K_multiome"
HF_SHA = "a9baa08bc22e5fb4fca204e00df4e65534285423"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_SHA}"

CELL_TYPES = ("B_naive", "CD4_Naive", "CD8_Naive", "NK", "CD14_Mono", "CD16_Mono")
# 600 bp is the causal inclusion radius; the extra 50 bp is the control distance-matching slack.
NEAR_CENTER_BP = 650
_COLUMNS = ["phenotype_id", "variant_id", "af", "pval_nominal", "slope", "slope_se"]
_RETRIES = 6

NOMINAL_DIR = PROCESSED / "tenk10k_nominal_near"


def nominal_url(cell_type: str, chrom: str) -> str:
    return f"{HF_BASE}/caQTL_common_TensorQTL/{cell_type}/TenK10K.cis_qtl_pairs.{chrom}.parquet"


def sig_url(cell_type: str, chrom: str) -> str:
    return f"{HF_BASE}/caQTL_common_TensorQTL/{cell_type}/TenK10K.sig_cis_qtl_pairs.{chrom}.csv"


def parse_peak(ids: pd.Series) -> pd.DataFrame:
    """'chr2:100-900' -> chrom, start, end, center (center = floor midpoint)."""
    parts = ids.str.extract(r"^(chr[^:]+):(\d+)-(\d+)$")
    start = parts[1].astype(np.int64)
    end = parts[2].astype(np.int64)
    return pd.DataFrame(
        {"chrom": parts[0], "start": start, "end": end, "center": (start + end) // 2},
        index=ids.index,
    )


def parse_variant(ids: pd.Series) -> pd.DataFrame:
    """'2:12345:A:G' -> chrom, pos (1-based), ref, alt."""
    parts = ids.str.split(":", n=3, expand=True)
    chrom = parts[0].where(parts[0].str.startswith("chr"), "chr" + parts[0])
    return pd.DataFrame(
        {"chrom": chrom, "pos": parts[1].astype(np.int64), "ref": parts[2], "alt": parts[3]},
        index=ids.index,
    )


def _filter_near(table: pa.Table) -> tuple[pd.DataFrame, np.ndarray]:
    df = table.to_pandas()
    tested = df["phenotype_id"].unique()
    var = parse_variant(df["variant_id"])
    snv = (var["ref"].str.len() == 1) & (var["alt"].str.len() == 1)
    df, var = df[snv], var[snv]
    peak = parse_peak(df["phenotype_id"])
    dist = var["pos"].to_numpy() - 1 - peak["center"].to_numpy()
    keep = np.abs(dist) <= NEAR_CENTER_BP
    out = df[keep].copy()
    out["dist_to_center"] = dist[keep]
    return out, tested


def _read_group(url: str, i: int) -> pa.Table:
    for attempt in range(_RETRIES):
        try:
            fs = fsspec.filesystem("https")
            with fs.open(url, block_size=8 << 20) as fh:
                return pq.ParquetFile(fh).read_row_group(i, columns=_COLUMNS)
        except Exception as exc:  # network errors surface as many exception types
            wait = 2**attempt
            log.warning("row group %d of %s failed (%s); retry in %ds", i, url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"giving up on row group {i} of {url}")


def extract(cell_type: str, chrom: str, workers: int = 4) -> None:
    out = NOMINAL_DIR / f"{cell_type}.{chrom}.parquet"
    tested_out = NOMINAL_DIR / f"{cell_type}.{chrom}.tested_peaks.txt"
    if out.exists() and tested_out.exists():
        return
    url = nominal_url(cell_type, chrom)
    with fsspec.filesystem("https").open(url) as fh:
        n_groups = pq.ParquetFile(fh).metadata.num_row_groups
    t0 = time.time()

    def work(i: int) -> tuple[pd.DataFrame, np.ndarray]:
        return _filter_near(_read_group(url, i))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(work, range(n_groups)))
    near = pd.concat([p[0] for p in parts], ignore_index=True)
    tested = np.unique(np.concatenate([p[1] for p in parts]))
    NOMINAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    near.to_parquet(tmp, index=False)
    tmp.rename(out)
    tested_out.write_text("\n".join(tested) + "\n")
    log.info(
        "%s %s: %d groups, %d near-center SNV rows, %d tested peaks, %.0fs",
        cell_type,
        chrom,
        n_groups,
        len(near),
        len(tested),
        time.time() - t0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroms", nargs="+", default=["chr2", "chr8", "chr13"])
    ap.add_argument("--cell-types", nargs="+", default=list(CELL_TYPES))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    for chrom in args.chroms:
        for ct in args.cell_types:
            extract(ct, chrom, args.workers)


if __name__ == "__main__":
    main()
