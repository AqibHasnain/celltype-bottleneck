"""Idempotent downloads of every raw source, verified against recorded checksums.

A file's SHA256 is recorded in data/OBSERVED_CHECKSUMS.json on first download and every later
run verifies against it, so a silently changed upstream file fails loudly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import requests

from ctb.data.tenk10k import CELL_TYPES, HF_BASE, sig_url
from ctb.paths import RAW, ROOT
from ctb.utils.logging import get_logger

log = get_logger(__name__)

CHECKSUMS = ROOT / "data" / "OBSERVED_CHECKSUMS.json"
TENX_BASE = (
    "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/"
    "pbmc_granulocyte_sorted_10k_"
)
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]


@dataclass(frozen=True)
class Source:
    url: str
    dest: Path


def all_sources() -> list[Source]:
    tenx = [
        Source(TENX_BASE + name, RAW / "10x" / name)
        for name in ("atac_peaks.bed", "filtered_feature_bc_matrix.h5", "per_barcode_metrics.csv")
    ]
    tk = RAW / "tenk10k"
    tenk = [
        Source(f"{HF_BASE}/Fine_mapping/all_celltypes_caQTL_susie_annotated.zip", tk / "susie.zip"),
        Source(
            f"{HF_BASE}/ChromBPNet/TenK10K.caQTL.Final1Mb.susie_finemapped_results_w_"
            "ChromBPNetScores.parquet",
            tk / "cbp_susie.parquet",
        ),
        Source(
            f"{HF_BASE}/Miscellaneous/TenK10K_ATAC_MACS3_Combined_Peaks_Annotated.bed",
            tk / "consensus_peaks.bed",
        ),
        Source(
            f"{HF_BASE}/Miscellaneous/TenK10K_ATAC_ct_specific_peaks.zip",
            tk / "ct_specific_peaks.zip",
        ),
    ]
    tenk += [
        Source(sig_url(ct, chrom), tk / "sig" / f"{ct}.{chrom}.csv")
        for ct in CELL_TYPES
        for chrom in AUTOSOMES
    ]
    ref = [
        Source(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
            RAW / "genome" / "hg38.fa.gz",
        ),
        Source(
            "https://jaspar2024.elixir.no/download/data/2024/CORE/"
            "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt",
            RAW / "jaspar" / "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt",
        ),
    ]
    return tenx + tenk + ref


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    part.rename(dest)


def fetch_all() -> None:
    recorded = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    for src in all_sources():
        key = str(src.dest.relative_to(RAW))
        if not src.dest.exists():
            _download(src.url, src.dest)
        digest = sha256_of(src.dest)
        if key in recorded and recorded[key]["sha256"] != digest:
            raise RuntimeError(f"checksum mismatch for {key}: upstream file changed")
        recorded[key] = {"url": src.url, "sha256": digest, "bytes": src.dest.stat().st_size}
    CHECKSUMS.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
    log.info("%d sources present and verified", len(recorded))


if __name__ == "__main__":
    fetch_all()
