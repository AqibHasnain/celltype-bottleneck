"""Causal caQTL pairs, matched controls and observed effect profiles (PROJECT_PLAN.md 3.3).

Units are variant-peak pairs. Every pair is scored in the 1,344 bp window centered on its
TenK10K peak. Observed profiles B come from the near-center nominal pairs (ctb.data.tenk10k),
which exist for the test chromosomes only.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from ctb.data.genome import bases_at, gc_fraction, windows
from ctb.data.splits import load_splits, split_array
from ctb.data.tenk10k import CELL_TYPES, NOMINAL_DIR, parse_peak, parse_variant
from ctb.paths import PROCESSED, RAW, RESULTS
from ctb.utils.logging import get_logger
from ctb.utils.manifest import write_json

log = get_logger(__name__)

VARIANTS_DIR = PROCESSED / "variants"
TK = RAW / "tenk10k"

CAUSAL_PIP = 0.5
CONTROL_MAX_PIP = 0.01
CAUSAL_RADIUS = 600
N_CONTROLS = 5
MATCH_DIST = 50
MATCH_GC = 0.02
MATCH_AF = 0.05
MIN_CALLED = 3
SIG_QVAL = 0.05
NEG_MAX_ABS_SLOPE = 0.05
LOCUS_GAP = 100_000
SEED = 0


def load_susie() -> pd.DataFrame:
    z = zipfile.ZipFile(TK / "susie.zip")
    s = pd.concat(
        [
            pd.read_csv(z.open(f"all_celltypes_caQTL_susie_annotated/{ct}_susie_annotated.csv"))
            for ct in CELL_TYPES
        ],
        ignore_index=True,
    )
    v = parse_variant(s["SNP"])
    p = parse_peak(s["peak"])
    s["chrom"], s["pos"], s["ref"], s["alt"] = v["chrom"], v["pos"], v["ref"], v["alt"]
    s["center"] = p["center"]
    s["dist"] = s["pos"] - 1 - s["center"]
    s["snv"] = (s["ref"].str.len() == 1) & (s["alt"].str.len() == 1)
    return s


def load_called() -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Per cell type and chromosome: MACS3 peak starts (sorted) and the running max of ends."""
    z = zipfile.ZipFile(TK / "ct_specific_peaks.zip")
    out = {}
    for ct in CELL_TYPES:
        name = f"TenK10K_ATAC_ct_specific_peaks/{ct}.csv"
        first = z.open(name).readline().decode()
        # Files come either as a pandas CSV with a header or as headerless tab-separated BED.
        if ",chrom,start,end" in first:
            df = pd.read_csv(z.open(name), index_col=0)[["chrom", "start", "end"]]
        else:
            df = pd.read_csv(z.open(name), sep="\t", header=None).iloc[:, :3]
            df.columns = ["chrom", "start", "end"]
        per_chrom = {}
        for chrom, g in df.groupby("chrom"):
            g = g.sort_values("start")
            per_chrom[chrom] = (g["start"].to_numpy(), np.maximum.accumulate(g["end"].to_numpy()))
        out[ct] = per_chrom
    return out


def overlaps(index: tuple[np.ndarray, np.ndarray] | None, start: np.ndarray, end: np.ndarray):
    if index is None:
        return np.zeros(len(start), dtype=bool)
    starts, max_end = index
    i = np.searchsorted(starts, end)  # intervals starting before `end`
    return (i > 0) & (max_end[np.maximum(i - 1, 0)] > start)


def load_nominal(chroms: list[str]) -> pd.DataFrame:
    parts = []
    for ct in CELL_TYPES:
        for chrom in chroms:
            df = pd.read_parquet(NOMINAL_DIR / f"{ct}.{chrom}.parquet")
            df["celltype"] = ct
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_tested(chroms: list[str]) -> dict[str, set[str]]:
    return {
        ct: {
            p
            for chrom in chroms
            for p in (NOMINAL_DIR / f"{ct}.{chrom}.tested_peaks.txt").read_text().split()
        }
        for ct in CELL_TYPES
    }


def load_sig(chroms: list[str]) -> pd.DataFrame:
    parts = []
    for ct in CELL_TYPES:
        for chrom in chroms:
            df = pd.read_csv(
                TK / "sig" / f"{ct}.{chrom}.csv",
                sep=None,
                engine="python",
                usecols=["phenotype_id", "qval", "pval_nominal_threshold"],
            )
            df["celltype"] = ct
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def assign_loci(chrom: np.ndarray, pos: np.ndarray, gap: int = LOCUS_GAP) -> np.ndarray:
    """Single-linkage clusters of positions closer than `gap` on the same chromosome."""
    order = np.lexsort((pos, chrom))
    c, p = chrom[order], pos[order]
    new = np.ones(len(order), dtype=bool)
    new[1:] = (c[1:] != c[:-1]) | (np.diff(p) > gap)
    ids = np.empty(len(order), dtype=np.int64)
    ids[order] = np.cumsum(new) - 1
    return ids


def ref_matches(chrom: np.ndarray, pos: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return bases_at(chrom, pos - 1) == ref


def window_gc(chrom: pd.Series, center: pd.Series) -> np.ndarray:
    """GC of the 1,344 bp window per pair, computed once per distinct peak."""
    key = pd.DataFrame({"chrom": chrom.to_numpy(), "center": center.to_numpy()})
    uniq = key.drop_duplicates().reset_index(drop=True)
    uniq["gc"] = gc_fraction(windows(uniq["chrom"].to_numpy(), uniq["center"].to_numpy()))
    return key.merge(uniq, on=["chrom", "center"], how="left")["gc"].to_numpy()


def build_causal(susie: pd.DataFrame) -> pd.DataFrame:
    hit = susie[
        (susie["PIP"] >= CAUSAL_PIP) & susie["snv"] & (susie["dist"].abs() <= CAUSAL_RADIUS)
    ]
    lead = hit.sort_values("PIP", ascending=False).drop_duplicates(["peak", "SNP"])
    causal = lead[["peak", "SNP", "chrom", "pos", "ref", "alt", "center", "dist"]].copy()
    causal["max_pip"] = lead["PIP"].to_numpy()
    causal["lead_celltype"] = lead["celltype"].to_numpy()
    causal["lead_slope"] = lead["slope"].to_numpy()
    af = susie.groupby("SNP")["af"].mean()
    causal["af"] = af.loc[causal["SNP"]].to_numpy()
    causal["n_types_causal"] = (
        hit.groupby(["peak", "SNP"])
        .size()
        .loc[list(zip(causal["peak"], causal["SNP"], strict=True))]
        .to_numpy()
    )
    causal["split"] = split_array(causal["chrom"].to_numpy())
    return causal[causal["split"] != "none"].reset_index(drop=True)


def attach_profiles(
    pairs: pd.DataFrame, nominal: pd.DataFrame, sig: pd.DataFrame, tested: dict[str, set[str]]
) -> pd.DataFrame:
    key = nominal.set_index(["phenotype_id", "variant_id", "celltype"])
    thr = sig.set_index(["phenotype_id", "celltype"])
    for ct in CELL_TYPES:
        idx = pd.MultiIndex.from_arrays([pairs["peak"], pairs["SNP"], [ct] * len(pairs)])
        found = key.reindex(idx)
        pairs[f"slope:{ct}"] = found["slope"].to_numpy()
        pairs[f"pval:{ct}"] = found["pval_nominal"].to_numpy()
        t = thr.reindex(pd.MultiIndex.from_arrays([pairs["peak"], [ct] * len(pairs)]))
        sig_ct = (t["qval"].to_numpy() < SIG_QVAL) & (
            pairs[f"pval:{ct}"].to_numpy() < t["pval_nominal_threshold"].to_numpy()
        )
        pairs[f"sig:{ct}"] = sig_ct
        pairs[f"tested:{ct}"] = pairs["peak"].isin(tested[ct]).to_numpy()
    return pairs


def match_controls(causal: pd.DataFrame, pool: pd.DataFrame, rng: np.random.Generator):
    """Greedy matching without replacement, causal pairs visited in random order."""
    used = np.zeros(len(pool), dtype=bool)
    pd_abs = pool["dist"].abs().to_numpy()
    p_gc, p_af = pool["gc"].to_numpy(), pool["af"].to_numpy()
    rows = []
    for i in rng.permutation(len(causal)):
        c = causal.iloc[i]
        ok = (
            ~used
            & (np.abs(pd_abs - abs(c["dist"])) <= MATCH_DIST)
            & (np.abs(p_gc - c["gc"]) <= MATCH_GC)
            & (np.abs(p_af - c["af"]) <= MATCH_AF)
        )
        cand = np.flatnonzero(ok)
        pick = rng.choice(cand, size=min(N_CONTROLS, len(cand)), replace=False) if len(cand) else []
        used[pick] = True
        rows += [(j, c["pair_id"]) for j in pick]
    ctrl = pool.iloc[[j for j, _ in rows]].copy()
    ctrl["matched_to"] = [m for _, m in rows]
    return ctrl.reset_index(drop=True)


def build() -> None:
    test_chroms = sorted(load_splits()["test"])
    susie = load_susie()
    causal = build_causal(susie)
    causal["pair_id"] = causal["peak"] + "|" + causal["SNP"]
    causal["ref_ok"] = ref_matches(
        causal["chrom"].to_numpy(), causal["pos"].to_numpy(), causal["ref"].to_numpy()
    )
    n_ref_bad = int((~causal["ref_ok"]).sum())
    causal = causal[causal["ref_ok"]].drop(columns="ref_ok").reset_index(drop=True)
    causal["gc"] = window_gc(causal["chrom"], causal["center"])

    called = load_called()
    p = parse_peak(causal["peak"])
    for ct in CELL_TYPES:
        causal[f"called:{ct}"] = np.zeros(len(causal), dtype=bool)
        for chrom, g in p.groupby("chrom"):
            causal.loc[g.index, f"called:{ct}"] = overlaps(
                called[ct].get(chrom), g["start"].to_numpy(), g["end"].to_numpy()
            )
    causal["n_called"] = causal[[f"called:{ct}" for ct in CELL_TYPES]].sum(axis=1)
    causal["locus"] = assign_loci(causal["chrom"].to_numpy(), causal["pos"].to_numpy())

    test = causal[causal["split"] == "test"].copy()
    nominal = load_nominal(test_chroms)
    sig = load_sig(test_chroms)
    tested = load_tested(test_chroms)
    test = attach_profiles(test, nominal, sig, tested)
    test["specific_set"] = test["n_called"] >= MIN_CALLED

    # Control pool: near-center SNV pairs on test chromosomes whose variant has PIP < 0.01 in
    # every SuSiE output of the six types (variants absent from every credible set count as
    # PIP < 0.01) and is not itself causal anywhere.
    high_pip = set(susie.loc[susie["PIP"] >= CONTROL_MAX_PIP, "SNP"])
    pool = (
        nominal.groupby(["phenotype_id", "variant_id"])
        .agg(af=("af", "mean"), dist=("dist_to_center", "first"))
        .reset_index()
        .rename(columns={"phenotype_id": "peak", "variant_id": "SNP"})
    )
    pool = pool[~pool["SNP"].isin(high_pip) & (pool["dist"].abs() <= CAUSAL_RADIUS + MATCH_DIST)]
    v = parse_variant(pool["SNP"])
    pool["chrom"], pool["pos"], pool["ref"], pool["alt"] = v["chrom"], v["pos"], v["ref"], v["alt"]
    pool["center"] = parse_peak(pool["peak"])["center"]
    pool = pool[
        ref_matches(pool["chrom"].to_numpy(), pool["pos"].to_numpy(), pool["ref"].to_numpy())
    ].reset_index(drop=True)
    pool["gc"] = window_gc(pool["chrom"], pool["center"])
    controls = match_controls(test, pool, np.random.default_rng(SEED))
    controls["pair_id"] = controls["peak"] + "|" + controls["SNP"]
    controls["locus"] = test.set_index("pair_id").loc[controls["matched_to"], "locus"].to_numpy()

    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    causal.to_parquet(VARIANTS_DIR / "causal_all.parquet", index=False)
    test.to_parquet(VARIANTS_DIR / "causal_test.parquet", index=False)
    controls.to_parquet(VARIANTS_DIR / "controls_test.parquet", index=False)

    spec = test[test["specific_set"]]
    n_ctrl = controls.groupby("matched_to").size().reindex(test["pair_id"], fill_value=0)
    summary = {
        "causal_pairs_all_chroms": int(len(causal)),
        "causal_pairs_by_split": causal["split"].value_counts().to_dict(),
        "causal_dropped_ref_mismatch": n_ref_bad,
        "test_causal_pairs": int(len(test)),
        "test_causal_unique_variants": int(test["SNP"].nunique()),
        "test_causal_loci": int(test["locus"].nunique()),
        "test_pairs_specific_set": int(len(spec)),
        "test_causal_by_lead_celltype": test["lead_celltype"].value_counts().to_dict(),
        "test_causal_pip_ge_05_by_celltype": {
            ct: int(
                susie[(susie["celltype"] == ct) & susie["snv"] & (susie["PIP"] >= CAUSAL_PIP)]
                .merge(test[["peak", "SNP"]], on=["peak", "SNP"])
                .shape[0]
            )
            for ct in CELL_TYPES
        },
        "specific_set_called_by_celltype": {
            ct: int(spec[f"called:{ct}"].sum()) for ct in CELL_TYPES
        },
        "specific_set_significant_by_celltype": {
            ct: int(spec[f"sig:{ct}"].sum()) for ct in CELL_TYPES
        },
        "test_pairs_missing_any_slope": int(
            test[[f"slope:{ct}" for ct in CELL_TYPES]].isna().any(axis=1).sum()
        ),
        "control_pool_pairs": int(len(pool)),
        "controls": int(len(controls)),
        "causal_with_5_controls": int((n_ctrl == N_CONTROLS).sum()),
        "causal_with_0_controls": int((n_ctrl == 0).sum()),
    }
    write_json(summary, RESULTS / "phase0" / "variant_sets.json")
    log.info("variant sets: %s", summary)


if __name__ == "__main__":
    build()
