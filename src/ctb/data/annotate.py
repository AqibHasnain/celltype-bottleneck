"""Cell-type labels from the RNA modality: Leiden clusters labelled by marker-gene scores.

Every Cell Ranger ARC-called cell is kept for training. A cluster is labelled "other" when its
best marker set is a decoy (DCs, platelets, plasma cells, MAIT), when it is a doublet cluster
(Scrublet), or when its median gene count is below `MIN_MEDIAN_GENES`. "other" cells still train
the model but contribute to no cell-type effect.

Clusters are first split by lineage on mean CD3E, because cytotoxic CD8 T and MAIT cells
express NKG7 and GNLY as strongly as NK cells and win the NK score by argmax alone.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scanpy as sc

from ctb.paths import PROCESSED, RAW, RESULTS
from ctb.utils.logging import get_logger

log = get_logger(__name__)

# Label here -> (TenK10K label, marker genes), from PROJECT_PLAN.md section 3.1.
TYPES: dict[str, tuple[str, list[str]]] = {
    "B": ("B_naive", ["MS4A1", "CD79A", "TCL1A", "IGHD"]),
    "CD4 T": ("CD4_Naive", ["CD3E", "CD4", "CCR7", "SELL"]),
    "CD8 T": ("CD8_Naive", ["CD3E", "CD8A", "CCR7"]),
    "NK": ("NK", ["NKG7", "GNLY", "KLRD1"]),
    "CD14 Mono": ("CD14_Mono", ["CD14", "LYZ", "S100A8"]),
    "CD16 Mono": ("CD16_Mono", ["FCGR3A", "MS4A7"]),
}
LABELS = list(TYPES)
TENK_LABELS = [TYPES[k][0] for k in LABELS]
DECOYS: dict[str, list[str]] = {
    "pDC": ["LILRA4", "IL3RA", "CLEC4C"],
    "cDC": ["FCER1A", "CD1C", "CLEC10A"],
    "Platelet": ["PPBP", "PF4"],
    "Plasma": ["JCHAIN", "MZB1"],
    "MAIT": ["SLC4A10", "TRAV1-2"],
}
T_LINEAGE = {"CD4 T", "CD8 T", "MAIT"}
CD3E_T_MIN = 0.5
MIN_MEDIAN_GENES = 700
DOUBLET_CLUSTER_FRACTION = 0.25
# Small decoy sets score noisily around zero; a decoy claims a cluster only above this.
DECOY_MIN_SCORE = 0.3
OTHER = "other"
LEIDEN_RESOLUTION = 1.0
SEED = 0

LABELS_PATH = PROCESSED / "cell_labels.tsv"


def load_rna() -> sc.AnnData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.read_10x_h5(RAW / "10x" / "filtered_feature_bc_matrix.h5", gex_only=True)
    adata.var_names_make_unique()
    return adata


def annotate() -> pd.DataFrame:
    adata = load_rna()
    sc.pp.scrublet(adata, random_state=SEED)
    n_genes = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    work = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(work, max_value=10)
    sc.tl.pca(work, n_comps=50, random_state=SEED)
    sc.pp.neighbors(work, n_neighbors=15, random_state=SEED)
    sc.tl.leiden(
        work, resolution=LEIDEN_RESOLUTION, random_state=SEED, flavor="igraph", n_iterations=2
    )

    marker_sets = {**{k: v[1] for k, v in TYPES.items()}, **DECOYS}
    for name, genes in marker_sets.items():
        sc.tl.score_genes(adata, genes, score_name=f"score:{name}", random_state=SEED)
    scores = adata.obs[[f"score:{n}" for n in marker_sets]].copy()
    scores.columns = list(marker_sets)
    scores["cluster"] = work.obs["leiden"].to_numpy()
    by_cluster = scores.groupby("cluster", observed=True).mean()
    cd3e = np.asarray(adata[:, "CD3E"].X.todense()).ravel()
    qc = (
        pd.DataFrame(
            {
                "cd3e": cd3e,
                "n_genes": n_genes,
                "doublet": adata.obs["predicted_doublet"].to_numpy(),
                "cluster": scores["cluster"].to_numpy(),
            }
        )
        .groupby("cluster", observed=True)
        .agg(
            cd3e=("cd3e", "mean"),
            median_genes=("n_genes", "median"),
            doublet_frac=("doublet", "mean"),
        )
    )
    is_t = qc["cd3e"] >= CD3E_T_MIN
    eligible = by_cluster.copy()
    decoy_cols = list(DECOYS)
    eligible[decoy_cols] = eligible[decoy_cols].where(eligible[decoy_cols] >= DECOY_MIN_SCORE)
    t_cols = [c for c in eligible.columns if c in T_LINEAGE]
    non_t_cols = [c for c in eligible.columns if c not in T_LINEAGE]
    eligible.loc[is_t, non_t_cols] = np.nan
    eligible.loc[~is_t, t_cols] = np.nan
    best = eligible.idxmax(axis=1)
    best_score = eligible.max(axis=1)
    failed_qc = (qc["median_genes"] < MIN_MEDIAN_GENES) | (
        qc["doublet_frac"] >= DOUBLET_CLUSTER_FRACTION
    )
    cluster_label = best.where(best.isin(LABELS) & ~failed_qc & (best_score > 0), OTHER)

    out = pd.DataFrame(
        {
            "barcode": adata.obs_names,
            "cluster": scores["cluster"].to_numpy(),
            "label": cluster_label.loc[scores["cluster"]].to_numpy(),
        }
    )
    summary = by_cluster.round(2)
    summary = summary.join(qc.round(2))
    summary["best_set"] = best
    summary["n_cells"] = scores.groupby("cluster", observed=True).size()
    summary["label"] = cluster_label
    log.info("cluster marker scores:\n%s", summary.to_string())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(LABELS_PATH, sep="\t", index=False)
    (RESULTS / "phase0").mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "phase0" / "annotation_clusters.tsv", sep="\t")
    counts = out["label"].value_counts().reindex([*LABELS, OTHER], fill_value=0)
    counts.rename("n_cells").to_csv(RESULTS / "phase0" / "cell_counts.tsv", sep="\t")
    log.info("cells per label:\n%s", counts.to_string())
    return out


def load_labels() -> pd.DataFrame:
    return pd.read_csv(LABELS_PATH, sep="\t")


def mean_expression(genes: list[str]) -> pd.DataFrame:
    """Mean log-normalized expression per cell type (rows: genes present, columns: LABELS)."""
    adata = load_rna()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    labels = load_labels().set_index("barcode").loc[adata.obs_names, "label"].to_numpy()
    genes = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    sub = adata[:, genes].X
    sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
    return pd.DataFrame({lab: sub[labels == lab].mean(axis=0) for lab in LABELS}, index=genes)


if __name__ == "__main__":
    annotate()
