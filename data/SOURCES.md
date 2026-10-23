# Data sources

Raw files live under `data/raw/` (not committed). `make data` downloads them idempotently and
`data/OBSERVED_CHECKSUMS.json` records the SHA256 of every file; a later run fails if an upstream
file changes. All sources are CC BY 4.0 except hg38 (UCSC, free for all uses) and JASPAR
(CC BY 4.0).

| Source | Files | Version | Use |
|---|---|---|---|
| 10x Genomics PBMC granulocyte-sorted 10k multiome | `atac_peaks.bed`, `filtered_feature_bc_matrix.h5`, `per_barcode_metrics.csv` from `cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/` | Cell Ranger ARC 2.0.0, GRCh38 | Training: RNA for cell labels, `Peaks` features for the peak x cell matrix |
| TenK10K phase 1 multiome (Xue et al., medRxiv 2025, doi 10.1101/2025.08.27.25334533) | Hugging Face `anglixue/TenK10K_multiome`: `Fine_mapping/all_celltypes_caQTL_susie_annotated.zip`; `caQTL_common_TensorQTL/<type>/TenK10K.sig_cis_qtl_pairs.chr*.csv`; `caQTL_common_TensorQTL/<type>/TenK10K.cis_qtl_pairs.chr{2,8,13}.parquet` (streamed, not stored); `Miscellaneous/TenK10K_ATAC_ct_specific_peaks.zip`; `Miscellaneous/TenK10K_ATAC_MACS3_Combined_Peaks_Annotated.bed`; `ChromBPNet/<type>/chrombpnet_model_full_bias_NKFull1_fold{0-4}/models/chrombpnet_nobias.h5`; `ChromBPNet/TenK10K.caQTL.Final1Mb.susie_finemapped_results_w_ChromBPNetScores.parquet` | commit `a9baa08bc22e5fb4fca204e00df4e65534285423` | Ground truth (SuSiE, nominal slopes, significance thresholds, per-type peak calls) and the ChromBPNet reference |
| hg38 reference | `hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz` | UCSC, 2014-01-16 | Sequence windows and reference-allele checks |
| JASPAR 2024 CORE vertebrates, non-redundant | `jaspar2024.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt` | JASPAR 2024 | Motif probes and the motif baseline |

The six TenK10K types used are B_naive, CD4_Naive, CD8_Naive, NK, CD14_Mono and CD16_Mono. The
nominal-pair parquet files are reduced while streaming to SNVs within 650 bp of their peak
center (`src/ctb/data/tenk10k.py`); the reduced tables are rebuilt by `make data`.
