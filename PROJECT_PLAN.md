# Does a shared sequence bottleneck limit cell-type-specific variant effects in single-cell accessibility models?

An executable project plan, written before any experiment is run. Hypotheses, metrics, thresholds and splits are fixed here. Changes after results are seen go in `docs/DECISIONS.md` with the reason and the date.

## 1. Summary

scBasset (Yuan and Kelley, Nat Methods 2022) predicts the accessibility of every peak in every cell from DNA sequence. A convolutional tower maps a 1,344 bp sequence to a 32-dimensional bottleneck, and each cell is a learned vector that reads that bottleneck out linearly. The predicted effect of a variant in cell c is therefore `w_c . (z_alt - z_ref)`: every cell type sees the same 32-dimensional change, weighted differently. A variant that opens chromatin in monocytes and closes it in B cells must be represented by a direction in that 32-dimensional space that the two cell vectors project with opposite signs. Whether 32 dimensions leave enough room for this has not been measured.

Gilfeather, Chikina and Kostka (bioRxiv 2026, "When Less Is Enough") apply post hoc SVD to the output layers of Sei, Borzoi, Enformer and AlphaGenome and report that rank-1 approximations can match or beat the full models on variant-effect benchmarks. This project asks the complementary question: whether the benchmarks hide the cost. Aggregate variant benchmarks reward getting the shared component of an effect right (is this variant causal, which direction does it push accessibility). They are close to insensitive to whether the model gets the cell-type-specific component right. This project separates the two, trains models at a range of bottleneck widths rather than truncating trained ones, and tests a cell-type-conditioned head as a fix.

Phase 1 trains scBasset at several bottleneck widths on public PBMC multiome data and measures the specific component of predicted effects against cell-type-resolved caQTLs. Phase 2 tests a conditioned head at matched parameter count. Phase 2 runs regardless of the Phase 1 outcome, because a negative Phase 1 result makes it a check that the fix costs nothing.

## 2. Hypotheses

**H1 (aggregate metrics are flat in width).** On held-out chromosomes, causal-versus-matched-control caQTL discrimination (AUROC of `|mean predicted effect|`) and sign concordance at fine-mapped caQTLs change by less than 0.02 between bottleneck width 8 and width 512.

**H2 (the specific component is width-limited).** The cell-type-specific metrics in section 5.2 (centered profile correlation and specificity AUROC) increase with bottleneck width, with the 512 arm exceeding the 32 arm by a margin whose paired-bootstrap 95% interval excludes zero.

**H3 (the fix).** A cell-type-conditioned head (section 6) at the parameter count of the width-32 arm raises the specific-component metrics over width 32 by the same criterion as H2, and lowers held-out-chromosome peak AUROC by no more than 0.01.

**Non-claims.** The project makes no claim about scBasset's cell embeddings or clustering quality, no state-of-the-art claim for variant effect prediction, and no claim beyond PBMC cell types and one training donor.

Any of H1 to H3 may come out negative. A negative H2 means that within PBMC, 32 dimensions are enough, which is a finding worth reporting with the same care.

## 3. Data

All sources are public under CC BY 4.0. `data/SOURCES.md` records each URL, version, checksum and license.

| Source | Use | Verified |
|---|---|---|
| 10x Genomics PBMC 10k multiome, granulocytes removed, Cell Ranger ARC 2.0.0, GRCh38 | Training data: ATAC fragments (2.4 GB), peaks BED (143,887 peaks), filtered feature-barcode matrix (RNA + ATAC, 192 MB) | HEAD requests return 200 at `cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/` |
| TenK10K phase 1 multiome caQTLs (Xue et al., medRxiv 2025), Hugging Face `anglixue/TenK10K_multiome`, GRCh38 | Ground truth: per-cell-type SuSiE fine-mapping (`Fine_mapping/all_celltypes_caQTL_susie_annotated.zip`, 195 MB, columns celltype, peak, SNP, PIP, Credible_Set, slope, af, pval_nominal), per-cell-type significant and independent cis pairs, and full nominal pairs as parquet | Repo listing and fine-mapping README read; not gated |
| TenK10K ChromBPNet variant scores, same repo | External baseline: one ChromBPNet model per cell type, so full rank by construction | Directory present; score file format to confirm in Phase 0 |
| hg38 reference FASTA (UCSC) | Sequence extraction | Standard |
| JASPAR 2024 CORE vertebrates | Motif-insertion probes and the motif baseline | Standard |

The TenK10K data are a preprint release that may change. Record the Hugging Face commit SHA used, and pin it.

### 3.1 Cell types

The 10x cells are labelled from the RNA modality by marker-gene scoring and Leiden clustering, then mapped to six TenK10K labels with enough cells in both datasets:

| Label here | TenK10K label | Marker genes for annotation |
|---|---|---|
| B | B_naive | MS4A1, CD79A, TCL1A, IGHD |
| CD4 T | CD4_Naive | CD3E, CD4, CCR7, SELL |
| CD8 T | CD8_Naive | CD3E, CD8A, CCR7 |
| NK | NK | NKG7, GNLY, KLRD1 |
| CD14 Mono | CD14_Mono | CD14, LYZ, S100A8 |
| CD16 Mono | CD16_Mono | FCGR3A, MS4A7 |

The training donor's naive and memory compartments are not separated in the ATAC labels, so "CD4 T" in the training data contains memory cells while the TenK10K match is naive only. This mismatch is the same for every arm, so it can depress absolute numbers but cannot favour one arm.

A cell-type effect is the mean of predicted logit differences over the cells of that type.

### 3.2 Splits

Chromosomes are held out for every peak and every variant. Validation: chr7. Test: chr2, chr8, chr13. Training: all other autosomes. chrX, chrY and chrM are excluded. `configs/splits.yaml` is the single source of truth and `tests/test_splits.py` asserts that no test-chromosome peak or variant appears in a training shard.

Variants on training chromosomes are scored in a secondary analysis only, because the model has seen the reference sequence around them.

### 3.3 Variant sets

From the SuSiE results, restricted to the six cell types and to SNVs (indels excluded):
- **Causal:** variant-peak pairs with PIP >= 0.5 in at least one of the six cell types, with the variant within 600 bp of the TenK10K peak center, so that it falls inside the 1,344 bp window centered on that peak.
- **Matched controls:** for each causal variant, up to 5 variants with PIP < 0.01 in all six cell types, matched on distance to the TenK10K peak center (within 50 bp), GC content of the 1,344 bp window (within 0.02), and allele frequency (within 0.05).
- **Effect profile:** for each causal variant-peak pair, the TenK10K slope in each of the six cell types, read from the full nominal pairs. The parquet files are 1 to 6 GB per chromosome per cell type, so they are read with predicate pushdown over HTTP or downloaded only for the three test chromosomes. Phase 0 decides which.

A causal pair enters the specificity analyses only if its peak is called in at least three of the six cell types, so that a missing effect means no effect and not no peak.

## 4. Model

A PyTorch reimplementation of scBasset following the published architecture: one-hot 1,344 bp input with reverse-complement and +/-3 bp shift augmentation, a stem convolution, a tower of 6 convolution blocks, a dense pre-bottleneck layer of 256, a bottleneck of width d, and a per-cell linear readout with a per-cell intercept that absorbs sequencing depth. Binary cross-entropy against the binarized peak-by-cell matrix.

scvi-tools ships a PyTorch port (`scvi.external.SCBASSET`, default `n_bottleneck_layer=32`). It is used as a reference implementation, not a dependency: the Phase 0 gate in section 8 checks that this reimplementation reaches comparable held-out performance on the same split.

Peaks are filtered to those accessible in at least 1% of cells, and that threshold is fixed in `configs/base.yaml`.

## 5. Metrics

### 5.1 Model quality

- **Peak AUROC:** held-out-chromosome AUROC per cell, averaged over cells, as in scBasset.
- **Pseudobulk correlation:** per cell type, Pearson correlation between predicted and observed pseudobulk accessibility over test-chromosome peaks.

### 5.2 Variant effects

Write `E` for the variants x 6 matrix of predicted cell-type effects and `B` for the matching matrix of TenK10K slopes. Each row splits into a shared part (the row mean) and a specific part (the row with its mean subtracted).

Aggregate metrics, the ones existing benchmarks measure:
- **Causal AUROC:** causal versus matched controls, scored by `|row mean of E|`.
- **Sign concordance:** fraction of causal variants where the sign of the row mean of `E` matches the sign of the slope in the cell type with the highest PIP.

Specific-component metrics:
- **Centered profile correlation (CPC):** for each causal variant, Pearson correlation between the centered rows of `E` and `B`, averaged over variants whose centered `B` has a norm above the 50th percentile, so that variants with flat observed profiles do not dominate.
- **Specificity AUROC:** for variant-cell-type pairs labelled specific-positive (significant in that cell type at the TenK10K threshold) or specific-negative (not significant there, `|slope| < 0.05`, and significant in another of the six), AUROC of the centered entry of `|E|`.
- **Specificity fraction (SF):** `||E - rowmean(E)||_F^2 / ||E||_F^2`, computed for predictions and for observed slopes on the same variants. SF in the observed data sets the target. A model with predicted SF far below the observed SF is compressing cell-type differences.

### 5.3 Representation rank

The entropy effective rank (Roy and Vetterli 2007) of three matrices on the test chromosomes: the predicted pseudobulk cell-type x peak matrix, the observed one, and the in-silico mutagenesis effect matrix (positions x cell types) over 500 randomly sampled test peaks. This is descriptive, not a hypothesis test.

### 5.4 Motif-insertion probes

Insert the JASPAR consensus for PAX5, EBF1 (B), TBX21, EOMES (NK), SPI1, CEBPB (monocytes) and TCF7, LEF1 (T) at the center of 1,000 dinucleotide-shuffled test-chromosome backgrounds. A probe passes if the cell type carrying the factor has the largest predicted effect among the six. Report the pass rate per arm. No labels from TenK10K are needed, so this is a check on the model that is independent of the caQTL data.

### 5.5 Uncertainty

Every interval is a 95% bootstrap over loci (variant-peak clusters within 100 kb are resampled together), 2,000 resamples. Between-arm comparisons are paired by seed and by variant.

## 6. Arms

| Arm | Bottleneck | Readout | Notes |
|---|---|---|---|
| W2, W8, W32, W128, W512 | d = 2, 8, 32, 128, 512 | per-cell linear | W32 is published scBasset |
| COND | d = 32 | per-cell linear on `z * (1 + gamma_t) + beta_t`, where `(gamma_t, beta_t)` is a learned FiLM vector for the cell's type t | Parameters matched to W32 by shrinking the pre-bottleneck layer; the match is asserted in a test |
| COND-SHUF | as COND | type labels permuted across cells | Control: any gain must disappear when the type labels carry no information |

Three seeds per arm. Optimizer, schedule, batch size, augmentation and stopping rule (best validation peak AUROC, patience 10 epochs) are identical across arms and live in `configs/base.yaml`.

The FiLM vectors use the RNA-derived labels, which are fixed before training. COND therefore uses information W32 does not have, and COND-SHUF exists to show whether the gain comes from that information or from the extra parameterization.

External references, scored on the same variants: TenK10K ChromBPNet (one model per cell type) and a motif baseline (change in best JASPAR match score at the variant, weighted by the TF's mean expression in each cell type from the multiome RNA).

## 7. Compute

All runs use one Apple M4 Max (128 GB, MPS). Phase 0 times one epoch at W32 and projects the full grid (7 arms x 3 seeds). If the projection exceeds 48 hours of wall clock, cut to two seeds for W2 and W8 first, and record the decision. `docs/COMPUTE_LOG.md` records every run: start time, git SHA, config hash, wall clock, outcome.

## 8. Phases and gates

**Phase 0: foundation.**
1. Download and checksum all sources, idempotently.
2. Annotate cell types from RNA; record cell counts per type.
3. Build the peak x cell matrix and the splits; leakage test passes.
4. Build the variant sets and record their sizes per cell type.
5. Train W32 with one seed.

**Gate G0.** W32 reaches a mean held-out peak AUROC within 0.02 of `scvi.external.SCBASSET` trained on the same split and data. At least 300 causal test-chromosome variants pass the specificity filter in 3.3. If either fails, stop and record why before continuing.

**Phase 1: width sweep.** Train W2 to W512 with three seeds. Compute every metric in section 5. Test H1 and H2.

**Phase 2: conditioned head.** Train COND and COND-SHUF with three seeds. Test H3.

**Phase 3: report.** `docs/REPORT.md` with figures: aggregate and specific metrics against width (the central figure), SF predicted versus observed, motif-probe pass rates, and the H3 comparison. Every claim made during the work that is later withdrawn is listed with its reason.

## 9. Repository layout

```
celltype-bottleneck/
  src/ctb/
    data/     downloads, cell annotation, peak matrix, splits, variant sets
    model/    scBasset reimplementation, FiLM head
    train/    training loop, arms, parameter matching
    eval/     metrics, variant scoring, motif probes, bootstrap, figures
    utils/    seeding, run manifests, logging
  configs/    base.yaml, splits.yaml, one file per arm
  docs/       DECISIONS.md, COMPUTE_LOG.md, REPORT.md
  data/       SOURCES.md (raw and processed data are not committed)
  results/    committed metrics, tables, figures
  tests/
```

`make setup`, `make data`, `make train ARM=...`, `make eval`, `make report` and `make test`, as in protein-interface-models.

## 10. Tests

Tests cover what would silently corrupt a result:
- No test-chromosome peak or variant in any training shard.
- Reference allele at each variant matches the hg38 base, and alt-allele sequences differ from reference at exactly one position.
- Reverse-complement scoring of a variant gives the same effect as forward scoring after averaging, on a small fixed model.
- COND and W32 have equal trainable parameter counts.
- Each metric in section 5 on small hand-computed matrices, including SF = 0 for a rank-1 row-constant matrix.

## 11. Risks

- **One training donor.** The model learns this donor's accessibility. caQTLs from 922 donors test whether it generalizes to sequence variation, which is the intended use, but a second 10x PBMC dataset is the first robustness check if time allows.
- **Label mismatch.** Naive-only TenK10K labels against mixed naive and memory training labels (section 3.1).
- **Few specific caQTLs.** Most caQTLs are shared across cell types. If fewer than 300 test-chromosome variants pass the filter, G0 fails and the fallback is to score training-chromosome variants as the primary set, with that weakness stated.
- **The TenK10K release changes.** Pinned commit SHA.
- **Peak-set mismatch.** TenK10K consensus peaks and 10x peaks differ. Each variant is scored in a 1,344 bp window centered on its TenK10K peak, matching how training windows are centered on 10x peaks, so the peak set only affects which variants are kept.
