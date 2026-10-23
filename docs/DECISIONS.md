# Decision log

Append-only, newest last. Each entry records what was decided, when, and why. Entries are written
at the time of the decision. Any entry that follows a seen result says so.

## 2026-10-22 - Novelty check against Gilfeather, Chikina and Kostka (bioRxiv 2026)

Read the v2 full text (JATS XML, posted 2026-07-28, DOI 10.64898/2026.01.21.700827). The paper
truncates linear layers of four trained bulk models (Sei, Borzoi, Enformer, AlphaGenome) with
post hoc SVD at ranks 1 to 2,048. It trains no model at a chosen width, analyses no single-cell
model, and does not separate shared from cell-type-specific components of variant effects. Its
cell-type analysis is zero-shot AUROC on bulk context-specific QTL sets (GM12878, microglia,
smooth muscle caQTLs; SPI1 bQTLs), and its Discussion defers what different ranks capture.
The trained-width sweep, the single-cell readout and the shared/specific split here are not
covered by it. The project proceeds as planned.

## 2026-10-22 - TenK10K release pinned

Hugging Face `anglixue/TenK10K_multiome` at commit
`a9baa08bc22e5fb4fca204e00df4e65534285423` (verified as current on 2026-10-22, last modified
2026-10-08). Every TenK10K URL in `src/ctb/data/tenk10k.py` resolves through that commit.

## 2026-10-22 - Nominal pairs are streamed over HTTP and reduced to near-center SNVs

The plan left the choice between predicate pushdown over HTTP and downloading the test
chromosomes to Phase 0. The parquet files are sorted by peak in 1,048,576-row groups. Peaks
needed for the test-chromosome analyses are spread over almost every row group, so row-group
pushdown would skip little. Each file is instead read row group by row group over HTTP with
column projection (6 of 9 columns) and filtered in memory to SNVs within 650 bp of the peak
center (600 bp causal radius plus the 50 bp distance-matching slack). Only the reduced table and
the list of tested peaks are written to disk. This covers the three test chromosomes for the
six cell types, about 60 GB of source files.

Consequence: observed effect profiles exist only for test chromosomes. The secondary
training-chromosome analysis (section 3.2) is limited to the metrics that need no profile or
controls, which is sign concordance against the lead-type SuSiE slope.

## 2026-10-22 - Peak x cell matrix taken from the Cell Ranger ARC filtered matrix

`filtered_feature_bc_matrix.h5` carries the 143,887 ATAC peaks as a `Peaks` feature type for the
11,898 called cells, which is the input scBasset and `scvi.external.SCBASSET` use. The 2.4 GB
fragments file adds nothing to that matrix and is not downloaded. Peaks on chrX, chrY, chrM and
non-reference contigs are dropped, then peaks accessible in fewer than 1% of cells. 98,507 peaks
remain: 79,443 train, 4,859 validation (chr7), 14,205 test (chr2, chr8, chr13).

## 2026-10-22 - Cell annotation rule

Leiden clustering (resolution 1.0) on 2,000 highly variable genes, then the mean
`scanpy.tl.score_genes` score of each marker set per cluster. The first pass, labelling each
cluster by the argmax over the six marker sets, called a CD3E+ CD8A+ GNLY+ cytotoxic CD8 T
cluster and a SLC4A10+ MAIT cluster "NK", and kept B-monocyte and T-monocyte doublet clusters.
The rule used is:

- A cluster is T lineage when its mean log-normalized CD3E is >= 0.5. T clusters choose among
  CD4 T, CD8 T and MAIT; other clusters among B, NK, CD14 Mono, CD16 Mono and the decoys.
- Decoy sets (pDC, cDC, platelet, plasma, MAIT on SLC4A10 and TRAV1-2) can claim a cluster only
  with a mean score >= 0.3, because small sets score noisily around zero.
- A cluster is "other" when a decoy wins, when the best score is <= 0, when >= 25% of its cells
  are Scrublet doublets, or when its median gene count is below 700.

Counts: B 923, CD4 T 3,187, CD8 T 2,600, NK 492, CD14 Mono 3,172, CD16 Mono 494, other 1,030
(`results/phase0/cell_counts.tsv`). All 11,898 cells train every model; "other" cells contribute
to no cell-type effect. CD8 T includes effector and memory CD8 clusters, the same kind of
naive/memory mismatch with TenK10K CD8_Naive that the plan records for CD4. One CD14 Mono
cluster (222 cells) has a cDC2 signature at 0.65 against 0.85 for CD14 Mono and keeps the CD14
label.

## 2026-10-22 - The pre-registered COND head cannot widen the bottleneck; COND-PRE is added

Written before any variant or COND result. With a per-cell linear readout,
`w_c . (z * (1 + gamma_t) + beta_t) + b_c = (w_c * (1 + gamma_t)) . z + (b_c + w_c . beta_t)`.
`w_c` is a free vector per cell, so `w_c * (1 + gamma_t)` ranges over the same set as `w_c`, and
`w_c . beta_t` is a per-cell constant that cancels in an alt-minus-ref effect. COND therefore
has exactly the function class of W32: every cell-type effect is still a fixed linear readout of
one shared 32-dimensional change. Any COND difference from W32 can only come from optimization.

COND and COND-SHUF are run as pre-registered and H3 is tested on COND as written. Two arms are
added to test the question H3 was meant to ask:

- COND-PRE: FiLM on the 256-channel pre-bottleneck activations, one (gamma, beta) per channel
  and type, so the bottleneck vector itself depends on the cell's type,
  `z_t = bottleneck(h * (1 + gamma_t) + beta_t)`.
- COND-PRE-SHUF: the same with type labels permuted across cells.

Both are reported separately from the pre-registered H3 test, as additional arms.

A related point for interpretation: with six cell types the effect matrix has at most six
columns, so any width d >= 6 removes the algebraic rank limit on effect profiles. W2 is the only
arm whose profiles are rank-limited (rank <= 2). A width effect between 8 and 512 has to come from
what the encoder learns at that width.

## 2026-10-22 - Parameter matching is to within one pre-bottleneck channel

One pre-bottleneck channel costs 738 parameters at d = 32 (512 conv weights, 2 BatchNorm, 224
bottleneck weights). The FiLM vectors add 448 parameters to COND (7 types including "other",
2 x 32) and 3,584 to COND-PRE at full width, so exact equality is not reachable by changing the
width. Each FiLM arm uses the largest pre-bottleneck width whose trainable count does not exceed
W32's: COND at 255 channels (4,898,560, 290 fewer than W32) and COND-PRE at 251 channels
(4,898,674, 176 fewer). `tests/test_param_matching.py` asserts this rule in place of exact
equality.

## 2026-10-22 - G0 reference settings

`scvi.external.SCBASSET` (scvi-tools 1.5.1, n_bottleneck_layer = 32) is trained on the same
peaks, cells and chromosome split through `external_indexing`, with its own optimizer defaults
(identical to ctb's), early stopping on its validation AUROC with patience 10, and best-epoch
weights restored. Its validation AUROC is computed over flattened batches, not per cell, so only
the test metric is shared code: per-cell AUROC over the 14,205 test peaks, averaged over cells.
N bases are drawn at random from A, C, G for scvi (its one-hot has no N) and are all-zero for
ctb. 24 of 98,507 windows contain an N.

## 2026-10-22 - Variant-set definitions not fixed by the plan

Set before any variant was scored.

- "Called" in a cell type means the TenK10K peak overlaps that type's MACS3 peak set
  (`Miscellaneous/TenK10K_ATAC_ct_specific_peaks.zip`). Being tested in TensorQTL does not
  discriminate: 91% to 99% of consensus peaks are tested in each of the six types.
- "Significant in a cell type at the TenK10K threshold" means the peak has permutation
  q < 0.05 in that type and the pair's nominal p-value is below that peak's
  `pval_nominal_threshold`, the TensorQTL definition of a significant pair.
- Variants absent from every SuSiE credible set of the six types count as PIP < 0.01 for the
  control pool, since SuSiE outputs list credible-set members only. Control candidates are
  near-center SNV pairs from the nominal files whose variant is not in the causal set.
- Distance matching uses absolute distance to the peak center. Allele frequency is the mean
  TenK10K alt-allele frequency over the six types. Controls are drawn without replacement,
  visiting causal pairs in a seeded random order.
- Variants whose reference allele does not match hg38 are dropped and counted.
- Specificity-AUROC score: for each variant and type, |E| minus the row mean of |E|.
- Rows of B with a missing slope (peak not tested in a type) use the observed types only, and
  the same mask is applied to E for every specific-component metric.

## 2026-10-22 - How H1 to H3 are judged, and other analysis details

Written before any variant was scored.

- H1 is supported when the 95% interval of W512 - W8 lies inside (-0.02, 0.02) for both causal
  AUROC and sign concordance. Point estimates within 0.02 with a wider interval are reported as
  "not demonstrated", since an interval that reaches 0.02 does not show a change below 0.02.
- H2 is supported when the W512 - W32 interval is above zero for both CPC and specificity AUROC,
  and partially supported when only one is. Arm means across all five widths are reported to
  show whether the change is monotone.
- H3 is supported when COND - W32 meets the H2 criterion and the mean over seeds of the paired
  test peak AUROC difference is >= -0.01. The same rule is applied to the added COND-PRE arm and
  reported separately. A gain counts as label-driven only if the shuffled control's interval
  against W32 is not above zero.
- An arm's value is the mean over its seeds of the per-seed metric. Seeds are paired by index
  across arms.
- The CPC row set (specificity-set rows whose centered B norm is above the median) is fixed once
  on the full data and reweighted in each bootstrap resample, so a resample cannot change which
  variants count as having a non-flat profile.
- Matched controls take the locus of the causal pair they were matched to.
- ISM rows are every (peak, position, alternative base) over 500 test peaks chosen with seed 0,
  forward strand only: 2,016,000 x 6 per run.
- Motif probes use the JASPAR 2024 non-redundant matrices named PAX5, EBF1, TBX21, EOMES, SPI1
  (MA0080.7, listed as Spi1), CEBPB, TCF7 and LEF1 (MA0768.3, listed as Lef1). TCF7 and LEF1
  pass when either T type is top, SPI1 and CEBPB when either monocyte type is top. The pass rate
  is over the 8 probes on the effect averaged over 1,000 backgrounds; the per-background pass
  rate is reported beside it.
- ChromBPNet reference: the TenK10K bias-free models were loaded into PyTorch from their HDF5
  weights (`src/ctb/eval/chrombpnet.py`), because the release's precomputed scores exist only for
  each variant's own credible-set cell types and not for controls. Scores are the 5-fold mean of
  the log-count difference in the 2,114 bp window centered on the variant, strand-averaged.
  Against TenK10K's own precomputed `log_counts_diff` on 60 random SNVs per type, the port gives
  r = 0.998 (slope 1.03) for CD14_Mono and r = 0.93 (slope 0.99) for B_naive. The ChromBPNet
  models were trained on TenK10K accessibility with folds that include this project's test
  chromosomes, so it is a reference point and not a held-out competitor.

## 2026-10-23 - scvi reference: AUROC metric computed on CPU

torchmetrics' binary AUROC fails with an out-of-bounds index on MPS for scvi's 128 x 11,898
batches, which aborts `SCBASSET.train` in the first epoch. `src/ctb/train/scvi_reference.py`
replaces the metric inside scvi's module: during validation it is computed on CPU, and the
training-batch AUROC, which scvi only logs, returns 0. Loss, gradients and the stopping monitor
are unchanged.

## 2026-10-23 - Grid projection: no seed cut

W32 seed 0 ran 63 epochs (best at 52) at about 70 s per epoch, 74 minutes in total. The
pre-registered grid (7 arms x 3 seeds) projects to about 26 hours and the two added arms to about
8 more, under the 48-hour limit, so all arms keep three seeds.

## 2026-10-23 - Grid paused after seed 0 of seven arms

Stopped at Aqib's request before a laptop shutdown. Finished and scored: seed 0 of W2, W8, W32,
W128, W512, COND and COND-SHUF, plus the ChromBPNet and motif references. COND-PRE seed 0 was
killed mid-training and is rerun from scratch on resume; `ctb.train.grid` skips every run that
has a `summary.json`. No hypothesis has been evaluated: H1 to H3 are judged on three seeds.

## 2026-10-23 - Seeds cut to two for the hypothesis arms; COND-PRE not run

Aqib set a time budget of about 7 hours for the remaining training, against about 22 hours for
the full grid. Decided before any hypothesis was evaluated. Seed 1 is trained for W8, W32,
W512, COND and COND-SHUF, the arms that H1 (W512 - W8), H2 (W512 - W32) and H3 (COND - W32, with
COND-SHUF as its control) compare, so every pre-registered hypothesis is tested on two seeds.
W2 and W128 keep one seed and are reported as points on the width curve only. Seed 2 is not run
for any arm. COND-PRE and COND-PRE-SHUF, added on 2026-10-22, are not run; the report states
the function-class equivalence of COND and W32 and leaves the pre-bottleneck head as future
work. Alternatives considered: two concurrent processes on the GPU (contention slowed the scvi
reference from 65 s to 188-400 s per epoch, so no throughput gain) and bfloat16 autocast (about
20% faster, numerically inconsistent with the finished seed-0 runs).

## 2026-10-23 - Exploratory within-type specificity AUROC (after results were seen)

Added after the pre-registered results were computed, and reported as exploratory. W2 scored a
pooled specificity AUROC of 0.39, below chance. Inspection showed W2 places its largest |E| in
CD4 T for every variant, so its centered |E| ranks cell types in the same fixed order for every
variant, and the pooled AUROC then measures that order against the types' positive-label rates
(0.27 for CD8 T to 0.88 for CD14 Mono). The within-type version computes the AUROC in each cell
type separately and averages over types (`src/ctb/eval/exploratory.py`), with the same locus
bootstrap. It does not replace the pre-registered metric, and H2 and H3 are judged on the
pooled version as planned.
