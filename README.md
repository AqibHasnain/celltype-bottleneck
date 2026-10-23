# Bottleneck width and cell-type-specific variant effects in scBasset

scBasset (Yuan and Kelley, Nat Methods 2022) predicts the accessibility of every peak in every
cell from DNA sequence. The sequence passes through a 32-dimensional bottleneck, and each cell
reads that bottleneck out linearly. The predicted effect of a variant in a cell type is therefore
`w_t . (z_alt - z_ref)`: every cell type sees the same 32-dimensional change, weighted
differently.

This repository asks whether that shared bottleneck limits the cell-type-specific part of
predicted variant effects. It trains scBasset at bottleneck widths 2 to 512 on public PBMC
multiome data and scores the specific component against cell-type-resolved caQTLs from TenK10K
(922 donors). The hypotheses, metrics and thresholds were fixed in advance in
[`PROJECT_PLAN.md`](PROJECT_PLAN.md).

Gilfeather, Chikina and Kostka (bioRxiv 2026, "When Less Is Enough") truncate the output layers
of trained bulk models with SVD and find that very low ranks match full models on aggregate
variant benchmarks. This project instead trains each width from scratch, uses a single-cell
readout, and separates each predicted effect into its shared and cell-type-specific parts.

The full write-up is in [`docs/REPORT.md`](docs/REPORT.md).

## Summary of results

Within six PBMC cell types and one training donor, a 32-dimensional bottleneck does not
measurably limit cell-type-specific variant effects. None of the three pre-registered hypotheses
is supported.

- **Gate G0 passed.** The reimplementation reaches test peak AUROC 0.8196 against 0.8189 for
  `scvi.external.SCBASSET` on the same split. 1,089 causal test-chromosome pairs enter the
  specificity analyses (threshold 300).
- **H1, aggregate metrics flat in width: not supported as stated.** Sign concordance is flat from
  width 8 to 512, at +0.003 [-0.016, +0.020]. Causal-versus-control AUROC falls slightly with
  width, at -0.017 [-0.028, -0.006], so the interval leaves the pre-registered +/-0.02 margin.
- **H2, specific component width-limited: not supported.** From width 32 to 512, centered
  profile correlation changes by +0.032 [-0.002, +0.067] and specificity AUROC by +0.010
  [-0.007, +0.026]. Both metrics rise from width 8 to 512, which was not the pre-registered
  comparison, so width matters below 32 and is not detectable above it.
- **H3, cell-type-conditioned head: not supported.** The pre-registered conditioned head changes
  centered profile correlation by +0.037 [+0.001, +0.072] and specificity AUROC by -0.020
  [-0.039, -0.0005], and does not separate from its shuffled-label control. The head turned out
  to be a reparameterization of the width-32 model: a FiLM layer on z feeding a free per-cell
  readout has the same function class. It was kept as pre-registered and the flaw is reported.
- **Width 2 collapses.** At width 2 the effect matrix has rank 2 at most, and its predicted
  specificity fraction is 0.03 against 0.40 observed.
- **No compression from width 32 up.** Predicted specificity fraction is 0.43 to 0.49 at
  widths 32 to 512, at or above the observed 0.40.
- **The gap lies elsewhere.** The TenK10K ChromBPNet models, one per cell type, reach
  specificity AUROC 0.72 and centered profile correlation 0.48, against 0.56 to 0.59 and 0.27 to
  0.32 for every scBasset width >= 8. ChromBPNet is a reference only: it was trained on many
  donors with folds that include the test chromosomes.

With six cell types, any width >= 6 already removes the algebraic rank limit on effect profiles.
Atlases with many more cell types than bottleneck dimensions are where the question could have a
different answer.

![Metrics against bottleneck width](results/figures/fig1_width_sweep.png)

## Data and models

- **Training data.** 10x Genomics PBMC granulocyte-sorted 10k multiome (Cell Ranger ARC 2.0.0):
  11,898 cells and 98,507 peaks. Chromosome hold-out: chr7 validation; chr2, chr8 and chr13 test
  ([`configs/splits.yaml`](configs/splits.yaml)).
- **Cell labels.** Six labels come from the RNA modality, matched to TenK10K B_naive, CD4_Naive,
  CD8_Naive, NK, CD14_Mono and CD16_Mono.
- **Ground truth.** TenK10K phase 1 caQTLs (Xue et al., medRxiv 2025), Hugging Face commit
  `a9baa08`:
  - SuSiE fine-mapping defines 1,664 causal pairs and 8,282 matched controls on the test
    chromosomes.
  - Per-type TensorQTL slopes, streamed from the nominal-pair files, give the observed effect
    profiles.
- **Model.** A PyTorch reimplementation of scBasset with the bottleneck width as the only change
  between W arms. COND and COND-SHUF are parameter-matched to W32 to within one pre-bottleneck
  channel.
- **References.**
  - The TenK10K ChromBPNet bias-free models, loaded into PyTorch from their released weights. The
    port reproduces TenK10K's own scores (r = 0.998 and 0.93 on two types).
  - A JASPAR motif baseline weighted by expression.
- **Sources.** Sources, versions and checksums are in [`data/SOURCES.md`](data/SOURCES.md).
- **Compute.** All runs used one Apple M4 Max, about 22 hours in total
  ([`docs/COMPUTE_LOG.md`](docs/COMPUTE_LOG.md)).

## Limitations

- One training donor and six PBMC cell types.
- Two seeds for the arms each hypothesis compares (W8, W32, W512, COND, COND-SHUF) and one for
  W2 and W128, cut from three under a time budget. The two seeds disagree on the sign of the W512
  minus W32 change in centered profile correlation.
- The pre-bottleneck conditioned head that would test H3's intended question (COND-PRE) is
  implemented and parameter-matched but was not trained.
- Training labels mix naive with memory and effector cells, while three TenK10K labels are naive
  only. TBX21 and EOMES motif probes rank CD8 T above NK, which may reflect this.
- Hyperparameters are scBasset's defaults at every width. No learning-rate or regularization
  sweep was run.
- Four claims were withdrawn or corrected during the work, and one exploratory metric was added
  after results were seen. [`docs/REPORT.md`](docs/REPORT.md) lists them with the reasons, and
  [`docs/DECISIONS.md`](docs/DECISIONS.md) records each change when it was made.

## Reproducing

```bash
make setup                    # create the virtual environment (Python 3.12, uv)
make data                     # download, stream TenK10K, annotate cells, build peaks and variant sets
make g0                       # scvi.external.SCBASSET reference on the same split
make train ARM=W32 SEED=0     # one arm and seed; arms are in configs/arms/
make eval                     # score variants, probes and ISM, then aggregate with bootstrap intervals
make report                   # regenerate figures and tables from committed results
make test                     # test suite, CPU only
```

`python -m ctb.train.grid --arms W8 W32 W512 COND COND-SHUF --seeds 0 1` runs a set of arms one
after another and logs each to the compute log. One W32 run takes about 75 minutes on the
hardware above.

## Repository layout

```
src/ctb/
  data/     downloads, TenK10K streaming, cell annotation, peak matrix, splits, variant sets
  model/    scBasset reimplementation with linear and FiLM readouts
  train/    training loop, arms and parameter matching, grid runner, scvi reference
  eval/     metrics, variant scoring, motif probes, ChromBPNet port, baselines, bootstrap, figures
  utils/    seeding, run manifests, logging
configs/    base.yaml, splits.yaml, one file per arm
docs/       REPORT.md, DECISIONS.md, COMPUTE_LOG.md
data/       SOURCES.md and recorded checksums (raw and processed data are not committed)
results/    metrics, comparisons, hypothesis verdicts, per-run summaries, figures
tests/
```

[`PROJECT_PLAN.md`](PROJECT_PLAN.md) is the original plan, written before the experiments.

## License

MIT. See [`LICENSE`](LICENSE).
