# Compute log

Every run on one Apple M4 Max (16 cores, 128 GB unified memory, PyTorch MPS backend), at
`nice -n 15`. No cloud compute was used and nothing was spent. A result with no row here is not a
result.

| Start (UTC) | Git SHA | Config hash | Wall-clock min | Purpose | Outcome |
|---|---|---|---|---|---|
| 2026-10-23T02:48Z | 70829a8 | 68380a167d4d | 74 | train W32 seed 0 (Phase 0, G0) | best epoch 52 of 63; test peak AUROC 0.8196 |
| 2026-10-23T02:31Z | 70829a8 | - | 51 | Stream TenK10K nominal pairs, chr2/8/13 x 6 types (network, CPU) | 18 files reduced to near-center SNVs |
| 2026-10-23T04:18Z | ab6acf3 | - | 80 | ChromBPNet reference: 6 types x 5 folds on 1,664 causal test pairs and 8,282 controls | Complete |
| 2026-10-23T04:05Z | ab6acf3 | - | 13 | Score W32 seed 0 (variants, probes, ISM), shared GPU | Complete; 4 of 8 probes pass |
| 2026-10-23T04:05Z | 7445bd0 | 8cf0cd34bbde | 133 | G0 reference: scvi.external.SCBASSET d=32 seed 0 (shared GPU for 80 min) | 45 epochs; test peak AUROC 0.8189 |
| 2026-10-23T06:16Z | 22d6c36 | a21448fa7597 | 39 | train W2 seed 0 | best epoch 23 of 34; test peak AUROC 0.7789 |
| 2026-10-23T06:54Z | 22d6c36 | 7f42bb216b3b | 66 | train W8 seed 0 | best epoch 39 of 50; test peak AUROC 0.8180 |
| 2026-10-23T08:01Z | 22d6c36 | 4fbdf4999ec2 | 67 | train W128 seed 0 | best epoch 39 of 50; test peak AUROC 0.8204 |
| 2026-10-23T09:08Z | 22d6c36 | 53b2e8704263 | 96 | train W512 seed 0 | best epoch 63 of 74; test peak AUROC 0.8172 |
| 2026-10-23T10:44Z | 22d6c36 | 04602131bf9b | 83 | train COND seed 0 | best epoch 52 of 63; test peak AUROC 0.8210 |
| 2026-10-23T12:06Z | 22d6c36 | be5fd3340a16 | 68 | train COND-SHUF seed 0 | best epoch 39 of 50; test peak AUROC 0.8205 |
| 2026-10-23T14:31Z | bd3d403 | - | - | train COND-PRE seed 0 | Stopped at epoch ~58 at Aqib's request (laptop shutdown); no result, rerun on resume |
| 2026-10-23T15:43Z | f8fdded | aecbeedb4ee1 | 50 | train W8 seed 1 | best epoch 27 of 38; test peak AUROC 0.8150 |
| 2026-10-23T16:32Z | f8fdded | 3668cda3add2 | 93 | train W32 seed 1 | best epoch 50 of 61; test peak AUROC 0.8197 |
| 2026-10-23T18:05Z | f8fdded | 7874ab8115bf | 100 | train W512 seed 1 | best epoch 50 of 61; test peak AUROC 0.8185 |
| 2026-10-23T19:45Z | f8fdded | 55c063ca0557 | 75 | train COND seed 1 | best epoch 34 of 45; test peak AUROC 0.8197 |
| 2026-10-23T21:00Z | f8fdded | 68b7deaf378d | 93 | train COND-SHUF seed 1 | best epoch 41 of 52; test peak AUROC 0.8189 |
| 2026-10-23T06:20Z | 22d6c36 | - | ~150 | Score W2 to COND-SHUF seed 0 and five seed-1 runs (variants, probes, ISM), concurrent with training at about 15 min each | Complete |
| 2026-10-23T22:50Z | 222d7f8 | - | 4 | Aggregate with 2,000 locus resamples; exploratory within-type AUROC | Complete |
