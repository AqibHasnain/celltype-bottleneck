# Compute log

Every run on one Apple M4 Max (16 cores, 128 GB unified memory, PyTorch MPS backend), at
`nice -n 15`. No cloud compute was used and nothing was spent. A result with no row here is not a
result.

| Start (UTC) | Git SHA | Config hash | Wall-clock min | Purpose | Outcome |
|---|---|---|---|---|---|
| 2026-10-23T02:48Z | 70829a8 | 68380a167d4d | 74 | train W32 seed 0 (Phase 0, G0) | best epoch 52 of 63; test peak AUROC 0.8196 |
| 2026-10-23T02:31Z | 70829a8 | - | 51 | Stream TenK10K nominal pairs, chr2/8/13 x 6 types (network, CPU) | 18 files reduced to near-center SNVs |
