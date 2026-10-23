"""Gate G0 reference: scvi.external.SCBASSET (n_bottleneck_layer=32) on the same data and split.

scvi takes its own default training recipe (Adam lr 0.01, betas 0.95/0.9995, batch 128) with
early stopping on its validation AUROC, patience 10, so only the split and stopping patience are
aligned with ctb. The test metric is computed with the same per-cell AUROC code as every arm.
"""

from __future__ import annotations

import argparse
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from ctb.data.peaks import PEAKS_DIR
from ctb.eval.metrics import per_column_auroc
from ctb.paths import RUNS
from ctb.train.data import load_peak_data
from ctb.utils.logging import get_logger
from ctb.utils.manifest import build_manifest, write_json
from ctb.utils.seeding import set_all_seeds

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=1000)
    args = ap.parse_args()

    import scvi
    from scvi.external import SCBASSET
    from scvi.train._callbacks import SaveCheckpoint

    set_all_seeds(args.seed)
    scvi.settings.seed = args.seed
    data = load_peak_data(torch.device("cpu"))
    codes = data.codes.numpy().astype(np.int64)
    # scvi one-hots with 4 classes; N gets a random base, as scBasset and scvi do.
    n_mask = codes == 4
    codes[n_mask] = np.random.default_rng(args.seed).integers(0, 3, n_mask.sum())
    adata = ad.AnnData(X=sp.load_npz(PEAKS_DIR / "y.npz").astype(np.float32).tocsr())
    adata.obsm["dna_code"] = codes
    SCBASSET.setup_anndata(adata, dna_code_key="dna_code")
    model = SCBASSET(adata, n_bottleneck_layer=32)

    out_dir = RUNS / "SCVI-W32" / f"seed{args.seed}"
    manifest = build_manifest(
        {
            "reference": "scvi.external.SCBASSET",
            "n_bottleneck_layer": 32,
            "seed": args.seed,
            "patience": 10,
        },
        device="mps",
    )
    write_json(manifest, out_dir / "manifest.json")
    t0 = time.time()
    model.train(
        max_epochs=args.max_epochs,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1,
        datasplitter_kwargs={
            "external_indexing": [
                data.split_idx["train"],
                data.split_idx["validation"],
                data.split_idx["test"],
            ]
        },
        early_stopping=True,
        early_stopping_monitor="auroc_validation",
        early_stopping_mode="max",
        early_stopping_patience=10,
        # Restores the best-validation weights, matching ctb's stopping rule.
        callbacks=[
            SaveCheckpoint(
                dirpath=str(out_dir / "ckpt"),
                monitor="auroc_validation",
                mode="max",
                load_best_on_end=True,
            )
        ],
    )
    history = {k: v.iloc[:, 0].tolist() for k, v in model.history.items()}
    module = model.module.cpu().eval()
    test = torch.as_tensor(codes[data.split_idx["test"]])
    with torch.no_grad():
        logits = torch.cat(
            [module._get_accessibility(test[i : i + 512]) for i in range(0, len(test), 512)]
        )
    auc = per_column_auroc(logits, data.y[torch.as_tensor(data.split_idx["test"])])
    summary = {
        "arm": "SCVI-W32",
        "seed": args.seed,
        "test": {"peak_auroc": float(torch.nanmean(auc))},
        "epochs_run": len(history.get("auroc_validation", [])),
        "best_val_auroc_scvi": max(history.get("auroc_validation", [float("nan")])),
        "wall_clock_min": (time.time() - t0) / 60,
        "git_sha": manifest["git_sha"],
        "config_hash": manifest["config_hash"],
    }
    write_json(summary, out_dir / "summary.json")
    write_json(history, out_dir / "history.json")
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()
