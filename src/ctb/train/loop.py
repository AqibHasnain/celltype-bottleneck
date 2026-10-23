"""Train one arm with one seed: early stopping on validation peak AUROC, then test metrics.

Usage: python -m ctb.train.loop --arm W32 --seed 0 [--max-epochs N]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ctb.data.annotate import LABELS
from ctb.eval.metrics import per_column_auroc, pseudobulk_correlation
from ctb.model.scbasset import ScBasset, n_trainable, one_hot, reverse_complement, shift
from ctb.paths import RUNS
from ctb.train.arms import build_model_config, load_arm, type_indices
from ctb.train.data import PeakData, load_peak_data
from ctb.utils.logging import JsonlLogger, get_logger
from ctb.utils.manifest import build_manifest, write_json
from ctb.utils.seeding import set_all_seeds

log = get_logger(__name__)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def run_dir(arm: str, seed: int):
    return RUNS / arm / f"seed{seed}"


def build_model(cfg: dict, data: PeakData, seed: int) -> ScBasset:
    mc = build_model_config(cfg, n_cells=data.y.shape[1])
    types = None
    if mc.readout != "linear":
        types = type_indices(
            data.labels, shuffle_seed=1000 + seed if cfg["model"]["shuffle_types"] else None
        )
    return ScBasset(mc, types)


@torch.no_grad()
def predict_logits(
    model: ScBasset, codes: torch.Tensor, batch: int = 256, rc_average: bool = False
) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, codes.shape[0], batch):
        x = one_hot(codes[i : i + batch])
        logit = model(x)
        if rc_average:
            logit = 0.5 * (logit + model(reverse_complement(x)))
        out.append(logit)
    return torch.cat(out)


def evaluate_split(model: ScBasset, data: PeakData, split: str, full: bool = False) -> dict:
    idx = torch.as_tensor(data.split_idx[split], device=data.codes.device)
    logits = predict_logits(model, data.codes[idx])
    y = data.y[idx]
    auc = per_column_auroc(logits, y)
    out = {"peak_auroc": float(torch.nanmean(auc))}
    if full:
        prob = torch.sigmoid(logits).cpu().numpy()
        obs = y.cpu().numpy().astype(np.float32)
        pred_pb = np.stack([prob[:, data.labels == t].mean(1) for t in LABELS])
        obs_pb = np.stack([obs[:, data.labels == t].mean(1) for t in LABELS])
        out["pseudobulk_r"] = dict(
            zip(LABELS, pseudobulk_correlation(pred_pb, obs_pb).tolist(), strict=True)
        )
        out["per_cell_auroc"] = auc.cpu().numpy()
        out["pred_pseudobulk"] = pred_pb
        out["obs_pseudobulk"] = obs_pb
    return out


def train(arm: str, seed: int, max_epochs: int | None = None, tag: str = "") -> dict:
    cfg = load_arm(arm)
    tcfg = cfg["train"]
    if max_epochs is not None:
        tcfg["max_epochs"] = max_epochs
    dev = device()
    set_all_seeds(seed)
    data = load_peak_data(dev)
    model = build_model(cfg, data, seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], betas=tuple(tcfg["betas"]))
    out_dir = run_dir(arm + tag, seed)
    manifest = build_manifest({**cfg, "seed": seed}, device=str(dev))
    manifest["n_trainable"] = n_trainable(model)
    manifest["pre_bottleneck_width"] = model.cfg.n_filters_pre_bottleneck
    write_json(manifest, out_dir / "manifest.json")
    events = JsonlLogger(out_dir)
    rng = np.random.default_rng(seed)
    train_idx = data.split_idx["train"]
    shifts = np.arange(-tcfg["shift_max"], tcfg["shift_max"] + 1)
    bce = torch.nn.BCEWithLogitsLoss()

    best, best_epoch, wait = -np.inf, -1, 0
    t_start = time.time()
    for epoch in range(tcfg["max_epochs"]):
        t0 = time.time()
        model.train()
        perm = rng.permutation(train_idx)
        total, n_batches = 0.0, 0
        for i in range(0, len(perm), tcfg["batch_size"]):
            b = torch.as_tensor(perm[i : i + tcfg["batch_size"]], device=dev)
            x = one_hot(data.codes[b])
            if rng.random() > 0.5:
                x = reverse_complement(x)
            x = shift(x, int(rng.choice(shifts)))
            loss = bce(model(x), data.y[b].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        val = evaluate_split(model, data, "validation")["peak_auroc"]
        improved = val > best
        if improved:
            best, best_epoch, wait = val, epoch, 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            wait += 1
        events.log(
            "epoch",
            epoch=epoch,
            train_loss=total / n_batches,
            val_peak_auroc=val,
            seconds=time.time() - t0,
        )
        log.info(
            "%s seed %d epoch %d loss %.4f val AUROC %.4f%s (%.0fs)",
            arm + tag,
            seed,
            epoch,
            total / n_batches,
            val,
            " *" if improved else "",
            time.time() - t0,
        )
        if wait >= tcfg["patience"]:
            break

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=dev))
    test = evaluate_split(model, data, "test", full=True)
    np.savez_compressed(
        out_dir / "test_predictions.npz",
        per_cell_auroc=test.pop("per_cell_auroc"),
        pred_pseudobulk=test.pop("pred_pseudobulk"),
        obs_pseudobulk=test.pop("obs_pseudobulk"),
    )
    summary = {
        "arm": arm + tag,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "val_peak_auroc": best,
        "test": test,
        "wall_clock_min": (time.time() - t_start) / 60,
        "git_sha": manifest["git_sha"],
        "config_hash": manifest["config_hash"],
        "n_trainable": manifest["n_trainable"],
    }
    write_json(summary, out_dir / "summary.json")
    events.log("done", **{k: v for k, v in summary.items() if k != "test"})
    events.close()
    log.info("done: %s", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--tag", default="", help="suffix for the run directory (timing runs)")
    args = ap.parse_args()
    train(args.arm, args.seed, args.max_epochs, args.tag)


if __name__ == "__main__":
    main()
