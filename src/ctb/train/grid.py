"""Run (arm, seed) pairs one after another, skipping finished runs, logging each to
docs/COMPUTE_LOG.md. Seed 0 of every listed arm runs before any seed 1.

Usage: python -m ctb.train.grid --arms W2 W8 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime

from ctb.paths import ROOT, RUNS

COMPUTE_LOG = ROOT / "docs" / "COMPUTE_LOG.md"


def log_row(summary: dict, started: str, purpose: str) -> None:
    row = (
        f"| {started} | {summary['git_sha'][:7]} | {summary['config_hash']} | "
        f"{summary['wall_clock_min']:.0f} | {purpose} | best epoch {summary['best_epoch']} of "
        f"{summary['epochs_run']}; test peak AUROC {summary['test']['peak_auroc']:.4f} |\n"
    )
    with COMPUTE_LOG.open("a") as fh:
        fh.write(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    for seed in args.seeds:
        for arm in args.arms:
            out = RUNS / arm / f"seed{seed}" / "summary.json"
            if out.exists():
                continue
            started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ")
            cmd = [sys.executable, "-m", "ctb.train.loop", "--arm", arm, "--seed", str(seed)]
            log_path = ROOT / "logs" / f"{arm}_seed{seed}.log"
            log_path.parent.mkdir(exist_ok=True)
            with log_path.open("w") as fh:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT).returncode
            if rc != 0 or not out.exists():
                with COMPUTE_LOG.open("a") as fh:
                    fh.write(
                        f"| {started} | - | - | - | train {arm} seed {seed} | FAILED rc={rc} |\n"
                    )
                continue
            log_row(json.loads(out.read_text()), started, f"train {arm} seed {seed}")


if __name__ == "__main__":
    main()
