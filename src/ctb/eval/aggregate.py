"""Section 5 metrics for every run, locus-bootstrap intervals, paired arm comparisons, H1-H3.

Every interval is the 2.5 and 97.5 percentiles of 2,000 resamples of loci (causal test pairs
within 100 kb, with their matched controls). Resamples are shared by all runs, so an arm
difference is paired by variant; an arm's value is the mean over its seeds, so a difference is
also paired by seed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ctb.data.annotate import LABELS
from ctb.data.tenk10k import CELL_TYPES
from ctb.data.variants import NEG_MAX_ABS_SLOPE, VARIANTS_DIR
from ctb.eval.metrics import (
    causal_auroc,
    centered_profile_correlation,
    cpc_rows,
    sign_concordance,
    specificity_auroc,
    specificity_fraction,
)
from ctb.paths import RESULTS, RUNS
from ctb.utils.logging import get_logger
from ctb.utils.manifest import write_json

log = get_logger(__name__)

N_BOOT = 2000
BOOT_SEED = 0
WIDTH_ARMS = ["W2", "W8", "W32", "W128", "W512"]
COND_ARMS = ["COND", "COND-SHUF", "COND-PRE", "COND-PRE-SHUF"]
BASELINES = ["CHROMBPNET", "MOTIF"]
VARIANT_METRICS = [
    "causal_auroc",
    "sign_concordance",
    "cpc",
    "specificity_auroc",
    "sf_pred",
    "sign_concordance_train",
]
COMPARISONS = [
    ("W512", "W8"),
    ("W512", "W32"),
    ("W128", "W32"),
    ("W32", "W8"),
    ("W8", "W2"),
    ("COND", "W32"),
    ("COND-SHUF", "W32"),
    ("COND", "COND-SHUF"),
    ("COND-PRE", "W32"),
    ("COND-PRE-SHUF", "W32"),
    ("COND-PRE", "COND-PRE-SHUF"),
]
H1_MARGIN = 0.02
H3_PEAK_AUROC_MAX_DROP = 0.01


class Truth:
    """Observed quantities and row bookkeeping shared by all runs."""

    def __init__(self):
        self.all = pd.read_parquet(VARIANTS_DIR / "causal_all.parquet")
        self.test = pd.read_parquet(VARIANTS_DIR / "causal_test.parquet")
        self.ctrl = pd.read_parquet(VARIANTS_DIR / "controls_test.parquet")
        pos = pd.Series(np.arange(len(self.all)), index=self.all["pair_id"])
        self.test_rows = pos.loc[self.test["pair_id"]].to_numpy()
        self.train_rows = np.flatnonzero(self.all["split"].to_numpy() == "train")
        self.b = self.test[[f"slope:{ct}" for ct in CELL_TYPES]].to_numpy(dtype=float)
        self.mask = ~np.isnan(self.b)
        self.spec = self.test["specific_set"].to_numpy(dtype=bool)
        sig = self.test[[f"sig:{ct}" for ct in CELL_TYPES]].to_numpy(dtype=bool)
        neg = ~sig & (np.abs(self.b) < NEG_MAX_ABS_SLOPE) & (sig.sum(1, keepdims=True) > 0)
        label = np.full(self.b.shape, np.nan)
        label[sig] = 1.0
        label[neg] = 0.0
        label[~self.spec[:, None] | ~self.mask] = np.nan
        self.label = label
        # Fixed once on the full specificity set; a resample reweights these rows.
        self.cpc_rows = cpc_rows(self.b, self.spec)
        self.lead = self.test["lead_slope"].to_numpy()
        self.lead_train = self.all["lead_slope"].to_numpy()[self.train_rows]
        loci, self.locus = np.unique(self.test["locus"].to_numpy(), return_inverse=True)
        self.n_loci = len(loci)
        self.ctrl_locus = (
            pd.Series(self.locus, index=self.test["pair_id"])
            .loc[self.ctrl["matched_to"]]
            .to_numpy()
        )
        tl, self.train_locus = np.unique(
            self.all["locus"].to_numpy()[self.train_rows], return_inverse=True
        )
        self.n_train_loci = len(tl)

    def sf_observed(self, w: np.ndarray) -> float:
        return specificity_fraction(self.b[self.spec], self.mask[self.spec], w[self.spec])


class RunMetrics:
    """Variant metrics of one run under any row weighting."""

    def __init__(self, truth: Truth, scores: dict):
        e_all = scores["e_causal_all"]
        self.e = e_all[truth.test_rows]
        self.ec = scores["e_controls"]
        self.ok = ~np.isnan(self.e).any(1)
        self.ok_ctrl = ~np.isnan(self.ec).any(1)
        self.e_train = e_all[truth.train_rows]
        self.has_train = not np.isnan(self.e_train).any()
        self.truth = truth

    def compute(self, w: np.ndarray, w_ctrl: np.ndarray, w_train: np.ndarray) -> dict:
        t, ok = self.truth, self.ok
        spec = t.spec & ok
        return {
            "causal_auroc": causal_auroc(
                self.e[ok], self.ec[self.ok_ctrl], w[ok], w_ctrl[self.ok_ctrl]
            ),
            "sign_concordance": sign_concordance(self.e[ok], t.lead[ok], w[ok]),
            "cpc": centered_profile_correlation(self.e[ok], t.b[ok], t.cpc_rows[ok], w[ok]),
            "specificity_auroc": specificity_auroc(self.e[ok], t.mask[ok], t.label[ok], w[ok]),
            "sf_pred": specificity_fraction(np.nan_to_num(self.e[spec]), t.mask[spec], w[spec]),
            "sign_concordance_train": (
                sign_concordance(self.e_train, t.lead_train, w_train) if self.has_train else np.nan
            ),
        }


def run_dirs() -> dict[str, list]:
    arms = {}
    for p in sorted(RUNS.glob("*/seed*/scores.npz")):
        arms.setdefault(p.parent.parent.name, []).append(p.parent)
    return arms


def bootstrap_weights(truth: Truth, rng: np.random.Generator):
    counts = rng.multinomial(truth.n_loci, np.full(truth.n_loci, 1 / truth.n_loci))
    tcounts = rng.multinomial(
        truth.n_train_loci, np.full(truth.n_train_loci, 1 / truth.n_train_loci)
    )
    return counts[truth.locus], counts[truth.ctrl_locus], tcounts[truth.train_locus]


def main() -> None:
    truth = Truth()
    arms = run_dirs()
    runs = {
        (arm, d.name): RunMetrics(truth, dict(np.load(d / "scores.npz")))
        for arm, dirs in arms.items()
        for d in dirs
    }
    ones = (np.ones(len(truth.test)), np.ones(len(truth.ctrl)), np.ones(len(truth.train_rows)))
    point = {k: rm.compute(*ones) for k, rm in runs.items()}
    rng = np.random.default_rng(BOOT_SEED)
    boot = {k: {m: np.empty(N_BOOT) for m in VARIANT_METRICS} for k in runs}
    sf_obs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        w = bootstrap_weights(truth, rng)
        sf_obs[i] = truth.sf_observed(w[0])
        for k, rm in runs.items():
            for m, v in rm.compute(*w).items():
                boot[k][m][i] = v
        if i % 500 == 0:
            log.info("bootstrap %d/%d", i, N_BOOT)

    # Run-level table, adding model-quality numbers, ranks and probes.
    rows = []
    for (arm, seed), m in point.items():
        d = RUNS / arm / seed
        row = {"arm": arm, "seed": seed, **m}
        if (d / "summary.json").exists():
            s = json.loads((d / "summary.json").read_text())
            row |= {
                "test_peak_auroc": s["test"]["peak_auroc"],
                "best_epoch": s["best_epoch"],
                "n_trainable": s["n_trainable"],
                **{f"pseudobulk_r:{t}": s["test"]["pseudobulk_r"][t] for t in LABELS},
            }
        sc = np.load(d / "scores.npz")
        for key in (
            "ism_effective_rank",
            "pred_pseudobulk_effective_rank",
            "obs_pseudobulk_effective_rank",
        ):
            if key in sc:
                row[key] = float(sc[key])
        if (d / "probes.tsv").exists():
            pr = pd.read_csv(d / "probes.tsv", sep="\t")
            row["probe_pass_rate"] = pr["pass"].mean()
            row["probe_per_background_pass"] = pr["per_background_pass"].mean()
        rows.append(row)
    by_run = pd.DataFrame(rows)
    by_run["pseudobulk_r_mean"] = (
        by_run[[f"pseudobulk_r:{t}" for t in LABELS]].mean(1)
        if "pseudobulk_r:B" in by_run
        else np.nan
    )

    def arm_boot(arm: str, metric: str) -> np.ndarray:
        return np.mean([boot[(arm, s)][metric] for s in sorted(d.name for d in arms[arm])], axis=0)

    def arm_point(arm: str, metric: str) -> float:
        return float(by_run.loc[by_run["arm"] == arm, metric].mean())

    summary = []
    for arm in arms:
        for metric in VARIANT_METRICS:
            b = arm_boot(arm, metric)
            lo, hi = np.nanpercentile(b, [2.5, 97.5]) if not np.isnan(b).all() else (np.nan, np.nan)
            summary.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "value": arm_point(arm, metric),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_seeds": len(arms[arm]),
                }
            )
    summary.append(
        {
            "arm": "OBSERVED",
            "metric": "sf_obs",
            "value": truth.sf_observed(np.ones(len(truth.test))),
            "ci_lo": np.percentile(sf_obs, 2.5),
            "ci_hi": np.percentile(sf_obs, 97.5),
            "n_seeds": 0,
        }
    )
    summary = pd.DataFrame(summary)

    comps = []
    for a, b in COMPARISONS:
        if a not in arms or b not in arms:
            continue
        seeds = sorted({d.name for d in arms[a]} & {d.name for d in arms[b]})
        for metric in [*VARIANT_METRICS, "test_peak_auroc"]:
            per_seed = [
                float(
                    by_run.query("arm == @a and seed == @s")[metric].iloc[0]
                    - by_run.query("arm == @b and seed == @s")[metric].iloc[0]
                )
                for s in seeds
            ]
            row = {
                "a": a,
                "b": b,
                "metric": metric,
                "diff": float(np.mean(per_seed)),
                "per_seed": per_seed,
                "n_paired_seeds": len(seeds),
            }
            if metric in VARIANT_METRICS:
                d = np.mean([boot[(a, s)][metric] - boot[(b, s)][metric] for s in seeds], axis=0)
                row["ci_lo"], row["ci_hi"] = np.nanpercentile(d, [2.5, 97.5])
            comps.append(row)
    comps = pd.DataFrame(comps)

    def ci(a: str, b: str, metric: str) -> dict:
        r = comps.query("a == @a and b == @b and metric == @metric")
        return r.iloc[0][["diff", "ci_lo", "ci_hi"]].astype(float).to_dict() if len(r) else {}

    verdicts = {}
    h1 = {m: ci("W512", "W8", m) for m in ("causal_auroc", "sign_concordance")}
    if all(h1.values()):
        verdicts["H1"] = {
            "criterion": "95% CI of W512 - W8 inside (-0.02, 0.02) for causal AUROC and sign "
            "concordance",
            "comparisons": h1,
            "supported": all(
                -H1_MARGIN < v["ci_lo"] and v["ci_hi"] < H1_MARGIN for v in h1.values()
            ),
            "point_estimates_within_margin": all(abs(v["diff"]) < H1_MARGIN for v in h1.values()),
        }
    h2 = {m: ci("W512", "W32", m) for m in ("cpc", "specificity_auroc")}
    if all(h2.values()):
        n_pos = sum(v["ci_lo"] > 0 for v in h2.values())
        verdicts["H2"] = {
            "criterion": "W512 - W32 95% CI above zero for CPC and specificity AUROC",
            "comparisons": h2,
            "n_metrics_supporting": n_pos,
            "supported": n_pos == 2,
        }
    for cond, shuf in (("COND", "COND-SHUF"), ("COND-PRE", "COND-PRE-SHUF")):
        h3 = {m: ci(cond, "W32", m) for m in ("cpc", "specificity_auroc")}
        peak = comps.query("a == @cond and b == 'W32' and metric == 'test_peak_auroc'")
        if all(h3.values()) and len(peak):
            drop = float(peak["diff"].iloc[0])
            n_pos = sum(v["ci_lo"] > 0 for v in h3.values())
            verdicts["H3" if cond == "COND" else "H3-added-COND-PRE"] = {
                "criterion": f"{cond} - W32 95% CI above zero for CPC and specificity AUROC, "
                "and test peak AUROC drop <= 0.01",
                "comparisons": h3,
                "peak_auroc_diff": drop,
                "n_metrics_supporting": n_pos,
                "peak_auroc_ok": drop >= -H3_PEAK_AUROC_MAX_DROP,
                "supported": n_pos == 2 and drop >= -H3_PEAK_AUROC_MAX_DROP,
                "shuffled_control": {m: ci(shuf, "W32", m) for m in ("cpc", "specificity_auroc")},
            }

    RESULTS.mkdir(parents=True, exist_ok=True)
    by_run.to_csv(RESULTS / "metrics_by_run.tsv", sep="\t", index=False, float_format="%.5g")
    summary.to_csv(RESULTS / "arm_summary.tsv", sep="\t", index=False, float_format="%.5g")
    comps.to_csv(RESULTS / "comparisons.tsv", sep="\t", index=False, float_format="%.5g")
    counts = {
        "test_causal_pairs": int(len(truth.test)),
        "controls": int(len(truth.ctrl)),
        "specificity_set": int(truth.spec.sum()),
        "cpc_rows": int(truth.cpc_rows.sum()),
        "specificity_labels_pos": int((truth.label == 1).sum()),
        "specificity_labels_neg": int((truth.label == 0).sum()),
        "test_loci": truth.n_loci,
        "train_causal_pairs": int(len(truth.train_rows)),
        "train_loci": truth.n_train_loci,
        "n_boot": N_BOOT,
    }
    write_json({"counts": counts, "verdicts": verdicts}, RESULTS / "hypotheses.json")
    log.info("verdicts: %s", json.dumps(verdicts, indent=1, default=float))


if __name__ == "__main__":
    main()
