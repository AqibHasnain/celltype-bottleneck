"""Every figure and table in docs/REPORT.md, regenerated from committed results/ files."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ctb.paths import RESULTS  # noqa: E402

FIG = RESULTS / "figures"
WIDTHS = [2, 8, 32, 128, 512]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
LABELS = {
    "causal_auroc": "Causal AUROC",
    "sign_concordance": "Sign concordance",
    "cpc": "Centered profile correlation",
    "specificity_auroc": "Specificity AUROC",
    "sf_pred": "Specificity fraction (predicted)",
    "sign_concordance_train": "Sign concordance, training chromosomes",
    "test_peak_auroc": "Test peak AUROC",
    "probe_pass_rate": "Motif-probe pass rate",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _arm(summary: pd.DataFrame, arm: str, metric: str):
    r = summary.query("arm == @arm and metric == @metric")
    return r.iloc[0] if len(r) else None


def fig_width_sweep(summary: pd.DataFrame, by_run: pd.DataFrame) -> None:
    """Central figure: aggregate and specific metrics against bottleneck width."""
    panels = [
        ("causal_auroc", BLUE),
        ("sign_concordance", BLUE),
        ("cpc", ORANGE),
        ("specificity_auroc", ORANGE),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.1), facecolor=SURFACE)
    for ax, (metric, color) in zip(axes, panels, strict=True):
        _style(ax)
        rows = [_arm(summary, f"W{w}", metric) for w in WIDTHS]
        x = [w for w, r in zip(WIDTHS, rows, strict=True) if r is not None]
        rows = [r for r in rows if r is not None]
        y = np.array([r["value"] for r in rows])
        lo = np.array([r["ci_lo"] for r in rows])
        hi = np.array([r["ci_hi"] for r in rows])
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5)
        for w in x:
            seeds = by_run.loc[by_run["arm"] == f"W{w}", metric]
            ax.scatter([w] * len(seeds), seeds, s=9, color=MUTED, zorder=3, linewidths=0)
        for name, c in (("CHROMBPNET", VIOLET), ("MOTIF", GREEN)):
            r = _arm(summary, name, metric)
            if r is not None and np.isfinite(r["value"]):
                ax.axhline(r["value"], color=c, linewidth=1, linestyle="--")
                ax.text(
                    x[-1],
                    r["value"],
                    f" {name.title()}",
                    color=MUTED,
                    fontsize=7,
                    va="bottom",
                    ha="right",
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(WIDTHS, [str(w) for w in WIDTHS])
        ax.set_xlabel("Bottleneck width d", color=INK, fontsize=8)
        ax.set_title(LABELS[metric], color=INK, fontsize=9, loc="left")
    fig.suptitle(
        "Aggregate (blue) and cell-type-specific (orange) metrics against width; "
        "band = 95% locus bootstrap, dots = seeds",
        color=MUTED,
        fontsize=8,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig1_width_sweep.png", dpi=200)
    plt.close(fig)


def fig_sf(summary: pd.DataFrame) -> None:
    arms = [
        a
        for a in [f"W{w}" for w in WIDTHS] + ["COND", "COND-PRE", "CHROMBPNET", "MOTIF"]
        if _arm(summary, a, "sf_pred") is not None
    ]
    obs = _arm(summary, "OBSERVED", "sf_obs")
    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor=SURFACE)
    _style(ax)
    vals = [_arm(summary, a, "sf_pred") for a in arms]
    xs = np.arange(len(arms))
    ax.bar(xs, [v["value"] for v in vals], color=BLUE, width=0.6)
    ax.errorbar(
        xs,
        [v["value"] for v in vals],
        yerr=[[v["value"] - v["ci_lo"] for v in vals], [v["ci_hi"] - v["value"] for v in vals]],
        fmt="none",
        ecolor=INK,
        elinewidth=1,
        capsize=2,
    )
    ax.axhspan(obs["ci_lo"], obs["ci_hi"], color=ORANGE, alpha=0.2, linewidth=0)
    ax.axhline(obs["value"], color=ORANGE, linewidth=2)
    ax.text(
        len(arms) - 0.5,
        obs["value"],
        f"observed SF {obs['value']:.2f}",
        color=INK,
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.set_xticks(xs, arms, fontsize=8)
    ax.set_ylabel("Specificity fraction", color=INK, fontsize=8)
    ax.set_title(
        "Share of effect variance that is cell-type-specific, predicted vs observed",
        color=INK,
        fontsize=9,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig2_specificity_fraction.png", dpi=200)
    plt.close(fig)


def fig_probes(by_run: pd.DataFrame) -> None:
    order = [
        a
        for a in [f"W{w}" for w in WIDTHS] + ["COND", "COND-SHUF", "COND-PRE", "COND-PRE-SHUF"]
        if a in set(by_run["arm"])
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3), facecolor=SURFACE)
    _style(ax)
    xs = np.arange(len(order))
    for i, arm in enumerate(order):
        v = by_run.loc[by_run["arm"] == arm, "probe_pass_rate"] * 8
        ax.scatter([i] * len(v), v, color=MUTED, s=10, zorder=3, linewidths=0)
        ax.bar(i, v.mean(), color=AQUA, width=0.6)
    ax.set_xticks(xs, order, fontsize=8)
    ax.set_ylim(0, 8.3)
    ax.set_ylabel("Probes passed (of 8)", color=INK, fontsize=8)
    ax.set_title(
        "Motif-insertion probes: target cell type has the largest effect",
        color=INK,
        fontsize=9,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig3_motif_probes.png", dpi=200)
    plt.close(fig)


def fig_comparisons(comps: pd.DataFrame) -> None:
    """Paired differences against W32 (and W8 for H1) with 95% intervals."""
    rows = [
        ("W512", "W8", "causal_auroc"),
        ("W512", "W8", "sign_concordance"),
        ("W512", "W32", "cpc"),
        ("W512", "W32", "specificity_auroc"),
        ("COND", "W32", "cpc"),
        ("COND", "W32", "specificity_auroc"),
        ("COND-SHUF", "W32", "cpc"),
        ("COND-SHUF", "W32", "specificity_auroc"),
        ("COND-PRE", "W32", "cpc"),
        ("COND-PRE", "W32", "specificity_auroc"),
        ("COND-PRE-SHUF", "W32", "cpc"),
        ("COND-PRE-SHUF", "W32", "specificity_auroc"),
    ]
    data = []
    for a, b, m in rows:
        r = comps.query("a == @a and b == @b and metric == @m")
        if len(r):
            data.append((f"{a} - {b}: {LABELS[m]}", r.iloc[0]))
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(data) + 0.9), facecolor=SURFACE)
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ys = np.arange(len(data))[::-1]
    for y, (_, r) in zip(ys, data, strict=True):
        color = ORANGE if r["ci_lo"] > 0 or r["ci_hi"] < 0 else BLUE
        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=color, linewidth=2)
        ax.scatter([r["diff"]], [y], color=color, s=22, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(ys, [d[0] for d in data], fontsize=7.5)
    ax.set_xlabel(
        "Paired difference (95% locus bootstrap); orange = interval excludes 0",
        color=INK,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig4_paired_differences.png", dpi=200)
    plt.close(fig)


def fig_quality(by_run: pd.DataFrame) -> None:
    order = [
        a
        for a in [f"W{w}" for w in WIDTHS] + ["COND", "COND-SHUF", "COND-PRE", "COND-PRE-SHUF"]
        if a in set(by_run["arm"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), facecolor=SURFACE)
    for ax, metric, title in (
        (axes[0], "test_peak_auroc", "Test peak AUROC (mean over cells)"),
        (axes[1], "pseudobulk_r_mean", "Pseudobulk Pearson r (mean over 6 types)"),
    ):
        _style(ax)
        for i, arm in enumerate(order):
            v = by_run.loc[by_run["arm"] == arm, metric]
            ax.scatter([i] * len(v), v, color=BLUE, s=14, linewidths=0)
            ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=INK, linewidth=1.5)
        ax.set_xticks(range(len(order)), order, fontsize=7, rotation=30)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_model_quality.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS / "arm_summary.tsv", sep="\t")
    by_run = pd.read_csv(RESULTS / "metrics_by_run.tsv", sep="\t")
    comps = pd.read_csv(RESULTS / "comparisons.tsv", sep="\t")
    fig_width_sweep(summary, by_run)
    fig_sf(summary)
    fig_probes(by_run.dropna(subset=["probe_pass_rate"]))
    fig_comparisons(comps)
    fig_quality(by_run.dropna(subset=["test_peak_auroc"]))
    print(json.dumps(json.loads((RESULTS / "hypotheses.json").read_text()), indent=1))


if __name__ == "__main__":
    main()
