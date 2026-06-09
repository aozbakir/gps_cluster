"""Dendrogram and gap-statistic plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from gps_cluster.domain.services.gap_statistic import GapResult
    from gps_cluster.application.euler_clustering import FTestResult


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    truncate_p: int = 12,
    max_d: float | None = None,
    title: str = "Hierarchical Clustering Dendrogram",
) -> "Figure":
    """Plot a (truncated) dendrogram with optional cut-line."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(12, 5))
    ddata = dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=truncate_p,
        show_contracted=True,
        leaf_rotation=90.0,
        leaf_font_size=9.0,
        ax=ax,
        color_threshold=max_d,
    )

    if max_d is not None:
        ax.axhline(y=max_d, color="k", linestyle="--", alpha=0.5, label=f"cut = {max_d:.1f}")
        # Annotate merge heights above max_d
        for icoord, dcoord, color in zip(ddata["icoord"], ddata["dcoord"], ddata["color_list"]):
            y = dcoord[1]
            if y > max_d:
                x = 0.5 * sum(icoord[1:3])
                ax.annotate(f"{y:.2g}", (x, y), xytext=(0, -6),
                            textcoords="offset points", fontsize=8, ha="center", va="top")
        ax.legend(fontsize=9)

    ax.set_xlabel("Sample index (or cluster size)", fontsize=11)
    ax.set_ylabel("Distance", fontsize=11)
    ax.set_title(title, fontsize=13)
    return fig


def plot_gap_statistic(gap_result: "GapResult") -> "Figure":
    """Plot observed ln(Wk), reference ln(Wk), and the gap statistic with error bars."""
    import matplotlib.pyplot as plt

    ks = gap_result.k_values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: log dispersions
    ax1.plot(ks, gap_result.observed_log_w, "-o", label="Observed", color="steelblue",
             markerfacecolor="white", linewidth=2)
    ax1.plot(ks, gap_result.reference_log_w, "-o", label="Reference (null)",
             color="tomato", markerfacecolor="white", linewidth=2)
    ax1.set_xlabel("Number of clusters k", fontsize=12)
    ax1.set_ylabel("ln(Wk)", fontsize=12)
    ax1.set_title("Within-cluster dispersion", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Panel 2: gap statistic
    ax2.errorbar(ks, gap_result.gap, yerr=gap_result.sk,
                 fmt="-o", capsize=5, linewidth=2, markerfacecolor="white",
                 color="steelblue", label="Gap(k) ± s_k")
    opt_k = gap_result.optimal_k
    ax2.axvline(opt_k, color="tomato", linestyle="--", label=f"Optimal k = {opt_k}")
    ax2.set_xlabel("Number of clusters k", fontsize=12)
    ax2.set_ylabel("Gap statistic", fontsize=12)
    ax2.set_title("Gap statistic (Tibshirani 2001)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    return fig


def plot_ftest(ftest_result: "FTestResult") -> "Figure":
    """Plot reduced chi-squared and F-test p-values for Euler-vector clustering."""
    import matplotlib.pyplot as plt

    ks = ftest_result.k_values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ks, ftest_result.chi2_reduced, "-o", linewidth=2,
             markerfacecolor="white", color="steelblue")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="χ²_red = 1")
    ax1.set_xlabel("Number of clusters k", fontsize=12)
    ax1.set_ylabel("Reduced χ²", fontsize=12)
    ax1.set_title("Euler-vector clustering fit quality", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax2.plot(ks[1:], ftest_result.p_values, "-o", linewidth=2,
             markerfacecolor="white", color="tomato")
    ax2.axhline(0.05, color="gray", linestyle="--", linewidth=1, label="α = 0.05")
    ax2.set_xlabel("Number of clusters k", fontsize=12)
    ax2.set_ylabel("F-test p-value (k vs k+1)", fontsize=12)
    ax2.set_title("Significance of adding one more cluster", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    return fig
