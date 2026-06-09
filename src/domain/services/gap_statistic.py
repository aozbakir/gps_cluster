"""Tibshirani et al. (2001) gap statistic for selecting optimal cluster count.

Reference: Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the
number of clusters in a data set via the gap statistic.
Journal of the Royal Statistical Society: Series B, 63(2), 411-423.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist


@dataclass
class GapResult:
    k_values: np.ndarray
    observed_log_w: np.ndarray  # log(Wk) for observed data
    reference_log_w: np.ndarray  # mean log(Wk) over B reference datasets
    gap: np.ndarray  # E*[log(Wk_ref)] - log(Wk_obs)
    sk: np.ndarray  # simulation standard error * sqrt(1 + 1/B)

    @property
    def optimal_k(self) -> int:
        """Smallest k such that Gap(k) >= Gap(k+1) - s_{k+1} (Tibshirani criterion)."""
        for i in range(len(self.k_values) - 1):
            if self.gap[i] >= self.gap[i + 1] - self.sk[i + 1]:
                return int(self.k_values[i])
        return int(self.k_values[-1])

    @property
    def k_max_gap(self) -> int:
        """k that maximises Gap(k) — appropriate when the first-crossing rule
        trivially selects k=1 (e.g. strong-gradient datasets where Gap(1) is
        already large relative to Gap(2)-s_2)."""
        return int(self.k_values[np.argmax(self.gap)])


def _within_dispersion(data: np.ndarray, labels: np.ndarray) -> float:
    """Pooled within-cluster sum of squared distances: Wk = sum_r D_r / (2 n_r)."""
    unique = np.unique(labels)
    wk = 0.0
    for k in unique:
        cluster_pts = data[labels == k]
        if len(cluster_pts) == 0:
            continue
        centroid = cluster_pts.mean(axis=0, keepdims=True)
        wk += (cdist(cluster_pts, centroid) ** 2).sum() / 2.0
    return wk


def _reference_dataset(data_pca: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a reference dataset uniform in the PCA-aligned bounding box.

    Following Tibshirani: sample uniformly in the box aligned with the principal
    components of the data, then rotate back to the original space.
    """
    n, d = data_pca.shape
    lo = data_pca.min(axis=0)
    hi = data_pca.max(axis=0)
    return rng.uniform(lo, hi, size=(n, d))


def compute_gap_statistic(
    data: np.ndarray,
    max_k: int = 10,
    n_ref: int = 20,
    linkage_method: str = "centroid",
    random_seed: int | None = 42,
) -> GapResult:
    """Compute Tibshirani gap statistic for k = 1..max_k.

    Parameters
    ----------
    data:
        N x 2 array of velocity vectors (Ve, Vn).
    max_k:
        Maximum number of clusters to evaluate.
    n_ref:
        Number of Monte Carlo reference datasets (B in Tibshirani).
    linkage_method:
        Scipy linkage method for HAC (default: centroid, matching Simpson 2012).
    random_seed:
        Seed for reproducibility.
    """
    rng = np.random.default_rng(random_seed)

    # PCA-align the data (rotate to principal component axes)
    _, _, Vh = np.linalg.svd(data, full_matrices=False)
    V = Vh.T  # principal component directions (columns)
    data_pca = data @ V

    # Observed dispersions
    Z_obs = linkage(data, method=linkage_method, metric="euclidean")
    obs_log_w = np.array([
        np.log(_within_dispersion(data, fcluster(Z_obs, t=k, criterion="maxclust")))
        for k in range(1, max_k + 1)
    ])

    # Reference dispersions (B Monte Carlo draws)
    ref_log_w = np.zeros((n_ref, max_k))
    for b in range(n_ref):
        ref_pca = _reference_dataset(data_pca, rng)
        ref_orig = ref_pca @ V.T
        Z_ref = linkage(ref_orig, method=linkage_method, metric="euclidean")
        for ki, k in enumerate(range(1, max_k + 1)):
            labels = fcluster(Z_ref, t=k, criterion="maxclust")
            ref_log_w[b, ki] = np.log(_within_dispersion(ref_orig, labels))

    mean_ref = ref_log_w.mean(axis=0)
    sdk = ref_log_w.std(axis=0)
    sk = sdk * np.sqrt(1.0 + 1.0 / n_ref)
    gap = mean_ref - obs_log_w

    return GapResult(
        k_values=np.arange(1, max_k + 1),
        observed_log_w=obs_log_w,
        reference_log_w=mean_ref,
        gap=gap,
        sk=sk,
    )
