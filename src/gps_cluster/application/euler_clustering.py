"""Euler-vector clustering use case (Savage 2018).

Reference: Savage, J. C. (2018). Euler-vector clustering of GPS velocities
defines microplate geometry in southwest Japan.
Journal of Geophysical Research: Solid Earth, 123, 1437–1454.
https://doi.org/10.1002/2017JB014874

Algorithm
---------
For a given k:
1. Initial partition via velocity-space HAC (fast, well-conditioned start).
2. Invert for a best-fit Euler vector per cluster (weighted least squares).
3. Reassign each station to the cluster whose Euler vector minimises its
   weighted velocity residual.
4. Repeat 2–3 until labels converge or max_iter is reached.

Optimal k
---------
Run the algorithm for k = 1..max_k.  For each k, compute the total reduced
chi-squared:

    chi2_red(k) = sum_clusters chi2(cluster) / (2*N - 3*k)

where 2*N are the observations and 3*k the free parameters.

Use an F-test between consecutive k values to decide when adding a cluster
no longer yields a statistically significant improvement:

    F = [chi2(k) - chi2(k+1)] * dof(k+1) / chi2(k+1) / 3

which is F-distributed with (3, dof(k+1)) under the null that k clusters
are sufficient.  Reject the null at p < alpha to prefer k+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import f as f_dist

from gps_cluster.domain.entities import EulerVector, GpsStation, VelocityCluster
from gps_cluster.domain.services.euler_math import (
    invert_euler_vector,
    total_chi_squared,
    weighted_residual_sq,
)

_MIN_STATIONS_PER_CLUSTER = 2  # minimum for a determined Euler inversion


@dataclass
class FTestResult:
    k_values: np.ndarray
    chi2_total: np.ndarray
    chi2_reduced: np.ndarray
    f_statistics: np.ndarray  # length max_k - 1
    p_values: np.ndarray


class EulerVectorClustering:
    """Euler-vector clustering (Savage 2018).

    Parameters
    ----------
    max_iter:
        Maximum number of reassignment iterations per k.
    min_stations:
        Minimum stations required to invert a valid Euler vector for a cluster.
        Clusters that fall below this threshold are merged into the nearest
        cluster by velocity residual.
    """

    def __init__(self, max_iter: int = 100, min_stations: int = _MIN_STATIONS_PER_CLUSTER) -> None:
        self.max_iter = max_iter
        self.min_stations = min_stations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(self, stations: list[GpsStation], k: int) -> list[VelocityCluster]:
        """Partition stations into k clusters via Euler-vector iteration.

        Returns a list of VelocityCluster objects with euler_vector set.
        """
        if k >= len(stations):
            raise ValueError(f"k={k} must be less than the number of stations ({len(stations)})")

        labels = self._initial_labels(stations, k)
        labels = self._iterate(stations, labels, k)
        return self._build_clusters(stations, labels, k)

    def find_optimal_k(
        self,
        stations: list[GpsStation],
        max_k: int = 9,
        alpha: float = 0.05,
    ) -> tuple[int, FTestResult]:
        """Return (optimal_k, FTestResult) via sequential F-tests.

        The optimal k is the smallest k such that adding one more cluster
        does not yield a statistically significant chi-squared improvement
        at significance level alpha.
        """
        n_total = len(stations)
        chi2_vals = np.zeros(max_k)

        for ki, k in enumerate(range(1, max_k + 1)):
            clusters = self.cluster(stations, k)
            chi2_vals[ki] = self._total_chi2(clusters)

        dof = np.array([max(2 * n_total - 3 * k, 1) for k in range(1, max_k + 1)])
        chi2_red = chi2_vals / dof

        # F-test: H0 = k clusters are sufficient, H1 = k+1 is better
        f_stats = np.zeros(max_k - 1)
        p_vals = np.zeros(max_k - 1)
        for i in range(max_k - 1):
            delta = chi2_vals[i] - chi2_vals[i + 1]
            if chi2_vals[i + 1] > 0:
                f_stats[i] = delta * dof[i + 1] / chi2_vals[i + 1] / 3
            p_vals[i] = 1.0 - f_dist.cdf(f_stats[i], dfn=3, dfd=dof[i + 1])

        # Optimal k: first k where the improvement is not significant
        optimal_k = max_k
        for i, p in enumerate(p_vals):
            if p >= alpha:
                optimal_k = i + 1  # k = i+1 is sufficient
                break

        result = FTestResult(
            k_values=np.arange(1, max_k + 1),
            chi2_total=chi2_vals,
            chi2_reduced=chi2_red,
            f_statistics=f_stats,
            p_values=p_vals,
        )
        return optimal_k, result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_labels(self, stations: list[GpsStation], k: int) -> np.ndarray:
        """Velocity-space HAC as the initial partition."""
        X = np.array([[s.velocity.ve, s.velocity.vn] for s in stations])
        Z = linkage(X, method="centroid", metric="euclidean")
        return fcluster(Z, t=k, criterion="maxclust")

    def _euler_per_cluster(
        self, stations: list[GpsStation], labels: np.ndarray, k: int
    ) -> dict[int, EulerVector]:
        """Invert one Euler vector per cluster. Falls back to zero vector for tiny clusters."""
        euler_map: dict[int, EulerVector] = {}
        for cid in range(1, k + 1):
            members = [s for s, lbl in zip(stations, labels) if lbl == cid]
            if len(members) >= self.min_stations:
                euler_map[cid] = invert_euler_vector(members)
            else:
                euler_map[cid] = EulerVector(0.0, 0.0, 0.0)
        return euler_map

    def _reassign(
        self,
        stations: list[GpsStation],
        euler_map: dict[int, EulerVector],
    ) -> np.ndarray:
        """Assign each station to the cluster with the smallest weighted residual."""
        cluster_ids = sorted(euler_map)
        new_labels = np.zeros(len(stations), dtype=int)
        for i, s in enumerate(stations):
            residuals = {cid: weighted_residual_sq(s, euler_map[cid]) for cid in cluster_ids}
            new_labels[i] = min(residuals, key=residuals.get)
        return new_labels

    def _iterate(
        self, stations: list[GpsStation], labels: np.ndarray, k: int
    ) -> np.ndarray:
        """Run inversion + reassignment until convergence."""
        for _ in range(self.max_iter):
            euler_map = self._euler_per_cluster(stations, labels, k)
            new_labels = self._reassign(stations, euler_map)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        return labels

    def _build_clusters(
        self, stations: list[GpsStation], labels: np.ndarray, k: int
    ) -> list[VelocityCluster]:
        clusters = []
        for cid in range(1, k + 1):
            members = [s for s, lbl in zip(stations, labels) if lbl == cid]
            euler = invert_euler_vector(members) if len(members) >= self.min_stations else None
            clusters.append(VelocityCluster(id=cid, stations=members, euler_vector=euler))
        return clusters

    @staticmethod
    def _total_chi2(clusters: list[VelocityCluster]) -> float:
        total = 0.0
        for c in clusters:
            if c.euler_vector is not None and len(c.stations) > 0:
                total += total_chi_squared(c.stations, c.euler_vector)
        return total
