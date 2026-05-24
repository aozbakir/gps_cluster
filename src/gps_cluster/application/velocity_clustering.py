"""Velocity-space HAC clustering use case (baseline: Simpson et al. 2012).

Clusters GPS stations by their raw (Ve, Vn) velocity vectors using
Hierarchical Agglomerative Clustering (HAC) with centroid linkage.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from gps_cluster.domain.entities import GpsStation, VelocityCluster
from gps_cluster.domain.services.gap_statistic import GapResult, compute_gap_statistic


class VelocityHACClustering:
    """HAC on raw (Ve, Vn) velocity vectors.

    Parameters
    ----------
    linkage_method:
        Scipy linkage method. 'centroid' matches Simpson et al. (2012).
    """

    def __init__(self, linkage_method: str = "centroid") -> None:
        self.linkage_method = linkage_method
        self._linkage_matrix: np.ndarray | None = None

    def _velocity_array(self, stations: list[GpsStation]) -> np.ndarray:
        return np.array([[s.velocity.ve, s.velocity.vn] for s in stations])

    def fit(self, stations: list[GpsStation]) -> np.ndarray:
        """Compute and cache the HAC linkage matrix. Returns the matrix."""
        X = self._velocity_array(stations)
        self._linkage_matrix = linkage(X, method=self.linkage_method, metric="euclidean")
        return self._linkage_matrix

    def cluster(self, stations: list[GpsStation], k: int) -> list[VelocityCluster]:
        """Partition stations into k clusters. Calls fit() if not already done."""
        if self._linkage_matrix is None:
            self.fit(stations)
        labels = fcluster(self._linkage_matrix, t=k, criterion="maxclust")
        return [
            VelocityCluster(
                id=cluster_id,
                stations=[s for s, lbl in zip(stations, labels) if lbl == cluster_id],
            )
            for cluster_id in range(1, k + 1)
        ]

    def find_optimal_k(
        self,
        stations: list[GpsStation],
        max_k: int = 10,
        n_ref: int = 20,
        random_seed: int | None = 42,
    ) -> tuple[int, GapResult]:
        """Return (optimal_k, GapResult) using Tibshirani 2001 gap statistic."""
        X = self._velocity_array(stations)
        gap_result = compute_gap_statistic(
            X,
            max_k=max_k,
            n_ref=n_ref,
            linkage_method=self.linkage_method,
            random_seed=random_seed,
        )
        return gap_result.optimal_k, gap_result
