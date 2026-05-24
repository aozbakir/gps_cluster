"""Unit tests for application/euler_clustering.py."""

from __future__ import annotations

import pytest

from gps_cluster.application.euler_clustering import EulerVectorClustering
from gps_cluster.domain.services.euler_math import weighted_residual_sq


class TestEulerVectorClustering:
    def test_cluster_count(self, two_cluster_stations):
        model = EulerVectorClustering()
        clusters = model.cluster(two_cluster_stations, k=2)
        assert len(clusters) == 2

    def test_total_stations_preserved(self, two_cluster_stations):
        model = EulerVectorClustering()
        clusters = model.cluster(two_cluster_stations, k=2)
        assert sum(c.size for c in clusters) == len(two_cluster_stations)

    def test_euler_vector_set_on_clusters(self, two_cluster_stations):
        model = EulerVectorClustering()
        clusters = model.cluster(two_cluster_stations, k=2)
        for c in clusters:
            assert c.euler_vector is not None

    def test_low_residuals_for_synthetic_data(self, synthetic_cluster_a, synthetic_cluster_b):
        """Noiseless synthetic data from 2 Euler poles → residuals near zero after clustering."""
        _, stations_a = synthetic_cluster_a
        _, stations_b = synthetic_cluster_b
        all_stations = stations_a + stations_b

        model = EulerVectorClustering()
        clusters = model.cluster(all_stations, k=2)

        for c in clusters:
            if c.euler_vector is not None:
                max_res = max(weighted_residual_sq(s, c.euler_vector) for s in c.stations)
                assert max_res < 1e-6, f"Cluster {c.id} has large residuals: {max_res}"

    def test_optimal_k_finds_two_for_two_populations(self, two_cluster_stations):
        model = EulerVectorClustering()
        k_opt, result = model.find_optimal_k(two_cluster_stations, max_k=5)
        assert k_opt == 2

    def test_raises_when_k_exceeds_stations(self, two_cluster_stations):
        model = EulerVectorClustering()
        with pytest.raises(ValueError, match="k="):
            model.cluster(two_cluster_stations, k=len(two_cluster_stations))

    def test_chi2_decreases_with_more_clusters(self, two_cluster_stations):
        """Adding a cluster should not increase total chi-squared."""
        model = EulerVectorClustering()
        _, result = model.find_optimal_k(two_cluster_stations, max_k=4)
        for i in range(len(result.chi2_total) - 1):
            assert result.chi2_total[i] >= result.chi2_total[i + 1] - 1e-6
