"""Unit tests for domain/services/gap_statistic.py."""

from __future__ import annotations

import numpy as np
import pytest

from gps_cluster.domain.services.gap_statistic import GapResult, compute_gap_statistic


def _two_blob_data(n_per_cluster: int = 30, seed: int = 0) -> np.ndarray:
    """Two well-separated Gaussian blobs in velocity space."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[-10, 0], scale=0.5, size=(n_per_cluster, 2))
    b = rng.normal(loc=[10, 0], scale=0.5, size=(n_per_cluster, 2))
    return np.vstack([a, b])


class TestGapStatistic:
    def test_returns_gap_result(self):
        data = _two_blob_data()
        result = compute_gap_statistic(data, max_k=5, n_ref=5)
        assert isinstance(result, GapResult)
        assert len(result.k_values) == 5
        assert len(result.gap) == 5
        assert len(result.sk) == 5

    def test_optimal_k_for_two_blobs(self):
        """Gap statistic should identify k=2 for two well-separated clusters."""
        data = _two_blob_data(n_per_cluster=50, seed=42)
        result = compute_gap_statistic(data, max_k=6, n_ref=20, random_seed=42)
        assert result.optimal_k == 2

    def test_gap_increases_then_flattens(self):
        """For blob data, gap(k=2) should exceed gap(k=1)."""
        data = _two_blob_data()
        result = compute_gap_statistic(data, max_k=4, n_ref=10)
        assert result.gap[1] > result.gap[0]

    def test_reproducible_with_same_seed(self):
        data = _two_blob_data()
        r1 = compute_gap_statistic(data, max_k=3, n_ref=5, random_seed=99)
        r2 = compute_gap_statistic(data, max_k=3, n_ref=5, random_seed=99)
        np.testing.assert_array_equal(r1.gap, r2.gap)

    def test_different_seeds_differ(self):
        data = _two_blob_data()
        r1 = compute_gap_statistic(data, max_k=3, n_ref=5, random_seed=1)
        r2 = compute_gap_statistic(data, max_k=3, n_ref=5, random_seed=2)
        assert not np.allclose(r1.gap, r2.gap)
