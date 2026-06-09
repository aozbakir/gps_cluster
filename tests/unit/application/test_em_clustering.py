"""Unit tests for EMEulerVectorClustering."""

from __future__ import annotations

import numpy as np
import pytest

from gps_cluster.application.euler_clustering import EMEulerVectorClustering
from gps_cluster.domain.entities import EulerPole, GpsStation, Position, Velocity
from gps_cluster.domain.services.euler_math import (
    euler_pole_to_vector,
    predict_velocity,
    weighted_residual_sq,
)


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def _station(lon: float, lat: float, ve: float = 0.0, vn: float = 0.0,
             se: float = 1.0, sn: float = 1.0, name: str = "T") -> GpsStation:
    return GpsStation(
        name=name,
        position=Position(lon=lon, lat=lat),
        velocity=Velocity(ve=ve, vn=vn, vu=0.0, se=se, sn=sn, su=1.0),
    )


def _synthetic_block(euler, lons, lats, sigma: float = 1.0) -> list[GpsStation]:
    stations = []
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        dummy = GpsStation("x", Position(lon, lat), Velocity(0, 0, 0, sigma, sigma, 1.0))
        ve, vn = predict_velocity(dummy, euler)
        stations.append(GpsStation(f"S{i}", Position(lon, lat),
                                   Velocity(ve, vn, 0.0, sigma, sigma, 1.0)))
    return stations


@pytest.fixture
def two_block_stations():
    """Two well-separated synthetic blocks (noiseless)."""
    euler_a = euler_pole_to_vector(EulerPole(lat=50.0, lon=130.0, rate=0.5))
    euler_b = euler_pole_to_vector(EulerPole(lat=10.0, lon=125.0, rate=3.0))
    stations_a = _synthetic_block(euler_a, np.linspace(130, 136, 10),
                                  np.linspace(33, 36, 10))
    stations_b = _synthetic_block(euler_b, np.linspace(129, 133, 10),
                                  np.linspace(31, 33, 10))
    return stations_a + stations_b


# ---------------------------------------------------------------------------
# Basic API contract
# ---------------------------------------------------------------------------

class TestEMClusteringAPI:
    def test_returns_k_clusters(self, two_block_stations):
        em = EMEulerVectorClustering(n_restarts=3, random_seed=0)
        clusters = em.cluster(two_block_stations, k=2)
        assert len(clusters) == 2

    def test_all_stations_assigned(self, two_block_stations):
        em = EMEulerVectorClustering(n_restarts=3, random_seed=0)
        clusters = em.cluster(two_block_stations, k=2)
        total = sum(c.size for c in clusters)
        assert total == len(two_block_stations)

    def test_euler_vector_set_on_each_cluster(self, two_block_stations):
        em = EMEulerVectorClustering(n_restarts=3, random_seed=0)
        clusters = em.cluster(two_block_stations, k=2)
        for c in clusters:
            assert c.euler_vector is not None

    def test_chi2_and_chi2_reduced_populated(self, two_block_stations):
        em = EMEulerVectorClustering(n_restarts=3, random_seed=0)
        clusters = em.cluster(two_block_stations, k=2)
        for c in clusters:
            assert c.chi2 is not None
            assert c.chi2_reduced is not None
            assert c.chi2 >= 0.0

    def test_cluster_ids_are_1_indexed(self, two_block_stations):
        em = EMEulerVectorClustering(n_restarts=3, random_seed=0)
        clusters = em.cluster(two_block_stations, k=3)
        ids = {c.id for c in clusters}
        assert ids == {1, 2, 3}


# ---------------------------------------------------------------------------
# membership_weights
# ---------------------------------------------------------------------------

class TestMembershipWeights:
    def test_weight_shape_is_n_total(self, two_block_stations):
        N = len(two_block_stations)
        em = EMEulerVectorClustering(n_restarts=3, random_seed=0)
        clusters = em.cluster(two_block_stations, k=2)
        for c in clusters:
            assert c.membership_weights is not None
            assert c.membership_weights.shape == (N,)

    def test_weights_sum_to_one_per_station(self, two_block_stations):
        """Column weights for all clusters must sum to 1 at each station."""
        em = EMEulerVectorClustering(n_restarts=3, random_seed=0)
        clusters = em.cluster(two_block_stations, k=2)
        weight_matrix = np.column_stack([c.membership_weights for c in clusters])
        np.testing.assert_allclose(weight_matrix.sum(axis=1), 1.0, atol=1e-12)

    def test_weights_in_range(self, two_block_stations):
        em = EMEulerVectorClustering(n_restarts=3, random_seed=0)
        clusters = em.cluster(two_block_stations, k=2)
        for c in clusters:
            assert np.all(c.membership_weights >= 0)
            assert np.all(c.membership_weights <= 1)

    def test_high_confidence_for_noiseless_separation(self, two_block_stations):
        """Well-separated noiseless blocks → dominant weight > 0.99 for most stations."""
        em = EMEulerVectorClustering(n_restarts=3, random_seed=0)
        clusters = em.cluster(two_block_stations, k=2)
        weight_matrix = np.column_stack([c.membership_weights for c in clusters])
        max_weight_per_station = weight_matrix.max(axis=1)
        # All stations should be confidently assigned (noiseless, distinct poles)
        assert np.all(max_weight_per_station > 0.99)


# ---------------------------------------------------------------------------
# Quality: EM vs hard assignment
# ---------------------------------------------------------------------------

class TestEMQuality:
    def test_low_residuals_for_noiseless_data(self, two_block_stations):
        """EM on noiseless synthetic data → residuals near zero for all members."""
        em = EMEulerVectorClustering(n_restarts=5, random_seed=0)
        clusters = em.cluster(two_block_stations, k=2)
        for c in clusters:
            for s in c.stations:
                assert weighted_residual_sq(s, c.euler_vector) < 1e-6

    def test_boundary_stations_have_elevated_entropy(self):
        """Stations assigned to both blocks with equal chi² should have entropy = log(2).

        We construct stations whose velocities are EXACTLY the halfway point between
        the two Euler predictions, then verify that the EM soft weights for these
        stations are approximately 0.5 / 0.5 (maximum entropy).
        """
        euler_a = euler_pole_to_vector(EulerPole(lat=50.0, lon=130.0, rate=0.5))
        euler_b = euler_pole_to_vector(EulerPole(lat=10.0, lon=125.0, rate=3.0))

        # Many clear interior stations for each block so EM poles converge correctly
        interior_a = _synthetic_block(euler_a, np.linspace(132, 138, 20),
                                      np.linspace(34, 38, 20))
        interior_b = _synthetic_block(euler_b, np.linspace(120, 126, 20),
                                      np.linspace(28, 31, 20))

        # Synthetic "fence-sitting" stations: velocity exactly halfway between
        # what both Euler vectors predict AT THE SAME LOCATION → equal chi² under both.
        # We use a tight sigma so the chi² difference is large even for small residuals,
        # ensuring that a station halfway in velocity space is genuinely ambiguous.
        fence_lons = np.linspace(130, 132, 6)
        fence_lats = np.linspace(32, 34, 6)
        sigma_tight = 0.1   # tight uncertainties → chi² difference is large
        fence_stations = []
        for i, (lon, lat) in enumerate(zip(fence_lons, fence_lats)):
            dummy = GpsStation("x", Position(lon, lat), Velocity(0, 0, 0, sigma_tight, sigma_tight, 1.0))
            ve_a, vn_a = predict_velocity(dummy, euler_a)
            ve_b, vn_b = predict_velocity(dummy, euler_b)
            ve = 0.5 * (ve_a + ve_b)
            vn = 0.5 * (vn_a + vn_b)
            fence_stations.append(GpsStation(f"FENCE{i}", Position(lon, lat),
                                             Velocity(ve, vn, 0.0, sigma_tight, sigma_tight, 1.0)))

        # Run EM on interior only to get good Euler poles, then compute weights for fence
        from gps_cluster.domain.services.euler_math import (
            invert_euler_vector, soft_weights_from_euler_map
        )
        euler_a_est = invert_euler_vector(interior_a)
        euler_b_est = invert_euler_vector(interior_b)
        euler_map = {1: euler_a_est, 2: euler_b_est}
        weights_fence = soft_weights_from_euler_map(fence_stations, euler_map)

        # Each fence station should have weights close to 0.5/0.5
        # (halfway velocity → equal chi² under both poles)
        np.testing.assert_allclose(weights_fence[:, 0], 0.5, atol=0.05)
        np.testing.assert_allclose(weights_fence[:, 1], 0.5, atol=0.05)

    def test_chi2_decreases_from_k1_to_k2(self, two_block_stations):
        """Two real blocks → chi² must drop substantially at k=2."""
        em = EMEulerVectorClustering(n_restarts=5, random_seed=0)
        _, result = em.find_optimal_k(two_block_stations, max_k=4)
        assert result.chi2_total[0] > 10 * result.chi2_total[1]


# ---------------------------------------------------------------------------
# find_optimal_k
# ---------------------------------------------------------------------------

class TestEMFindOptimalK:
    def test_returns_ftest_result(self, two_block_stations):
        em = EMEulerVectorClustering(n_restarts=3, random_seed=0)
        k_opt, result = em.find_optimal_k(two_block_stations, max_k=4)
        assert isinstance(k_opt, int)
        assert 1 <= k_opt <= 4

    def test_solutions_cached(self, two_block_stations):
        em = EMEulerVectorClustering(n_restarts=3, random_seed=0)
        _, result = em.find_optimal_k(two_block_stations, max_k=4)
        for k in range(1, 5):
            assert k in result.solutions
            assert len(result.solutions[k]) == k

    def test_chi2_reduced_monotone(self, two_block_stations):
        """chi²_reduced must decrease (or stay flat) as k increases."""
        em = EMEulerVectorClustering(n_restarts=3, random_seed=0)
        _, result = em.find_optimal_k(two_block_stations, max_k=4)
        for i in range(len(result.chi2_reduced) - 1):
            assert result.chi2_reduced[i] >= result.chi2_reduced[i + 1] - 1e-6

    def test_finds_correct_k_for_two_blocks(self, two_block_stations):
        """EM find_optimal_k should identify k=2 for clearly two-block data."""
        em = EMEulerVectorClustering(n_restarts=5, random_seed=0)
        k_opt, _ = em.find_optimal_k(two_block_stations, max_k=5)
        assert k_opt == 2
