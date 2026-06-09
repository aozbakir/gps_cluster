"""Tests for SpatialBayesianEulerClustering and spatial_soft_weights."""

from __future__ import annotations

import numpy as np
import pytest

from gps_cluster.domain.entities import EulerVector, GpsStation, Position, Velocity
from gps_cluster.domain.services.euler_math import (
    predict_velocity,
    spatial_soft_weights,
)
from gps_cluster.domain.services.spatial import adjacency_matrix, delaunay_graph
from gps_cluster.application.euler_clustering import SpatialBayesianEulerClustering


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_station(lon, lat, ve, vn, se=0.5, sn=0.5, name=None):
    name = name or f"S{lon:.0f}{lat:.0f}"
    return GpsStation(
        name=name,
        position=Position(lon=lon, lat=lat),
        velocity=Velocity(ve=ve, vn=vn, vu=0.0, se=se, sn=sn, su=1.0),
    )


def _noiseless_stations_two_blocks(n_per_block=30, seed=0):
    """
    Two geographically separated blocks driven by distinct Euler vectors.
    Block A: lon in [26, 32], Euler = (0.3, -0.1, 0.2) mm/yr
    Block B: lon in [36, 42], Euler = (-0.1, 0.4, -0.3) mm/yr
    Velocities are exact (no noise) so hard assignment should be perfect
    and spatial prior should leave entropy near zero.
    """
    rng = np.random.default_rng(seed)
    omega_a = EulerVector(ox=0.3, oy=-0.1, oz=0.2)
    omega_b = EulerVector(ox=-0.1, oy=0.4, oz=-0.3)

    stations = []
    # Block A: western region
    for i in range(n_per_block):
        lon = rng.uniform(26.0, 32.0)
        lat = rng.uniform(37.0, 42.0)
        s = _make_station(lon, lat, 0.0, 0.0, name=f"A{i:02d}")
        ve, vn = predict_velocity(s, omega_a)
        stations.append(_make_station(lon, lat, ve, vn, name=f"A{i:02d}"))

    # Block B: eastern region
    for i in range(n_per_block):
        lon = rng.uniform(36.0, 42.0)
        lat = rng.uniform(37.0, 42.0)
        s = _make_station(lon, lat, 0.0, 0.0, name=f"B{i:02d}")
        ve, vn = predict_velocity(s, omega_b)
        stations.append(_make_station(lon, lat, ve, vn, name=f"B{i:02d}"))

    return stations, omega_a, omega_b


# ── spatial_soft_weights ──────────────────────────────────────────────────────

class TestSpatialSoftWeights:

    def _two_block_setup(self, n=20, seed=0):
        stations, omega_a, omega_b = _noiseless_stations_two_blocks(n, seed)
        euler_map = {1: omega_a, 2: omega_b}
        graph = delaunay_graph(stations)
        adj = adjacency_matrix(graph, len(stations))
        return stations, euler_map, adj

    def test_output_shape(self):
        stations, euler_map, adj = self._two_block_setup()
        w = spatial_soft_weights(stations, euler_map, adj, beta=1.0)
        assert w.shape == (len(stations), 2)

    def test_rows_sum_to_one(self):
        stations, euler_map, adj = self._two_block_setup()
        w = spatial_soft_weights(stations, euler_map, adj, beta=1.0)
        assert np.allclose(w.sum(axis=1), 1.0)

    def test_weights_in_unit_interval(self):
        stations, euler_map, adj = self._two_block_setup()
        w = spatial_soft_weights(stations, euler_map, adj, beta=1.0)
        assert np.all(w >= 0.0) and np.all(w <= 1.0)

    def test_beta_zero_equals_no_spatial(self):
        """beta=0 must produce same result as soft_weights_from_euler_map."""
        from gps_cluster.domain.services.euler_math import soft_weights_from_euler_map
        stations, euler_map, adj = self._two_block_setup()
        w_spatial = spatial_soft_weights(stations, euler_map, adj, beta=0.0)
        w_plain   = soft_weights_from_euler_map(stations, euler_map)
        assert np.allclose(w_spatial, w_plain, atol=1e-6)

    def test_noiseless_two_blocks_high_confidence(self):
        """Noiseless well-separated blocks: mean dominant weight > 0.95.
        Stations near the geographic boundary (lon ≈ 32–36) may have slightly
        lower confidence due to the spatial prior pulling both ways."""
        stations, euler_map, adj = self._two_block_setup(n=30)
        w = spatial_soft_weights(stations, euler_map, adj, beta=1.0)
        dominant = w.max(axis=1)
        assert dominant.mean() > 0.95

    def test_spatial_prior_increases_confidence(self):
        """
        For mildly noisy data, adding beta > 0 should push high-confidence
        stations to even higher confidence (spatial regularisation reinforces
        dominant cluster), measured by mean dominant weight.
        """
        rng = np.random.default_rng(42)
        stations, omega_a, omega_b = _noiseless_stations_two_blocks(30, seed=1)
        # Add small noise so beta=0 leaves some ambiguity
        noisy = []
        for s in stations:
            noise_e = rng.normal(0, 0.3)
            noise_n = rng.normal(0, 0.3)
            noisy.append(_make_station(
                s.position.lon, s.position.lat,
                s.velocity.ve + noise_e, s.velocity.vn + noise_n,
                name=s.name,
            ))

        euler_map = {1: omega_a, 2: omega_b}
        graph = delaunay_graph(noisy)
        adj   = adjacency_matrix(graph, len(noisy))

        w0 = spatial_soft_weights(noisy, euler_map, adj, beta=0.0)
        w1 = spatial_soft_weights(noisy, euler_map, adj, beta=2.0)

        mean_conf_0 = w0.max(axis=1).mean()
        mean_conf_1 = w1.max(axis=1).mean()
        assert mean_conf_1 >= mean_conf_0

    def test_uniform_pi_versus_none(self):
        """Explicit uniform pi should give same result as pi=None."""
        stations, euler_map, adj = self._two_block_setup()
        K = len(euler_map)
        pi = np.ones(K) / K
        w_none = spatial_soft_weights(stations, euler_map, adj, beta=1.0, pi=None)
        w_unif = spatial_soft_weights(stations, euler_map, adj, beta=1.0, pi=pi)
        assert np.allclose(w_none, w_unif, atol=1e-6)


# ── SpatialBayesianEulerClustering ───────────────────────────────────────────

class TestSpatialBayesianEulerClustering:

    def test_cluster_returns_k_clusters(self):
        stations, _, _ = _noiseless_stations_two_blocks(25)
        vbc = SpatialBayesianEulerClustering(beta=1.0, n_restarts=3)
        clusters = vbc.cluster(stations, k=2)
        assert len(clusters) == 2

    def test_all_stations_assigned(self):
        stations, _, _ = _noiseless_stations_two_blocks(25)
        vbc = SpatialBayesianEulerClustering(beta=1.0, n_restarts=3)
        clusters = vbc.cluster(stations, k=2)
        total = sum(len(c.stations) for c in clusters)
        assert total == len(stations)

    def test_euler_vectors_set(self):
        stations, _, _ = _noiseless_stations_two_blocks(25)
        vbc = SpatialBayesianEulerClustering(beta=1.0, n_restarts=3)
        clusters = vbc.cluster(stations, k=2)
        assert all(c.euler_vector is not None for c in clusters)

    def test_chi2_populated(self):
        stations, _, _ = _noiseless_stations_two_blocks(25)
        vbc = SpatialBayesianEulerClustering(beta=1.0, n_restarts=3)
        clusters = vbc.cluster(stations, k=2)
        assert all(c.chi2 is not None for c in clusters)

    def test_membership_weights_shape(self):
        """membership_weights should have shape (N,) — all stations."""
        stations, _, _ = _noiseless_stations_two_blocks(25)
        N = len(stations)
        vbc = SpatialBayesianEulerClustering(beta=1.0, n_restarts=3)
        clusters = vbc.cluster(stations, k=2)
        for c in clusters:
            assert c.membership_weights is not None
            assert c.membership_weights.shape == (N,)

    def test_membership_weights_sum_to_one(self):
        """Column weights across clusters sum to 1 at each station."""
        stations, _, _ = _noiseless_stations_two_blocks(25)
        vbc = SpatialBayesianEulerClustering(beta=1.0, n_restarts=3)
        clusters = vbc.cluster(stations, k=2)
        W = np.column_stack([c.membership_weights for c in clusters])
        assert np.allclose(W.sum(axis=1), 1.0, atol=1e-6)

    def test_noiseless_two_blocks_correct_assignment(self):
        """
        Noiseless well-separated blocks: spatial VB should assign each
        station to the geographically correct block (A-stations to one
        cluster, B-stations to the other).
        """
        stations, _, _ = _noiseless_stations_two_blocks(30)
        vbc = SpatialBayesianEulerClustering(beta=1.0, n_restarts=5)
        clusters = vbc.cluster(stations, k=2)

        # Find which cluster corresponds to block A (lon < 34)
        # and block B (lon > 34)
        def centroid_lon(c):
            return np.mean([s.position.lon for s in c.stations])

        c_west, c_east = sorted(clusters, key=centroid_lon)

        # All A-stations should be in the western cluster
        a_names = {f"A{i:02d}" for i in range(30)}
        b_names = {f"B{i:02d}" for i in range(30)}
        west_names = {s.name for s in c_west.stations}
        east_names = {s.name for s in c_east.stations}

        assert len(a_names & west_names) == 30
        assert len(b_names & east_names) == 30

    def test_low_residuals_noiseless(self):
        """Noiseless data: chi2_reduced should be near zero."""
        stations, _, _ = _noiseless_stations_two_blocks(30)
        vbc = SpatialBayesianEulerClustering(beta=1.0, n_restarts=5)
        clusters = vbc.cluster(stations, k=2)
        for c in clusters:
            assert c.chi2_reduced is not None
            assert c.chi2_reduced < 0.01

    def test_entropy_elevated_at_boundary(self):
        """
        Station placed exactly at the geographic boundary between the two
        blocks (lon ≈ 34) with velocity halfway between the two Euler
        predictions should have higher entropy than interior stations.
        """
        stations, omega_a, omega_b = _noiseless_stations_two_blocks(30, seed=7)

        # Add a boundary station with halfway velocity
        lon_b, lat_b = 34.0, 39.5
        s_tmp = _make_station(lon_b, lat_b, 0.0, 0.0, name="BNDRY")
        ve_a, vn_a = predict_velocity(s_tmp, omega_a)
        ve_b, vn_b = predict_velocity(s_tmp, omega_b)
        boundary = _make_station(lon_b, lat_b,
                                  0.5 * (ve_a + ve_b), 0.5 * (vn_a + vn_b),
                                  se=0.5, sn=0.5, name="BNDRY")
        all_stations = stations + [boundary]

        vbc = SpatialBayesianEulerClustering(beta=1.0, n_restarts=5)
        clusters = vbc.cluster(all_stations, k=2)

        # Compute per-station entropy from membership weights
        W = np.column_stack([c.membership_weights for c in clusters])
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.nansum(W * np.where(W > 0, np.log(W), 0.0), axis=1)

        bndry_idx = len(all_stations) - 1
        mean_interior = entropy[:-1].mean()
        assert entropy[bndry_idx] > mean_interior, (
            f"Boundary entropy {entropy[bndry_idx]:.4f} not above "
            f"interior mean {mean_interior:.4f}"
        )

    def test_beta_zero_close_to_em(self):
        """
        beta=0 should produce near-identical results to EMEulerVectorClustering
        (both use same likelihood, same init, no spatial prior).
        """
        from gps_cluster.application.euler_clustering import EMEulerVectorClustering
        stations, _, _ = _noiseless_stations_two_blocks(20, seed=3)

        em  = EMEulerVectorClustering(n_restarts=5, random_seed=0)
        vb0 = SpatialBayesianEulerClustering(beta=0.0, n_restarts=5, random_seed=0)

        c_em  = em.cluster(stations, k=2)
        c_vb0 = vb0.cluster(stations, k=2)

        chi2_em  = sum(c.chi2 for c in c_em  if c.chi2 is not None)
        chi2_vb0 = sum(c.chi2 for c in c_vb0 if c.chi2 is not None)

        # chi2 should agree to within 1% (same algorithm, same init)
        assert abs(chi2_em - chi2_vb0) / max(chi2_em, 1e-9) < 0.01

    def test_find_optimal_k_returns_ftest(self):
        from gps_cluster.application.euler_clustering import FTestResult
        stations, _, _ = _noiseless_stations_two_blocks(25)
        vbc = SpatialBayesianEulerClustering(beta=1.0, n_restarts=3)
        k_opt, result = vbc.find_optimal_k(stations, max_k=4)
        assert isinstance(result, FTestResult)
        assert 1 <= k_opt <= 4

    def test_find_optimal_k_identifies_two_blocks(self):
        """Noiseless two-block data: optimal k should be 2."""
        stations, _, _ = _noiseless_stations_two_blocks(30)
        vbc = SpatialBayesianEulerClustering(beta=1.0, n_restarts=5)
        k_opt, _ = vbc.find_optimal_k(stations, max_k=5)
        assert k_opt == 2
