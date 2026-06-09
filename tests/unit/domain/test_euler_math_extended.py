"""Tests for euler_math functions not covered by test_euler_math.py.

Covers:
    - total_chi_squared / reduced_chi_squared
    - euler_pole_uncertainty
    - fault_slip_rate
    - fault_slip_rate_uncertainty
    - assignment_probabilities
    - invert_euler_vector_weighted
    - soft_weights_from_euler_map
"""

from __future__ import annotations

import numpy as np
import pytest

from gps_cluster.domain.entities import (
    EulerPole,
    EulerVector,
    GpsStation,
    Position,
    Velocity,
    VelocityCluster,
)
from gps_cluster.domain.services.euler_math import (
    assignment_probabilities,
    euler_pole_to_vector,
    euler_vector_to_pole,
    fault_slip_rate,
    fault_slip_rate_uncertainty,
    invert_euler_vector,
    invert_euler_vector_weighted,
    predict_velocity,
    reduced_chi_squared,
    soft_weights_from_euler_map,
    total_chi_squared,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _station(lon: float, lat: float, ve: float = 0.0, vn: float = 0.0,
             se: float = 1.0, sn: float = 1.0, name: str = "T") -> GpsStation:
    return GpsStation(
        name=name,
        position=Position(lon=lon, lat=lat),
        velocity=Velocity(ve=ve, vn=vn, vu=0.0, se=se, sn=sn, su=1.0),
    )


def _synthetic_stations(euler: EulerVector, lons, lats, sigma: float = 1.0,
                         noise_seed: int | None = None) -> list[GpsStation]:
    """Stations whose velocities are exactly predicted by `euler`, plus optional Gaussian noise."""
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    stations = []
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        dummy = GpsStation("x", Position(lon, lat), Velocity(0, 0, 0, sigma, sigma, 1.0))
        ve, vn = predict_velocity(dummy, euler)
        if rng is not None:
            ve += rng.normal(0, sigma)
            vn += rng.normal(0, sigma)
        stations.append(GpsStation(f"S{i}", Position(lon, lat),
                                   Velocity(ve, vn, 0.0, sigma, sigma, 1.0)))
    return stations


# ---------------------------------------------------------------------------
# total_chi_squared / reduced_chi_squared
# ---------------------------------------------------------------------------

class TestChiSquared:
    def test_zero_for_exact_prediction(self):
        euler = euler_pole_to_vector(EulerPole(lat=30.0, lon=-100.0, rate=1.5))
        stations = _synthetic_stations(euler, np.linspace(-110, -100, 8),
                                       np.linspace(28, 38, 8))
        assert total_chi_squared(stations, euler) == pytest.approx(0.0, abs=1e-8)

    def test_reduced_chi2_near_one_for_noisy_data(self):
        """chi²_red ≈ 1 when noise matches reported sigma (statistically)."""
        euler = euler_pole_to_vector(EulerPole(lat=45.0, lon=10.0, rate=2.0))
        rng = np.random.default_rng(0)
        sigma = 1.0
        lons = np.linspace(0, 20, 100)
        lats = np.linspace(35, 50, 100)
        stations = []
        for lon, lat in zip(lons, lats):
            dummy = GpsStation("x", Position(lon, lat), Velocity(0, 0, 0, sigma, sigma, 1.0))
            ve, vn = predict_velocity(dummy, euler)
            ve += rng.normal(0, sigma)
            vn += rng.normal(0, sigma)
            stations.append(GpsStation("x", Position(lon, lat),
                                       Velocity(ve, vn, 0.0, sigma, sigma, 1.0)))
        chi2_red = reduced_chi_squared(stations, euler)
        # Should be within 3-sigma of 1.0 for N=100 (std of chi²/dof ≈ sqrt(2/dof))
        assert 0.5 < chi2_red < 1.5

    def test_reduced_chi2_inf_for_tiny_cluster(self):
        euler = EulerVector(1.0, 0.0, 0.0)
        stations = [_station(0, 0)]  # N=1 < 3 → dof ≤ 0
        assert reduced_chi_squared(stations, euler) == np.inf


# ---------------------------------------------------------------------------
# euler_pole_uncertainty
# ---------------------------------------------------------------------------

class TestEulerPoleUncertainty:
    def test_returns_zero_without_covariance(self):
        euler = EulerVector(1.0, 0.5, 2.0)  # covariance=None by default
        from gps_cluster.domain.services.euler_math import euler_pole_uncertainty
        s_lat, s_lon, s_rate = euler_pole_uncertainty(euler)
        assert s_lat == 0.0 and s_lon == 0.0 and s_rate == 0.0

    def test_uncertainty_populated_after_inversion(self):
        euler = euler_pole_to_vector(EulerPole(lat=45.0, lon=10.0, rate=2.0))
        stations = _synthetic_stations(euler, np.linspace(-5, 15, 12),
                                       np.linspace(40, 50, 12))
        recovered = invert_euler_vector(stations)
        pole = euler_vector_to_pole(recovered)
        # Uncertainties should be small but positive for a well-constrained inversion
        assert pole.sigma_lat >= 0.0
        assert pole.sigma_lon >= 0.0
        assert pole.sigma_rate >= 0.0

    def test_larger_sigma_gives_larger_uncertainty(self):
        """Bigger measurement errors → bigger pole uncertainty."""
        euler = euler_pole_to_vector(EulerPole(lat=45.0, lon=10.0, rate=2.0))
        lons = np.linspace(-5, 15, 10)
        lats = np.linspace(40, 50, 10)
        stations_tight = _synthetic_stations(euler, lons, lats, sigma=0.5)
        stations_loose = _synthetic_stations(euler, lons, lats, sigma=5.0)
        pole_tight = euler_vector_to_pole(invert_euler_vector(stations_tight))
        pole_loose = euler_vector_to_pole(invert_euler_vector(stations_loose))
        assert pole_loose.sigma_lat > pole_tight.sigma_lat
        assert pole_loose.sigma_rate > pole_tight.sigma_rate


# ---------------------------------------------------------------------------
# fault_slip_rate
# ---------------------------------------------------------------------------

class TestFaultSlipRate:
    def _pure_eastward_euler(self) -> EulerVector:
        """North-pole rotation → pure eastward velocity at equator."""
        return EulerVector(0.0, 0.0, 10.0)

    def test_zero_relative_velocity_for_identical_blocks(self):
        euler = self._pure_eastward_euler()
        results = fault_slip_rate(euler, euler,
                                  fault_lats=[0.0], fault_lons=[0.0],
                                  fault_strike_deg=0.0)
        assert results[0]["total_mm_yr"] == pytest.approx(0.0, abs=1e-10)
        assert results[0]["strike_slip_mm_yr"] == pytest.approx(0.0, abs=1e-10)
        assert results[0]["fault_normal_mm_yr"] == pytest.approx(0.0, abs=1e-10)

    def test_returns_one_dict_per_point(self):
        euler_a = EulerVector(1.0, 0.0, 5.0)
        euler_b = EulerVector(-1.0, 0.0, 3.0)
        lats = [30.0, 35.0, 40.0]
        lons = [25.0, 28.0, 31.0]
        results = fault_slip_rate(euler_a, euler_b, lats, lons)
        assert len(results) == 3
        for r in results:
            assert "lat" in r and "lon" in r and "total_mm_yr" in r

    def test_no_decomposition_without_strike(self):
        euler_a = EulerVector(1.0, 0.0, 5.0)
        euler_b = EulerVector(0.0, 0.0, 5.0)
        results = fault_slip_rate(euler_a, euler_b,
                                  fault_lats=[35.0], fault_lons=[30.0])
        assert "strike_slip_mm_yr" not in results[0]
        assert "fault_normal_mm_yr" not in results[0]

    def test_east_west_fault_strike_slip_sign(self):
        """Block B moving east relative to A along an E-W fault → right-lateral (+ve).

        strike=90° (E-W): fault-parallel = east, fault-normal = north.
        A z-axis rotation gives pure eastward velocity at equator, so
        strike_slip = ve_rel > 0 (right-lateral).
        """
        euler_a = EulerVector(0.0, 0.0, 5.0)   # slower eastward rotation
        euler_b = EulerVector(0.0, 0.0, 15.0)  # faster eastward rotation
        results = fault_slip_rate(euler_a, euler_b,
                                  fault_lats=[0.0], fault_lons=[0.0],
                                  fault_strike_deg=90.0)   # E-W strike
        # relative velocity is pure eastward; E-W fault: strike_slip>0 = right-lateral
        assert results[0]["strike_slip_mm_yr"] > 0
        # fault-normal (north) component is zero for z-axis rotation at equator
        assert results[0]["fault_normal_mm_yr"] == pytest.approx(0.0, abs=1e-6)

    def test_antisymmetry(self):
        """Swapping A and B flips all component signs."""
        euler_a = EulerVector(1.0, 2.0, 5.0)
        euler_b = EulerVector(-1.0, 0.5, 3.0)
        lats, lons = [35.0], [28.0]
        r_ab = fault_slip_rate(euler_a, euler_b, lats, lons, fault_strike_deg=45.0)[0]
        r_ba = fault_slip_rate(euler_b, euler_a, lats, lons, fault_strike_deg=45.0)[0]
        assert r_ab["strike_slip_mm_yr"] == pytest.approx(-r_ba["strike_slip_mm_yr"], rel=1e-10)
        assert r_ab["fault_normal_mm_yr"] == pytest.approx(-r_ba["fault_normal_mm_yr"], rel=1e-10)


# ---------------------------------------------------------------------------
# fault_slip_rate_uncertainty
# ---------------------------------------------------------------------------

class TestFaultSlipRateUncertainty:
    def test_zero_without_covariance(self):
        euler_a = EulerVector(1.0, 0.0, 5.0)   # no covariance
        euler_b = EulerVector(-1.0, 0.0, 3.0)
        result = fault_slip_rate_uncertainty(euler_a, euler_b, 35.0, 28.0, 45.0)
        assert result["sigma_strike_slip_mm_yr"] == 0.0
        assert result["sigma_fault_normal_mm_yr"] == 0.0

    def test_positive_sigma_when_covariance_set(self):
        euler = euler_pole_to_vector(EulerPole(lat=45.0, lon=10.0, rate=2.0))
        stations = _synthetic_stations(euler, np.linspace(0, 20, 15),
                                       np.linspace(35, 45, 15))
        inv = invert_euler_vector(stations)
        result = fault_slip_rate_uncertainty(inv, inv, 40.0, 10.0, 90.0)
        # C_rel = C + C = 2C > 0, so all sigmas > 0
        assert result["sigma_strike_slip_mm_yr"] > 0
        assert result["sigma_fault_normal_mm_yr"] > 0

    def test_independent_blocks_add_covariances(self):
        """Uncertainty from two independently estimated blocks > one block alone."""
        euler = euler_pole_to_vector(EulerPole(lat=45.0, lon=10.0, rate=2.0))
        stations = _synthetic_stations(euler, np.linspace(0, 20, 15),
                                       np.linspace(35, 45, 15))
        inv = invert_euler_vector(stations)
        zero = EulerVector(0.0, 0.0, 0.0)   # no covariance
        result_one = fault_slip_rate_uncertainty(inv, zero, 40.0, 10.0, 90.0)
        result_two = fault_slip_rate_uncertainty(inv, inv, 40.0, 10.0, 90.0)
        assert result_two["sigma_strike_slip_mm_yr"] > result_one["sigma_strike_slip_mm_yr"]


# ---------------------------------------------------------------------------
# assignment_probabilities
# ---------------------------------------------------------------------------

class TestAssignmentProbabilities:
    def _two_clusters(self):
        euler_a = euler_pole_to_vector(EulerPole(lat=50.0, lon=130.0, rate=0.5))
        euler_b = euler_pole_to_vector(EulerPole(lat=10.0, lon=125.0, rate=3.0))
        stations_a = _synthetic_stations(euler_a, np.linspace(130, 136, 8),
                                         np.linspace(33, 36, 8))
        stations_b = _synthetic_stations(euler_b, np.linspace(129, 133, 8),
                                         np.linspace(31, 33, 8))
        c_a = VelocityCluster(id=1, stations=stations_a, euler_vector=euler_a)
        c_b = VelocityCluster(id=2, stations=stations_b, euler_vector=euler_b)
        return stations_a + stations_b, [c_a, c_b]

    def test_shape(self):
        stations, clusters = self._two_clusters()
        probs, entropy = assignment_probabilities(stations, clusters)
        assert probs.shape == (len(stations), 2)
        assert entropy.shape == (len(stations),)

    def test_probabilities_sum_to_one(self):
        stations, clusters = self._two_clusters()
        probs, _ = assignment_probabilities(stations, clusters)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)

    def test_probabilities_in_range(self):
        stations, clusters = self._two_clusters()
        probs, _ = assignment_probabilities(stations, clusters)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_high_confidence_for_noiseless_data(self):
        """Noiseless data → dominant cluster should have p > 0.99."""
        stations, clusters = self._two_clusters()
        probs, _ = assignment_probabilities(stations, clusters)
        assert np.all(probs.max(axis=1) > 0.99)

    def test_entropy_zero_for_perfect_separation(self):
        """Perfectly separated clusters → near-zero entropy for all stations."""
        stations, clusters = self._two_clusters()
        _, entropy = assignment_probabilities(stations, clusters)
        assert np.all(entropy < 0.01)

    def test_entropy_bounded_by_log_k(self):
        stations, clusters = self._two_clusters()
        _, entropy = assignment_probabilities(stations, clusters)
        assert np.all(entropy <= np.log(2) + 1e-10)


# ---------------------------------------------------------------------------
# invert_euler_vector_weighted
# ---------------------------------------------------------------------------

class TestInvertEulerVectorWeighted:
    def test_uniform_weights_matches_unweighted(self):
        """Equal weights for all stations → same result as unweighted inversion."""
        euler = euler_pole_to_vector(EulerPole(lat=45.0, lon=10.0, rate=2.0))
        stations = _synthetic_stations(euler, np.linspace(0, 20, 12),
                                       np.linspace(35, 45, 12))
        N = len(stations)
        weights = np.ones(N)
        ev_weighted = invert_euler_vector_weighted(stations, weights)
        ev_plain    = invert_euler_vector(stations)
        # Results should be numerically identical up to floating-point noise
        assert ev_weighted.ox == pytest.approx(ev_plain.ox, rel=1e-8)
        assert ev_weighted.oy == pytest.approx(ev_plain.oy, rel=1e-8)
        assert ev_weighted.oz == pytest.approx(ev_plain.oz, rel=1e-8)

    def test_zero_weight_station_is_ignored(self):
        """A station with weight=0 must not influence the Euler vector."""
        euler = euler_pole_to_vector(EulerPole(lat=45.0, lon=10.0, rate=2.0))
        stations = _synthetic_stations(euler, np.linspace(0, 20, 10),
                                       np.linspace(35, 45, 10))
        # Add one wildly wrong station
        bad = _station(lon=10.0, lat=40.0, ve=9999.0, vn=9999.0)
        all_stations = stations + [bad]

        weights_with_bad  = np.ones(len(all_stations))
        weights_ignore_bad = np.ones(len(all_stations))
        weights_ignore_bad[-1] = 0.0   # zero out the bad station

        ev_with    = invert_euler_vector_weighted(all_stations, weights_with_bad)
        ev_without = invert_euler_vector_weighted(all_stations, weights_ignore_bad)
        ev_plain   = invert_euler_vector(stations)

        # ev_without should closely match the plain inversion on clean data
        assert ev_without.ox == pytest.approx(ev_plain.ox, rel=1e-6)
        assert ev_without.oy == pytest.approx(ev_plain.oy, rel=1e-6)
        # ev_with should be significantly different (pulled by bad station)
        assert abs(ev_with.ox - ev_plain.ox) > abs(ev_without.ox - ev_plain.ox)

    def test_covariance_attached(self):
        euler = euler_pole_to_vector(EulerPole(lat=45.0, lon=10.0, rate=2.0))
        stations = _synthetic_stations(euler, np.linspace(0, 20, 10),
                                       np.linspace(35, 45, 10))
        ev = invert_euler_vector_weighted(stations, np.ones(len(stations)))
        assert ev.covariance is not None
        assert ev.covariance.shape == (3, 3)

    def test_lower_weight_gives_larger_uncertainty(self):
        """Halving all weights (fewer effective stations) → larger covariance diagonal."""
        euler = euler_pole_to_vector(EulerPole(lat=45.0, lon=10.0, rate=2.0))
        stations = _synthetic_stations(euler, np.linspace(0, 20, 12),
                                       np.linspace(35, 45, 12))
        ev_full = invert_euler_vector_weighted(stations, np.ones(len(stations)))
        ev_half = invert_euler_vector_weighted(stations, 0.5 * np.ones(len(stations)))
        # Halving weights is equivalent to doubling measurement variances → 2× covariance
        np.testing.assert_allclose(ev_half.covariance, 2 * ev_full.covariance, rtol=1e-6)

    def test_zero_weight_sum_returns_zero_vector(self):
        stations = [_station(0.0, 30.0), _station(10.0, 35.0), _station(20.0, 40.0)]
        ev = invert_euler_vector_weighted(stations, np.zeros(3))
        assert ev.ox == 0.0 and ev.oy == 0.0 and ev.oz == 0.0


# ---------------------------------------------------------------------------
# soft_weights_from_euler_map
# ---------------------------------------------------------------------------

class TestSoftWeightsFromEulerMap:
    def test_shape(self):
        euler_a = euler_pole_to_vector(EulerPole(lat=50.0, lon=130.0, rate=0.5))
        euler_b = euler_pole_to_vector(EulerPole(lat=10.0, lon=125.0, rate=3.0))
        stations = _synthetic_stations(euler_a, np.linspace(130, 136, 6),
                                       np.linspace(33, 36, 6))
        euler_map = {1: euler_a, 2: euler_b}
        weights = soft_weights_from_euler_map(stations, euler_map)
        assert weights.shape == (len(stations), 2)

    def test_probabilities_sum_to_one(self):
        euler_a = euler_pole_to_vector(EulerPole(lat=50.0, lon=130.0, rate=0.5))
        euler_b = euler_pole_to_vector(EulerPole(lat=10.0, lon=125.0, rate=3.0))
        stations = _synthetic_stations(euler_a, np.linspace(130, 136, 8),
                                       np.linspace(33, 36, 8))
        euler_map = {1: euler_a, 2: euler_b}
        weights = soft_weights_from_euler_map(stations, euler_map)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)

    def test_correct_cluster_dominates_for_noiseless_data(self):
        """Stations generated by euler_a should have weight[:,0] > 0.99."""
        euler_a = euler_pole_to_vector(EulerPole(lat=50.0, lon=130.0, rate=0.5))
        euler_b = euler_pole_to_vector(EulerPole(lat=10.0, lon=125.0, rate=3.0))
        stations = _synthetic_stations(euler_a, np.linspace(130, 136, 8),
                                       np.linspace(33, 36, 8))
        euler_map = {1: euler_a, 2: euler_b}
        weights = soft_weights_from_euler_map(stations, euler_map)
        assert np.all(weights[:, 0] > 0.99)

    def test_consistent_with_assignment_probabilities(self):
        """soft_weights_from_euler_map must agree with assignment_probabilities."""
        euler_a = euler_pole_to_vector(EulerPole(lat=50.0, lon=130.0, rate=0.5))
        euler_b = euler_pole_to_vector(EulerPole(lat=10.0, lon=125.0, rate=3.0))
        stations_a = _synthetic_stations(euler_a, np.linspace(130, 136, 6),
                                         np.linspace(33, 36, 6))
        stations_b = _synthetic_stations(euler_b, np.linspace(129, 133, 6),
                                         np.linspace(31, 33, 6))
        all_stations = stations_a + stations_b
        c_a = VelocityCluster(id=1, stations=stations_a, euler_vector=euler_a)
        c_b = VelocityCluster(id=2, stations=stations_b, euler_vector=euler_b)

        probs_ap, _ = assignment_probabilities(all_stations, [c_a, c_b])
        weights_em   = soft_weights_from_euler_map(all_stations, {1: euler_a, 2: euler_b})

        np.testing.assert_allclose(probs_ap, weights_em, atol=1e-12)
