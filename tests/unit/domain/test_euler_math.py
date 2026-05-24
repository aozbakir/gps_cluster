"""Unit tests for domain/services/euler_math.py."""

from __future__ import annotations

import numpy as np
import pytest

from gps_cluster.domain.entities import EulerPole, EulerVector, GpsStation, Position, Velocity
from gps_cluster.domain.services.euler_math import (
    design_matrix,
    euler_angular_distance,
    euler_pole_to_vector,
    euler_vector_to_pole,
    invert_euler_vector,
    predict_velocity,
    weighted_residual_sq,
)


def _station(lon: float, lat: float, ve: float = 0.0, vn: float = 0.0) -> GpsStation:
    return GpsStation(
        name="T",
        position=Position(lon=lon, lat=lat),
        velocity=Velocity(ve=ve, vn=vn, vu=0.0, se=1.0, sn=1.0, su=1.0),
    )


class TestDesignMatrix:
    def test_shape(self):
        stations = [_station(0, 0), _station(90, 0), _station(0, 45)]
        G = design_matrix(stations)
        assert G.shape == (6, 3)

    def test_north_pole_rotation_equator(self):
        """Rotation about North Pole → eastward velocity at equator."""
        omega = np.array([0.0, 0.0, 10.0])  # rotation about z-axis
        s = _station(lon=0, lat=0)
        G = design_matrix([s])
        v = G @ omega
        # ve should be ~10, vn should be ~0
        assert pytest.approx(v[0], abs=1e-9) == 10.0
        assert pytest.approx(v[1], abs=1e-9) == 0.0

    def test_station_at_euler_pole_has_zero_velocity(self):
        """Station exactly at the Euler pole location → zero velocity (r ∥ Ω)."""
        from gps_cluster.domain.entities import EulerPole
        from gps_cluster.domain.services.euler_math import euler_pole_to_vector
        pole = EulerPole(lat=30.0, lon=45.0, rate=2.0)
        euler = euler_pole_to_vector(pole)
        # r is parallel to Ω, so v = Ω × r = 0
        s = _station(lon=45, lat=30)
        G = design_matrix([s])
        v = G @ euler.to_array()
        assert v[0] == pytest.approx(0.0, abs=1e-6)
        assert v[1] == pytest.approx(0.0, abs=1e-6)


class TestPredictVelocity:
    def test_zero_euler_vector(self):
        s = _station(lon=10, lat=20)
        ve, vn = predict_velocity(s, EulerVector(0, 0, 0))
        assert ve == pytest.approx(0.0, abs=1e-12)
        assert vn == pytest.approx(0.0, abs=1e-12)

    def test_north_pole_rotation(self):
        """Rotation about z-axis: eastward speed = omega * cos(lat)."""
        omega_z = 10.0  # mm/yr on unit sphere
        euler = EulerVector(0.0, 0.0, omega_z)
        for lat in [0, 30, 60]:
            s = _station(lon=0, lat=lat)
            ve, vn = predict_velocity(s, euler)
            expected_ve = omega_z * np.cos(np.radians(lat))
            assert ve == pytest.approx(expected_ve, rel=1e-6), f"lat={lat}"
            assert vn == pytest.approx(0.0, abs=1e-9), f"lat={lat}"

    def test_consistency_with_design_matrix(self):
        """predict_velocity and design_matrix @ omega must agree."""
        euler = EulerVector(3.0, -1.0, 7.0)
        s = _station(lon=45.0, lat=35.0)
        ve_pred, vn_pred = predict_velocity(s, euler)
        G = design_matrix([s])
        v = G @ euler.to_array()
        assert ve_pred == pytest.approx(v[0], rel=1e-10)
        assert vn_pred == pytest.approx(v[1], rel=1e-10)


class TestInvertEulerVector:
    def test_recovers_true_euler_vector(self):
        """Noiseless synthetic data → inversion must recover the true Euler vector."""
        from gps_cluster.domain.entities import EulerPole
        true_pole = EulerPole(lat=45.0, lon=10.0, rate=2.0)
        true_euler = euler_pole_to_vector(true_pole)

        lons = np.linspace(-125, -118, 15)
        lats = np.linspace(36, 40, 15)
        stations = []
        for lon, lat in zip(lons, lats):
            s_dummy = GpsStation("x", Position(lon, lat), Velocity(0, 0, 0, 1, 1, 1))
            ve, vn = predict_velocity(s_dummy, true_euler)
            stations.append(GpsStation("x", Position(lon, lat), Velocity(ve, vn, 0, 1, 1, 1)))

        recovered = invert_euler_vector(stations)

        assert recovered.ox == pytest.approx(true_euler.ox, rel=1e-5)
        assert recovered.oy == pytest.approx(true_euler.oy, rel=1e-5)
        assert recovered.oz == pytest.approx(true_euler.oz, rel=1e-5)

    def test_raises_for_single_station(self):
        with pytest.raises(ValueError, match=">="):
            invert_euler_vector([_station(0, 0)])

    def test_zero_residual_for_exact_data(self):
        """Inverted Euler vector on noiseless data must give zero residuals."""
        from gps_cluster.domain.entities import EulerPole
        true_euler = euler_pole_to_vector(EulerPole(lat=30.0, lon=-100.0, rate=1.5))
        stations = []
        for lon, lat in zip(np.linspace(-110, -100, 8), np.linspace(28, 38, 8)):
            s_dummy = GpsStation("x", Position(lon, lat), Velocity(0, 0, 0, 1, 1, 1))
            ve, vn = predict_velocity(s_dummy, true_euler)
            stations.append(GpsStation("x", Position(lon, lat), Velocity(ve, vn, 0, 1, 1, 1)))

        recovered = invert_euler_vector(stations)
        for s in stations:
            assert weighted_residual_sq(s, recovered) == pytest.approx(0.0, abs=1e-8)


class TestEulerPoleConversions:
    def test_pole_to_vector_and_back(self):
        pole = EulerPole(lat=45.0, lon=30.0, rate=2.5)
        vec = euler_pole_to_vector(pole)
        recovered = euler_vector_to_pole(vec)
        assert recovered.lat == pytest.approx(pole.lat, abs=1e-4)
        assert recovered.lon == pytest.approx(pole.lon, abs=1e-4)
        assert recovered.rate == pytest.approx(pole.rate, rel=1e-4)

    def test_north_pole(self):
        pole = EulerPole(lat=90.0, lon=0.0, rate=1.0)
        vec = euler_pole_to_vector(pole)
        assert vec.ox == pytest.approx(0.0, abs=1e-10)
        assert vec.oy == pytest.approx(0.0, abs=1e-10)
        assert vec.oz > 0

    def test_zero_rate(self):
        pole = EulerPole(lat=45.0, lon=45.0, rate=0.0)
        vec = euler_pole_to_vector(pole)
        recovered = euler_vector_to_pole(vec)
        assert recovered.rate == pytest.approx(0.0, abs=1e-10)


class TestEulerAngularDistance:
    def test_same_vector_is_zero(self):
        v = EulerVector(1.0, 2.0, 3.0)
        assert euler_angular_distance(v, v) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_vectors(self):
        a = EulerVector(1.0, 0.0, 0.0)
        b = EulerVector(0.0, 1.0, 0.0)
        assert euler_angular_distance(a, b) == pytest.approx(np.pi / 2, rel=1e-9)

    def test_antiparallel_vectors(self):
        a = EulerVector(1.0, 0.0, 0.0)
        b = EulerVector(-1.0, 0.0, 0.0)
        assert euler_angular_distance(a, b) == pytest.approx(np.pi, rel=1e-9)
