"""Shared pytest fixtures for gps_cluster tests."""

from __future__ import annotations

import numpy as np
import pytest

from gps_cluster.domain.entities import EulerPole, EulerVector, GpsStation, Position, Velocity
from gps_cluster.domain.services.euler_math import euler_pole_to_vector, predict_velocity


def _make_stations(
    euler: EulerVector,
    lons: list[float],
    lats: list[float],
    sigma: float = 1.0,
) -> list[GpsStation]:
    """Generate noiseless synthetic stations consistent with a given Euler vector."""
    stations = []
    for lon, lat in zip(lons, lats):
        dummy = GpsStation("x", Position(lon, lat), Velocity(0, 0, 0, sigma, sigma, 1.0))
        ve, vn = predict_velocity(dummy, euler)
        stations.append(
            GpsStation("x", Position(lon, lat), Velocity(ve, vn, 0.0, sigma, sigma, 1.0))
        )
    return stations


@pytest.fixture
def synthetic_cluster_a():
    """Euler pole A — northeast of the map, slow rotation."""
    pole = EulerPole(lat=50.0, lon=130.0, rate=0.5)
    euler = euler_pole_to_vector(pole)
    lons = list(np.linspace(130, 136, 8))
    lats = list(np.linspace(33, 36, 8))
    return euler, _make_stations(euler, lons, lats)


@pytest.fixture
def synthetic_cluster_b():
    """Euler pole B — far south, fast rotation (Ryukyu-like)."""
    pole = EulerPole(lat=10.0, lon=125.0, rate=3.0)
    euler = euler_pole_to_vector(pole)
    lons = list(np.linspace(129, 133, 8))
    lats = list(np.linspace(31, 33, 8))
    return euler, _make_stations(euler, lons, lats)


@pytest.fixture
def two_cluster_stations(synthetic_cluster_a, synthetic_cluster_b):
    """Combined station list from two distinct synthetic Euler poles."""
    _, stations_a = synthetic_cluster_a
    _, stations_b = synthetic_cluster_b
    return stations_a + stations_b
