"""Euler vector math for GPS velocity analysis.

Key identity used throughout:
    Ve = (Omega x r) . e = Omega . (r x e) = Omega . n
    Vn = (Omega x r) . n = Omega . (r x n) = Omega . (-e)

where r is the unit position vector, e is local east, n is local north.
Proof: scalar triple product a.(b x c) = b.(c x a) = det[a,b,c]
    => (Omega x r).e = Omega.(r x e)

The design matrix rows are therefore:
    G_east  = r x e = n   (verified analytically)
    G_north = r x n = -e  (verified analytically)

Omega units: mm/yr (same as velocity), since G entries are dimensionless.
"""

from __future__ import annotations

import numpy as np

from gps_cluster.domain.entities import EulerPole, EulerVector, GpsStation

# Earth radius in mm (6371 km * 1e6 mm/km)
_EARTH_RADIUS_MM = 6_371_000.0 * 1_000.0


def _unit_position(lat_deg: float, lon_deg: float) -> np.ndarray:
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    return np.array([
        np.cos(phi) * np.cos(lam),
        np.cos(phi) * np.sin(lam),
        np.sin(phi),
    ])


def _local_east(lon_deg: float) -> np.ndarray:
    lam = np.radians(lon_deg)
    return np.array([-np.sin(lam), np.cos(lam), 0.0])


def _local_north(lat_deg: float, lon_deg: float) -> np.ndarray:
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    return np.array([
        -np.sin(phi) * np.cos(lam),
        -np.sin(phi) * np.sin(lam),
        np.cos(phi),
    ])


def design_matrix(stations: list[GpsStation]) -> np.ndarray:
    """Build 2N x 3 design matrix for N stations.

    Row order: [G_east_1, G_north_1, G_east_2, G_north_2, ...]
    G_east  = r x e = n
    G_north = r x n = -e
    """
    rows = []
    for s in stations:
        r = _unit_position(s.position.lat, s.position.lon)
        e = _local_east(s.position.lon)
        n = _local_north(s.position.lat, s.position.lon)
        rows.append(np.cross(r, e))  # == n
        rows.append(np.cross(r, n))  # == -e
    return np.vstack(rows)


def _obs_and_weights(
    stations: list[GpsStation],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (d, W) where d is the 2N observation vector and W the diagonal weight matrix."""
    d = np.array([v for s in stations for v in (s.velocity.ve, s.velocity.vn)])
    sigmas = np.array([v for s in stations for v in (s.velocity.se, s.velocity.sn)])
    W = np.diag(1.0 / sigmas**2)
    return d, W


def invert_euler_vector(stations: list[GpsStation]) -> EulerVector:
    """Weighted least-squares Euler vector inversion.

    Solves: Omega = (G^T W G)^{-1} G^T W d

    Raises ValueError if fewer than 3 stations are provided (system would be
    rank-deficient: 3 unknowns require at least 2 stations for 4 equations,
    but 3 stations give a robustly over-determined system).
    """
    if len(stations) < 2:
        raise ValueError(f"Need >= 2 stations to invert, got {len(stations)}")

    G = design_matrix(stations)
    d, W = _obs_and_weights(stations)

    GtW = G.T @ W
    try:
        omega = np.linalg.solve(GtW @ G, GtW @ d)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Design matrix is singular; check station geometry") from exc

    return EulerVector(ox=float(omega[0]), oy=float(omega[1]), oz=float(omega[2]))


def predict_velocity(station: GpsStation, euler: EulerVector) -> tuple[float, float]:
    """Predicted (Ve, Vn) at station from an Euler vector."""
    r = _unit_position(station.position.lat, station.position.lon)
    e = _local_east(station.position.lon)
    n = _local_north(station.position.lat, station.position.lon)
    omega = euler.to_array()
    ve = float(np.dot(omega, np.cross(r, e)))  # Omega . n
    vn = float(np.dot(omega, np.cross(r, n)))  # Omega . (-e)
    return ve, vn


def weighted_residual_sq(station: GpsStation, euler: EulerVector) -> float:
    """Weighted squared velocity residual: ((Ve_pred-Ve)/Se)^2 + ((Vn_pred-Vn)/Sn)^2."""
    ve_pred, vn_pred = predict_velocity(station, euler)
    re = (ve_pred - station.velocity.ve) / station.velocity.se
    rn = (vn_pred - station.velocity.vn) / station.velocity.sn
    return re**2 + rn**2


def total_chi_squared(stations: list[GpsStation], euler: EulerVector) -> float:
    """Sum of weighted squared residuals over a cluster."""
    return sum(weighted_residual_sq(s, euler) for s in stations)


def reduced_chi_squared(stations: list[GpsStation], euler: EulerVector) -> float:
    """chi^2 / dof, where dof = 2*N - 3 (3 Euler vector parameters)."""
    n = len(stations)
    if n < 3:
        return np.inf
    chi2 = total_chi_squared(stations, euler)
    return chi2 / (2 * n - 3)


def euler_vector_to_pole(euler: EulerVector) -> EulerPole:
    """Convert Cartesian Omega to geographic Euler pole (lat, lon, rate in deg/Myr)."""
    ox, oy, oz = euler.ox, euler.oy, euler.oz
    magnitude = np.sqrt(ox**2 + oy**2 + oz**2)
    if magnitude == 0.0:
        return EulerPole(lat=0.0, lon=0.0, rate=0.0)
    lon = float(np.degrees(np.arctan2(oy, ox)))
    lat = float(np.degrees(np.arcsin(np.clip(oz / magnitude, -1.0, 1.0))))
    # magnitude [mm/yr] / R_earth [mm] = angular rate [rad/yr]; convert to deg/Myr
    rate = float(np.degrees(magnitude / _EARTH_RADIUS_MM) * 1e6)
    return EulerPole(lat=lat, lon=lon, rate=rate)


def euler_pole_to_vector(pole: EulerPole) -> EulerVector:
    """Convert geographic Euler pole to Cartesian Omega vector."""
    omega_mag = np.radians(pole.rate / 1e6) * _EARTH_RADIUS_MM  # mm/yr
    phi = np.radians(pole.lat)
    lam = np.radians(pole.lon)
    return EulerVector(
        ox=float(omega_mag * np.cos(phi) * np.cos(lam)),
        oy=float(omega_mag * np.cos(phi) * np.sin(lam)),
        oz=float(omega_mag * np.sin(phi)),
    )


def euler_angular_distance(a: EulerVector, b: EulerVector) -> float:
    """Angular separation (radians) between two rotation axes."""
    va, vb = a.to_array(), b.to_array()
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_angle = np.clip(np.dot(va, vb) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.arccos(cos_angle))
