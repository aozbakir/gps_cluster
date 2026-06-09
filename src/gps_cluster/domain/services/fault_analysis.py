"""Domain services for fault geometry and kinematics.

Functions here operate on pure domain objects (VelocityCluster, GpsStation)
and have no dependency on plotting libraries or I/O.
"""

from __future__ import annotations

import numpy as np

from gps_cluster.domain.entities import VelocityCluster


def assign_sides(
    clusters: list[VelocityCluster],
    flon: float,
    flat: float,
    strike_deg: float,
    search_radius_deg: float = 2.5,
) -> tuple[VelocityCluster, VelocityCluster]:
    """Determine which cluster lies on the right vs left side of a fault point.

    Uses a station-vote approach: each station within *search_radius_deg* of
    the fault point casts a signed vote proportional to its perpendicular
    distance from the fault trace.  The cluster accumulating the most negative
    perpendicular sum is on the right; the most positive is on the left.

    Falls back to cluster centroids when fewer than 4 nearby stations exist.

    Parameters
    ----------
    clusters : list[VelocityCluster]
        All clusters in the current solution.
    flon, flat : float
        Longitude and latitude of the fault sample point (degrees).
    strike_deg : float
        Local fault strike, measured clockwise from north (degrees).
        Must be in [0, 180).
    search_radius_deg : float
        Radius (degrees, ~km on the surface) within which to count votes.

    Returns
    -------
    (right_cluster, left_cluster)
        The cluster on the right side of the fault (positive strike direction)
        and the cluster on the left side.
    """
    strike_rad = np.deg2rad(strike_deg)
    cos_s = np.cos(strike_rad)
    sin_s = np.sin(strike_rad)

    # Flatten all stations across clusters for vectorised distance computation
    all_stations = [(s, c.id) for c in clusters for s in c.stations]
    sta_pos = np.array([[s.position.lon, s.position.lat] for s, _ in all_stations])
    sta_cid = np.array([cid for _, cid in all_stations])

    dists = np.hypot(sta_pos[:, 0] - flon, sta_pos[:, 1] - flat)
    nearby_idx = np.where(dists < search_radius_deg)[0]

    votes: dict[int, float] = {c.id: 0.0 for c in clusters}

    if len(nearby_idx) >= 4:
        for i in nearby_idx:
            dx = sta_pos[i, 0] - flon
            dy = sta_pos[i, 1] - flat
            perp = dx * (-cos_s) + dy * sin_s   # positive = left side
            votes[sta_cid[i]] += perp
    else:
        # Fallback: use cluster centroids
        for c in clusters:
            clon = float(np.mean([s.position.lon for s in c.stations]))
            clat = float(np.mean([s.position.lat for s in c.stations]))
            votes[c.id] += (clon - flon) * (-cos_s) + (clat - flat) * sin_s

    c_by_id = {c.id: c for c in clusters}
    sorted_votes = sorted(votes.items(), key=lambda x: x[1])
    right_cluster = c_by_id[sorted_votes[0][0]]   # most negative perp → right
    left_cluster  = c_by_id[sorted_votes[-1][0]]  # most positive perp → left
    return right_cluster, left_cluster


def fault_strike_from_geom(
    xs: list[float],
    ys: list[float],
    idx: int,
    half_win: int = 5,
) -> float:
    """Smoothed local fault strike at vertex *idx*.

    Uses a ±half_win vertex window to suppress jagged short-segment noise.
    Always returns the strike in [0, 180) so that the perpendicular-distance
    sign convention is consistent regardless of vertex ordering.

    Parameters
    ----------
    xs, ys : sequence of float
        Longitude and latitude coordinates of the fault polyline vertices.
    idx : int
        Vertex index at which to evaluate the local strike.
    half_win : int
        Half-width of the smoothing window in vertex count.

    Returns
    -------
    float
        Strike in degrees, clockwise from north, in [0, 180).
    """
    n = len(xs)
    i0 = max(0, idx - half_win)
    i1 = min(n - 1, idx + half_win)
    dx = xs[i1] - xs[i0]
    dy = ys[i1] - ys[i0]
    az = float(np.degrees(np.arctan2(dx, dy)) % 360)
    if az >= 180.0:
        az -= 180.0
    return az


def compute_rake(ss: float, fn: float) -> float:
    """Geological rake angle in degrees from strike-slip and fault-normal rates.

    Convention
    ----------
    ss > 0, fn = 0  → dextral      (+180°)
    ss < 0, fn = 0  → sinistral    (  0°)
    ss = 0, fn < 0  → thrust       ( +90°)
    ss = 0, fn > 0  → normal       ( -90°)

    Parameters
    ----------
    ss : float
        Strike-slip rate (mm/yr); positive = right-lateral.
    fn : float
        Fault-normal rate (mm/yr); positive = opening.

    Returns
    -------
    float
        Rake in degrees, in (-180, 180].
    """
    raw = float(np.degrees(np.arctan2(fn, ss))) + 180.0
    if raw > 180.0:
        raw -= 360.0
    return raw
