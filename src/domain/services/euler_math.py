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


def invert_euler_vector_weighted(
    stations: list[GpsStation],
    weights: "np.ndarray",
) -> EulerVector:
    """Soft-weighted least-squares Euler vector inversion.

    Each station i contributes to the normal equations with weight ``weights[i]``
    on top of the measurement precision weights ``1/sigma²``.  This is the M-step
    of the EM algorithm: stations near block boundaries (low ``weights[i]``) pull
    less on the Euler vector estimate.

    Parameters
    ----------
    stations:
        Full list of GPS stations (length N).
    weights:
        1-D array of shape (N,) with non-negative soft-assignment probabilities
        for this cluster.  Need not sum to 1.

    Returns
    -------
    EulerVector with covariance attached.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.sum() < 1e-12:
        return EulerVector(0.0, 0.0, 0.0)

    G = design_matrix(stations)               # (2N, 3)
    d = np.array([v for s in stations for v in (s.velocity.ve, s.velocity.vn)])
    sigmas = np.array([v for s in stations for v in (s.velocity.se, s.velocity.sn)])

    # Expand per-station weights to per-observation (east, north pairs)
    w_obs = np.repeat(weights, 2) / sigmas**2  # shape (2N,)

    W = np.diag(w_obs)
    GtW = G.T @ W
    GtWG = GtW @ G
    try:
        omega = np.linalg.solve(GtWG, GtW @ d)
    except np.linalg.LinAlgError:
        return EulerVector(0.0, 0.0, 0.0)

    # Covariance: (G^T W G)^{-1} — same formula, now W carries soft weights
    try:
        C = np.linalg.inv(GtWG)
    except np.linalg.LinAlgError:
        C = None

    return EulerVector(ox=float(omega[0]), oy=float(omega[1]), oz=float(omega[2]),
                       covariance=C)


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

    C = np.linalg.inv(GtW @ G)   # 3×3 covariance in (mm/yr)²
    return EulerVector(ox=float(omega[0]), oy=float(omega[1]), oz=float(omega[2]), covariance=C)


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
    """Weighted squared velocity residual: ((Ve_pred-Ve)/Se)^2 + ((Vn_pred-Vn)/Sn)^2.

    Kept for reference / chi² bookkeeping; NOT used in Savage (2018) reassignment.
    """
    ve_pred, vn_pred = predict_velocity(station, euler)
    re = (ve_pred - station.velocity.ve) / station.velocity.se
    rn = (vn_pred - station.velocity.vn) / station.velocity.sn
    return re**2 + rn**2


def unweighted_residual_sq(station: GpsStation, euler: EulerVector) -> float:
    """Unweighted squared velocity residual: (Ve_pred-Ve)^2 + (Vn_pred-Vn)^2  [mm²/yr²].

    This is the reassignment criterion used by Savage (2018) — minimise the
    Euclidean distance in velocity space, ignoring measurement uncertainties.
    """
    ve_pred, vn_pred = predict_velocity(station, euler)
    dve = ve_pred - station.velocity.ve
    dvn = vn_pred - station.velocity.vn
    return dve * dve + dvn * dvn


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


def euler_pole_uncertainty(euler: EulerVector) -> tuple[float, float, float]:
    """Propagate Euler vector covariance to pole (lat, lon, rate) 1-sigma uncertainties.

    Uses first-order error propagation: C_pole = J @ C_omega @ J^T
    where J is the 3×3 Jacobian d(lat,lon,rate)/d(ox,oy,oz).

    Returns
    -------
    (sigma_lat_deg, sigma_lon_deg, sigma_rate_deg_myr)
    All in degrees / degrees / deg per Myr.
    Returns (0,0,0) if covariance is not set.
    """
    if euler.covariance is None:
        return (0.0, 0.0, 0.0)

    ox, oy, oz = euler.ox, euler.oy, euler.oz
    magnitude = np.sqrt(ox**2 + oy**2 + oz**2)
    if magnitude == 0.0:
        return (0.0, 0.0, 0.0)

    r_xy = np.sqrt(ox**2 + oy**2)
    if r_xy == 0.0:
        return (0.0, 0.0, 0.0)   # pole at geographic pole — degenerate

    # Jacobian rows: d(lat)/d(omega), d(lon)/d(omega), d(rate)/d(omega)
    # lat = arcsin(oz / magnitude)
    dlat_dox = -ox * oz / (magnitude**2 * r_xy)
    dlat_doy = -oy * oz / (magnitude**2 * r_xy)
    dlat_doz =  r_xy / magnitude**2
    # lon = atan2(oy, ox)
    dlon_dox = -oy / r_xy**2
    dlon_doy =  ox / r_xy**2
    dlon_doz =  0.0
    # rate = degrees(magnitude / R) * 1e6  => d(rate)/d(omega) in (deg/Myr)/(mm/yr)
    drate_dox = ox / (magnitude * _EARTH_RADIUS_MM)
    drate_doy = oy / (magnitude * _EARTH_RADIUS_MM)
    drate_doz = oz / (magnitude * _EARTH_RADIUS_MM)

    J = np.array([
        [dlat_dox, dlat_doy, dlat_doz],
        [dlon_dox, dlon_doy, dlon_doz],
        [drate_dox, drate_doy, drate_doz],
    ])  # shape (3, 3)

    C_pole = J @ euler.covariance @ J.T   # (3, 3)

    # lat and lon Jacobian rows give result in radians; convert to degrees
    rad_to_deg = np.degrees(1.0)
    yr_to_myr = 1e6

    sigma_lat  = rad_to_deg * np.sqrt(max(C_pole[0, 0], 0.0))
    sigma_lon  = rad_to_deg * np.sqrt(max(C_pole[1, 1], 0.0))
    sigma_rate = rad_to_deg * yr_to_myr * np.sqrt(max(C_pole[2, 2], 0.0))

    return (sigma_lat, sigma_lon, sigma_rate)


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
    sigma_lat, sigma_lon, sigma_rate = euler_pole_uncertainty(euler)
    return EulerPole(lat=lat, lon=lon, rate=rate,
                     sigma_lat=sigma_lat, sigma_lon=sigma_lon, sigma_rate=sigma_rate)


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


def fault_slip_rate(
    euler_a: EulerVector,
    euler_b: EulerVector,
    fault_lats: list[float],
    fault_lons: list[float],
    fault_strike_deg: float | None = None,
) -> list[dict]:
    """Compute fault slip rate at points along a fault from two adjacent block Euler vectors.

    The relative angular velocity is omega_rel = omega_b - omega_a, giving the
    velocity of block B relative to block A at each fault point.

    Parameters
    ----------
    euler_a, euler_b:
        Euler vectors of the two blocks on either side of the fault.
    fault_lats, fault_lons:
        Sampling points along the fault trace.
    fault_strike_deg:
        Along-fault azimuth (degrees CW from N). If provided, decompose the
        relative velocity into fault-parallel (strike-slip) and fault-normal
        (opening/convergence) components. If None, only total rate is returned.

    Returns
    -------
    List of dicts, one per fault point, with keys:
        lat, lon, ve_rel, vn_rel, total_mm_yr,
        strike_slip_mm_yr (+ = right-lateral),
        fault_normal_mm_yr (+ = opening)
    """
    omega_rel = EulerVector(
        ox=euler_b.ox - euler_a.ox,
        oy=euler_b.oy - euler_a.oy,
        oz=euler_b.oz - euler_a.oz,
    )

    results = []
    for lat, lon in zip(fault_lats, fault_lons):
        r = _unit_position(lat, lon)
        e = _local_east(lon)
        n = _local_north(lat, lon)
        om = omega_rel.to_array()

        ve = float(np.dot(om, np.cross(r, e)))
        vn = float(np.dot(om, np.cross(r, n)))
        total = float(np.sqrt(ve**2 + vn**2))

        entry: dict = dict(lat=lat, lon=lon, ve_rel=ve, vn_rel=vn,
                           total_mm_yr=total)

        if fault_strike_deg is not None:
            # Rotate velocity into fault-parallel / fault-normal frame
            # strike = azimuth of fault (CW from N)
            strike_rad = np.radians(fault_strike_deg)
            # fault-parallel unit vector (along strike, pointing in strike direction)
            fp_e =  np.sin(strike_rad)
            fp_n =  np.cos(strike_rad)
            # fault-normal unit vector (90° CCW from strike = left of travel)
            fn_e = -np.cos(strike_rad)
            fn_n =  np.sin(strike_rad)

            strike_slip   = ve * fp_e + vn * fp_n   # +ve = right-lateral
            fault_normal  = ve * fn_e + vn * fn_n   # +ve = opening

            entry["strike_slip_mm_yr"]  = float(strike_slip)
            entry["fault_normal_mm_yr"] = float(fault_normal)

        results.append(entry)

    return results


def fault_slip_rate_uncertainty(
    euler_a: EulerVector,
    euler_b: EulerVector,
    lat: float,
    lon: float,
    fault_strike_deg: float,
) -> dict:
    """1-sigma uncertainty on strike-slip and fault-normal rates at a fault point.

    Assumes the two block Euler vectors are independent (uncorrelated), so
    C_rel = C_a + C_b.  Propagates through the linear velocity decomposition.

    Returns
    -------
    dict with keys:
        sigma_strike_slip_mm_yr, sigma_fault_normal_mm_yr, sigma_total_mm_yr
    Returns zeros if neither euler has a covariance set.
    """
    if euler_a.covariance is None and euler_b.covariance is None:
        return dict(sigma_strike_slip_mm_yr=0.0,
                    sigma_fault_normal_mm_yr=0.0,
                    sigma_total_mm_yr=0.0)

    C_a = euler_a.covariance if euler_a.covariance is not None else np.zeros((3, 3))
    C_b = euler_b.covariance if euler_b.covariance is not None else np.zeros((3, 3))
    C_rel = C_a + C_b

    r = _unit_position(lat, lon)
    e = _local_east(lon)
    n = _local_north(lat, lon)

    # Design matrix rows for this point: G_e = r×e, G_n = r×n  (both 3-vectors)
    G_e = np.cross(r, e)   # ∂ve_rel/∂omega_rel
    G_n = np.cross(r, n)   # ∂vn_rel/∂omega_rel

    strike_rad = np.radians(fault_strike_deg)
    fp_e =  np.sin(strike_rad)
    fp_n =  np.cos(strike_rad)
    fn_e = -np.cos(strike_rad)
    fn_n =  np.sin(strike_rad)

    # Sensitivity of ss and fn to omega_rel (1×3 row vectors)
    J_ss = fp_e * G_e + fp_n * G_n   # ∂ss/∂omega_rel
    J_fn = fn_e * G_e + fn_n * G_n   # ∂fn/∂omega_rel

    # Total rate direction sensitivity
    omega_rel = np.array([euler_b.ox - euler_a.ox,
                           euler_b.oy - euler_a.oy,
                           euler_b.oz - euler_a.oz])
    ve_rel = float(np.dot(omega_rel, G_e))
    vn_rel = float(np.dot(omega_rel, G_n))
    total  = np.sqrt(ve_rel**2 + vn_rel**2)
    if total > 0:
        J_tot = (ve_rel * G_e + vn_rel * G_n) / total
    else:
        J_tot = np.zeros(3)

    sigma_ss  = np.sqrt(max(float(J_ss @ C_rel @ J_ss), 0.0))
    sigma_fn  = np.sqrt(max(float(J_fn @ C_rel @ J_fn), 0.0))
    sigma_tot = np.sqrt(max(float(J_tot @ C_rel @ J_tot), 0.0))

    return dict(sigma_strike_slip_mm_yr=sigma_ss,
                sigma_fault_normal_mm_yr=sigma_fn,
                sigma_total_mm_yr=sigma_tot)


def assignment_probabilities(
    stations: list[GpsStation],
    clusters: list,            # list of VelocityCluster (or any object with .euler_vector)
) -> tuple[np.ndarray, np.ndarray]:
    """Soft cluster membership probabilities for each station.

    Uses the chi² residual under each cluster's Euler vector as a
    log-likelihood:

        log P(station s → cluster j) ∝ -χ²_j(s) / 2

    Converted to probabilities via softmax:

        P(s, j) = exp(-χ²_j / 2) / Σ_k exp(-χ²_k / 2)

    This is the Bayesian posterior under equal priors and Gaussian
    measurement errors.  High-entropy stations (H = -Σ p·log p ≈ log k)
    are kinematically ambiguous — they lie near block boundaries or carry
    elastic strain signals from locked faults.

    Parameters
    ----------
    stations : list of GpsStation
    clusters : list of objects with .euler_vector (EulerVector | None)

    Returns
    -------
    probs : ndarray shape (N, k)
        Probability matrix; probs[i, j] = P(station i → cluster j).
    entropy : ndarray shape (N,)
        Shannon entropy per station in nats.  Max = log(k) (fully uncertain).
    """
    valid = [c for c in clusters if c.euler_vector is not None]
    N  = len(stations)
    k  = len(valid)
    chi2_mat = np.zeros((N, k))

    for j, c in enumerate(valid):
        for i, s in enumerate(stations):
            chi2_mat[i, j] = weighted_residual_sq(s, c.euler_vector)

    # Softmax over -chi²/2 (subtract row max for numerical stability)
    log_p = -0.5 * chi2_mat
    log_p -= log_p.max(axis=1, keepdims=True)
    p = np.exp(log_p)
    p /= p.sum(axis=1, keepdims=True)

    # Shannon entropy per station (nats)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(p * np.where(p > 0, np.log(p), 0.0), axis=1)

    return p, ent


def soft_weights_from_euler_map(
    stations: list[GpsStation],
    euler_map: dict,      # {cluster_id: EulerVector}
) -> np.ndarray:
    """Compute soft assignment weight matrix from an euler_map dict.

    Returns
    -------
    weights : ndarray shape (N, k)
        weights[i, j] = P(station i → cluster j), via softmax over -χ²/2.
    """
    cluster_ids = sorted(euler_map)
    N = len(stations)
    k = len(cluster_ids)
    chi2_mat = np.zeros((N, k))
    for j, cid in enumerate(cluster_ids):
        ev = euler_map[cid]
        for i, s in enumerate(stations):
            chi2_mat[i, j] = weighted_residual_sq(s, ev)

    log_p = -0.5 * chi2_mat
    log_p -= log_p.max(axis=1, keepdims=True)
    p = np.exp(log_p)
    p /= p.sum(axis=1, keepdims=True)
    return p   # (N, k)


def distance_soft_weights(
    stations: list[GpsStation],
    euler_map: dict,
    gamma: float = 4e-6,
    pi: "np.ndarray | None" = None,
    tol: float = 1e-4,
    max_iter: int = 50,
    chi2_scale: float | None = None,
) -> np.ndarray:
    """Variational Bayes E-step with distance-to-centroid spatial prior.

    Posterior over cluster assignments:

        log w[i, k] = -chi²[i,k] / (2·s) + log(π[k]) - γ · d²(i, x̄_k)

    where s = chi2_scale (the within-cluster reduced chi² from the previous
    M-step), x̄_k is the soft-weighted geographic centroid of cluster k in km,
    and γ (nats/km²) is the spatial penalty strength.

    When chi2_scale is provided (recommended for χ²_red >> 1 datasets), chi²
    is divided by chi2_scale before computing weights.  This makes the
    likelihood contribution per station ≈ 1 nat regardless of absolute sigma
    magnitude, so the spatial prior γ·d² always competes on equal footing.
    Without scaling, datasets with very small measurement uncertainties
    (χ²_red >> 1) collapse to hard assignment because exp(−chi²/2) ≈ 0.

    Physical interpretation
    -----------------------
    γ = 1/R₀² where R₀ is the characteristic block radius.
    Default R₀ = 500 km → γ = 4×10⁻⁶ nats/km²:

    - Station 200 km from centroid:  penalty = 0.16 nats  (negligible)
    - Station 500 km from centroid:  penalty = 1.00 nats  (moderate)
    - Station 900 km from centroid:  penalty = 3.24 nats  (strong)

    Unlike the Potts model, this prior does *not* penalise geographic
    neighbours that correctly belong to different clusters (i.e. stations
    on opposite sides of a fault).  It only penalises teleportation: a
    station joining a cluster whose centroid is far away.

    The centroid and weights are coupled (centroid depends on weights and
    vice versa), so they are iterated to a fixed point with the chi²
    matrix held fixed.  Given fixed centroids the per-station weight
    updates are independent, so no inner coupling loop is required —
    convergence is guaranteed.

    Parameters
    ----------
    stations:
        All GPS stations (length N).
    euler_map:
        Dict mapping cluster id → EulerVector.  Iterated in sorted key
        order so column j always refers to the same cluster.
    gamma:
        Spatial penalty (nats/km²).  Default 4×10⁻⁶ = 1/(500 km)².
        Set gamma=0 to recover pure kinematic EM (no spatial prior).
    pi:
        Mixing proportions shape (K,).  None → uniform.
    tol:
        Convergence tolerance on max element-wise weight change.
    max_iter:
        Maximum iterations for the centroid ↔ weight fixed point.

    Returns
    -------
    ndarray shape (N, K) — posterior assignment probabilities.  Rows
    sum to 1.
    """
    N = len(stations)
    keys = sorted(euler_map.keys())
    K = len(keys)

    # ── Station positions ────────────────────────────────────────────────────
    st_lats = np.array([s.position.lat for s in stations])   # (N,)
    st_lons = np.array([s.position.lon for s in stations])   # (N,)

    # ── Vectorised chi² matrix  (N, K) ──────────────────────────────────────
    G      = design_matrix(stations)                          # (2N, 3)
    v_obs  = np.array([v for s in stations
                       for v in (s.velocity.ve, s.velocity.vn)])  # (2N,)
    sigmas = np.array([v for s in stations
                       for v in (s.velocity.se, s.velocity.sn)])  # (2N,)
    omegas = np.array([euler_map[k].to_array() for k in keys])    # (K, 3)
    V_pred = G @ omegas.T                                          # (2N, K)
    resid  = (v_obs[:, np.newaxis] - V_pred) / sigmas[:, np.newaxis]
    chi2   = resid[0::2] ** 2 + resid[1::2] ** 2              # (N, K)

    # ── Normalise chi² so that the spatial prior can compete ─────────────────
    # When chi2_scale is provided (the within-cluster chi²_red from the last
    # M-step) chi² is rescaled to O(1) per station.  Without this, datasets
    # where chi²_red >> 1 (precise measurements, imperfect plate model) produce
    # exp(−chi²/2) ≈ 0 even for the best-fit cluster, collapsing weights to
    # hard {0, 1} and defeating the spatial prior.
    scale  = float(chi2_scale) if (chi2_scale is not None and chi2_scale > 0) else 1.0
    eff_chi2 = chi2 / scale                                    # (N, K), O(1)/station

    # ── Log mixing proportions ───────────────────────────────────────────────
    log_pi = (np.zeros(K) if pi is None
              else np.log(np.clip(pi, 1e-300, None)))

    # ── Initialise from likelihood only (gamma = 0 solution) ────────────────
    log_w = -0.5 * eff_chi2 + log_pi[np.newaxis, :]
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)

    if gamma == 0.0:
        return w

    # ── Centroid ↔ weight fixed-point loop ───────────────────────────────────
    for _ in range(max_iter):
        w_prev = w

        # Soft-weighted centroids  (K,)
        total_w   = w.sum(axis=0) + 1e-12            # (K,)
        c_lats = (w.T @ st_lats) / total_w           # (K,)
        c_lons = (w.T @ st_lons) / total_w           # (K,)

        # Great-circle distance² (N, K) in km²
        phi1   = np.radians(st_lats[:, np.newaxis])  # (N, 1)
        phi2   = np.radians(c_lats[np.newaxis, :])   # (1, K)
        dphi   = np.radians(c_lats[np.newaxis, :] - st_lats[:, np.newaxis])
        dlam   = np.radians(c_lons[np.newaxis, :] - st_lons[:, np.newaxis])
        a      = (np.sin(dphi / 2) ** 2
                  + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2)
        d_km   = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        d2_km2 = d_km ** 2                            # (N, K)

        # Updated weights (use scaled chi² throughout the inner loop)
        log_w = -0.5 * eff_chi2 + log_pi[np.newaxis, :] - gamma * d2_km2
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        w /= w.sum(axis=1, keepdims=True)

        if np.max(np.abs(w - w_prev)) < tol:
            break

    return w   # (N, K)


def euler_angular_distance(a: EulerVector, b: EulerVector) -> float:
    """Angular separation (radians) between two rotation axes."""
    va, vb = a.to_array(), b.to_array()
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_angle = np.clip(np.dot(va, vb) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.arccos(cos_angle))
