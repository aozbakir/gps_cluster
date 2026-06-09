"""Euler-vector clustering use case (Savage 2018).

Reference: Savage, J. C. (2018). Euler-vector clustering of GPS velocities
defines microplate geometry in southwest Japan.
Journal of Geophysical Research: Solid Earth, 123, 1437–1454.
https://doi.org/10.1002/2017JB014874

Algorithm
---------
For a given k:
1. Initial partition via velocity-space HAC (fast, well-conditioned start).
2. Invert for a best-fit Euler vector per cluster (weighted least squares).
3. Reassign each station to the cluster whose Euler vector minimises its
   weighted velocity residual.
4. Repeat 2–3 until labels converge or max_iter is reached.

Optimal k
---------
Run the algorithm for k = 1..max_k.  For each k, compute the total reduced
chi-squared:

    chi2_red(k) = sum_clusters chi2(cluster) / (2*N - 3*k)

where 2*N are the observations and 3*k the free parameters.

Use an F-test between consecutive k values to decide when adding a cluster
no longer yields a statistically significant improvement:

    F = [chi2(k) - chi2(k+1)] * dof(k+1) / chi2(k+1) / 3

which is F-distributed with (3, dof(k+1)) under the null that k clusters
are sufficient.  Reject the null at p < alpha to prefer k+1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import f as f_dist

from gps_cluster.domain.entities import EulerVector, GpsStation, VelocityCluster
from gps_cluster.domain.services.euler_math import (
    distance_soft_weights,
    invert_euler_vector,
    invert_euler_vector_weighted,
    soft_weights_from_euler_map,
    total_chi_squared,
    unweighted_residual_sq,
    weighted_residual_sq,  # kept for chi² bookkeeping; not used in Savage reassignment
)

_MIN_STATIONS_PER_CLUSTER = 2  # minimum for a determined Euler inversion


@dataclass
class FTestResult:
    k_values: np.ndarray
    chi2_total: np.ndarray
    chi2_reduced: np.ndarray
    f_statistics: np.ndarray  # length max_k - 1
    p_values: np.ndarray
    solutions: dict = field(default_factory=dict)  # k → list[VelocityCluster]

    @property
    def k_elbow(self) -> int:
        """k at which chi²_red drops the most (largest single-step improvement).

        This is the 'elbow' of the chi²_red vs k curve — a useful complement
        to the F-test optimal k when the curve is shallow and the F-test keeps
        adding clusters past the obvious knee.
        """
        if len(self.chi2_reduced) < 2:
            return int(self.k_values[0])
        improvement = -np.diff(self.chi2_reduced)   # positive = improvement
        return int(self.k_values[1:][np.argmax(improvement)])


class EulerVectorClustering:
    """Euler-vector clustering (Savage 2018).

    Parameters
    ----------
    max_iter:
        Maximum number of reassignment iterations per k.
    min_stations:
        Minimum stations required to invert a valid Euler vector for a cluster.
    init:
        Initialization strategy for the iterative algorithm.
        ``"velocity"`` (default) — velocity-space HAC; finds the global chi²
        minimum but may converge to velocity-magnitude bands in datasets where
        interseismic elastic loading dominates (e.g. subduction zones).
        ``"multiscale"`` — runs *n_restarts* random restarts plus geographic
        and velocity initializations; returns the partition with minimum total
        chi²; more expensive but more robust.
    n_restarts:
        Number of additional random restarts used when ``init="multiscale"``.
        Savage (2018) uses 3,000; default here is 100 for practical run-time.
    random_seed:
        Seed for random restarts.
    weighted_reassign:
        If True, reassign stations using the chi² criterion
        ``(dVe/se)²+(dVn/sn)²`` (weighted by measurement uncertainties).
        If False (default), use the unweighted Euclidean criterion
        ``dVe²+dVn²``, matching Savage (2018).
    """

    def __init__(
        self,
        max_iter: int = 100,
        min_stations: int = _MIN_STATIONS_PER_CLUSTER,
        init: str = "velocity",
        n_restarts: int = 100,
        random_seed: int = 0,
        weighted_reassign: bool = False,
    ) -> None:
        self.max_iter = max_iter
        self.min_stations = min_stations
        self.init = init
        self.n_restarts = n_restarts
        self.random_seed = random_seed
        self.weighted_reassign = weighted_reassign

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(
        self,
        stations: list[GpsStation],
        k: int,
        init_labels: np.ndarray | None = None,
    ) -> list[VelocityCluster]:
        """Partition stations into k clusters via Euler-vector iteration.

        Parameters
        ----------
        stations:
            List of GPS stations to cluster.
        k:
            Number of clusters.
        init_labels:
            Optional 1-indexed integer array of length ``len(stations)`` that
            provides the initial cluster assignment.  When supplied, the
            ``init`` strategy is ignored.  Useful for passing domain-specific
            starting partitions (e.g. geographic zones for subduction-zone data
            where the default velocity-HAC initialization converges to
            iso-velocity bands rather than tectonic plates).

        Returns a list of VelocityCluster objects with euler_vector set.
        """
        if k >= len(stations):
            raise ValueError(f"k={k} must be less than the number of stations ({len(stations)})")

        if init_labels is not None:
            labels = np.asarray(init_labels, dtype=int)
        elif self.init == "multiscale":
            labels = self._multiscale_init(stations, k)
        else:
            labels = self._initial_labels(stations, k)
        labels, euler_map = self._iterate(stations, labels, k)
        return self._build_clusters(stations, labels, k, euler_map=euler_map)

    def find_optimal_k(
        self,
        stations: list[GpsStation],
        max_k: int = 9,
        alpha: float = 0.05,
    ) -> tuple[int, FTestResult]:
        """Return (optimal_k, FTestResult) via sequential F-tests.

        The optimal k is the smallest k such that adding one more cluster
        does not yield a statistically significant chi-squared improvement
        at significance level alpha.
        """
        n_total = len(stations)
        chi2_vals = np.zeros(max_k)
        solutions: dict[int, list[VelocityCluster]] = {}

        for ki, k in enumerate(range(1, max_k + 1)):
            clusters = self.cluster(stations, k)
            solutions[k] = clusters
            chi2_vals[ki] = self._total_chi2(clusters)

        dof = np.array([max(2 * n_total - 3 * k, 1) for k in range(1, max_k + 1)])
        chi2_red = chi2_vals / dof

        # F-test: H0 = k clusters are sufficient, H1 = k+1 is better
        f_stats = np.zeros(max_k - 1)
        p_vals = np.zeros(max_k - 1)
        for i in range(max_k - 1):
            delta = chi2_vals[i] - chi2_vals[i + 1]
            if chi2_vals[i + 1] > 0:
                f_stats[i] = delta * dof[i + 1] / chi2_vals[i + 1] / 3
            p_vals[i] = 1.0 - f_dist.cdf(f_stats[i], dfn=3, dfd=dof[i + 1])

        # Optimal k: first k where the improvement is not significant
        optimal_k = max_k
        for i, p in enumerate(p_vals):
            if p >= alpha:
                optimal_k = i + 1  # k = i+1 is sufficient
                break

        result = FTestResult(
            k_values=np.arange(1, max_k + 1),
            chi2_total=chi2_vals,
            chi2_reduced=chi2_red,
            f_statistics=f_stats,
            p_values=p_vals,
            solutions=solutions,
        )
        return optimal_k, result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_labels(self, stations: list[GpsStation], k: int) -> np.ndarray:
        """Velocity-space HAC as the initial partition."""
        X = np.array([[s.velocity.ve, s.velocity.vn] for s in stations])
        Z = linkage(X, method="centroid", metric="euclidean")
        return fcluster(Z, t=k, criterion="maxclust")

    def _candidate_inits(self, stations: list[GpsStation], k: int) -> list[np.ndarray]:
        """Return a list of candidate initial label arrays for multiscale init."""
        from gps_cluster.domain.services.euler_math import invert_euler_vector, predict_velocity

        candidates: list[np.ndarray] = []
        n = len(stations)

        # 1. Velocity HAC (centroid)
        X_vel = np.array([[s.velocity.ve, s.velocity.vn] for s in stations])
        Z = linkage(X_vel, method="centroid")
        candidates.append(fcluster(Z, t=k, criterion="maxclust"))

        # 2. Velocity HAC (ward)
        Z = linkage(X_vel, method="ward")
        candidates.append(fcluster(Z, t=k, criterion="maxclust"))

        # 3. Geographic HAC (ward on lon/lat)
        X_geo = np.array([[s.position.lon, s.position.lat] for s in stations])
        Z = linkage(X_geo, method="ward")
        candidates.append(fcluster(Z, t=k, criterion="maxclust"))

        # 4. Residual HAC: velocity residuals relative to single best-fit Euler vector
        k1_euler = invert_euler_vector(stations)
        X_res = np.array([
            [s.velocity.ve - predict_velocity(s, k1_euler)[0],
             s.velocity.vn - predict_velocity(s, k1_euler)[1]]
            for s in stations
        ])
        Z = linkage(X_res, method="ward")
        candidates.append(fcluster(Z, t=k, criterion="maxclust"))

        # 5. Combined velocity + geographic (scaled equally)
        v_scale = max(X_vel.std(axis=0).max(), 1e-9)
        g_scale = max(X_geo.std(axis=0).max(), 1e-9)
        X_comb = np.hstack([X_vel / v_scale, X_geo / g_scale])
        Z = linkage(X_comb, method="ward")
        candidates.append(fcluster(Z, t=k, criterion="maxclust"))

        # 6. Random restarts
        rng = np.random.default_rng(self.random_seed)
        for _ in range(self.n_restarts):
            lbl = rng.integers(1, k + 1, size=n)
            candidates.append(lbl)

        return candidates

    def _multiscale_init(self, stations: list[GpsStation], k: int) -> np.ndarray:
        """Try many initializations; return the one with minimum total chi² after convergence."""
        best_labels: np.ndarray | None = None
        best_chi2 = np.inf

        for init_labels in self._candidate_inits(stations, k):
            labels, euler_map = self._iterate(stations, init_labels.copy(), k)
            clusters = self._build_clusters(stations, labels, k, euler_map)
            chi2 = self._total_chi2(clusters)
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_labels = labels.copy()

        return best_labels  # type: ignore[return-value]

    def _euler_per_cluster(
        self, stations: list[GpsStation], labels: np.ndarray, k: int
    ) -> dict[int, EulerVector]:
        """Invert one Euler vector per cluster. Falls back to zero vector for tiny clusters."""
        euler_map: dict[int, EulerVector] = {}
        for cid in range(1, k + 1):
            members = [s for s, lbl in zip(stations, labels) if lbl == cid]
            if len(members) >= self.min_stations:
                euler_map[cid] = invert_euler_vector(members)
            else:
                euler_map[cid] = EulerVector(0.0, 0.0, 0.0)
        return euler_map

    def _reassign(
        self,
        stations: list[GpsStation],
        euler_map: dict[int, EulerVector],
    ) -> np.ndarray:
        """Assign each station to the cluster with the smallest velocity residual.

        Default (weighted_reassign=False): unweighted Euclidean distance in
        velocity space — sqrt(dVe²+dVn²) — matching Savage (2018).
        weighted_reassign=True uses the chi² criterion ((dVe/se)²+(dVn/sn)²),
        kept for reference and comparison.
        """
        cluster_ids = sorted(euler_map)
        new_labels = np.zeros(len(stations), dtype=int)
        resid_fn = weighted_residual_sq if self.weighted_reassign else unweighted_residual_sq
        for i, s in enumerate(stations):
            residuals = {cid: resid_fn(s, euler_map[cid]) for cid in cluster_ids}
            new_labels[i] = min(residuals, key=residuals.get)
        return new_labels

    def _iterate(
        self, stations: list[GpsStation], labels: np.ndarray, k: int
    ) -> tuple[np.ndarray, dict]:
        """Run inversion + reassignment until convergence."""
        for _ in range(self.max_iter):
            euler_map = self._euler_per_cluster(stations, labels, k)
            new_labels = self._reassign(stations, euler_map)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        # Final inversion with converged labels
        euler_map = self._euler_per_cluster(stations, labels, k)
        return labels, euler_map

    def _build_clusters(
        self, stations: list[GpsStation], labels: np.ndarray, k: int,
        euler_map: dict | None = None,
    ) -> list[VelocityCluster]:
        from gps_cluster.domain.services.euler_math import reduced_chi_squared
        if euler_map is None:
            euler_map = self._euler_per_cluster(stations, labels, k)
        clusters = []
        for cid in range(1, k + 1):
            members = [s for s, lbl in zip(stations, labels) if lbl == cid]
            euler = euler_map.get(cid) if len(members) >= self.min_stations else None
            chi2 = total_chi_squared(members, euler) if euler is not None else None
            chi2_red = reduced_chi_squared(members, euler) if euler is not None else None
            clusters.append(VelocityCluster(
                id=cid, stations=members, euler_vector=euler,
                chi2=chi2, chi2_reduced=chi2_red,
            ))
        return clusters

    @staticmethod
    def _total_chi2(clusters: list[VelocityCluster]) -> float:
        total = 0.0
        for c in clusters:
            if c.chi2 is not None:
                total += c.chi2
            elif c.euler_vector is not None and len(c.stations) > 0:
                total += total_chi_squared(c.stations, c.euler_vector)
        return total


# ---------------------------------------------------------------------------
# EM Euler-vector clustering
# ---------------------------------------------------------------------------

class EMEulerVectorClustering:
    """Expectation-Maximisation Euler-vector clustering.

    Extends the hard-assignment Savage (2018) algorithm to a proper EM loop:

    E-step
        Compute soft assignment probabilities w_{ij} = P(station i → cluster j)
        via softmax over −χ²/2.  This is the exact Bayesian posterior under
        Gaussian measurement errors and a uniform cluster prior.

    M-step
        Re-invert each Euler vector using *weighted* least squares, where each
        station i contributes with weight ``w_{ij} / σ_i²``.  Stations near
        block boundaries (high entropy) pull less on both adjacent Euler vectors,
        reducing the elastic-loading bias of hard assignment.

    The algorithm converges when the maximum element-wise change in the weight
    matrix drops below ``tol``, or after ``max_iter`` iterations.

    Initialization
    --------------
    The first E-step is seeded from the converged hard-assignment solution
    (``EulerVectorClustering`` with ``init="multiscale"``), which provides a
    physically reasonable starting point and avoids the label-switching
    degeneracy of random EM initialization.

    Parameters
    ----------
    max_iter : int
        Maximum EM iterations (default 100).
    tol : float
        Convergence tolerance on the weight matrix (default 1e-4).
    min_weight_sum : float
        Minimum effective station count per cluster (sum of weights) needed to
        attempt a WLS inversion.  Below this, the Euler vector is set to zero.
    n_restarts : int
        Passed to the hard-clustering initializer.
    random_seed : int
        Seed for the hard-clustering initializer.
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-4,
        min_weight_sum: float = 2.0,
        n_restarts: int = 20,
        random_seed: int = 0,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.min_weight_sum = min_weight_sum
        self.n_restarts = n_restarts
        self.random_seed = random_seed
        # Hard-assignment clusterer used for initialization
        self._hard = EulerVectorClustering(
            init="multiscale",
            n_restarts=n_restarts,
            random_seed=random_seed,
        )

    # ------------------------------------------------------------------
    # Public API  (mirrors EulerVectorClustering)
    # ------------------------------------------------------------------

    def cluster(
        self,
        stations: list[GpsStation],
        k: int,
        init_labels: np.ndarray | None = None,
    ) -> list[VelocityCluster]:
        """Partition stations into k clusters via EM.

        Returns
        -------
        list[VelocityCluster]
            Hard-assigned clusters (argmax of final weight matrix), with:
            - ``euler_vector`` estimated by soft-weighted WLS
            - ``chi2`` / ``chi2_reduced`` computed on hard-assigned members
            - ``membership_weights`` : ndarray shape (N,) — the column of the
              final weight matrix for this cluster (over *all* stations)
        """
        N = len(stations)

        # ── Initialization: seed weights from hard-assignment solution ──────
        if init_labels is not None:
            labels0 = np.asarray(init_labels, dtype=int)
            euler_map = self._hard._euler_per_cluster(stations, labels0, k)
        else:
            hard_clusters = self._hard.cluster(stations, k)
            labels0 = np.array([
                next(c.id for c in hard_clusters if s in c.stations)
                for s in stations
            ], dtype=int)
            euler_map = {c.id: c.euler_vector for c in hard_clusters
                         if c.euler_vector is not None}

        # One-hot initial weights from hard labels
        weights = np.zeros((N, k), dtype=float)
        for i, lbl in enumerate(labels0):
            if 1 <= lbl <= k:
                weights[i, lbl - 1] = 1.0

        # ── EM loop ─────────────────────────────────────────────────────────
        cluster_ids = list(range(1, k + 1))

        for iteration in range(self.max_iter):
            # M-step: re-invert each Euler vector with current soft weights
            new_euler_map: dict[int, EulerVector] = {}
            for j, cid in enumerate(cluster_ids):
                w_j = weights[:, j]
                if w_j.sum() >= self.min_weight_sum:
                    new_euler_map[cid] = invert_euler_vector_weighted(stations, w_j)
                else:
                    new_euler_map[cid] = EulerVector(0.0, 0.0, 0.0)

            # E-step: recompute soft weights from new Euler vectors
            new_weights = soft_weights_from_euler_map(stations, new_euler_map)

            # Convergence check
            delta = float(np.max(np.abs(new_weights - weights)))
            weights = new_weights
            euler_map = new_euler_map
            if delta < self.tol:
                break

        # ── Build output clusters via hard argmax ────────────────────────────
        labels = np.argmax(weights, axis=1) + 1   # 1-indexed
        return self._build_clusters(stations, labels, k, euler_map, weights)

    def find_optimal_k(
        self,
        stations: list[GpsStation],
        max_k: int = 9,
        alpha: float = 0.05,
    ) -> tuple[int, FTestResult]:
        """Return (optimal_k, FTestResult) using F-test on hard-assigned chi².

        Runs full EM for each k=1..max_k.  Hard-assigned chi² is used for the
        F-test (same criterion as the Savage hard-assignment version) so the
        results are directly comparable.
        """
        n_total = len(stations)
        chi2_vals = np.zeros(max_k)
        solutions: dict[int, list[VelocityCluster]] = {}

        for ki, k in enumerate(range(1, max_k + 1)):
            clusters = self.cluster(stations, k)
            solutions[k] = clusters
            chi2_vals[ki] = _total_chi2_static(clusters)

        dof = np.array([max(2 * n_total - 3 * k, 1) for k in range(1, max_k + 1)])
        chi2_red = chi2_vals / dof

        f_stats = np.zeros(max_k - 1)
        p_vals = np.zeros(max_k - 1)
        for i in range(max_k - 1):
            delta = chi2_vals[i] - chi2_vals[i + 1]
            if chi2_vals[i + 1] > 0:
                f_stats[i] = delta * dof[i + 1] / chi2_vals[i + 1] / 3
            p_vals[i] = 1.0 - f_dist.cdf(f_stats[i], dfn=3, dfd=dof[i + 1])

        optimal_k = max_k
        for i, p in enumerate(p_vals):
            if p >= alpha:
                optimal_k = i + 1
                break

        result = FTestResult(
            k_values=np.arange(1, max_k + 1),
            chi2_total=chi2_vals,
            chi2_reduced=chi2_red,
            f_statistics=f_stats,
            p_values=p_vals,
            solutions=solutions,
        )
        return optimal_k, result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_clusters(
        self,
        stations: list[GpsStation],
        labels: np.ndarray,       # 1-indexed, hard argmax
        k: int,
        euler_map: dict[int, EulerVector],
        weights: np.ndarray,       # (N, k) full weight matrix
    ) -> list[VelocityCluster]:
        from gps_cluster.domain.services.euler_math import reduced_chi_squared
        clusters = []
        for j, cid in enumerate(range(1, k + 1)):
            members = [s for s, lbl in zip(stations, labels) if lbl == cid]
            euler = euler_map.get(cid)
            if euler is not None and (euler.ox == 0 and euler.oy == 0 and euler.oz == 0
                                      and len(members) < 2):
                euler = None
            chi2 = total_chi_squared(members, euler) if euler is not None and members else None
            chi2_red = reduced_chi_squared(members, euler) if euler is not None and members else None
            clusters.append(VelocityCluster(
                id=cid,
                stations=members,
                euler_vector=euler,
                chi2=chi2,
                chi2_reduced=chi2_red,
                membership_weights=weights[:, j].copy(),
            ))
        return clusters


def bootstrap_pole_uncertainty(
    stations: list[GpsStation],
    n_boot: int = 300,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 1-sigma uncertainties on Euler pole (lat, lon, rate).

    Resamples *stations* with replacement *n_boot* times, re-inverts the
    Euler vector each time, and returns the standard deviation of the
    resulting pole parameters.  This gives realistic uncertainties that
    reflect block non-rigidity and uneven station distribution — unlike
    the formal WLS covariance which is overoptimistic for large datasets.

    Parameters
    ----------
    stations : list[GpsStation]
        Stations belonging to a single cluster.
    n_boot : int
        Number of bootstrap resamples.
    random_seed : int
        Seed for the random number generator.

    Returns
    -------
    (sigma_lat_deg, sigma_lon_deg, sigma_rate_deg_myr)
        Bootstrap standard deviations.  Returns (0, 0, 0) if fewer than
        10 valid resamples are obtained.
    """
    from gps_cluster.domain.services.euler_math import (
        euler_vector_to_pole,
        invert_euler_vector,
    )

    rng = np.random.default_rng(random_seed)
    n   = len(stations)
    lats, lons, rates = [], [], []

    for _ in range(n_boot):
        idx    = rng.integers(0, n, size=n)
        sample = [stations[i] for i in idx]
        try:
            ev   = invert_euler_vector(sample)
            pole = euler_vector_to_pole(ev)
            lats.append(pole.lat)
            lons.append(pole.lon)
            rates.append(pole.rate)
        except Exception:
            pass

    if len(lats) < 10:
        return 0.0, 0.0, 0.0
    return float(np.std(lats)), float(np.std(lons)), float(np.std(rates))


class SpatialBayesianEulerClustering:
    """Fully Bayesian Euler-vector clustering with distance-to-centroid prior.

    Extends the EM algorithm (EMEulerVectorClustering) to a Variational
    Bayes EM (VBEM) that encodes the physical prior that tectonic blocks
    are spatially compact.

    Model
    -----
    Likelihood:
        p(v_i | z_i=k, ω_k) = N(G_i ω_k, Σ_i)   [Gaussian GPS errors]

    Prior on cluster assignments — distance to geographic centroid:
        log p(z_i=k | x̄_k) ∝ -γ · d²(x_i, x̄_k)
        where x̄_k is the soft-weighted centroid of cluster k and
        d is great-circle distance in km.

    Prior on Euler vectors:
        p(ω_k) ∝ 1  (flat) → posterior is Gaussian N(ω̂_k, C_k).

    Prior on mixing proportions:
        p(π) = Dir(α=1, …)  (uniform over simplex).

    Inference — Variational Bayes EM
    ---------------------------------
    Approximate posterior: q(z) = Π_i q_i(z_i)

    E-step (iterated centroid ↔ weight fixed point):
        x̄_k   = Σ_i w_{ik} · x_i / Σ_i w_{ik}          (centroid update)
        log q_i(k) = -χ²_{ik}/2 + log π̂_k - γ · d²(x_i, x̄_k)
        w_{ik} = softmax_k(log q_i(k))                   (weight update)

    M-step:
        ω̂_k, C_k = weighted WLS with weights w_{ik}/σ_i²
        π̂_k      = (1 + Σ_i w_{ik}) / (K + N)

    Advantages over the Potts model
    --------------------------------
    - Does NOT penalise geographic neighbours in different clusters, so
      real fault boundaries are correctly resolved.
    - Only penalises teleportation: a station joining a cluster whose
      centroid is far away.
    - Parameter γ has physical units (nats/km²) and a clear meaning:
      γ = 1/R₀² where R₀ is the characteristic block radius.
    - No graph structure required; no inner mean-field coupling loop.
    - At γ = 0 reduces exactly to EMEulerVectorClustering.

    Parameters
    ----------
    gamma:
        Distance penalty (nats/km²).  Default 4×10⁻⁶ = 1/(500 km)².
        Recommended range: [1e-6, 1e-4].  Cross-validate on held-out
        stations for a principled choice.
    max_iter:
        Maximum VB-EM iterations.
    tol:
        Convergence tolerance on max element-wise weight change.
    e_max_iter / e_tol:
        Convergence parameters for the E-step centroid ↔ weight loop.
    n_restarts:
        Passed to the hard-clustering multiscale initialiser.
    random_seed:
        Seed for multiscale initialiser.
    min_weight_sum:
        Minimum effective station count per cluster (Σ_i w_{ik}) to
        attempt WLS inversion; below this the Euler vector is zeroed.
    """

    def __init__(
        self,
        gamma: float = 4e-6,
        max_iter: int = 100,
        tol: float = 1e-4,
        e_tol: float = 1e-4,
        e_max_iter: int = 50,
        n_restarts: int = 20,
        random_seed: int = 0,
        min_weight_sum: float = 2.0,
        normalize_chi2: bool = True,
    ) -> None:
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.e_tol = e_tol
        self.e_max_iter = e_max_iter
        self.n_restarts = n_restarts
        self.random_seed = random_seed
        self.min_weight_sum = min_weight_sum
        self.normalize_chi2 = normalize_chi2
        self._hard = EulerVectorClustering(
            init="multiscale",
            n_restarts=n_restarts,
            random_seed=random_seed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(
        self,
        stations: list[GpsStation],
        k: int,
    ) -> list[VelocityCluster]:
        """Partition stations into k clusters via distance-prior VB-EM.

        Returns
        -------
        list[VelocityCluster]
            Hard-assigned clusters (argmax of final weight matrix) with:
            - ``euler_vector`` : soft-weighted WLS estimate
            - ``chi2`` / ``chi2_reduced`` : computed on hard-assigned members
            - ``membership_weights`` : ndarray (N,) — VB posterior w_{ik}
              over all N stations.  Shannon entropy of these weights is
              geophysically informative: elevated at block boundaries and
              at strain-contaminated stations near locked faults.
        """
        N = len(stations)

        # ── Initialise from hard-assignment clustering ────────────────────────
        hard_clusters = self._hard.cluster(stations, k)
        euler_map: dict[int, EulerVector] = {
            c.id: c.euler_vector
            for c in hard_clusters
            if c.euler_vector is not None
        }
        for cid in range(1, k + 1):
            if cid not in euler_map:
                euler_map[cid] = EulerVector(0.0, 0.0, 0.0)

        pi = np.array([
            next((c.size for c in hard_clusters if c.id == cid), 1)
            for cid in range(1, k + 1)
        ], dtype=float)
        pi /= pi.sum()

        # ── Initial chi²_scale from hard clustering ───────────────────────────
        # Normalises the log-likelihood contribution to ~1 nat/station so that
        # the spatial prior γ·d² can compete even when chi²_red >> 1.
        chi2_scale = self._chi2_scale(hard_clusters, N, k)

        # ── VB-EM loop ────────────────────────────────────────────────────────
        weights: np.ndarray | None = None

        for _iter in range(self.max_iter):
            # E-step: distance-to-centroid posterior (with optional chi² scaling)
            w_new = distance_soft_weights(
                stations, euler_map,
                gamma=self.gamma, pi=pi,
                tol=self.e_tol, max_iter=self.e_max_iter,
                chi2_scale=chi2_scale,
            )

            # M-step: weighted WLS per cluster
            new_euler_map: dict[int, EulerVector] = {}
            for j, cid in enumerate(sorted(euler_map.keys())):
                w_j = w_new[:, j]
                if w_j.sum() >= self.min_weight_sum:
                    new_euler_map[cid] = invert_euler_vector_weighted(stations, w_j)
                else:
                    new_euler_map[cid] = EulerVector(0.0, 0.0, 0.0)

            # Dirichlet posterior on mixing proportions (α = 1)
            pi = (w_new.sum(axis=0) + 1.0) / (N + k)

            # Update chi²_scale from current (soft-assigned) chi²_red
            chi2_scale = self._chi2_scale_from_map(
                stations, new_euler_map, np.argmax(w_new, axis=1) + 1, N, k
            )

            if weights is not None and np.max(np.abs(w_new - weights)) < self.tol:
                weights = w_new
                euler_map = new_euler_map
                break

            weights = w_new
            euler_map = new_euler_map

        if weights is None:
            weights = w_new  # type: ignore[possibly-undefined]

        labels = np.argmax(weights, axis=1) + 1   # 1-indexed
        return self._build_clusters(stations, labels, k, euler_map, weights)

    def find_optimal_k(
        self,
        stations: list[GpsStation],
        max_k: int = 9,
        alpha: float = 0.05,
    ) -> tuple[int, FTestResult]:
        """Return (optimal_k, FTestResult) using F-test on VB-EM chi².

        Runs full spatial VB-EM for k = 1..max_k.  Hard-assigned chi² is
        used for the F-test so results are comparable with the hard and
        plain-EM clusterers.
        """
        n_total = len(stations)
        chi2_vals = np.zeros(max_k)
        solutions: dict[int, list[VelocityCluster]] = {}

        for ki, k in enumerate(range(1, max_k + 1)):
            clusters = self.cluster(stations, k)
            solutions[k] = clusters
            chi2_vals[ki] = _total_chi2_static(clusters)

        dof = np.array([max(2 * n_total - 3 * k, 1) for k in range(1, max_k + 1)])
        chi2_red = chi2_vals / dof

        f_stats = np.zeros(max_k - 1)
        p_vals  = np.zeros(max_k - 1)
        for i in range(max_k - 1):
            delta = chi2_vals[i] - chi2_vals[i + 1]
            if chi2_vals[i + 1] > 0:
                f_stats[i] = delta * dof[i + 1] / chi2_vals[i + 1] / 3
            p_vals[i] = 1.0 - f_dist.cdf(f_stats[i], dfn=3, dfd=dof[i + 1])

        optimal_k = max_k
        for i, p in enumerate(p_vals):
            if p >= alpha:
                optimal_k = i + 1
                break

        return optimal_k, FTestResult(
            k_values=np.arange(1, max_k + 1),
            chi2_total=chi2_vals,
            chi2_reduced=chi2_red,
            f_statistics=f_stats,
            p_values=p_vals,
            solutions=solutions,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chi2_scale(self, clusters: list, N: int, k: int) -> "float | None":
        """Return chi²_red from a list of VelocityCluster (hard assignment).

        Returns None when normalize_chi2 is False, which passes through to
        distance_soft_weights as no scaling (original behaviour).
        """
        if not self.normalize_chi2:
            return None
        chi2_total = sum((c.chi2 or 0.0) for c in clusters)
        dof = max(2 * N - 3 * k, 1)
        return max(chi2_total / dof, 1.0)

    def _chi2_scale_from_map(
        self,
        stations: list,
        euler_map: dict,
        labels: "np.ndarray",
        N: int,
        k: int,
    ) -> "float | None":
        """Return chi²_red from current euler_map + hard-assigned labels."""
        if not self.normalize_chi2:
            return None
        chi2_total = 0.0
        for cid in sorted(euler_map.keys()):
            ev = euler_map[cid]
            if ev is None or (ev.ox == 0.0 and ev.oy == 0.0 and ev.oz == 0.0):
                continue
            members = [s for s, lbl in zip(stations, labels) if lbl == cid]
            if members:
                chi2_total += total_chi_squared(members, ev)
        dof = max(2 * N - 3 * k, 1)
        return max(chi2_total / dof, 1.0)

    def _build_clusters(
        self,
        stations: list[GpsStation],
        labels: np.ndarray,
        k: int,
        euler_map: dict[int, EulerVector],
        weights: np.ndarray,
    ) -> list[VelocityCluster]:
        from gps_cluster.domain.services.euler_math import reduced_chi_squared
        clusters = []
        for j, cid in enumerate(range(1, k + 1)):
            members = [s for s, lbl in zip(stations, labels) if lbl == cid]
            euler   = euler_map.get(cid)
            if euler is not None and euler.ox == 0 and euler.oy == 0 and euler.oz == 0:
                euler = None if len(members) < 2 else euler
            chi2     = total_chi_squared(members, euler) if euler and members else None
            chi2_red = reduced_chi_squared(members, euler) if euler and members else None
            clusters.append(VelocityCluster(
                id=cid,
                stations=members,
                euler_vector=euler,
                chi2=chi2,
                chi2_reduced=chi2_red,
                membership_weights=weights[:, j].copy(),
            ))
        return clusters


def _total_chi2_static(clusters: list[VelocityCluster]) -> float:
    """Module-level helper so EMEulerVectorClustering doesn't need self."""
    total = 0.0
    for c in clusters:
        if c.chi2 is not None:
            total += c.chi2
        elif c.euler_vector is not None and len(c.stations) > 0:
            total += total_chi_squared(c.stations, c.euler_vector)
    return total
