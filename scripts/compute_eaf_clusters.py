"""Compute Euler-vector clustering for the EAF region (k=1..6) and write JSON cache.

Output: results/eaf/clusters.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gps_cluster.application.euler_clustering import SpatialBayesianEulerClustering
from gps_cluster.application.preprocess import filter_by_extent, preprocess
from gps_cluster.application.serialise import clusters_to_list, station_to_dict
from gps_cluster.application.velocity_clustering import VelocityHACClustering
from gps_cluster.infrastructure.readers.velocity_vel import read_vel_file

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data/external/globk_vel_igs14_ITRF_M2E_11JAN2021_CMBND_improved_reformat.vel"
OUT    = ROOT / "results/eaf"
CACHE  = OUT / "clusters.json"
OUT.mkdir(parents=True, exist_ok=True)

EXTENT     = [34.0, 42.0, 36.0, 39.0]
MAX_K      = 6
N_RESTARTS = 50
N_REF      = 50
VB_GAMMA   = 4e-6        # distance-to-centroid prior: 1 / (500 km)²

raw      = read_vel_file(DATA)
all_sta  = preprocess(raw, max_sigma=99, zscore_threshold=99)
stations = filter_by_extent(all_sta, *EXTENT)
N = len(stations)
print(f"EAF region: {N} stations")

# ── gap statistic (velocity-space HAC, unchanged) ─────────────────────────────
print(f"Gap statistic (max_k={MAX_K}, n_ref={N_REF}) …")
hac = VelocityHACClustering()
_, gap_result = hac.find_optimal_k(stations, max_k=MAX_K, n_ref=N_REF)
print(f"  k_max_gap={gap_result.k_max_gap}  k_first_cross={gap_result.optimal_k}")

# ── VB F-test ─────────────────────────────────────────────────────────────────
print(f"VB F-test (max_k={MAX_K}, n_restarts={N_RESTARTS}, γ={VB_GAMMA:.1e}) …")
vbc = SpatialBayesianEulerClustering(
    gamma=VB_GAMMA, n_restarts=N_RESTARTS, random_seed=0
)
_, ftest = vbc.find_optimal_k(stations, max_k=MAX_K)

# ── station records ───────────────────────────────────────────────────────────
station_records = [station_to_dict(s) for s in stations]

# ── k-by-k solutions: serialise + extract genuine VB entropy ──────────────────
print(f"Serialising k=1…{MAX_K} solutions …")
solutions: dict[int, list[dict]] = {}

def _vb_entropy(clusters: list) -> np.ndarray:
    W = np.column_stack([c.membership_weights for c in clusters])  # (N, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(W * np.where(W > 0, np.log(W), 0.0), axis=1)

for k in range(1, MAX_K + 1):
    sol     = clusters_to_list(ftest.solutions[k])
    entropy = _vb_entropy(ftest.solutions[k])
    for s_rec, ent in zip(station_records, entropy):
        s_rec[f"vb_entropy_k{k}"] = float(ent)
    solutions[k] = sol
    dof    = max(2 * N - 3 * k, 1)
    chi2_t = sum(c["chi2"] for c in sol)
    print(f"  k={k}  χ²_red={chi2_t/dof:.3f}  mean_vb_entropy={entropy.mean():.3f} nats")

# ── cache ─────────────────────────────────────────────────────────────────────
cache = {
    "meta": {
        "n_stations": N,
        "extent":     EXTENT,
        "max_k":      MAX_K,
        "n_restarts": N_RESTARTS,
        "n_ref_gap":  N_REF,
        "vb_gamma":   VB_GAMMA,
        "frame":      "ITRF14",
        "source":     "GLOBK Jan 2021 — EAF region",
    },
    "stations": station_records,
    "gap": {
        "k_values":      gap_result.k_values.tolist(),
        "gap":           gap_result.gap.tolist(),
        "sk":            gap_result.sk.tolist(),
        "k_first_cross": gap_result.optimal_k,
        "k_max_gap":     gap_result.k_max_gap,
    },
    "ftest": {
        "k_values":     ftest.k_values.tolist(),
        "chi2_reduced": ftest.chi2_reduced.tolist(),
        "f_statistics": ftest.f_statistics.tolist(),
        "p_values":     ftest.p_values.tolist(),
    },
    "solutions": {str(k): v for k, v in solutions.items()},
}

with open(CACHE, "w") as f:
    json.dump(cache, f, indent=2)

print(f"\nCache written → {CACHE}")
