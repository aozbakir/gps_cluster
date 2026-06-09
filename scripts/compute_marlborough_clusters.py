"""Compute Euler-vector clustering for the Marlborough Fault System and write results to JSON.

Run this once (takes ~5 min with 50 restarts).  The JSON cache is then
read by plot_marlborough_clusters.py for fast, compute-free plotting.

Output: reports/marlborough/clusters.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gps_cluster.application.euler_clustering import EulerVectorClustering
from gps_cluster.application.preprocess import filter_by_extent
from gps_cluster.application.serialise import clusters_to_list, station_to_dict
from gps_cluster.application.velocity_clustering import VelocityHACClustering
from gps_cluster.infrastructure.readers.velocity_csv_nz import read_nz_csv

# Fault data: Litchfield et al. 2014, NZJGG 57(1):32-56
#   doi:10.1080/00288306.2013.854256
#   data set: https://doi.org/10.21420/W08T-TY11?x=y

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data/external/nz_velocities.csv"
OUT    = ROOT / "reports/marlborough"
CACHE  = OUT / "clusters.json"
OUT.mkdir(parents=True, exist_ok=True)

EXTENT  = [171.0, 175.0, -43.5, -41.0]
MAX_K   = 8
N_REF   = 50
N_RESTARTS = 50

# ── load & filter ──────────────────────────────────────────────────────────────
all_sta  = read_nz_csv(DATA, frame="itrf")
stations = filter_by_extent(all_sta, *EXTENT)
N = len(stations)
print(f"Marlborough box: {N} stations")

# ── gap statistic (HAC on velocity space) ──────────────────────────────────────
print(f"Running gap statistic (max_k={MAX_K}, n_ref={N_REF}) …")
hac = VelocityHACClustering()
_, gap_result = hac.find_optimal_k(stations, max_k=MAX_K, n_ref=N_REF)
print(f"  first-crossing k = {gap_result.optimal_k}   max-gap k = {gap_result.k_max_gap}")

# ── Euler-vector F-test ────────────────────────────────────────────────────────
print(f"Running Euler F-test (max_k={MAX_K}) …")
evc = EulerVectorClustering(init="multiscale", n_restarts=N_RESTARTS, random_seed=0)
_, ftest = evc.find_optimal_k(stations, max_k=MAX_K)

k_elbow = ftest.k_elbow
print(f"  k_elbow (largest Δχ²_red) = {k_elbow}")

# ── k-by-k cluster solutions ───────────────────────────────────────────────────
print(f"Serialising k=2…{MAX_K} solutions …")
solutions: dict[int, list[dict]] = {}
for k in range(2, MAX_K + 1):
    sol = clusters_to_list(ftest.solutions[k])
    solutions[k] = sol
    dof        = max(2 * N - 3 * k, 1)
    chi2_total = sum(c["chi2"] for c in sol)
    print(f"  k={k}  χ²_red={chi2_total/dof:.2f}")

# ── serialise ─────────────────────────────────────────────────────────────────
station_records = [station_to_dict(s) for s in stations]

cache = {
    "meta": {
        "n_stations":  N,
        "extent":      EXTENT,
        "max_k":       MAX_K,
        "n_restarts":  N_RESTARTS,
        "n_ref_gap":   N_REF,
        "frame":       "ITRF2008",
        "source":      "Beavan et al. 2016",
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
        "k_elbow":      k_elbow,
    },
    "solutions": {str(k): v for k, v in solutions.items()},
}

with open(CACHE, "w") as f:
    json.dump(cache, f, indent=2)

print(f"\nCache written to {CACHE}")
