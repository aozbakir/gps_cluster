"""Compute Euler-vector clustering for the Kurt et al. (2022) Euro-fixed
GNSS velocity field and write JSON cache.

Runs both the Savage (2018) hard-assignment algorithm and the new
EM soft-assignment extension.

Input:  data/external/gnss_vel_euro_kurt2022.dat  (lon lat ve vn se sn)
Output: reports/anatolia_eur/clusters.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gps_cluster.application.euler_clustering import (
    EMEulerVectorClustering,
    EulerVectorClustering,
)
from gps_cluster.application.preprocess import preprocess
from gps_cluster.application.serialise import clusters_to_list, station_to_dict
from gps_cluster.application.velocity_clustering import VelocityHACClustering
from gps_cluster.domain.services.euler_math import assignment_probabilities
from gps_cluster.infrastructure.readers.velocity_dat import read_dat_file

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data/external/gnss_vel_euro_kurt2022.dat"
OUT    = ROOT / "reports/anatolia_eur"
CACHE  = OUT / "clusters.json"
OUT.mkdir(parents=True, exist_ok=True)

MAX_K      = 7
N_RESTARTS = 20
N_REF      = 30

raw      = read_dat_file(DATA)
stations = preprocess(raw, max_sigma=99, zscore_threshold=99)
N = len(stations)
print(f"Stations: raw={len(raw)}, clean={N}")

# ── gap statistic ──────────────────────────────────────────────────────────────
print(f"Gap statistic (max_k={MAX_K}, n_ref={N_REF}) …")
hac = VelocityHACClustering()
_, gap_result = hac.find_optimal_k(stations, max_k=MAX_K, n_ref=N_REF)
print(f"  k_max_gap={gap_result.k_max_gap}  k_first_cross={gap_result.optimal_k}")

# ── F-test ─────────────────────────────────────────────────────────────────────
print(f"F-test (max_k={MAX_K}, n_restarts={N_RESTARTS}) …")
evc = EulerVectorClustering(init="multiscale", n_restarts=N_RESTARTS, random_seed=0)
_, ftest = evc.find_optimal_k(stations, max_k=MAX_K)

# ── station records (needed inside the per-k loop below) ─────────────────────
station_records = [station_to_dict(s) for s in stations]

# ── serialise solutions (reuse cached results from find_optimal_k) ─────────────
print(f"Serialising k=1…{MAX_K} solutions …")
solutions: dict[int, list[dict]] = {}
for k in range(1, MAX_K + 1):
    sol = clusters_to_list(ftest.solutions[k])
    probs, entropy = assignment_probabilities(stations, ftest.solutions[k])
    for i, s_dict in enumerate(sol):
        s_dict["prob_vector"] = probs[i].tolist()
    for s_rec, ent in zip(station_records, entropy):
        s_rec[f"entropy_k{k}"] = float(ent)
    solutions[k] = sol
    dof    = max(2 * N - 3 * k, 1)
    chi2_t = sum(c["chi2"] for c in sol)
    print(f"  k={k}  χ²_red={chi2_t/dof:.2f}  mean_entropy={entropy.mean():.3f}")

# ── EM clustering at the gap-statistic optimal k ──────────────────────────────
k_em = gap_result.k_max_gap
print(f"\nEM clustering at k={k_em} …")
em = EMEulerVectorClustering(max_iter=100, n_restarts=N_RESTARTS, random_seed=0)
em_clusters = em.cluster(stations, k=k_em)
em_sol = clusters_to_list(em_clusters)

em_weights = np.column_stack([
    c.membership_weights for c in em_clusters
    if c.membership_weights is not None
])  # (N, k_em)
with np.errstate(divide="ignore", invalid="ignore"):
    em_entropy = -np.nansum(
        em_weights * np.where(em_weights > 0, np.log(em_weights), 0.0), axis=1
    )
for s_rec, ent in zip(station_records, em_entropy):
    s_rec["em_entropy"] = float(ent)

dof_em = max(2 * N - 3 * k_em, 1)
chi2_em = sum(c["chi2"] for c in em_sol)
print(f"  EM k={k_em}  χ²_red={chi2_em/dof_em:.2f}  mean_entropy={em_entropy.mean():.3f}")

cache = {
    "meta": {
        "n_stations": N, "max_k": MAX_K,
        "n_restarts": N_RESTARTS, "n_ref_gap": N_REF,
        "frame": "Euro-fixed (Kurt et al. 2022)",
        "source": "gnss_vel_euro_kurt2022.dat",
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
        "chi2_total":   ftest.chi2_total.tolist(),
        "chi2_reduced": ftest.chi2_reduced.tolist(),
        "f_statistics": ftest.f_statistics.tolist(),
        "p_values":     ftest.p_values.tolist(),
    },
    "solutions": {str(k): v for k, v in solutions.items()},
    "em": {
        "k":        k_em,
        "solution": em_sol,
    },
}

with open(CACHE, "w") as f:
    json.dump(cache, f, indent=2)

print(f"\nCache written → {CACHE}")
