"""Compute Euler-vector clustering for southwest Japan (Savage 2018 dataset).

Run once (~15 min with 100 restarts).  Cache is read by plot_japan_clusters.py.

Output: results/japan/clusters.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gps_cluster.application.euler_clustering import SpatialBayesianEulerClustering
from gps_cluster.application.preprocess import preprocess
from gps_cluster.application.serialise import clusters_to_list, station_to_dict
from gps_cluster.application.velocity_clustering import VelocityHACClustering
from gps_cluster.infrastructure.readers.velocity_csv import read_velocity_file

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data/external/gji_4600_SI_TablesS1.csv"
OUT    = ROOT / "results/japan"
CACHE  = OUT / "clusters.json"
OUT.mkdir(parents=True, exist_ok=True)

MAX_K      = 9
N_RESTARTS = 100
N_REF      = 30
VB_GAMMA   = 4e-6        # distance-to-centroid prior: 1 / (500 km)²

# ── load data ─────────────────────────────────────────────────────────────────
raw      = read_velocity_file(DATA)
stations = preprocess(raw, max_sigma=99, zscore_threshold=99)
N        = len(stations)
print(f"Stations: raw={len(raw)}, clean={N}")

# ── gap statistic + linkage matrix ────────────────────────────────────────────
print(f"Gap statistic (max_k={MAX_K}, n_ref={N_REF}) …")
hac = VelocityHACClustering()
Z   = hac.fit(stations)                                    # linkage matrix for dendrogram
_, gap_result = hac.find_optimal_k(stations, max_k=MAX_K, n_ref=N_REF)
k_gap = gap_result.k_max_gap
print(f"  k_max_gap={k_gap}  k_first_cross={gap_result.optimal_k}")

# ── VB F-test ─────────────────────────────────────────────────────────────────
print(f"VB F-test (max_k={MAX_K}, n_restarts={N_RESTARTS}, γ={VB_GAMMA:.1e}) …")
vbc = SpatialBayesianEulerClustering(
    gamma=VB_GAMMA, n_restarts=N_RESTARTS, random_seed=0
)
_, ftest = vbc.find_optimal_k(stations, max_k=MAX_K)
k_euler = int(ftest.k_values[np.argmin(ftest.chi2_reduced)])
print(f"  k_euler (min chi²_red) = {k_euler}")

# ── station records (preprocessed + raw for velocity scatter) ─────────────────
station_records     = [station_to_dict(s) for s in stations]
raw_station_records = [station_to_dict(s) for s in raw]

# ── k-by-k solutions (k = 2..MAX_K) ──────────────────────────────────────────
print(f"Serialising k=2…{MAX_K} solutions …")
solutions: dict[int, list[dict]] = {}


def _vb_entropy(clusters: list) -> np.ndarray:
    W = np.column_stack([c.membership_weights for c in clusters])  # (N, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(W * np.where(W > 0, np.log(W), 0.0), axis=1)


for k in range(2, MAX_K + 1):
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
        "n_stations":     N,
        "n_raw_stations": len(raw),
        "max_k":          MAX_K,
        "n_restarts":     N_RESTARTS,
        "n_ref_gap":      N_REF,
        "vb_gamma":       VB_GAMMA,
        "frame":          "ITRF2000",
        "source":         "Savage (2018) Suppl. Table S1",
        "k_gap":          k_gap,
        "k_euler":        k_euler,
    },
    "stations":     station_records,
    "raw_stations": raw_station_records,
    "linkage":      Z.tolist(),              # scipy linkage matrix for dendrogram
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
