# gps_cluster

Bayesian Euler-vector clustering of GPS velocities for tectonic block identification.
Applied to Anatolia (Turkey), the East Anatolian Fault system, the Marlborough Fault
System (New Zealand), and southwest Japan (Savage 2018 reproduction).

## Reference

Savage, J. C. (2018). Euler-vector clustering of GPS velocities defines microplate
geometry in southwest Japan. *JGR Solid Earth*, 123, 1437–1454.
https://doi.org/10.1002/2017JB014874

## Algorithms

Four algorithms are implemented, forming a methodological ladder from hard assignment
to fully Bayesian inference.

### 1. `VelocityHACClustering`
Hierarchical agglomerative clustering in 2D velocity space (vₑ, vₙ). No Euler physics
— treats each station's velocity as a point in ℝ². Used exclusively for the gap
statistic to bound the search over k. Not a block model.

### 2. `EulerVectorClustering` — Savage (2018)
Hard-assignment EM. E-step: assign each station to the cluster whose Euler vector gives
the lowest χ². M-step: weighted-least-squares inversion for one Euler vector per
cluster. Iterates until assignments stop changing. No soft weights — every station
belongs to exactly one cluster. Multiple random restarts; best solution (minimum total
χ²) is kept.

### 3. `EMEulerVectorClustering`
Soft-assignment EM (Gaussian mixture in velocity space). E-step: compute posterior
weight w[i,k] ∝ exp(−χ²[i,k]/2) for each station × cluster pair. M-step: weighted
WLS where each station contributes to all cluster inversions proportional to its weight.
Converges to a local maximum of the mixture likelihood. Weights are genuine posteriors
**under a flat prior on assignments** — which is where `SpatialBayesianEulerClustering`
improves.

### 4. `SpatialBayesianEulerClustering` ← default for all compute scripts
Variational Bayes EM with a distance-to-centroid prior. Same likelihood as #3. Added
prior:

```
log p(zᵢ = k) ∝ −γ · d²(xᵢ, x̄ₖ)
```

where d is great-circle distance from station i to the soft-weighted centroid of cluster
k, and γ = 1/(500 km)² by default. The E-step is a fixed-point iteration: centroids
and weights are mutually dependent and updated together until convergence. Setting γ = 0
recovers #3 exactly.

**Why this prior?** It penalises teleportation (a station assigned to a geographically
distant cluster pays an extra cost) without penalising fault boundaries (adjacent
stations on opposite sides of the NAF are free to be in different clusters). The Potts
model — an alternative that rewards geographic neighbours for sharing a cluster — was
considered and rejected: it explicitly penalises the real fault boundaries we are trying
to detect.

The prior also enters the M-step indirectly: the weighted WLS uses the VB weights, so
the Euler vector estimates themselves change relative to #3.

**The key conceptual ladder:**

| Algorithm | Assignment | Prior | Entropy |
|-----------|-----------|-------|---------|
| Savage (2018) | Hard | — | 0 or log(k) |
| EM | Soft | Flat | Near-zero near faults (uninformative) |
| **VB** | **Soft** | **Distance-to-centroid** | **Elevated at true block boundaries** |

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Or activate the bundled venv:

```bash
source .venv/bin/activate
```

## Usage

```python
from gps_cluster.infrastructure.readers.velocity_vel import read_vel_file
from gps_cluster.application.preprocess import preprocess
from gps_cluster.application.euler_clustering import SpatialBayesianEulerClustering

stations = preprocess(read_vel_file("data/external/globk_vel_igs14_...vel"))
vbc = SpatialBayesianEulerClustering(gamma=4e-6, n_restarts=20)
k_opt, ftest = vbc.find_optimal_k(stations, max_k=7)
clusters = ftest.solutions[k_opt]
```

## Workflow

Each region has a compute script (slow, run once) and a plot script (fast, cache-only):

```
python scripts/compute_anatolia_clusters.py     # → results/anatolia/clusters.json
python scripts/plot_anatolia_clusters.py        # → results/anatolia/fig*.png

python scripts/compute_eaf_clusters.py          # → results/eaf/clusters.json
python scripts/plot_eaf_clusters.py             # → results/eaf/fig*.png

python scripts/compute_anatolia_eur_clusters.py # → results/anatolia_eur/clusters.json
python scripts/plot_anatolia_eur.py             # → results/anatolia_eur/fig*.png

python scripts/compute_marlborough_clusters.py  # → results/marlborough/clusters.json
python scripts/plot_marlborough_clusters.py     # → results/marlborough/fig*.png

python scripts/compute_japan_clusters.py        # → results/japan/clusters.json
python scripts/plot_japan_clusters.py           # → results/japan/fig*.png
```

The comparison figure (Turkey only):
```
python scripts/plot_comparison.py               # → results/anatolia/fig_comparison.png
```

## Tests

```bash
pytest          # 120 tests, all passing
```

## Deviations from Savage (2018)

| Item | Paper | This repo |
|------|-------|-----------|
| Restarts | 3,000 | 20–50 (configurable; see issue #9) |
| Station count | 469 | 468 (one absent from source CSV; see issue #10) |
| Optimal-k criterion | ω-space collinearity | F-test on χ² + gap statistic |
| Assignment | Hard | Soft (VB posterior weights) |
