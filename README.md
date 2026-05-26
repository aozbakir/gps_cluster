# gps_cluster

Python implementation of Euler-vector clustering for GPS velocities (Savage, 2018), applied to 468 GEONET stations in southwest Japan.

## Reference

Savage, J. C. (2018). Euler-vector clustering of GPS velocities defines microplate geometry in southwest Japan. *JGR Solid Earth*, 123, 1437–1454. https://doi.org/10.1002/2017JB014874

## What it does

Partitions GPS stations into *k* rigid blocks by iteratively inverting one Euler vector per cluster and reassigning each station to the cluster whose predicted velocity best fits its observed velocity. The best solution over many random restarts (minimum RMS) is kept.

## Install

```bash
pip install -e ".[viz,dev]"
```

Requires Python ≥ 3.10. Cartopy and Matplotlib are in the optional `viz` group.

## Usage

```python
from gps_cluster.infrastructure.readers.velocity_csv import read_velocity_file
from gps_cluster.application.preprocess import preprocess
from gps_cluster.application.euler_clustering import EulerVectorClustering

stations = preprocess(read_velocity_file("data/raw/gji_4600_SI_TablesS1.csv"))
evc = EulerVectorClustering(init="multiscale", n_restarts=100)
clusters = evc.cluster(stations, k=3)
```

## Figures

```bash
python scripts/plot_japan_clusters.py
# → reports/figures/fig1..fig8
```

## Tests

```bash
pytest
```

## Deviations from Savage (2018)

| Item | Paper | This repo |
|------|-------|-----------|
| Restarts | 3,000 | 100 (configurable) |
| Station count | 469 | 468 (one absent from source CSV) |
| Optimal-k criterion | ω-space collinearity | F-test on χ² (convenience) |
