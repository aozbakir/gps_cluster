"""Spatial graph utilities for station neighbourhood analysis.

Provides the geographic neighbour graph used by the Variational Bayes
Euler-vector clustering (SpatialBayesianEulerClustering) as its spatial
prior on cluster assignments (Potts model on the Delaunay triangulation).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay

from gps_cluster.domain.entities import GpsStation


def delaunay_graph(stations: list[GpsStation]) -> dict[int, set[int]]:
    """Delaunay triangulation neighbour graph on station positions.

    Builds the triangulation in geographic (lon, lat) coordinates.
    Every pair of stations that share a Delaunay edge are considered
    geographic neighbours.

    Parameters
    ----------
    stations:
        List of GPS stations.  At least 3 required for a non-degenerate
        triangulation.

    Returns
    -------
    dict mapping station index → set of neighbour indices.
    """
    if len(stations) < 3:
        return {i: set() for i in range(len(stations))}

    pts = np.array([[s.position.lon, s.position.lat] for s in stations])

    try:
        tri = Delaunay(pts)
        neighbours: dict[int, set[int]] = {i: set() for i in range(len(stations))}
        for simplex in tri.simplices:
            a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
            neighbours[a].update((b, c))
            neighbours[b].update((a, c))
            neighbours[c].update((a, b))
        return neighbours
    except Exception:
        # Degenerate geometry (e.g. collinear stations): fall back to
        # 6-nearest-neighbours by Euclidean distance in lon/lat.
        return _knn_graph(pts, k=min(6, len(stations) - 1))


def _knn_graph(pts: np.ndarray, k: int) -> dict[int, set[int]]:
    """k-nearest-neighbour fallback graph for degenerate Delaunay inputs."""
    from scipy.spatial import KDTree
    tree = KDTree(pts)
    neighbours: dict[int, set[int]] = {i: set() for i in range(len(pts))}
    # k+1 because query returns the point itself as nearest neighbour
    dists, indices = tree.query(pts, k=k + 1)
    for i, nbrs in enumerate(indices):
        for j in nbrs[1:]:   # skip self (index 0)
            neighbours[i].add(int(j))
            neighbours[int(j)].add(i)
    return neighbours


def adjacency_matrix(
    graph: dict[int, set[int]],
    n: int,
) -> csr_matrix:
    """Convert a neighbour-graph dict to a sparse CSR adjacency matrix.

    The matrix A has A[i, j] = 1 wherever station j is a Delaunay
    neighbour of station i.  Used for the vectorised spatial-prior term:

        spatial_term = A @ w          # shape (N, K)

    where w is the (N, K) weight matrix and spatial_term[i, k] =
    number of geographic neighbours of station i that are soft-assigned
    to cluster k.

    Parameters
    ----------
    graph:
        Neighbour graph returned by :func:`delaunay_graph`.
    n:
        Number of stations (matrix size n × n).

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n, n).
    """
    rows: list[int] = []
    cols: list[int] = []
    for i, nbrs in graph.items():
        for j in nbrs:
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=float)
    return csr_matrix((data, (rows, cols)), shape=(n, n))
