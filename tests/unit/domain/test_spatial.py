"""Tests for domain/services/spatial.py — Delaunay graph construction."""

from __future__ import annotations

import numpy as np
import pytest

from gps_cluster.domain.entities import GpsStation, Position, Velocity
from gps_cluster.domain.services.spatial import adjacency_matrix, delaunay_graph


def _station(lon: float, lat: float, name: str = "S") -> GpsStation:
    return GpsStation(
        name=name,
        position=Position(lon=lon, lat=lat),
        velocity=Velocity(ve=0.0, vn=0.0, vu=0.0, se=1.0, sn=1.0, su=1.0),
    )


# ── delaunay_graph ────────────────────────────────────────────────────────────

def test_delaunay_graph_square():
    """Four corners of a square: each corner has 2 or 3 neighbours."""
    stations = [
        _station(0.0, 0.0, "A"),
        _station(1.0, 0.0, "B"),
        _station(1.0, 1.0, "C"),
        _station(0.0, 1.0, "D"),
    ]
    g = delaunay_graph(stations)
    assert len(g) == 4
    # Every node has at least 2 neighbours
    for i, nbrs in g.items():
        assert len(nbrs) >= 2, f"Node {i} has only {len(nbrs)} neighbours"


def test_delaunay_graph_symmetric():
    """If j is a neighbour of i then i is a neighbour of j."""
    stations = [_station(float(i), float(j), f"{i}{j}")
                for i in range(4) for j in range(4)]
    g = delaunay_graph(stations)
    for i, nbrs in g.items():
        for j in nbrs:
            assert i in g[j], f"Graph is not symmetric: {i}->{j} but not {j}->{i}"


def test_delaunay_graph_no_self_loops():
    # Use a 2D grid — non-degenerate for Delaunay
    stations = [_station(float(i), float(j), f"{i}{j}")
                for i in range(3) for j in range(2)]
    g = delaunay_graph(stations)
    for i, nbrs in g.items():
        assert i not in nbrs, f"Self-loop at node {i}"


def test_delaunay_graph_fewer_than_3():
    """Degenerate input: fewer than 3 stations returns empty neighbour sets."""
    stations = [_station(0.0, 0.0, "A"), _station(1.0, 0.0, "B")]
    g = delaunay_graph(stations)
    assert all(len(v) == 0 for v in g.values())


def test_delaunay_graph_collinear_fallback():
    """Collinear stations trigger kNN fallback — no crash, graph is valid."""
    stations = [_station(float(i), 0.0, str(i)) for i in range(6)]
    g = delaunay_graph(stations)  # must not raise
    assert len(g) == 6
    # Symmetry still holds
    for i, nbrs in g.items():
        for j in nbrs:
            assert i in g[j]
    # No self-loops
    for i, nbrs in g.items():
        assert i not in nbrs


# ── adjacency_matrix ──────────────────────────────────────────────────────────

def test_adjacency_matrix_shape():
    stations = [_station(float(i), float(j)) for i in range(3) for j in range(3)]
    g = adjacency_matrix(delaunay_graph(stations), len(stations))
    assert g.shape == (9, 9)


def test_adjacency_matrix_symmetric():
    stations = [_station(float(i), float(j)) for i in range(4) for j in range(4)]
    g = delaunay_graph(stations)
    A = adjacency_matrix(g, len(stations))
    diff = (A - A.T).data
    assert len(diff) == 0 or np.allclose(diff, 0), "Adjacency matrix not symmetric"


def test_adjacency_matrix_no_diagonal():
    stations = [_station(float(i), float(j)) for i in range(3) for j in range(3)]
    g = delaunay_graph(stations)
    A = adjacency_matrix(g, len(stations)).toarray()
    assert np.all(np.diag(A) == 0), "Diagonal must be zero (no self-loops)"


def test_adjacency_matmul_spatial_term():
    """A @ w gives the correct neighbour weight sum."""
    stations = [
        _station(0.0, 0.0, "A"),
        _station(1.0, 0.0, "B"),
        _station(0.5, 1.0, "C"),
    ]
    g = delaunay_graph(stations)
    A = adjacency_matrix(g, 3)
    # Uniform weights: [1/2, 1/2] for K=2
    w = np.full((3, 2), 0.5)
    result = A @ w
    # Each node has 2 neighbours in a triangle → sum = 2 * 0.5 = 1.0 per cluster
    assert result.shape == (3, 2)
    assert np.allclose(result, 1.0)
