"""Unit tests for application/preprocess.py."""

from __future__ import annotations

import pytest

from gps_cluster.application.preprocess import preprocess, remove_fixed, remove_outliers, remove_uncertain
from gps_cluster.domain.entities import GpsStation, Position, Velocity


def _s(name: str, ve: float, vn: float, se: float = 0.1, sn: float = 0.1) -> GpsStation:
    return GpsStation(name, Position(0.0, 0.0), Velocity(ve, vn, 0.0, se, sn, 0.1))


class TestRemoveFixed:
    def test_removes_exact_zeros(self):
        stations = [_s("A", 0.0, 0.0), _s("B", 1.0, 0.0)]
        assert len(remove_fixed(stations)) == 1

    def test_keeps_near_zero_but_nonzero(self):
        stations = [_s("A", 1e-10, 0.0), _s("B", 0.0, 1e-10)]
        assert len(remove_fixed(stations)) == 2

    def test_empty_input(self):
        assert remove_fixed([]) == []


class TestRemoveUncertain:
    def test_removes_high_sigma(self):
        stations = [_s("A", 1.0, 1.0, se=0.7, sn=0.1), _s("B", 1.0, 1.0, se=0.1, sn=0.1)]
        result = remove_uncertain(stations, max_sigma=0.6)
        assert len(result) == 1
        assert result[0].name == "B"

    def test_boundary_is_exclusive(self):
        s = _s("A", 1.0, 1.0, se=0.6, sn=0.1)
        assert remove_uncertain([s], max_sigma=0.6) == []

    def test_keeps_all_when_low_sigma(self):
        stations = [_s("A", 1.0, 1.0, se=0.1, sn=0.1)] * 5
        assert len(remove_uncertain(stations)) == 5


class TestRemoveOutliers:
    def test_removes_clear_outlier(self):
        stations = [_s(f"S{i}", float(i), 0.0) for i in range(10)]
        stations.append(_s("Outlier", 1000.0, 0.0))
        result = remove_outliers(stations, zscore_threshold=2.0)
        names = {s.name for s in result}
        assert "Outlier" not in names

    def test_passthrough_for_uniform_data(self):
        stations = [_s(f"S{i}", 5.0, 3.0) for i in range(10)]
        assert len(remove_outliers(stations)) == len(stations)

    def test_single_station_unchanged(self):
        s = [_s("A", 100.0, 0.0)]
        assert remove_outliers(s) == s


class TestPreprocessPipeline:
    def test_chains_all_filters(self):
        stations = [
            _s("fixed", 0.0, 0.0),
            _s("uncertain", 1.0, 1.0, se=0.9, sn=0.1),
            _s("outlier", 1000.0, 0.0),
            _s("good1", 5.0, 3.0),
            _s("good2", 6.0, 4.0),
        ]
        result = preprocess(stations)
        names = {s.name for s in result}
        assert "fixed" not in names
        assert "uncertain" not in names
        assert "good1" in names
        assert "good2" in names
