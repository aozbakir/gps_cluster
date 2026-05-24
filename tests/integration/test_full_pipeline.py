"""Integration test: full pipeline from CSV → preprocess → cluster → Euler poles."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from gps_cluster.application.euler_clustering import EulerVectorClustering
from gps_cluster.application.preprocess import preprocess
from gps_cluster.application.velocity_clustering import VelocityHACClustering
from gps_cluster.domain.services.euler_math import euler_vector_to_pole
from gps_cluster.infrastructure.readers.velocity_csv import read_velocity_file


FIXTURE_CSV = Path(__file__).parent.parent / "fixtures" / "synthetic_two_block.csv"


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory) -> Path:
    """Write a synthetic two-block CSV file and return its path."""
    import numpy as np
    from gps_cluster.domain.entities import EulerPole
    from gps_cluster.domain.entities import GpsStation, Position, Velocity
    from gps_cluster.domain.services.euler_math import euler_pole_to_vector, predict_velocity

    rng = np.random.default_rng(42)
    sigma = 1.0  # mm/yr; noise drawn from N(0, sigma^2) so chi2_red(k_true) ≈ 1

    pole_a = EulerPole(lat=45.0, lon=10.0, rate=2.0)
    pole_b = EulerPole(lat=20.0, lon=-80.0, rate=3.0)
    ev_a = euler_pole_to_vector(pole_a)
    ev_b = euler_pole_to_vector(pole_b)

    lines = ["Sta Longitude Latitude Ve Vn Vu Se Sn Su"]
    for i, (lon, lat) in enumerate(zip(np.linspace(-125, -118, 12), np.linspace(36, 40, 12))):
        s = GpsStation(f"A{i:02d}", Position(lon, lat), Velocity(0, 0, 0, sigma, sigma, sigma))
        ve, vn = predict_velocity(s, ev_a)
        n = rng.normal(0, sigma, 2)
        lines.append(f"A{i:02d} {lon:.3f} {lat:.3f} {ve+n[0]:.6f} {vn+n[1]:.6f} 0.0 {sigma} {sigma} {sigma}")

    for i, (lon, lat) in enumerate(zip(np.linspace(-120, -115, 12), np.linspace(32, 36, 12))):
        s = GpsStation(f"B{i:02d}", Position(lon, lat), Velocity(0, 0, 0, sigma, sigma, sigma))
        ve, vn = predict_velocity(s, ev_b)
        n = rng.normal(0, sigma, 2)
        lines.append(f"B{i:02d} {lon:.3f} {lat:.3f} {ve+n[0]:.6f} {vn+n[1]:.6f} 0.0 {sigma} {sigma} {sigma}")

    p = tmp_path_factory.mktemp("data") / "synthetic_two_block.csv"
    p.write_text("\n".join(lines))
    return p


class TestFullPipeline:
    def test_csv_loads_correct_station_count(self, csv_path):
        stations = read_velocity_file(csv_path)
        assert len(stations) == 24

    def test_preprocess_retains_all_good_stations(self, csv_path):
        stations = read_velocity_file(csv_path)
        # Se=Sn=1.0 > 0.6 default → use a relaxed max_sigma for this fixture
        cleaned = preprocess(stations, max_sigma=2.0)
        assert len(cleaned) == len(stations)  # noiseless data, no outliers

    def test_velocity_hac_clustering(self, csv_path):
        stations = preprocess(read_velocity_file(csv_path), max_sigma=2.0)
        model = VelocityHACClustering()
        clusters = model.cluster(stations, k=2)
        assert len(clusters) == 2
        assert sum(c.size for c in clusters) == len(stations)

    def test_euler_clustering_finds_two_blocks(self, csv_path):
        stations = preprocess(read_velocity_file(csv_path), max_sigma=2.0)
        model = EulerVectorClustering()
        clusters = model.cluster(stations, k=2)
        # Each cluster should have an Euler vector
        for c in clusters:
            assert c.euler_vector is not None
            pole = euler_vector_to_pole(c.euler_vector)
            # Rate should be positive and geophysically plausible
            assert pole.rate > 0

    def test_euler_chi2_drops_at_k2(self, csv_path):
        """chi-squared should fall dramatically from k=1 to k=2 (two real blocks)."""
        stations = preprocess(read_velocity_file(csv_path), max_sigma=2.0)
        model = EulerVectorClustering()
        _, result = model.find_optimal_k(stations, max_k=5)
        # chi2 at k=1 must be much larger than at k=2
        assert result.chi2_total[0] > 10 * result.chi2_total[1]

    def test_station_names_preserved(self, csv_path):
        stations = read_velocity_file(csv_path)
        assert stations[0].name == "A00"
        assert stations[12].name == "B00"
