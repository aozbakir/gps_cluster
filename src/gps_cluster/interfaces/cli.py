"""Command-line interface for gps_cluster."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from gps_cluster.application.euler_clustering import EulerVectorClustering
from gps_cluster.application.preprocess import preprocess
from gps_cluster.application.velocity_clustering import VelocityHACClustering
from gps_cluster.domain.services.euler_math import euler_vector_to_pole
from gps_cluster.infrastructure.readers.velocity_csv import read_velocity_file

_logger = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False)
def cli(verbose: bool) -> None:
    """GPS velocity clustering tools."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--max-sigma", default=0.6, show_default=True,
              help="Maximum allowed 1-sigma uncertainty (mm/yr).")
@click.option("--zscore", default=2.0, show_default=True,
              help="Z-score threshold for outlier removal.")
def clean(input_file: Path, max_sigma: float, zscore: float) -> None:
    """Preprocess a velocity file and report station counts."""
    stations = read_velocity_file(input_file)
    click.echo(f"Loaded {len(stations)} stations from {input_file}")
    cleaned = preprocess(stations, max_sigma=max_sigma, zscore_threshold=zscore)
    click.echo(f"After preprocessing: {len(cleaned)} stations retained")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--algorithm", type=click.Choice(["velocity", "euler"]), default="euler",
              show_default=True, help="Clustering algorithm.")
@click.option("-k", "--num-clusters", default=None, type=int,
              help="Number of clusters. If omitted, optimal k is determined automatically.")
@click.option("--max-k", default=9, show_default=True, help="Maximum k to evaluate.")
@click.option("--plot", is_flag=True, default=False, help="Show diagnostic plots.")
def cluster(
    input_file: Path,
    algorithm: str,
    num_clusters: int | None,
    max_k: int,
    plot: bool,
) -> None:
    """Cluster GPS velocities using velocity-space HAC or Euler-vector iteration."""
    stations = read_velocity_file(input_file)
    stations = preprocess(stations)
    click.echo(f"Stations after preprocessing: {len(stations)}")

    if algorithm == "velocity":
        model = VelocityHACClustering()
        if num_clusters is None:
            k, gap_result = model.find_optimal_k(stations, max_k=max_k)
            click.echo(f"Optimal k (gap statistic) = {k}")
            if plot:
                from gps_cluster.infrastructure.visualization.dendrogram import plot_gap_statistic
                import matplotlib.pyplot as plt
                plot_gap_statistic(gap_result)
                plt.show()
        else:
            k = num_clusters
        clusters = model.cluster(stations, k)
        if plot:
            from gps_cluster.infrastructure.visualization.dendrogram import plot_dendrogram
            from gps_cluster.infrastructure.visualization.velocity_map import plot_velocity_scatter
            import matplotlib.pyplot as plt
            plot_dendrogram(model.fit(stations))
            plot_velocity_scatter(clusters)
            plt.show()

    else:  # euler
        model = EulerVectorClustering()
        if num_clusters is None:
            k, ftest_result = model.find_optimal_k(stations, max_k=max_k)
            click.echo(f"Optimal k (F-test) = {k}")
            if plot:
                from gps_cluster.infrastructure.visualization.dendrogram import plot_ftest
                import matplotlib.pyplot as plt
                plot_ftest(ftest_result)
                plt.show()
        else:
            k = num_clusters
        clusters = model.cluster(stations, k)

    _print_cluster_summary(clusters)

    if plot:
        from gps_cluster.infrastructure.visualization.velocity_map import plot_clusters
        import matplotlib.pyplot as plt
        plot_clusters(clusters)
        plt.show()


def _print_cluster_summary(clusters) -> None:
    click.echo(f"\n{'Cluster':>8}  {'N':>5}  {'Pole lat':>10}  {'Pole lon':>10}  {'Rate °/Myr':>12}")
    click.echo("-" * 55)
    for c in clusters:
        if c.euler_vector is not None:
            pole = euler_vector_to_pole(c.euler_vector)
            click.echo(
                f"{c.id:>8}  {c.size:>5}  {pole.lat:>10.2f}  {pole.lon:>10.2f}  {pole.rate:>12.3f}"
            )
        else:
            click.echo(f"{c.id:>8}  {c.size:>5}  {'—':>10}  {'—':>10}  {'—':>12}")
