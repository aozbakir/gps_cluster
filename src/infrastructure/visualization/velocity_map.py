"""Cartopy-based map visualisations for GPS velocity fields and clusters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure


def _get_extent(stations, pad: float = 2.0) -> list[float]:
    lons = [s.position.lon for s in stations]
    lats = [s.position.lat for s in stations]
    return [min(lons) - pad, max(lons) + pad, min(lats) - pad, max(lats) + pad]


def plot_velocity_field(
    stations,
    extent: list[float] | None = None,
    scale: float = 1.0,
    title: str = "GPS velocity field",
) -> "Figure":
    """Quiver map of GPS horizontal velocities.

    Parameters
    ----------
    stations:
        List of GpsStation objects.
    extent:
        [lon_min, lon_max, lat_min, lat_max]. Auto-computed if None.
    scale:
        Arrow scale factor passed to quiver.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(10, 8), subplot_kw={"projection": ccrs.Mercator()}
    )

    if extent is None:
        extent = _get_extent(stations)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(color="gray", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="ivory")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    lons = [s.position.lon for s in stations]
    lats = [s.position.lat for s in stations]
    ve = [s.velocity.ve for s in stations]
    vn = [s.velocity.vn for s in stations]

    ax.quiver(
        lons, lats, ve, vn,
        transform=ccrs.PlateCarree(),
        scale=scale,
        scale_units="xy",
        color="steelblue",
        alpha=0.8,
    )
    ax.set_title(title, fontsize=13)
    return fig


def plot_clusters(
    clusters,
    extent: list[float] | None = None,
    show_euler_poles: bool = True,
    title: str = "GPS clusters",
) -> "Figure":
    """Map view coloured by cluster membership, with optional Euler pole markers.

    Parameters
    ----------
    clusters:
        List of VelocityCluster objects.
    show_euler_poles:
        If True, mark the Euler pole location for each cluster that has one.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    from gps_cluster.domain.services.euler_math import euler_vector_to_pole

    all_stations = [s for c in clusters for s in c.stations]
    if extent is None:
        extent = _get_extent(all_stations)

    fig, ax = plt.subplots(
        figsize=(10, 8), subplot_kw={"projection": ccrs.Mercator()}
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(color="gray", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="ivory")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    cmap = get_cmap("tab10")
    for c in clusters:
        color = cmap(c.id % 10)
        lons = [s.position.lon for s in c.stations]
        lats = [s.position.lat for s in c.stations]
        ve = [s.velocity.ve for s in c.stations]
        vn = [s.velocity.vn for s in c.stations]
        ax.quiver(
            lons, lats, ve, vn,
            transform=ccrs.PlateCarree(),
            color=color,
            alpha=0.85,
            label=f"Cluster {c.id} (n={c.size})",
        )

        if show_euler_poles and c.euler_vector is not None:
            pole = euler_vector_to_pole(c.euler_vector)
            if extent[0] <= pole.lon <= extent[1] and extent[2] <= pole.lat <= extent[3]:
                ax.plot(
                    pole.lon, pole.lat,
                    marker="*",
                    markersize=14,
                    color=color,
                    transform=ccrs.PlateCarree(),
                    markeredgecolor="k",
                    markeredgewidth=0.5,
                    zorder=5,
                )
                ax.text(
                    pole.lon + 0.3, pole.lat + 0.3,
                    f"E{c.id}\n{pole.rate:.2f}°/Myr",
                    transform=ccrs.PlateCarree(),
                    fontsize=8,
                    color=color,
                )

    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(title, fontsize=13)
    return fig


def plot_velocity_scatter(clusters) -> "Figure":
    """Scatter plot of (Ve, Vn) coloured by cluster, with confidence ellipses."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.cm import get_cmap
    from matplotlib.patches import Ellipse
    from matplotlib.transforms import Affine2D

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = get_cmap("tab10")

    for c in clusters:
        color = cmap(c.id % 10)
        ve = np.array([s.velocity.ve for s in c.stations])
        vn = np.array([s.velocity.vn for s in c.stations])
        ax.scatter(ve, vn, color=color, edgecolor="k", linewidths=0.5, s=40, alpha=0.8)

        if len(ve) > 2:
            cov = np.cov(ve, vn)
            pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
            ell = Ellipse(
                (0, 0),
                width=np.sqrt(1 + pearson) * 2,
                height=np.sqrt(1 - pearson) * 2,
                facecolor=color,
                alpha=0.2,
                edgecolor=color,
            )
            sx = np.sqrt(cov[0, 0]) * 2
            sy = np.sqrt(cov[1, 1]) * 2
            t = Affine2D().rotate_deg(45).scale(sx, sy).translate(ve.mean(), vn.mean())
            ell.set_transform(t + ax.transData)
            ax.add_patch(ell)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Ve (mm/yr)", fontsize=12)
    ax.set_ylabel("Vn (mm/yr)", fontsize=12)
    ax.set_title("Velocity scatter by cluster", fontsize=13)
    return fig
