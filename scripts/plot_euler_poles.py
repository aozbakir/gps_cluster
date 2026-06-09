"""Euler poles figure — orthographic globe with 1σ/2σ uncertainty ellipses.

Loads the pre-computed k=5 solution from clusters.json (run
compute_anatolia_clusters.py first).  Pole uncertainties are estimated
by bootstrap resampling (N_BOOT iterations with replacement) of each
cluster's stations, giving realistic uncertainties that account for
block non-rigidity — not the over-optimistic formal WLS covariance which
is sub-millidegree and invisible at globe scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from gps_cluster.domain.entities import GpsStation, Position, Velocity
from gps_cluster.application.euler_clustering import bootstrap_pole_uncertainty
from gps_cluster.domain.services.euler_math import (
    EulerVector, euler_vector_to_pole, invert_euler_vector,
)

ROOT   = Path(__file__).parent.parent
CACHE  = ROOT / "reports/anatolia/clusters.json"
OUT    = ROOT / "reports/anatolia"

K_OPT  = 5
CMAP   = plt.colormaps["tab10"]
EXTENT = [25.0, 45.5, 35.5, 43.0]

if not CACHE.exists():
    raise FileNotFoundError(f"{CACHE} — run compute_anatolia_clusters.py first")

with open(CACHE) as f:
    cache = json.load(f)

N_BOOT = 300   # bootstrap iterations per cluster

stations = [
    GpsStation(r["name"], Position(r["lon"], r["lat"]),   # Position(lon, lat)
               Velocity(r["ve"], r["vn"], 0.0, r["se"], r["sn"], 1.0))
    for r in cache["stations"]
]
station_by_name = {s.name: s for s in stations}


# bootstrap_pole_uncertainty imported from gps_cluster.application.euler_clustering

clusters = []
for c in cache["solutions"][str(K_OPT)]:
    ev_d = c["euler"]
    ev   = EulerVector(ox=ev_d["ox"], oy=ev_d["oy"], oz=ev_d["oz"])
    pole = euler_vector_to_pole(ev)
    ns   = SimpleNamespace(
        id=c["id"], size=c["size"],
        euler_vector=ev, pole=pole,
        stations=[station_by_name[n] for n in c["stations"] if n in station_by_name],
    )
    clusters.append(ns)

clusters.sort(key=lambda c: c.id)

print("Computing bootstrap uncertainties …")
for c in clusters:
    s_lat, s_lon, s_rate = bootstrap_pole_uncertainty(c.stations, n_boot=N_BOOT)
    c.pole.sigma_lat  = s_lat
    c.pole.sigma_lon  = s_lon
    c.pole.sigma_rate = s_rate
    print(f"  C{c.id} N={c.size}: σ_lat={s_lat:.2f}° σ_lon={s_lon:.2f}° σ_rate={s_rate:.3f}°/Ma")

# ── helpers ────────────────────────────────────────────────────────────────────

def _small_circle(pole_lat, pole_lon, angular_radius_deg, n=360):
    """Lon/lat arrays of a small circle around (pole_lat, pole_lon)."""
    plat = np.radians(pole_lat)
    plon = np.radians(pole_lon)
    r    = np.radians(angular_radius_deg)
    az   = np.linspace(0, 2 * np.pi, n)
    slat = np.arcsin(np.cos(r)) * np.ones(n)
    slon = az

    x = np.cos(slat) * np.cos(slon)
    y = np.cos(slat) * np.sin(slon)
    z = np.sin(slat)

    def Ry(a):
        return np.array([[np.cos(a), 0, np.sin(a)],
                         [0,         1, 0         ],
                         [-np.sin(a),0, np.cos(a)]])
    def Rz(a):
        return np.array([[np.cos(a),-np.sin(a), 0],
                         [np.sin(a), np.cos(a), 0],
                         [0,         0,         1]])

    R   = Rz(plon) @ Ry(np.pi / 2 - plat)
    xyz = R @ np.vstack([x, y, z])
    lats = np.degrees(np.arcsin(np.clip(xyz[2], -1, 1)))
    lons = np.degrees(np.arctan2(xyz[1], xyz[0]))
    return lons, lats


def _mean_angular_radius(pole_lat, pole_lon, cluster_stations):
    plat = np.radians(pole_lat)
    plon = np.radians(pole_lon)
    dists = []
    for s in cluster_stations:
        slat = np.radians(s.position.lat)
        slon = np.radians(s.position.lon)
        cos_d = (np.sin(plat)*np.sin(slat) +
                 np.cos(plat)*np.cos(slat)*np.cos(slon - plon))
        dists.append(np.degrees(np.arccos(np.clip(cos_d, -1, 1))))
    return float(np.mean(dists)) if dists else 0.0


def _uncertainty_ellipse(pole_lat, pole_lon, sigma_lat, sigma_lon, scale=1.0, n=120):
    """Return (lons, lats) of an uncertainty ellipse in geographic space.

    Semi-axes are scale*sigma_lon (E-W) and scale*sigma_lat (N-S) in degrees.
    Longitude semi-axis accounts for cos(lat) compression so the ellipse
    looks circular in physical (km) space.
    """
    theta  = np.linspace(0, 2 * np.pi, n)
    cos_lat = max(np.cos(np.radians(pole_lat)), 0.01)
    dlons  = scale * sigma_lon / cos_lat * np.cos(theta)
    dlats  = scale * sigma_lat * np.sin(theta)
    return pole_lon + dlons, pole_lat + dlats


# ── figure ─────────────────────────────────────────────────────────────────────
ORTHO_LON, ORTHO_LAT = 12.0, 50.0

fig = plt.figure(figsize=(15, 10))

ax_globe = fig.add_axes([0.02, 0.05, 0.60, 0.90],
                        projection=ccrs.Orthographic(central_longitude=ORTHO_LON,
                                                     central_latitude=ORTHO_LAT))
ax_globe.set_global()
ax_globe.add_feature(cfeature.LAND,      facecolor="#f0ede6", zorder=0)
ax_globe.add_feature(cfeature.OCEAN,     facecolor="#cce4f0", zorder=0)
ax_globe.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="gray", zorder=1)
ax_globe.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="lightgray",
                     zorder=1, linestyle=":")
ax_globe.gridlines(linewidth=0.3, color="gray", alpha=0.5, linestyle="--")

# Anatolia domain box
lon0, lon1, lat0, lat1 = EXTENT
box_lons = [lon0, lon1, lon1, lon0, lon0]
box_lats = [lat0, lat0, lat1, lat1, lat0]
ax_globe.plot(box_lons, box_lats, transform=ccrs.PlateCarree(),
              color="k", linewidth=1.2, zorder=4, alpha=0.7)
ax_globe.fill(box_lons, box_lats, transform=ccrs.PlateCarree(),
              color="gold", alpha=0.15, zorder=3)

legend_handles = []

for c in clusters:
    col  = CMAP(c.id - 1)
    pole = c.pole

    # Small-circle arc (mean angular distance from pole to stations)
    r_deg = _mean_angular_radius(pole.lat, pole.lon, c.stations)
    try:
        sc_lons, sc_lats = _small_circle(pole.lat, pole.lon, r_deg)
        ax_globe.plot(sc_lons, sc_lats, transform=ccrs.Geodetic(),
                      color=col, linewidth=1.0, linestyle="--", alpha=0.45, zorder=3)
    except Exception:
        pass

    # 2σ ellipse (filled, light)
    if pole.sigma_lat > 0 and pole.sigma_lon > 0:
        e2_lons, e2_lats = _uncertainty_ellipse(
            pole.lat, pole.lon, pole.sigma_lat, pole.sigma_lon, scale=2.0)
        ax_globe.fill(e2_lons, e2_lats, transform=ccrs.PlateCarree(),
                      color=col, alpha=0.12, zorder=4)
        ax_globe.plot(e2_lons, e2_lats, transform=ccrs.PlateCarree(),
                      color=col, lw=0.7, linestyle=":", alpha=0.6, zorder=4)

        # 1σ ellipse (filled, medium)
        e1_lons, e1_lats = _uncertainty_ellipse(
            pole.lat, pole.lon, pole.sigma_lat, pole.sigma_lon, scale=1.0)
        ax_globe.fill(e1_lons, e1_lats, transform=ccrs.PlateCarree(),
                      color=col, alpha=0.22, zorder=5)
        ax_globe.plot(e1_lons, e1_lats, transform=ccrs.PlateCarree(),
                      color=col, lw=1.0, linestyle="-", alpha=0.75, zorder=5)

    # Pole star marker
    ax_globe.scatter(pole.lon, pole.lat,
                     transform=ccrs.PlateCarree(),
                     s=200, marker="*", color=col,
                     edgecolor="k", linewidth=0.8, zorder=6)

    # Label — offset to avoid overlap
    dlat = 2.0 if pole.lat < 60 else -3.0
    sigma_str = (f"σ=({pole.sigma_lat:.2f}°, {pole.sigma_lon:.2f}°)"
                 if pole.sigma_lat > 0 else "")
    ax_globe.text(
        pole.lon + 1.0, pole.lat + dlat,
        f"C{c.id}  {pole.lat:.1f}°N, {pole.lon:.1f}°E\n"
        f"{pole.rate:.2f}±{pole.sigma_rate:.3f} °/Ma\n{sigma_str}",
        transform=ccrs.PlateCarree(),
        fontsize=7, color=col, fontweight="bold", ha="left", va="bottom",
        zorder=7,
        bbox=dict(facecolor="white", alpha=0.70, edgecolor="none", pad=1.5),
    )

    legend_handles.append(Line2D(
        [0], [0], color=col, lw=0, marker="*", markersize=11,
        markeredgecolor="k", markeredgewidth=0.5,
        label=(f"C{c.id} (N={c.size})   "
               f"{pole.lat:.1f}°N, {pole.lon:.1f}°E,  "
               f"{pole.rate:.2f}±{pole.sigma_rate:.3f} °/Ma")))

# Ellipse scale legend
for sigma_fac, ls, lbl in [(1, "-", "1σ"), (2, ":", "2σ")]:
    legend_handles.append(Line2D([0], [0], color="gray", lw=1.2, linestyle=ls,
                                 label=f"{lbl} uncertainty ellipse"))

ax_globe.set_title(
    f"Euler poles — Euler-vector clustering k = {K_OPT}\n"
    "Stars = poles  ·  solid/dotted ellipses = 1σ / 2σ formal uncertainty\n"
    "Dashed arcs = mean small-circle radius per cluster  ·  Gold box = Anatolia GPS domain",
    fontsize=10)

# ── inset: Anatolia cluster map ────────────────────────────────────────────────
ax_inset = fig.add_axes([0.63, 0.08, 0.35, 0.42],
                         projection=ccrs.Mercator())
ax_inset.set_extent(EXTENT, crs=ccrs.PlateCarree())
ax_inset.add_feature(cfeature.LAND,      facecolor="#f5f1eb", zorder=0)
ax_inset.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
ax_inset.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="gray", zorder=1)
ax_inset.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="gray",
                     zorder=1, linestyle=":")
gl = ax_inset.gridlines(draw_labels=True, linewidth=0.25, color="gray",
                         alpha=0.4, linestyle="--", crs=ccrs.PlateCarree())
gl.top_labels = False; gl.right_labels = False
gl.xlocator = mticker.FixedLocator(range(26, 46, 6))
gl.ylocator = mticker.FixedLocator(range(36, 44, 3))
gl.xlabel_style = {"size": 6}; gl.ylabel_style = {"size": 6}

for c in clusters:
    lons = [s.position.lon for s in c.stations]
    lats = [s.position.lat for s in c.stations]
    ax_inset.scatter(lons, lats, s=8, color=CMAP(c.id - 1),
                     edgecolor="k", linewidths=0.2,
                     transform=ccrs.PlateCarree(), zorder=4)
ax_inset.set_title("Cluster assignments  (k = 5)", fontsize=8)

# ── legend panel ───────────────────────────────────────────────────────────────
ax_leg = fig.add_axes([0.63, 0.54, 0.35, 0.40])
ax_leg.axis("off")
ax_leg.legend(handles=legend_handles, loc="upper left",
              fontsize=7.5, framealpha=0.9,
              title="Cluster  |  Euler pole  ±  1σ rate",
              title_fontsize=8, handlelength=1.5)

fig.savefig(OUT / "fig_euler_poles.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}/fig_euler_poles.png")
