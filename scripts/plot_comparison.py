"""Three-way comparison figure.

Reads: results/anatolia/clusters.json   (written by compute_anatolia_clusters.py)
Writes: results/anatolia/fig_comparison.png

Panel A — Özdemir & Karslıoğlu (2019): GMM soft clustering, k=5 (Eurasia-fixed)
Panel B — Kılıç & Özarpacı (2022): MCLA ensemble clustering, k=5 (Eurasia-fixed)
Panel C — This study: VB Euler-vector clustering, k=5, raw ITRF velocities
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.image import imread
from matplotlib.lines import Line2D

from gps_cluster.domain.entities import EulerPole, GpsStation, Position, Velocity
from gps_cluster.domain.services.euler_math import EulerVector, predict_velocity

ROOT   = Path(__file__).parent.parent
CACHE  = ROOT / "results/anatolia/clusters.json"
OUT    = ROOT / "results/anatolia"

# Fault overlay — from project data/external (EMME fault map, GeoJSON)
FAULT_FILE = ROOT / "data/external/mta_emme_fault_map.geojson"

# Reference panel images (gitignored; script skips missing panels gracefully)
IMG_OZDEMIR = ROOT / "docs/references/OzdemirKarslioglu_final.png"
IMG_KILIC   = ROOT / "docs/references/KilicOzarpaci_final_b.png"

CMAP   = plt.colormaps["tab10"]
EXTENT = [25.0, 45.5, 35.5, 43.0]

if not CACHE.exists():
    raise FileNotFoundError(f"{CACHE} — run compute_anatolia_clusters.py first")

with open(CACHE) as f:
    _cache = json.load(f)

# ── reconstruct station & cluster objects ──────────────────────────────────────
stations = [
    GpsStation(r["name"], Position(r["lon"], r["lat"]),
               Velocity(r["ve"], r["vn"], 0.0, r["se"], r["sn"], 1.0))
    for r in _cache["stations"]
]
station_by_name = {s.name: s for s in stations}
N = len(stations)


def _load_solution(k: int) -> list:
    """Reconstruct cluster SimpleNamespaces from the JSON cache for a given k."""
    out = []
    for c in _cache["solutions"][str(k)]:
        ev_d = c["euler"]
        cov  = np.array(ev_d["covariance"]) if ev_d.get("covariance") else None
        ev   = EulerVector(ox=ev_d["ox"], oy=ev_d["oy"], oz=ev_d["oz"], covariance=cov)
        pd_  = c["pole"]
        pole = EulerPole(lat=pd_["lat"], lon=pd_["lon"], rate=pd_["rate"],
                         sigma_lat=pd_.get("sigma_lat", 0.0),
                         sigma_lon=pd_.get("sigma_lon", 0.0),
                         sigma_rate=pd_.get("sigma_rate", 0.0))
        ns = SimpleNamespace(
            id=c["id"], size=c["size"], chi2=c["chi2"],
            euler_vector=ev, pole=pole,
            stations=[station_by_name[n] for n in c["stations"] if n in station_by_name],
        )
        out.append(ns)
    return out


clusters = _load_solution(5)
print(f"Loaded k=5 solution from cache  (N={N} stations)")

# ── helpers ────────────────────────────────────────────────────────────────────
def _basemap(ax, extent=EXTENT):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#f5f1eb", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="gray", zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="gray", zorder=1, linestyle=":")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--", crs=ccrs.PlateCarree())
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(26, 46, 4))
    gl.ylocator = mticker.FixedLocator(range(36, 44, 2))
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}


def _load_turkey_polygon():
    shpfile = shpreader.natural_earth(resolution="10m", category="cultural",
                                      name="admin_0_countries")
    reader  = shpreader.Reader(shpfile)
    turkey  = next(c for c in reader.records() if c.attributes["NAME"] == "Turkey")
    return turkey.geometry


_TURKEY_POLY = _load_turkey_polygon()

_MAJOR_KEYWORDS = [
    "KUZEY ANADOLU", "DOGU ANADOLU", "ANTAKYA", "ÖLÜDENIZ",
    "BÜYÜK MENDERES", "GEDIZ GRABEN", "YENICE-GÖNEN", "MUSTAFA KEMAL",
    "ULUBAT FAYI", "BIGA-ÇAN", "MANYAS FAY", "IZNIK-MEKECE",
    "GEYVE FAYI", "DODURGA FAYI", "SULTANDAGI", "BURDUR GRABEN", "KOVADA FAYI",
]


def _is_major(sourcename) -> bool:
    if not sourcename or sourcename == "None":
        return False
    sn = str(sourcename).upper()
    return any(kw in sn for kw in _MAJOR_KEYWORDS)


def _plot_faults(ax):
    """Plot faults from EMME GeoJSON clipped to Turkey, major faults thicker."""
    if not FAULT_FILE.exists():
        return
    gdf = gpd.read_file(FAULT_FILE).to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notnull()]
    for _, row in gdf.iterrows():
        geom = row.geometry.intersection(_TURKEY_POLY)
        if geom.is_empty:
            continue
        major = _is_major(row.get("SOURCENAME"))
        lw    = 1.6 if major else 0.4
        color = "#222222" if major else "#888888"
        alpha = 0.9 if major else 0.55
        geoms = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for g in geoms:
            if g.geom_type not in ("LineString", "MultiLineString"):
                continue
            parts = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
            for part in parts:
                xs, ys = part.xy
                ax.plot(list(xs), list(ys), color=color, linewidth=lw,
                        transform=ccrs.PlateCarree(), zorder=5, alpha=alpha)


def _scatter(ax, st_list, color, s=20):
    lons = [s.position.lon for s in st_list]
    lats = [s.position.lat for s in st_list]
    ax.scatter(lons, lats, s=s, color=color,
               edgecolor="k", linewidths=0.25,
               transform=ccrs.PlateCarree(), zorder=4)


# ── figure layout ──────────────────────────────────────────────────────────────
# Row 0: A and B side by side (prior studies, if images present)
# Row 1: C full width (this study)
has_ozdemir = IMG_OZDEMIR.exists()
has_kilic   = IMG_KILIC.exists()

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.08,
                        height_ratios=[1, 1.2])

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :], projection=ccrs.Mercator())

# ── Panel A — Özdemir & Karslıoğlu ───────────────────────────────────────────
if has_ozdemir:
    ax_a.imshow(imread(IMG_OZDEMIR))
else:
    ax_a.text(0.5, 0.5, "Image not available\n(docs/references/ is gitignored)",
              ha="center", va="center", transform=ax_a.transAxes,
              fontsize=10, color="gray", style="italic")
ax_a.axis("off")
ax_a.set_title(
    "(a)  Özdemir & Karslıoğlu (2019) — GMM soft clustering, k = 5  (Eurasia-fixed)",
    fontsize=10, loc="left", fontweight="bold")

# ── Panel B — Kılıç & Özarpacı MCLA ─────────────────────────────────────────
if has_kilic:
    ax_b.imshow(imread(IMG_KILIC))
else:
    ax_b.text(0.5, 0.5, "Image not available\n(docs/references/ is gitignored)",
              ha="center", va="center", transform=ax_b.transAxes,
              fontsize=10, color="gray", style="italic")
ax_b.axis("off")
ax_b.set_title(
    "(b)  Kılıç & Özarpacı (2022) — MCLA ensemble clustering, k = 5  (Eurasia-fixed)",
    fontsize=10, loc="left", fontweight="bold")

# ── Panel C — This study ─────────────────────────────────────────────────────
_basemap(ax_c)
_plot_faults(ax_c)

legend_handles = []
sq = []
for c in clusters:
    col  = CMAP(c.id - 1)
    pole = c.pole
    _scatter(ax_c, c.stations, color=col)
    in_ext   = EXTENT[0] <= pole.lon <= EXTENT[1] and EXTENT[2] <= pole.lat <= EXTENT[3]
    pole_str = f"  pole {pole.lat:.0f}°N {pole.lon:.0f}°E" if not in_ext else ""
    legend_handles.append(
        Line2D([0], [0], color=col, lw=0, marker="o",
               markersize=8, markeredgecolor="k", markeredgewidth=0.4,
               label=f"Cluster {c.id}  (N={c.size}){pole_str}"))
    if c.euler_vector is not None:
        for s in c.stations:
            ve_p, vn_p = predict_velocity(s, c.euler_vector)
            sq.append((s.velocity.ve - ve_p)**2 + (s.velocity.vn - vn_p)**2)

rms = float(np.sqrt(np.mean(sq))) if sq else 0.0

ax_c.legend(handles=legend_handles, loc="upper left", fontsize=7.5, framealpha=0.9)
ax_c.set_title(
    f"(c)  This study — VB Euler-vector clustering, k = 5, raw ITRF velocities  "
    f"(N = {N},  RMS = {rms:.1f} mm/yr)",
    fontsize=10, loc="left", fontweight="bold")

fig.savefig(OUT / "fig_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {OUT}/fig_comparison.png")
