"""Euler-vector clustering for Anatolia/Turkey GPS velocities (ITRF14).

Applies the Savage (2018) workflow to a GLOBK combined velocity solution
covering Turkey and surrounding region (836 stations, Jan 2021).

Generates:
  Fig 1 — Raw velocity field
  Fig 2 — Velocity scatter (raw)
  Fig 3 — HAC dendrogram + gap statistic
  Fig 4 — Euler chi² vs k (F-test elbow)
  Fig 5 — Map: best-k clusters with velocity arrows + Euler poles
  Fig 6 — Map: residual vectors (observed − Euler-predicted)
  Fig 7 — Cluster comparison grid: k = 2..7

All figures saved to reports/anatolia/.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import ConvexHull

from gps_cluster.domain.entities import GpsStation, Position, Velocity
from gps_cluster.domain.services.euler_math import (
    EulerVector,
    euler_vector_to_pole,
    predict_velocity,
)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
CACHE      = ROOT / "reports/anatolia/clusters.json"
FAULT_FILE = ROOT / "data/raw/anatolia_slip_rate_faults_simplified.geojson"
OUT        = ROOT / "reports/anatolia"
OUT.mkdir(parents=True, exist_ok=True)

# ── load cache ────────────────────────────────────────────────────────────────
# LN 33-35 / 50-53 core issue: EulerVectorClustering.find_optimal_k() and
# VelocityHACClustering.find_optimal_k() were called at plot time (minutes).
# Fix: read gap/ftest/solutions from clusters.json written by compute script.
if not CACHE.exists():
    raise FileNotFoundError(f"{CACHE} — run compute_anatolia_clusters.py first")

with open(CACHE) as _f:
    _cache = json.load(_f)

# LN 50-53: read_vel_file + preprocess replaced by cache stations.
stations = [
    GpsStation(name=r["name"], position=Position(lat=r["lat"], lon=r["lon"]),
               velocity=Velocity(ve=r["ve"], vn=r["vn"], vu=0.0,
                                 se=r["se"], sn=r["sn"], su=1.0))
    for r in _cache["stations"]
]
station_by_name = {s.name: s for s in stations}
print(f"Stations loaded from cache: {len(stations)}")


def _load_solution(k: int):
    """Reconstruct cluster list from JSON cache for given k."""
    sol = []
    for c in _cache["solutions"][str(k)]:
        ev = EulerVector(ox=c["euler"]["ox"], oy=c["euler"]["oy"], oz=c["euler"]["oz"])
        sol.append(SimpleNamespace(
            id=c["id"], size=c["size"], chi2=c["chi2"],
            euler_vector=ev,
            pole=SimpleNamespace(**c["pole"]),
            stations=[station_by_name[n] for n in c["stations"] if n in station_by_name],
        ))
    return sol

EXTENT = [25.0, 45.5, 35.5, 43.0]   # [lon_min, lon_max, lat_min, lat_max]
CMAP   = plt.colormaps["tab10"]

# ── Earth radius for ω-space conversion ──────────────────────────────────────
_R_MM = 6_371_000.0 * 1_000.0  # mm

def _omega_deg_per_ma(euler_vec) -> np.ndarray:
    """Convert EulerVector (mm/yr) → Cartesian (ωx, ωy, ωz) in °/Ma."""
    return euler_vec.to_array() / _R_MM * np.degrees(1) * 1e6


# ── map helpers ───────────────────────────────────────────────────────────────

def _basemap(ax, extent=EXTENT):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#f5f1eb", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="gray",  zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="gray",  zorder=1, linestyle=":")
    ax.add_feature(cfeature.RIVERS,    linewidth=0.3, edgecolor="#aed6f1", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                      alpha=0.6, linestyle="--", crs=ccrs.PlateCarree())
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(26, 46, 2))
    gl.ylocator = mticker.FixedLocator(range(36, 44, 1))
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    return ax


def _quiver(ax, stations_list, color, scale=200, **kw):
    """GPS velocity arrows using scale_units='width' (avoids Mercator circle bug)."""
    lons = np.array([s.position.lon for s in stations_list])
    lats = np.array([s.position.lat for s in stations_list])
    ve   = np.array([s.velocity.ve  for s in stations_list])
    vn   = np.array([s.velocity.vn  for s in stations_list])
    q = ax.quiver(lons, lats, ve, vn,
                  transform=ccrs.PlateCarree(),
                  scale=scale, scale_units="width",
                  angles="uv",
                  width=0.003, headwidth=4, headlength=5, headaxislength=4,
                  minlength=0, minshaft=0.5,
                  color=color, alpha=0.9, zorder=3, **kw)
    return q


def _scatter(ax, stations_list, color, s=28):
    lons = np.array([s.position.lon for s in stations_list])
    lats = np.array([s.position.lat for s in stations_list])
    ax.scatter(lons, lats, s=s, color=color,
               edgecolor="k", linewidths=0.3,
               transform=ccrs.PlateCarree(), zorder=4)


def _ref_arrow(ax, length=20, scale=200, label=True):
    """Reference scale bar via quiverkey.  Anchor inside map extent (zorder=-1)."""
    _q = ax.quiver(np.array([35.0]), np.array([36.5]),
                   np.array([float(length)]), np.array([0.0]),
                   transform=ccrs.PlateCarree(),
                   scale=scale, scale_units="width",
                   angles="uv",
                   width=0.003, headwidth=4, headlength=5,
                   color="k", zorder=-1)
    if label:
        ax.quiverkey(_q, X=0.85, Y=0.06, U=length,
                     label=f"{length} mm/yr", labelpos="S",
                     fontproperties={"size": 7})


def _pole_marker(ax, pole, color, label=""):
    """Draw Euler pole symbol only when it falls inside the map extent."""
    in_extent = (EXTENT[0] <= pole.lon <= EXTENT[1] and
                 EXTENT[2] <= pole.lat <= EXTENT[3])
    if not in_extent:
        return
    ax.scatter(pole.lon, pole.lat,
               marker="*", s=200, color=color,
               edgecolor="k", linewidth=0.8,
               transform=ccrs.PlateCarree(), zorder=6)
    if label:
        ax.text(pole.lon + 0.15, pole.lat + 0.1, label,
                transform=ccrs.PlateCarree(), fontsize=7,
                color=color, fontweight="bold", zorder=7)


def _cluster_hull(ax, cluster_stations, color):
    """Draw convex hull of cluster station positions as a thick dashed outline."""
    if len(cluster_stations) < 3:
        return
    pts = np.array([[s.position.lon, s.position.lat] for s in cluster_stations])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return
    hull_pts = pts[hull.vertices]
    closed = np.vstack([hull_pts, hull_pts[0]])
    ax.plot(closed[:, 0], closed[:, 1],
            transform=ccrs.PlateCarree(),
            color=color, linewidth=2.2, linestyle="--",
            alpha=0.85, zorder=3)


def _rms(clusters_list):
    sq = []
    for c in clusters_list:
        if c.euler_vector is None:
            continue
        for s in c.stations:
            ve_p, vn_p = predict_velocity(s, c.euler_vector)
            sq.append((s.velocity.ve - ve_p) ** 2 + (s.velocity.vn - vn_p) ** 2)
    return np.sqrt(np.mean(sq)) if sq else np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Raw velocity field
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 1: raw velocity field …")
fig, ax = plt.subplots(figsize=(14, 8),
                       subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)
_quiver(ax, stations, color="steelblue")
_ref_arrow(ax)
ax.set_title("Anatolia GPS velocities (ITRF14)\n"
             f"N = {len(stations)} stations", fontsize=12)
fig.savefig(OUT / "fig1_velocity_field.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Velocity scatter
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 2: velocity scatter …")
fig, ax = plt.subplots(figsize=(7, 7))
ve_all = [s.velocity.ve for s in stations]
vn_all = [s.velocity.vn for s in stations]
ax.scatter(ve_all, vn_all, s=30, edgecolor="steelblue", facecolor="white",
           linewidths=0.8, alpha=0.8)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.axvline(0, color="gray", lw=0.5, ls="--")
ax.set_xlabel("Ve  (mm/yr)", fontsize=11)
ax.set_ylabel("Vn  (mm/yr)", fontsize=11)
ax.set_title(f"Velocity space — Anatolia GPS (ITRF14, N={len(stations)})", fontsize=11)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT / "fig2_velocity_scatter.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — HAC dendrogram + gap statistic
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 3: HAC dendrogram + gap statistic …")
# LN 207-209 core issue: VelocityHACClustering.find_optimal_k() reran gap
# statistic (n_ref=30 Monte Carlo references) — ~minutes.
# Fix: read gap results from cache; recompute only linkage Z for dendrogram
# (scipy linkage on 836 pts is sub-second — no cache entry for Z).
_vel_arr = np.array([[s.velocity.ve, s.velocity.vn] for s in stations])
Z        = linkage(_vel_arr, method="centroid", metric="euclidean")
gap_result = SimpleNamespace(**_cache["gap"])
k_gap      = gap_result.k_max_gap
print(f"  Gap optimal k = {k_gap}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
dendrogram(Z, ax=ax1, no_labels=True, color_threshold=0,
           above_threshold_color="steelblue", leaf_font_size=6)
ax1.set_xlabel("Station index", fontsize=11)
ax1.set_ylabel("Linkage distance  (mm/yr)", fontsize=11)
ax1.set_title("HAC dendrogram (centroid linkage)", fontsize=11)

ks = gap_result.k_values
ax2.errorbar(ks, gap_result.gap, yerr=gap_result.sk,
             fmt="-o", capsize=4, linewidth=1.8,
             markerfacecolor="white", color="steelblue", label="Gap(k) ± s_k")
ax2.axvline(k_gap, color="tomato", ls="--", lw=1.5, label=f"Max-gap k = {k_gap}")
ax2.set_xlabel("Number of clusters k", fontsize=11)
ax2.set_ylabel("Gap statistic", fontsize=11)
ax2.set_title("Gap statistic — max-gap criterion", fontsize=11)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.legend(fontsize=9)
fig.suptitle("Velocity-space HAC — Anatolia GPS", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig3_hac_gap.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Euler chi²_red vs k  +  marginal improvement (Δchi²_red)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 4: Euler chi² vs k …")
# LN 239-240 core issue: EulerVectorClustering.find_optimal_k() re-ran all
# k=1..7 with 20 multiscale restarts — the dominant slow step (~minutes).
# Fix: read ftest chi2_reduced from cache; solutions already cached too.
ftest = SimpleNamespace(**_cache["ftest"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ks = ftest.k_values  # [1, 2, ..., 7]

# Left: chi²_red on log scale.  Log scale reveals the elbow region (k=4-6)
# that is compressed when large k=1 value dominates a linear axis.
ax1.semilogy(ks, ftest.chi2_reduced, "-o", lw=2,
             markerfacecolor="white", color="steelblue")
ax1.axhline(1, color="gray", ls="--", lw=1, label="χ²_red = 1")
ax1.axvline(k_gap, color="tomato", ls="--", lw=1.5,
            label=f"Gap statistic k = {k_gap}")
ax1.set_xlabel("Number of clusters k", fontsize=11)
ax1.set_ylabel("Reduced χ²  (log scale)", fontsize=11)
ax1.set_title("Euler-vector clustering — fit quality", fontsize=11)
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.legend(fontsize=9)

# Right: marginal improvement Δchi²_red = chi²_red(k-1) − chi²_red(k).
# RMS always decreases with k (guaranteed); chi²_red accounts for the
# extra 3 parameters per cluster.  The sharp drop after k=k_gap shows
# where additional clusters stop buying meaningful fit improvement.
# F-test p-values are useless here: with N=836 stations all p ≈ 0
# regardless of k, so the panel is flat and contains no information.
improvement = -np.diff(ftest.chi2_reduced)   # positive: chi2_red(k-1) - chi2_red(k)
k_bars      = np.array(ks[1:])               # k = 2..7

ax2.bar(k_bars, improvement, color="steelblue", alpha=0.8,
        edgecolor="k", linewidth=0.5)
ax2.set_yscale("log")
ax2.axvline(k_gap, color="tomato", ls="--", lw=1.5,
            label=f"Gap statistic k = {k_gap}")
ax2.set_xlabel("Number of clusters k", fontsize=11)
ax2.set_ylabel("Δχ²_red  (log scale)", fontsize=11)
ax2.set_title("Marginal improvement per added cluster\n"
              "Elbow = natural cluster number", fontsize=11)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.legend(fontsize=9)

fig.suptitle("Euler-vector clustering — Anatolia GPS (ITRF14)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig4_euler_chi2.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Map: best-k clusters + velocity arrows
# ═══════════════════════════════════════════════════════════════════════════════
print(f"Plotting Fig 5: k={k_gap} cluster map (gap statistic) …")
clusters_best = _load_solution(k_gap)

fig, ax = plt.subplots(figsize=(16, 9),
                       subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)

legend_handles = []
for c in clusters_best:
    col  = CMAP(c.id - 1)
    pole = euler_vector_to_pole(c.euler_vector)
    _scatter(ax, c.stations, color=col, s=25)
    _quiver(ax, c.stations, color=col)
    _pole_marker(ax, pole, color=col,
                 label=f"E{c.id} ({pole.lat:.0f}°N,{pole.lon:.0f}°E)\n{pole.rate:.2f}°/Myr")
    in_ext = EXTENT[0] <= pole.lon <= EXTENT[1] and EXTENT[2] <= pole.lat <= EXTENT[3]
    pole_str = f"  pole {pole.lat:.0f}°N {pole.lon:.0f}°E" if not in_ext else ""
    legend_handles.append(Line2D([0], [0], color=col, lw=0, marker="o",
                                 markersize=9, markeredgecolor="k", markeredgewidth=0.5,
                                 label=f"Cluster {c.id}  (N={c.size}){pole_str}"))

_ref_arrow(ax)
ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)

chi2_best = sum(c.chi2 for c in clusters_best if c.chi2 is not None)
dof_best  = 2 * len(stations) - 3 * k_gap
rms_best  = _rms(clusters_best)
ax.set_title(f"Euler-vector clustering  k = {k_gap}  (gap statistic)\n"
             f"RMS = {rms_best:.1f} mm/yr   χ²_red = {chi2_best/dof_best:.0f}",
             fontsize=12)
fig.savefig(OUT / "fig5_clusters_best_k.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Residual velocity vectors
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 6: residual velocities …")
fig, axes = plt.subplots(1, 2, figsize=(22, 9),
                         subplot_kw={"projection": ccrs.Mercator()})
_basemap(axes[0])
_basemap(axes[1])

if FAULT_FILE.exists():
    _faults_gdf = gpd.read_file(FAULT_FILE).to_crs("EPSG:4326")
    for _, row in _faults_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        segs = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for seg in segs:
            xs, ys = seg.xy
            axes[0].plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
                         color="black", lw=1.2, alpha=0.7, zorder=3)

res_lons, res_lats, res_dve, res_dvn = [], [], [], []
for c in clusters_best:
    if c.euler_vector is None:
        continue
    col = CMAP(c.id - 1)
    _scatter(axes[0], c.stations, color=col, s=18)
    for s in c.stations:
        ve_p, vn_p = predict_velocity(s, c.euler_vector)
        res_lons.append(s.position.lon)
        res_lats.append(s.position.lat)
        res_dve.append(s.velocity.ve - ve_p)
        res_dvn.append(s.velocity.vn - vn_p)

res_lons = np.array(res_lons)
res_lats = np.array(res_lats)
res_dve  = np.array(res_dve)
res_dvn  = np.array(res_dvn)
rms_res  = np.sqrt(np.mean(res_dve ** 2 + res_dvn ** 2))

q_res = axes[1].quiver(
    res_lons, res_lats, res_dve, res_dvn,
    transform=ccrs.PlateCarree(),
    scale=80, scale_units="width",
    angles="uv",
    width=0.004, headwidth=4, headlength=5, headaxislength=4,
    minlength=0, minshaft=0.5,
    color="k", alpha=0.85, zorder=4,
)
axes[1].quiverkey(q_res, X=0.85, Y=0.06, U=5,
                  label="5 mm/yr", labelpos="S",
                  fontproperties={"size": 7})

axes[0].set_title(f"Cluster assignment  (k = {k_gap})", fontsize=10)
axes[1].set_title(f"Obs − Euler predicted  (RMS misfit = {rms_res:.1f} mm/yr)", fontsize=10)
fig.suptitle(f"Euler-vector clustering k = {k_gap} — Anatolia GPS (ITRF14)",
             fontsize=12)
fig.savefig(OUT / "fig6_residuals.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — 4×2 grid: k = 2..7  (dots only)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 7: k = 2–7 cluster maps …")
all_clusters_k = {}
fig, axes7 = plt.subplots(3, 2, figsize=(16, 18),
                           subplot_kw={"projection": ccrs.Mercator()})
axes7 = axes7.flatten()

for ax, k in zip(axes7, range(2, 8)):
    _basemap(ax)
    clusters_k = _load_solution(k)
    all_clusters_k[k] = clusters_k
    for c in clusters_k:
        _scatter(ax, c.stations, color=CMAP(c.id - 1), s=10)
    rms_k = _rms(clusters_k)
    ax.set_title(f"k = {k}   rms = {rms_k:.2f} mm/yr", fontsize=10)

fig.suptitle("Euler-vector clustering — Anatolia GPS (ITRF14)\n"
             "Multiscale init, 20 restarts", fontsize=12)
fig.savefig(OUT / "fig7_k2to7_clusters.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8 — EM soft-assignment entropy map
# Loaded from the JSON cache (computed by compute_anatolia_clusters.py).
# EM entropy reflects genuine kinematic ambiguity — stations near block
# boundaries get soft weights rather than a hard assignment, so their
# entropy is elevated even when the hard clustering is confident.
# ═══════════════════════════════════════════════════════════════════════════════
# _cache already loaded at module top — no re-read needed.
k_em   = _cache["em"]["k"]
_em_entropy_by_name = {
    r["name"]: r["em_entropy"]
    for r in _cache["stations"]
    if "em_entropy" in r
}
entropy     = np.array([_em_entropy_by_name.get(s.name, 0.0) for s in stations])
max_entropy = np.log(k_em)   # theoretical maximum for k_em clusters

print(f"Plotting Fig 8: EM entropy map (k_em={k_em}) …")

fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)

lons = np.array([s.position.lon for s in stations])
lats = np.array([s.position.lat for s in stations])

sc = ax.scatter(lons, lats,
                c=entropy, cmap="RdYlGn_r",
                vmin=0, vmax=max_entropy,
                s=22, edgecolors="k", linewidths=0.2,
                transform=ccrs.PlateCarree(), zorder=4)

cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("EM soft-assignment entropy  (nats)\n0 = certain   log(k) = fully ambiguous",
               fontsize=9)
cbar.set_ticks([0, max_entropy / 2, max_entropy])
cbar.set_ticklabels(["0\n(certain)", f"{max_entropy/2:.2f}",
                     f"{max_entropy:.2f}\n(uniform)"])

# Overlay fault traces
if FAULT_FILE.exists():
    _faults_gdf = gpd.read_file(FAULT_FILE).to_crs("EPSG:4326")
    for _, row in _faults_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        segs = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for seg in segs:
            xs, ys = seg.xy
            ax.plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
                    color="black", lw=1.0, alpha=0.6, zorder=5)

high_ent_pct = int(100 * np.mean(entropy > 0.5 * max_entropy))
ax.set_title(
    f"EM soft-assignment entropy — k = {k_em}  (ITRF14)\n"
    f"Red = ambiguous (≥½ max entropy: {high_ent_pct}% of stations)   "
    f"Green = certain   max entropy = log({k_em}) = {max_entropy:.2f} nats",
    fontsize=11)
fig.savefig(OUT / "fig8_entropy.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"\nAll figures saved to {OUT}/")
print(f"  Mean EM entropy: {entropy.mean():.3f} / {max_entropy:.3f} nats  "
      f"({100*entropy.mean()/max_entropy:.0f}% of max)")
