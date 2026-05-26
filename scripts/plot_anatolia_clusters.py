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

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage

from gps_cluster.application.euler_clustering import EulerVectorClustering
from gps_cluster.application.preprocess import preprocess
from gps_cluster.application.velocity_clustering import VelocityHACClustering
from gps_cluster.domain.services.euler_math import (
    euler_vector_to_pole,
    predict_velocity,
    total_chi_squared,
)
from gps_cluster.infrastructure.readers.velocity_vel import read_vel_file

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/raw/globk_vel_igs14_ITRF_M2E_11JAN2021_CMBND_improved_reformat.vel"
OUT  = ROOT / "reports/anatolia"
OUT.mkdir(parents=True, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
raw      = read_vel_file(DATA)
# Keep all stations: outlier removal by velocity magnitude would discard
# real tectonic signals (Arabia/Eurasia contrast).  Only flag excessive sigmas.
stations = preprocess(raw, max_sigma=99, zscore_threshold=99)
print(f"Stations: raw={len(raw)}, clean={len(stations)}")

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
hac      = VelocityHACClustering()
Z        = hac.fit(stations)
k_gap, gap_result = hac.find_optimal_k(stations, max_k=7, n_ref=30)
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
ax2.axvline(k_gap, color="tomato", ls="--", lw=1.5, label=f"Optimal k = {k_gap}")
ax2.set_xlabel("Number of clusters k", fontsize=11)
ax2.set_ylabel("Gap statistic", fontsize=11)
ax2.set_title("Tibshirani (2001) gap statistic", fontsize=11)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.legend(fontsize=9)
fig.suptitle("Velocity-space HAC — Anatolia GPS", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig3_hac_gap.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Euler chi² vs k (elbow / F-test)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 4: Euler chi² vs k …")
evc = EulerVectorClustering(init="multiscale", n_restarts=20, random_seed=0)
k_euler, ftest = evc.find_optimal_k(stations, max_k=7)
print(f"  F-test optimal k = {k_euler}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ks = ftest.k_values
ax1.plot(ks, ftest.chi2_reduced, "-o", lw=2,
         markerfacecolor="white", color="steelblue")
ax1.axhline(1, color="gray", ls="--", lw=1, label="χ²_red = 1")
ax1.set_xlabel("Number of clusters k", fontsize=11)
ax1.set_ylabel("Reduced χ²", fontsize=11)
ax1.set_title("Euler-vector clustering — fit quality", fontsize=11)
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.legend(fontsize=9)

ax2.plot(ks[1:], ftest.p_values, "-o", lw=2,
         markerfacecolor="white", color="tomato")
ax2.axhline(0.05, color="gray", ls="--", lw=1, label="α = 0.05")
ax2.set_xlabel("Number of clusters k", fontsize=11)
ax2.set_ylabel("F-test p-value  (k vs k+1)", fontsize=11)
ax2.set_title("Significance of adding one more cluster", fontsize=11)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.legend(fontsize=9)
fig.suptitle("Euler-vector clustering — Anatolia GPS (ITRF14)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig4_euler_chi2.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Map: best-k clusters + velocity arrows
# ═══════════════════════════════════════════════════════════════════════════════
print(f"Plotting Fig 5: k={k_euler} cluster map …")
clusters_best = evc.cluster(stations, k=k_euler)

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

chi2_best = sum(total_chi_squared(c.stations, c.euler_vector)
                for c in clusters_best if c.euler_vector is not None)
dof_best  = 2 * len(stations) - 3 * k_euler
rms_best  = _rms(clusters_best)
ax.set_title(f"Euler-vector clustering  k = {k_euler}\n"
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

res_lons, res_lats, res_dve, res_dvn = [], [], [], []
rms_obs_sq = []
for c in clusters_best:
    if c.euler_vector is None:
        continue
    col = CMAP(c.id - 1)
    _scatter(axes[0], c.stations, color=col, s=18)
    _quiver(axes[0], c.stations, color=col)
    for s in c.stations:
        ve_p, vn_p = predict_velocity(s, c.euler_vector)
        res_lons.append(s.position.lon)
        res_lats.append(s.position.lat)
        res_dve.append(s.velocity.ve - ve_p)
        res_dvn.append(s.velocity.vn - vn_p)
        rms_obs_sq.append(s.velocity.ve ** 2 + s.velocity.vn ** 2)

res_lons = np.array(res_lons)
res_lats = np.array(res_lats)
res_dve  = np.array(res_dve)
res_dvn  = np.array(res_dvn)
rms_obs  = np.sqrt(np.mean(rms_obs_sq))
rms_res  = np.sqrt(np.mean(res_dve ** 2 + res_dvn ** 2))

q_res = axes[1].quiver(
    res_lons, res_lats, res_dve, res_dvn,
    transform=ccrs.PlateCarree(),
    scale=40, scale_units="width",
    angles="uv",
    width=0.004, headwidth=4, headlength=5, headaxislength=4,
    minlength=0, minshaft=0.5,
    color="k", alpha=0.85, zorder=4,
)
axes[1].quiverkey(q_res, X=0.85, Y=0.06, U=10,
                  label="10 mm/yr", labelpos="S",
                  fontproperties={"size": 7})

_ref_arrow(axes[0], length=20)

axes[0].set_title(f"Observed velocities  (RMS = {rms_obs:.1f} mm/yr)", fontsize=10)
axes[1].set_title(f"Obs − Euler predicted  (RMS misfit = {rms_res:.1f} mm/yr)", fontsize=10)
fig.suptitle(f"Euler-vector clustering k = {k_euler} — Anatolia GPS (ITRF14)",
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
    clusters_k = evc.cluster(stations, k=k)
    all_clusters_k[k] = clusters_k
    for c in clusters_k:
        _scatter(ax, c.stations, color=CMAP(c.id - 1), s=10)
    rms_k = _rms(clusters_k)
    ax.set_title(f"k = {k}   rms = {rms_k:.2f} mm/yr", fontsize=10)

fig.suptitle("Euler-vector clustering — Anatolia GPS (ITRF14)\n"
             "Multiscale init, 20 restarts", fontsize=12)
fig.savefig(OUT / "fig7_k2to7_clusters.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nAll figures saved to {OUT}/")
