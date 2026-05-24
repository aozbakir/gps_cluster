"""Reproduce figures comparable to Savage (2018) for southwest Japan GPS clustering.

Generates:
  Fig 1 — Raw velocity field (ITRF2000)
  Fig 2 — Velocity scatter (Ve vs Vn), raw and after preprocessing
  Fig 3 — HAC dendrogram + gap statistic → optimal k
  Fig 4 — Euler-vector chi² vs k (F-test elbow)
  Fig 5 — Map: k=3 Euler clusters with velocity arrows + Euler poles
  Fig 6 — Map: k=3 clusters, residual vectors (observed − predicted)
  Fig 7 — Cluster comparison grid: k = 2, 3, 4, 5

All figures saved to reports/figures/.
"""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage

from gps_cluster.application.euler_clustering import EulerVectorClustering
from gps_cluster.application.preprocess import preprocess
from gps_cluster.application.velocity_clustering import VelocityHACClustering
from gps_cluster.domain.services.euler_math import euler_vector_to_pole, predict_velocity, total_chi_squared
from gps_cluster.infrastructure.readers.velocity_csv import read_velocity_file

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/raw/gji_4600_SI_TablesS1.csv"
OUT  = ROOT / "reports/figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
raw      = read_velocity_file(DATA)
# Paper (Savage 2018) uses all 469 stations from the supplementary table with
# minimal preprocessing.  Use max_sigma=99 to keep all 468 stations we have.
stations = preprocess(raw, max_sigma=99, zscore_threshold=99)
print(f"Stations: raw={len(raw)}, clean={len(stations)}")

EXTENT = [128.0, 139.5, 31.0, 38.5]   # [lon_min, lon_max, lat_min, lat_max]
CMAP   = plt.colormaps["tab10"]


def _geo_init_labels(stations_list, k):
    """Geographic block initialization for SW Japan (Savage 2018 comparison).

    In strongly loaded subduction-zone datasets the velocity-HAC initialization
    converges to iso-velocity bands rather than tectonic plates.  This function
    seeds clusters by velocity regime so the iterative Euler algorithm starts
    in the geographically coherent basin:

      * Vn < -20 mm/yr → Ryukyu high-loading zone (separate seed)
      * Ve > +20 mm/yr (and Vn >= -20) → fast-eastward zone (separate seed)
      * Remainder → main block

    For k > 3 the main block is further split by velocity-HAC.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(stations_list)
    vns = np.array([s.velocity.vn for s in stations_list])
    ves = np.array([s.velocity.ve  for s in stations_list])

    if k == 1 or n < 10:
        X = np.array([[s.velocity.ve, s.velocity.vn] for s in stations_list])
        return fcluster(linkage(X, method="centroid"), t=k, criterion="maxclust")

    # Identify the two distinctive velocity regimes
    is_loading  = vns < -20.0                        # Ryukyu extreme southward
    is_fast_e   = (~is_loading) & (ves > 20.0)      # fast-eastward (not already loading)
    is_main     = ~(is_loading | is_fast_e)

    labels = np.zeros(n, dtype=int)
    labels[is_main]    = 1
    labels[is_loading] = 2
    if k == 2:
        labels[is_fast_e] = 1   # merge fast-east into main for k=2
        return labels
    labels[is_fast_e]  = 3
    if k == 3:
        return labels

    # k >= 4: further split the main block by velocity HAC
    main_idx = np.where(is_main)[0]
    X_main = np.array([[stations_list[i].velocity.ve,
                        stations_list[i].velocity.vn] for i in main_idx])
    sub = fcluster(linkage(X_main, method="centroid"),
                   t=k - 2, criterion="maxclust")
    labels[main_idx] = sub + 2                       # shift past clusters 1, 2
    # Renumber labels 1..k consecutively
    for new_id, old_id in enumerate(np.unique(labels), start=1):
        labels[labels == old_id] = -new_id
    labels = -labels
    return labels

# ── helpers ───────────────────────────────────────────────────────────────────

def _basemap(ax, extent=EXTENT):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,       facecolor="#f5f1eb", zorder=0)
    ax.add_feature(cfeature.OCEAN,      facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.7, edgecolor="gray",  zorder=1)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.4, edgecolor="gray",  zorder=1, linestyle=":")
    ax.add_feature(cfeature.RIVERS,     linewidth=0.3, edgecolor="#aed6f1", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                      alpha=0.6, linestyle="--", crs=ccrs.PlateCarree())
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(128, 141, 2))
    gl.ylocator = mticker.FixedLocator(range(31, 39, 1))
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    return ax


def _quiver(ax, stations_list, color, scale=20, **kw):
    """scale=20 → 20 mm/yr arrow = 1° long on the map."""
    lons = np.array([s.position.lon for s in stations_list])
    lats = np.array([s.position.lat for s in stations_list])
    ve   = np.array([s.velocity.ve  for s in stations_list])
    vn   = np.array([s.velocity.vn  for s in stations_list])
    q = ax.quiver(lons, lats, ve, vn,
                  transform=ccrs.PlateCarree(),
                  scale=scale, scale_units="xy",
                  width=0.004, headwidth=5, headlength=6,
                  color=color, alpha=0.9, zorder=3, **kw)
    return q


def _scatter(ax, stations_list, color, s=28):
    """Plot colored station dots on a cartopy map."""
    lons = np.array([s.position.lon for s in stations_list])
    lats = np.array([s.position.lat for s in stations_list])
    ax.scatter(lons, lats, s=s, color=color,
               edgecolor="k", linewidths=0.3,
               transform=ccrs.PlateCarree(), zorder=4)


def _ref_arrow(ax, lon=138.5, lat=31.7, length=20, scale=20, label=True):
    """Draw a reference scale arrow (all args wrapped as arrays for cartopy)."""
    ax.quiver(np.array([lon]), np.array([lat]),
              np.array([float(length)]), np.array([0.0]),
              transform=ccrs.PlateCarree(),
              scale=scale, scale_units="xy",
              width=0.003, headwidth=4, headlength=5, color="k", zorder=5)
    if label:
        ax.text(lon + length / scale / 2, lat - 0.25, f"{length} mm/yr",
                transform=ccrs.PlateCarree(), fontsize=7, ha="center")


def _pole_marker(ax, pole, color, label=""):
    in_extent = (EXTENT[0] <= pole.lon <= EXTENT[1] and
                 EXTENT[2] <= pole.lat <= EXTENT[3])
    marker = "*" if in_extent else "D"
    size   = 200 if in_extent else 80
    ax.scatter(pole.lon, pole.lat,
               marker=marker, s=size, color=color,
               edgecolor="k", linewidth=0.8,
               transform=ccrs.PlateCarree(), zorder=6)
    if label:
        ax.text(pole.lon + 0.15, pole.lat + 0.1, label,
                transform=ccrs.PlateCarree(), fontsize=7,
                color=color, fontweight="bold", zorder=7)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Raw velocity field
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 1: raw velocity field …")
fig, ax = plt.subplots(figsize=(9, 6),
                       subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)
_quiver(ax, stations, color="steelblue")
_ref_arrow(ax)
ax.set_title("Southwest Japan GPS velocities (ITRF2000)\n"
             f"N = {len(stations)} stations after preprocessing", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig1_velocity_field.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Velocity scatter (raw vs clean)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 2: velocity scatter …")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for ax, s_list, title in [
        (ax1, raw,      f"Raw  (N={len(raw)})"),
        (ax2, stations, f"Preprocessed (N={len(stations)})")]:
    ve = [s.velocity.ve for s in s_list]
    vn = [s.velocity.vn for s in s_list]
    ax.scatter(ve, vn, s=50, edgecolor="steelblue", facecolor="white",
               linewidths=0.8, alpha=0.8)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Ve  (mm/yr)", fontsize=11)
    ax.set_ylabel("Vn  (mm/yr)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
fig.suptitle("GPS velocity space — southwest Japan (ITRF2000)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig2_velocity_scatter.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — HAC dendrogram + gap statistic
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 3: HAC dendrogram + gap statistic …")
hac  = VelocityHACClustering()
X    = np.array([[s.velocity.ve, s.velocity.vn] for s in stations])
Z    = hac.fit(stations)
k_gap, gap_result = hac.find_optimal_k(stations, max_k=9, n_ref=30)
print(f"  Gap optimal k = {k_gap}")

from scipy.cluster.hierarchy import dendrogram
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
ax2.axvline(k_gap, color="tomato", ls="--", lw=1.5,
            label=f"Optimal k = {k_gap}")
ax2.set_xlabel("Number of clusters k", fontsize=11)
ax2.set_ylabel("Gap statistic", fontsize=11)
ax2.set_title("Tibshirani (2001) gap statistic", fontsize=11)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.legend(fontsize=9)
fig.suptitle("Velocity-space HAC — optimal cluster selection", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig3_hac_gap.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Euler chi² vs k (elbow)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 4: Euler chi² vs k …")
# "multiscale" tries 5 deterministic + 20 random initializations and keeps the
# minimum-chi² result — more robust in datasets with strong interseismic loading.
evc = EulerVectorClustering(init="multiscale", n_restarts=20, random_seed=0)
k_euler, ftest = evc.find_optimal_k(stations, max_k=9)
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
fig.suptitle("Euler-vector clustering — southwest Japan", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig4_euler_chi2.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Map: k=3 clusters + Euler poles  (main comparison figure)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 5: k=3 cluster map …")
clusters3 = evc.cluster(stations, k=3,
                         init_labels=_geo_init_labels(stations, 3))

fig, ax = plt.subplots(figsize=(11, 8),
                       subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)

legend_handles = []
for c in clusters3:
    col  = CMAP(c.id - 1)
    pole = euler_vector_to_pole(c.euler_vector)
    _scatter(ax, c.stations, color=col, s=30)
    _quiver(ax, c.stations, color=col)
    _pole_marker(ax, pole, color=col,
                 label=f"E{c.id} ({pole.lat:.0f}°N,{pole.lon:.0f}°E)\n{pole.rate:.2f}°/Myr")
    legend_handles.append(Line2D([0], [0], color=col, lw=0, marker="o",
                                 markersize=9, markeredgecolor="k", markeredgewidth=0.5,
                                 label=f"Cluster {c.id}  (N={c.size})"))

_ref_arrow(ax)

# Euler pole legend entry
legend_handles.append(Line2D([0], [0], lw=0, marker="*",
                              markersize=11, markeredgecolor="k", markeredgewidth=0.5,
                              color="gray", label="Euler pole (if in region)"))
ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)
chi2_k3 = sum(total_chi_squared(c.stations, c.euler_vector)
              for c in clusters3 if c.euler_vector is not None)
dof_k3  = 2 * len(stations) - 3 * 3
ax.set_title(f"Euler-vector clustering  k = 3   χ²_red = {chi2_k3/dof_k3:.0f}\n"
             "Southwest Japan GPS velocities (ITRF2000)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig5_clusters_k3.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Residual velocity vectors (observed − Euler-predicted)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 6: residual velocities …")
fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                         subplot_kw={"projection": ccrs.Mercator()})

for ax, clusters, title in [
        (axes[0], clusters3, "k = 3  observed"),
        (axes[1], clusters3, "k = 3  residuals (obs − Euler predicted)")]:
    _basemap(ax)
    ax.set_title(title, fontsize=10)

for c in clusters3:
    col = CMAP(c.id - 1)
    # left panel: observed
    _scatter(axes[0], c.stations, color=col, s=22)
    _quiver(axes[0], c.stations, color=col, scale=20)
    # right panel: residuals
    lons, lats, dve, dvn = [], [], [], []
    for s in c.stations:
        ve_pred, vn_pred = predict_velocity(s, c.euler_vector)
        lons.append(s.position.lon)
        lats.append(s.position.lat)
        dve.append(s.velocity.ve - ve_pred)
        dvn.append(s.velocity.vn - vn_pred)
    _scatter(axes[1], c.stations, color=col, s=22)
    axes[1].quiver(np.array(lons), np.array(lats), np.array(dve), np.array(dvn),
                   transform=ccrs.PlateCarree(),
                   scale=5, scale_units="xy",
                   width=0.004, headwidth=5, headlength=6,
                   color=col, alpha=0.9, zorder=3)

_ref_arrow(axes[0], length=20, scale=20)
_ref_arrow(axes[1], length=5,  scale=5)

fig.suptitle("Observed vs. residual velocities — Euler-vector clustering k=3",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig6_residuals_k3.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — 2×2 grid: k = 2, 3, 4, 5
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 7: k = 2–5 comparison …")
fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                         subplot_kw={"projection": ccrs.Mercator()})
axes = axes.flatten()

for ax, k in zip(axes, [2, 3, 4, 5]):
    _basemap(ax)
    clusters_k = evc.cluster(stations, k=k,
                              init_labels=_geo_init_labels(stations, k))
    for c in clusters_k:
        col  = CMAP(c.id - 1)
        pole = euler_vector_to_pole(c.euler_vector)
        _scatter(ax, c.stations, color=col, s=18)
        _quiver(ax, c.stations, color=col, scale=20)
        _pole_marker(ax, pole, color=col)

    _ref_arrow(ax, label=False)

    chi2r = ftest.chi2_reduced[k - 1]
    ax.set_title(f"k = {k}   χ²_red = {chi2r:.1f}", fontsize=11)

fig.suptitle("Euler-vector clustering — southwest Japan GPS (ITRF2000)\n"
             "★ = Euler pole (if inside map region)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig7_k2345_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"\nAll figures saved to {OUT}/")
