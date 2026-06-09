"""Reproduce figures comparable to Savage (2018) for southwest Japan GPS clustering.

Reads: results/japan/clusters.json   (written by compute_japan_clusters.py)
Writes: results/japan/fig*.png

Generates:
  Fig 1 — Raw velocity field (ITRF2000)
  Fig 2 — Velocity scatter (Ve vs Vn), raw and after preprocessing
  Fig 3 — HAC dendrogram + gap statistic → optimal k
  Fig 4 — Euler-vector chi² vs k (F-test elbow)
  Fig 5 — Map: k=3 VB clusters with velocity arrows + Euler poles
  Fig 6 — Map: k=3 clusters, residual vectors (observed − predicted)
  Fig 7 — Cluster comparison grid: k = 2..9
  Fig 8 — ω-space Euler vectors: k = 2..9
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram

from gps_cluster.domain.entities import EulerPole, GpsStation, Position, Velocity
from gps_cluster.domain.services.euler_math import EulerVector, predict_velocity

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).parent.parent
CACHE = ROOT / "results/japan/clusters.json"
OUT   = ROOT / "results/japan"
OUT.mkdir(parents=True, exist_ok=True)

if not CACHE.exists():
    raise FileNotFoundError(f"{CACHE} — run compute_japan_clusters.py first")

with open(CACHE) as f:
    _cache = json.load(f)

EXTENT  = [128.0, 139.5, 31.0, 38.5]
CMAP    = plt.colormaps["tab10"]
N       = _cache["meta"]["n_stations"]
n_raw   = _cache["meta"]["n_raw_stations"]
k_gap   = _cache["meta"]["k_gap"]
k_euler = _cache["meta"]["k_euler"]
MAX_K   = _cache["meta"]["max_k"]

print(f"Japan: N={N}  k_gap={k_gap}  k_euler={k_euler}")

# ── reconstruct station objects ────────────────────────────────────────────────
stations = [
    GpsStation(r["name"], Position(r["lon"], r["lat"]),
               Velocity(r["ve"], r["vn"], 0.0, r["se"], r["sn"], 1.0))
    for r in _cache["stations"]
]
raw_stations = [
    GpsStation(r["name"], Position(r["lon"], r["lat"]),
               Velocity(r["ve"], r["vn"], 0.0, r["se"], r["sn"], 1.0))
    for r in _cache["raw_stations"]
]
station_by_name = {s.name: s for s in stations}


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


# ── Earth radius for ω-space conversion ──────────────────────────────────────
_R_MM = 6_371_000.0 * 1_000.0  # mm


def _omega_deg_per_ma(ev: EulerVector) -> np.ndarray:
    """Convert EulerVector (mm/yr) → Cartesian (ωx, ωy, ωz) in °/Ma."""
    return ev.to_array() / _R_MM * np.degrees(1) * 1e6


# ── map helpers ───────────────────────────────────────────────────────────────
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


def _quiver(ax, stations_list, color, scale=200, **kw):
    lons = np.array([s.position.lon for s in stations_list])
    lats = np.array([s.position.lat for s in stations_list])
    ve   = np.array([s.velocity.ve  for s in stations_list])
    vn   = np.array([s.velocity.vn  for s in stations_list])
    return ax.quiver(lons, lats, ve, vn,
                     transform=ccrs.PlateCarree(),
                     scale=scale, scale_units="width",
                     angles="uv",
                     width=0.003, headwidth=4, headlength=5, headaxislength=4,
                     minlength=0, minshaft=0.5,
                     color=color, alpha=0.9, zorder=3, **kw)


def _scatter(ax, stations_list, color, s=28):
    lons = np.array([s.position.lon for s in stations_list])
    lats = np.array([s.position.lat for s in stations_list])
    ax.scatter(lons, lats, s=s, color=color,
               edgecolor="k", linewidths=0.3,
               transform=ccrs.PlateCarree(), zorder=4)


def _ref_arrow(ax, length=20, scale=200, label=True):
    _q = ax.quiver(np.array([134.0]), np.array([34.5]),
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


def _rms(clusters_list) -> float:
    sq = []
    for c in clusters_list:
        if c.euler_vector is None:
            continue
        for s in c.stations:
            ve_p, vn_p = predict_velocity(s, c.euler_vector)
            sq.append((s.velocity.ve - ve_p)**2 + (s.velocity.vn - vn_p)**2)
    return float(np.sqrt(np.mean(sq))) if sq else np.nan


def _pole_marker(ax, pole, color, label=""):
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


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Raw velocity field
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 1: velocity field …")
fig, ax = plt.subplots(figsize=(9, 6),
                       subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)
_quiver(ax, stations, color="steelblue")
_ref_arrow(ax)
ax.set_title("Southwest Japan GPS velocities (ITRF2000)\n"
             f"N = {N} stations after preprocessing", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig1_velocity_field.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Velocity scatter (raw vs preprocessed)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 2: velocity scatter …")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for ax, s_list, title in [
        (ax1, raw_stations, f"Raw  (N={n_raw})"),
        (ax2, stations,     f"Preprocessed (N={N})")]:
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
gap_d = _cache["gap"]
Z     = np.array(_cache["linkage"])    # pre-computed linkage matrix

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

dendrogram(Z, ax=ax1, no_labels=True, color_threshold=0,
           above_threshold_color="steelblue", leaf_font_size=6)
ax1.set_xlabel("Station index", fontsize=11)
ax1.set_ylabel("Linkage distance  (mm/yr)", fontsize=11)
ax1.set_title("HAC dendrogram (centroid linkage)", fontsize=11)

ks = np.array(gap_d["k_values"])
ax2.errorbar(ks, gap_d["gap"], yerr=gap_d["sk"],
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
# Figure 4 — Euler chi² vs k (F-test elbow)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 4: Euler chi² vs k …")
ft_d  = _cache["ftest"]
ks_f  = np.array(ft_d["k_values"])
chi2r = np.array(ft_d["chi2_reduced"])
pvals = np.array(ft_d["p_values"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(ks_f, chi2r, "-o", lw=2,
         markerfacecolor="white", color="steelblue")
ax1.axhline(1, color="gray", ls="--", lw=1, label="χ²_red = 1")
ax1.set_xlabel("Number of clusters k", fontsize=11)
ax1.set_ylabel("Reduced χ²", fontsize=11)
ax1.set_title("VB Euler-vector clustering — fit quality", fontsize=11)
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.legend(fontsize=9)

ax2.plot(ks_f[1:len(pvals)+1], pvals, "-o", lw=2,
         markerfacecolor="white", color="tomato")
ax2.axhline(0.05, color="gray", ls="--", lw=1, label="α = 0.05")
ax2.set_xlabel("Number of clusters k", fontsize=11)
ax2.set_ylabel("F-test p-value  (k vs k+1)", fontsize=11)
ax2.set_title("Significance of adding one more cluster", fontsize=11)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.legend(fontsize=9)
fig.suptitle("VB Euler-vector clustering — southwest Japan", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig4_euler_chi2.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Map: k=3 clusters + Euler poles  (Savage 2018 comparison)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 5: k=3 cluster map …")
clusters3 = _load_solution(3)
rms_k3    = _rms(clusters3)
chi2_k3   = sum(c.chi2 for c in clusters3 if c.chi2 is not None)
dof_k3    = max(2 * N - 3 * 3, 1)

fig, ax = plt.subplots(figsize=(14, 10),
                       subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)

legend_handles = []
for c in clusters3:
    col  = CMAP(c.id - 1)
    pole = c.pole
    _scatter(ax, c.stations, color=col, s=30)
    _quiver(ax, c.stations, color=col)
    _pole_marker(ax, pole, color=col,
                 label=f"E{c.id} ({pole.lat:.0f}°N,{pole.lon:.0f}°E) {pole.rate:.2f}°/Myr")
    in_ext   = EXTENT[0] <= pole.lon <= EXTENT[1] and EXTENT[2] <= pole.lat <= EXTENT[3]
    pole_str = f"  pole {pole.lat:.0f}°N {pole.lon:.0f}°E" if not in_ext else ""
    legend_handles.append(Line2D([0], [0], color=col, lw=0, marker="o",
                                 markersize=9, markeredgecolor="k", markeredgewidth=0.5,
                                 label=f"Cluster {c.id}  (N={c.size}){pole_str}"))

_ref_arrow(ax)
ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)
ax.set_title(f"VB Euler-vector clustering  k = 3\n"
             f"RMS = {rms_k3:.1f} mm/yr   χ²_red = {chi2_k3/dof_k3:.0f}",
             fontsize=11)
fig.savefig(OUT / "fig5_clusters_k3.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Residual velocity vectors (observed − Euler-predicted)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 6: residual velocities …")
fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                         subplot_kw={"projection": ccrs.Mercator()})
_basemap(axes[0])
_basemap(axes[1])

res_lons, res_lats, res_dve, res_dvn = [], [], [], []
rms_obs_sq = []
for c in clusters3:
    col = CMAP(c.id - 1)
    _scatter(axes[0], c.stations, color=col, s=22)
    _quiver(axes[0], c.stations, color=col)
    if c.euler_vector is None:
        continue
    for s in c.stations:
        ve_pred, vn_pred = predict_velocity(s, c.euler_vector)
        res_lons.append(s.position.lon)
        res_lats.append(s.position.lat)
        res_dve.append(s.velocity.ve - ve_pred)
        res_dvn.append(s.velocity.vn - vn_pred)
        rms_obs_sq.append(s.velocity.ve**2 + s.velocity.vn**2)

res_lons = np.array(res_lons); res_lats = np.array(res_lats)
res_dve  = np.array(res_dve);  res_dvn  = np.array(res_dvn)

rms_obs = float(np.sqrt(np.mean(rms_obs_sq))) if rms_obs_sq else np.nan
rms_res = float(np.sqrt(np.mean(res_dve**2 + res_dvn**2))) if len(res_dve) else np.nan

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
axes[1].set_title(f"Obs − VB predicted  (RMS misfit = {rms_res:.1f} mm/yr)", fontsize=10)

fig.suptitle("VB Euler-vector clustering k = 3 — southwest Japan GPS (ITRF2000)",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig6_residuals_k3.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — k = 2..MAX_K grid (dots only, cf. Savage 2018 Fig 2)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"Plotting Fig 7: k=2–{MAX_K} cluster maps …")
n_panels = MAX_K - 1
n_cols   = 2
n_rows   = (n_panels + 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 5.5),
                         subplot_kw={"projection": ccrs.Mercator()})
axes = axes.flatten()

for ax, k in zip(axes, range(2, MAX_K + 1)):
    _basemap(ax)
    clusters_k = _load_solution(k)
    for c in clusters_k:
        _scatter(ax, c.stations, color=CMAP(c.id - 1), s=12)
    rms_k = _rms(clusters_k)
    ax.set_title(f"k = {k}   rms = {rms_k:.2f} mm/yr", fontsize=10)

for ax in axes[MAX_K - 1:]:
    ax.axis("off")

fig.suptitle("VB Euler-vector clustering — southwest Japan GPS (ITRF2000)\n"
             f"γ = {_cache['meta']['vb_gamma']:.1e}, {_cache['meta']['n_restarts']} restarts;"
             " cf. Savage (2018) Fig 2", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig7_k2to9_clusters.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8 — ω-space scatter (k = 2..MAX_K), cf. Savage (2018) Figs 3 & 4
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting Fig 8: ω-space Euler vectors …")
fig, axes8 = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 5),
                           subplot_kw={"projection": "3d"})
axes8 = axes8.flatten()

for ax, k in zip(axes8, range(2, MAX_K + 1)):
    clusters_k = _load_solution(k)
    for c in clusters_k:
        if c.euler_vector is None:
            continue
        col = CMAP(c.id - 1)
        wx, wy, wz = _omega_deg_per_ma(c.euler_vector)
        ax.scatter(wx, wy, wz, color=col, s=80, edgecolors="k", linewidths=0.5, zorder=5)

    ax.set_xlabel("ωx  (°/Ma)", fontsize=7, labelpad=2)
    ax.set_ylabel("ωy  (°/Ma)", fontsize=7, labelpad=2)
    ax.set_zlabel("ωz  (°/Ma)", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.set_title(f"k = {k}", fontsize=9)

for ax in axes8[MAX_K - 1:]:
    ax.set_visible(False)

fig.suptitle("VB Euler vectors in ω-space  (°/Ma, ITRF2000)\n"
             "cf. Savage (2018) Figs 3 & 4", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig8_omega_space.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nAll figures saved to {OUT}/")
