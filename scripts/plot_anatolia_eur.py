"""Cluster analysis figures for Kurt et al. (2022) Euro-fixed velocity field.

Reads: results/anatolia_eur/clusters.json
Writes: results/anatolia_eur/fig*.png

Figures produced:
  fig1_velocity_field.png   — raw velocity arrows
  fig2_velocity_scatter.png — Ve vs Vn scatter by cluster
  fig3_hac_gap.png          — gap statistic
  fig4_euler_chi2.png       — chi²_red vs k
  fig5_clusters_best_k.png  — k=k_gap map + velocity arrows
  fig6_residuals.png        — cluster map + residual vectors
  fig7_k2to7_clusters.png   — k=2..7 grid
  fig8_entropy.png          — VB soft-assignment entropy at k=k_gap
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from gps_cluster.domain.entities import EulerPole, GpsStation, Position, Velocity
from gps_cluster.domain.services.euler_math import EulerVector, predict_velocity

ROOT       = Path(__file__).parent.parent
CACHE      = ROOT / "results/anatolia_eur/clusters.json"
FAULT_FILE = ROOT / "data/external/anatolia_slip_rate_faults_simplified.geojson"
OUT        = ROOT / "results/anatolia_eur"
OUT.mkdir(parents=True, exist_ok=True)

EXTENT = [25.0, 45.5, 35.5, 43.0]
CMAP   = plt.colormaps["tab10"]

if not CACHE.exists():
    raise FileNotFoundError(f"{CACHE} — run compute_anatolia_eur_clusters.py first")

with open(CACHE) as f:
    cache = json.load(f)

# Reconstruct stations and solutions from cache
stations = [
    GpsStation(r["name"], Position(r["lon"], r["lat"]),
               Velocity(r["ve"], r["vn"], 0.0, r["se"], r["sn"], 1.0))
    for r in cache["stations"]
]
station_by_name = {s.name: s for s in stations}
N = len(stations)

k_gap = cache["gap"]["k_max_gap"]
print(f"k_gap = {k_gap}   N = {N}")

def _load_solution(k):
    out = []
    for c in cache["solutions"][str(k)]:
        ev_d = c["euler"]
        cov  = np.array(ev_d["covariance"]) if ev_d.get("covariance") else None
        ev   = EulerVector(ox=ev_d["ox"], oy=ev_d["oy"], oz=ev_d["oz"], covariance=cov)
        pole_d = c["pole"]
        pole = EulerPole(lat=pole_d["lat"], lon=pole_d["lon"], rate=pole_d["rate"],
                         sigma_lat=pole_d.get("sigma_lat", 0.0),
                         sigma_lon=pole_d.get("sigma_lon", 0.0),
                         sigma_rate=pole_d.get("sigma_rate", 0.0))
        ns = SimpleNamespace(
            id=c["id"], size=c["size"], chi2=c["chi2"],
            euler_vector=ev, pole=pole,
            stations=[station_by_name[n] for n in c["stations"] if n in station_by_name],
        )
        out.append(ns)
    return out

clusters_best = _load_solution(k_gap)

# ── map helpers ────────────────────────────────────────────────────────────────
def _basemap(ax, extent=EXTENT):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#f5f1eb", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="gray",  zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="gray",  zorder=1, linestyle=":")
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                      alpha=0.6, linestyle="--", crs=ccrs.PlateCarree())
    gl.top_labels = False; gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(26, 46, 2))
    gl.ylocator = mticker.FixedLocator(range(36, 44, 1))
    gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}

def _scatter(ax, slist, color, s=25):
    lons = [s.position.lon for s in slist]
    lats = [s.position.lat for s in slist]
    ax.scatter(lons, lats, s=s, color=color, edgecolor="k", linewidths=0.3,
               transform=ccrs.PlateCarree(), zorder=4)

def _quiver(ax, slist, color, scale=200):
    lons = np.array([s.position.lon for s in slist])
    lats = np.array([s.position.lat for s in slist])
    ve   = np.array([s.velocity.ve  for s in slist])
    vn   = np.array([s.velocity.vn  for s in slist])
    ax.quiver(lons, lats, ve, vn, transform=ccrs.PlateCarree(),
              scale=scale, scale_units="width", angles="uv",
              width=0.003, headwidth=4, headlength=5, headaxislength=4,
              color=color, alpha=0.9, zorder=3)

def _ref_arrow(ax, length=10, scale=200):
    _q = ax.quiver(np.array([35.0]), np.array([36.5]), np.array([float(length)]), np.array([0.0]),
                   transform=ccrs.PlateCarree(), scale=scale, scale_units="width",
                   angles="uv", width=0.003, headwidth=4, headlength=5, color="k", zorder=-1)
    ax.quiverkey(_q, X=0.85, Y=0.06, U=length, label=f"{length} mm/yr",
                 labelpos="S", fontproperties={"size": 7})

def _pole_marker(ax, pole, color, label=""):
    in_ext = EXTENT[0] <= pole.lon <= EXTENT[1] and EXTENT[2] <= pole.lat <= EXTENT[3]
    if not in_ext:
        return
    ax.scatter(pole.lon, pole.lat, marker="*", s=200, color=color,
               edgecolor="k", linewidth=0.8, transform=ccrs.PlateCarree(), zorder=6)
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
            vep, vnp = predict_velocity(s, c.euler_vector)
            sq.append((s.velocity.ve - vep)**2 + (s.velocity.vn - vnp)**2)
    return float(np.sqrt(np.mean(sq))) if sq else 0.0

def _draw_faults(ax):
    if not FAULT_FILE.exists():
        return
    gdf = gpd.read_file(FAULT_FILE).to_crs("EPSG:4326")
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        segs = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for seg in segs:
            xs, ys = seg.xy
            ax.plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
                    color="black", lw=1.2, alpha=0.7, zorder=3)

# ── Fig 1: Raw velocity field ──────────────────────────────────────────────────
print("Plotting Fig 1: velocity field …")
fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)
_quiver(ax, stations, color="steelblue", scale=150)
_ref_arrow(ax, length=10, scale=150)
ax.set_title(f"Kurt et al. (2022) GNSS — Euro-fixed  (N={N})", fontsize=12)
fig.savefig(OUT / "fig1_velocity_field.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Fig 2: Velocity scatter ────────────────────────────────────────────────────
print("Plotting Fig 2: velocity scatter …")
fig, ax = plt.subplots(figsize=(8, 7))
for c in clusters_best:
    col = CMAP(c.id - 1)
    ve  = [s.velocity.ve for s in c.stations]
    vn  = [s.velocity.vn for s in c.stations]
    ax.scatter(ve, vn, s=12, color=col, alpha=0.7, label=f"C{c.id} (N={c.size})")
ax.set_xlabel("Ve (mm/yr)"); ax.set_ylabel("Vn (mm/yr)")
ax.set_title(f"Velocity scatter — k={k_gap} clusters  (Euro-fixed)")
ax.legend(fontsize=9); ax.grid(lw=0.3, alpha=0.5); ax.set_aspect("equal")
fig.savefig(OUT / "fig2_velocity_scatter.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Fig 3: Gap statistic ───────────────────────────────────────────────────────
print("Plotting Fig 3: gap statistic …")
gap_d = cache["gap"]
k_vals = np.array(gap_d["k_values"])
gap    = np.array(gap_d["gap"])
sk     = np.array(gap_d["sk"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_vals, gap, "-o", color="steelblue", lw=2, label="Gap(k)")
ax.fill_between(k_vals, gap - sk, gap + sk, alpha=0.2, color="steelblue")
ax.axvline(gap_d["k_max_gap"], color="tomato", ls="--", lw=1.5,
           label=f"k = {gap_d['k_max_gap']} (max gap)")
ax.set_xlabel("k"); ax.set_ylabel("Gap statistic")
ax.set_title("Gap statistic — Euro-fixed velocity field")
ax.legend(); ax.grid(lw=0.3, alpha=0.5)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig(OUT / "fig3_hac_gap.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Fig 4: χ²_red vs k ────────────────────────────────────────────────────────
print("Plotting Fig 4: chi²_red …")
ft    = cache["ftest"]
ks    = np.array(ft["k_values"])
chi2r = np.array(ft["chi2_reduced"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.semilogy(ks, chi2r, "-o", lw=2, color="steelblue", markerfacecolor="white")
ax1.axhline(1, color="gray", ls="--", lw=1, label="χ²_red = 1")
ax1.axvline(k_gap, color="tomato", ls="--", lw=1.5, label=f"Gap k = {k_gap}")
ax1.set_xlabel("k"); ax1.set_ylabel("Reduced χ²  (log)")
ax1.set_title("Fit quality"); ax1.legend(fontsize=9)
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

improvement = -np.diff(chi2r)
ax2.bar(ks[1:], improvement, color="steelblue", alpha=0.8, edgecolor="k", lw=0.5)
ax2.set_yscale("log")
ax2.axvline(k_gap, color="tomato", ls="--", lw=1.5, label=f"Gap k = {k_gap}")
ax2.set_xlabel("k"); ax2.set_ylabel("Δχ²_red  (log)")
ax2.set_title("Marginal improvement per added cluster")
ax2.legend(fontsize=9)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

fig.suptitle("Euler-vector clustering — Euro-fixed GPS (Kurt et al. 2022)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig4_euler_chi2.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Fig 5: Best-k cluster map ─────────────────────────────────────────────────
print(f"Plotting Fig 5: k={k_gap} cluster map …")
fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)
legend_handles = []
for c in clusters_best:
    col  = CMAP(c.id - 1)
    pole = c.pole
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
dof_best  = 2 * N - 3 * k_gap
rms_best  = _rms(clusters_best)
ax.set_title(f"Euler-vector clustering  k = {k_gap}  (gap statistic) — Euro-fixed\n"
             f"RMS = {rms_best:.1f} mm/yr   χ²_red = {chi2_best/dof_best:.0f}", fontsize=12)
fig.savefig(OUT / "fig5_clusters_best_k.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Fig 6: Residuals ──────────────────────────────────────────────────────────
print("Plotting Fig 6: residual velocities …")
fig, axes = plt.subplots(1, 2, figsize=(22, 9), subplot_kw={"projection": ccrs.Mercator()})
_basemap(axes[0]); _basemap(axes[1])
_draw_faults(axes[0])

res_lons, res_lats, res_dve, res_dvn = [], [], [], []
for c in clusters_best:
    if c.euler_vector is None:
        continue
    col = CMAP(c.id - 1)
    _scatter(axes[0], c.stations, color=col, s=18)
    for s in c.stations:
        vep, vnp = predict_velocity(s, c.euler_vector)
        res_lons.append(s.position.lon); res_lats.append(s.position.lat)
        res_dve.append(s.velocity.ve - vep); res_dvn.append(s.velocity.vn - vnp)

res_lons = np.array(res_lons); res_lats = np.array(res_lats)
res_dve  = np.array(res_dve);  res_dvn  = np.array(res_dvn)
rms_res  = np.sqrt(np.mean(res_dve**2 + res_dvn**2))

q = axes[1].quiver(res_lons, res_lats, res_dve, res_dvn,
                   transform=ccrs.PlateCarree(), scale=80, scale_units="width",
                   angles="uv", width=0.004, headwidth=4, headlength=5,
                   headaxislength=4, color="k", alpha=0.85, zorder=4)
axes[1].quiverkey(q, X=0.85, Y=0.06, U=5, label="5 mm/yr",
                  labelpos="S", fontproperties={"size": 7})
axes[0].set_title(f"Cluster assignment  (k = {k_gap})", fontsize=10)
axes[1].set_title(f"Obs − Euler predicted  (RMS = {rms_res:.1f} mm/yr)", fontsize=10)
fig.suptitle(f"Euler-vector clustering k = {k_gap} — Euro-fixed GPS (Kurt et al. 2022)",
             fontsize=12)
fig.savefig(OUT / "fig6_residuals.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Fig 7: k=2..7 grid ────────────────────────────────────────────────────────
print("Plotting Fig 7: k=2–7 cluster maps …")
fig, axes7 = plt.subplots(3, 2, figsize=(16, 18),
                           subplot_kw={"projection": ccrs.Mercator()})
axes7 = axes7.flatten()
for ax, k in zip(axes7, range(2, 8)):
    _basemap(ax)
    clusters_k = _load_solution(k)
    for c in clusters_k:
        _scatter(ax, c.stations, color=CMAP(c.id - 1), s=10)
    rms_k = _rms(clusters_k)
    ax.set_title(f"k = {k}   rms = {rms_k:.2f} mm/yr", fontsize=10)
fig.suptitle("Euler-vector clustering — Euro-fixed GPS (Kurt et al. 2022)\n"
             "Multiscale init, 20 restarts", fontsize=12)
fig.savefig(OUT / "fig7_k2to7_clusters.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Fig 8: VB soft-assignment entropy map ─────────────────────────────────────
print(f"Plotting Fig 8: VB entropy map (k={k_gap}) …")
_ent_key = f"vb_entropy_k{k_gap}"
_ent_by_name = {
    r["name"]: r[_ent_key]
    for r in cache["stations"]
    if _ent_key in r
}
entropy     = np.array([_ent_by_name.get(s.name, 0.0) for s in stations])
max_entropy = np.log(k_gap)   # theoretical maximum = log(k) nats

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
cbar.set_label("VB soft-assignment entropy  (nats)\n0 = certain   log(k) = fully ambiguous",
               fontsize=9)
cbar.set_ticks([0, max_entropy / 2, max_entropy])
cbar.set_ticklabels(["0\n(certain)", f"{max_entropy/2:.2f}",
                     f"{max_entropy:.2f}\n(uniform)"])

_draw_faults(ax)

high_ent_pct = int(100 * np.mean(entropy > 0.5 * max_entropy))
ax.set_title(
    f"VB soft-assignment entropy — k = {k_gap}  (Euro-fixed, Kurt et al. 2022)\n"
    f"Red = ambiguous (≥½ max entropy: {high_ent_pct}% of stations)   "
    f"Green = certain   max = log({k_gap}) = {max_entropy:.2f} nats",
    fontsize=11)
fig.savefig(OUT / "fig8_entropy.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"\nAll figures saved to {OUT}/")
print(f"  Mean VB entropy: {entropy.mean():.4f} / {max_entropy:.3f} nats  "
      f"({100 * entropy.mean() / max_entropy:.1f}% of max)")
