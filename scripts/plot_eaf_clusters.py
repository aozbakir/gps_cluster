"""EAF case study — Euler-vector clustering in the East Anatolian Fault zone.

Reads: results/eaf/clusters.json   (written by compute_eaf_clusters.py)
Writes: results/eaf/fig*.png

Region: 34–42°E, 36–39°N
Focus: Can pre-earthquake GPS clustering recover the Sürgü-Çardak fault
       as a block boundary, and resolve kinematic peculiarities of the EAF
       that led to the 2023 Kahramanmaraş earthquake sequence?

Generates:
  fig1 — Raw velocity field (ITRF14) in the EAF box
  fig2 — Velocity scatter
  fig3 — Gap statistic + F-test model selection
  fig4 — Best-k cluster map with fault overlay
  fig5 — Residual velocities
  fig6 — k=2..6 cluster grid
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

ROOT   = Path(__file__).parent.parent
CACHE  = ROOT / "results/eaf/clusters.json"
OUT    = ROOT / "results/eaf"
OUT.mkdir(parents=True, exist_ok=True)

# Fault overlays — both confirmed in data/external/
EMME_FAULT_FILE       = ROOT / "data/external/mta_emme_fault_map.geojson"
SIMPLIFIED_FAULT_FILE = ROOT / "data/external/eaf_slip_rate_faults_simplified.geojson"

if not CACHE.exists():
    raise FileNotFoundError(f"{CACHE} — run compute_eaf_clusters.py first")

with open(CACHE) as f:
    _cache = json.load(f)

EXTENT = _cache["meta"]["extent"]
CMAP   = plt.colormaps["tab10"]
k_gap  = _cache["gap"]["k_max_gap"]
N      = _cache["meta"]["n_stations"]
print(f"EAF: k_gap={k_gap}  N={N}")

# ── reconstruct station objects ───────────────────────────────────────────────
stations = [
    GpsStation(r["name"], Position(r["lon"], r["lat"]),
               Velocity(r["ve"], r["vn"], 0.0, r["se"], r["sn"], 1.0))
    for r in _cache["stations"]
]
station_by_name = {s.name: s for s in stations}


def _load_solution(k: int) -> list:
    """Reconstruct cluster SimpleNamespaces from the JSON cache."""
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


clusters_best = _load_solution(k_gap)

# ── fault overlay helpers ─────────────────────────────────────────────────────
from shapely.geometry import box as _shp_box
_CLIP_POLY = _shp_box(EXTENT[0]-0.5, EXTENT[2]-0.5, EXTENT[1]+0.5, EXTENT[3]+0.5)

_SIMPLIFIED_COLORS = {
    "EAF":    "#c0392b",
    "DSF":    "#8e44ad",
    "Surgu":  "#e67e22",
    "Cardak": "#27ae60",
    "Ecemis": "#7f8c8d",
}

_SIMPLIFIED_GDF = (gpd.read_file(SIMPLIFIED_FAULT_FILE).to_crs("EPSG:4326")
                   if SIMPLIFIED_FAULT_FILE.exists() else None)
_EMME_GDF       = (gpd.read_file(EMME_FAULT_FILE).to_crs("EPSG:4326")
                   if EMME_FAULT_FILE.exists() else None)


def _plot_faults(ax):
    # Thin grey EMME reference network
    if _EMME_GDF is not None:
        for _, row in _EMME_GDF.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            geom = geom.intersection(_CLIP_POLY)
            if geom.is_empty:
                continue
            geoms = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
            for g in geoms:
                if g.geom_type not in ("LineString", "MultiLineString"):
                    continue
                parts = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
                for part in parts:
                    xs, ys = part.xy
                    ax.plot(list(xs), list(ys), color="#999999", linewidth=0.4,
                            transform=ccrs.PlateCarree(), zorder=4, alpha=0.3)

    # Bold simplified kinematic boundaries
    if _SIMPLIFIED_GDF is not None:
        for _, row in _SIMPLIFIED_GDF.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            col  = _SIMPLIFIED_COLORS.get(row.get("group", ""), "black")
            segs = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            for seg in segs:
                xs, ys = seg.xy
                ax.plot(list(xs), list(ys), color=col, linewidth=2.2,
                        transform=ccrs.PlateCarree(), zorder=6, alpha=0.9)


# ── map helpers ───────────────────────────────────────────────────────────────
def _basemap(ax, extent=None):
    ext = extent or EXTENT
    ax.set_extent(ext, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#f5f1eb", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="gray", zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="gray",
                   zorder=1, linestyle=":")
    ax.add_feature(cfeature.RIVERS,    linewidth=0.3, edgecolor="#aed6f1", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--", crs=ccrs.PlateCarree())
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(34, 43, 2))
    gl.ylocator = mticker.FixedLocator(range(36, 40, 1))
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}


def _quiver(ax, sta_list, color, scale=120):
    lons = np.array([s.position.lon for s in sta_list])
    lats = np.array([s.position.lat for s in sta_list])
    ve   = np.array([s.velocity.ve  for s in sta_list])
    vn   = np.array([s.velocity.vn  for s in sta_list])
    ax.quiver(lons, lats, ve, vn,
              transform=ccrs.PlateCarree(),
              scale=scale, scale_units="width",
              angles="uv", width=0.004,
              headwidth=4, headlength=5, headaxislength=4,
              color=color, alpha=0.85, zorder=4)


def _scatter(ax, sta_list, color, s=28):
    lons = [s.position.lon for s in sta_list]
    lats = [s.position.lat for s in sta_list]
    ax.scatter(lons, lats, s=s, color=color,
               edgecolor="k", linewidths=0.3,
               transform=ccrs.PlateCarree(), zorder=4)


def _pole_marker(ax, pole, color):
    if EXTENT[0] <= pole.lon <= EXTENT[1] and EXTENT[2] <= pole.lat <= EXTENT[3]:
        ax.scatter(pole.lon, pole.lat, marker="*", s=220, color=color,
                   edgecolor="k", linewidth=0.8,
                   transform=ccrs.PlateCarree(), zorder=6)


def _rms(clusters_list):
    sq = []
    for c in clusters_list:
        if c.euler_vector is None:
            continue
        for s in c.stations:
            ve_p, vn_p = predict_velocity(s, c.euler_vector)
            sq.append((s.velocity.ve - ve_p)**2 + (s.velocity.vn - vn_p)**2)
    return float(np.sqrt(np.mean(sq))) if sq else np.nan


def _ref_arrow(ax, length=10, scale=120, x=40.8, y=36.3):
    q = ax.quiver(np.array([x]), np.array([y]),
                  np.array([float(length)]), np.array([0.0]),
                  transform=ccrs.PlateCarree(),
                  scale=scale, scale_units="width", angles="uv",
                  width=0.004, headwidth=4, headlength=5,
                  color="k", zorder=3)
    ax.quiverkey(q, X=0.88, Y=0.06, U=length,
                 label=f"{length} mm/yr", labelpos="S",
                 fontproperties={"size": 7})


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Raw velocity field
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 1: velocity field …")
fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)
_plot_faults(ax)
_quiver(ax, stations, color="steelblue")
_ref_arrow(ax)
ax.set_title(f"EAF region — GPS velocities (ITRF14)  N = {N} stations", fontsize=12)
fig.savefig(OUT / "fig1_velocity_field.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Velocity scatter
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 2: velocity scatter …")
fig, ax = plt.subplots(figsize=(7, 7))
ve_all = [s.velocity.ve for s in stations]
vn_all = [s.velocity.vn for s in stations]
ax.scatter(ve_all, vn_all, s=30, edgecolor="steelblue",
           facecolor="white", linewidths=0.8, alpha=0.8)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.axvline(0, color="gray", lw=0.5, ls="--")
ax.set_xlabel("Ve  (mm/yr)", fontsize=11)
ax.set_ylabel("Vn  (mm/yr)", fontsize=11)
ax.set_title(f"Velocity space — EAF region (ITRF14, N={N})", fontsize=11)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT / "fig2_velocity_scatter.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Gap statistic + F-test
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 3: model selection …")
gap_d = _cache["gap"]
ft_d  = _cache["ftest"]

ks_gap = np.array(gap_d["k_values"])
ks_f   = np.array(ft_d["k_values"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.errorbar(ks_gap, gap_d["gap"], yerr=gap_d["sk"],
             fmt="-o", capsize=4, linewidth=1.8,
             markerfacecolor="white", color="steelblue", label="Gap(k) ± s_k")
ax1.axvline(k_gap, color="tomato", ls="--", lw=1.5, label=f"Max-gap k = {k_gap}")
ax1.set_xlabel("Number of clusters k", fontsize=11)
ax1.set_ylabel("Gap statistic", fontsize=11)
ax1.set_title("Gap statistic (velocity-space HAC)", fontsize=11)
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.legend(fontsize=9)

chi2r      = np.array(ft_d["chi2_reduced"])
improvement = -np.diff(chi2r)
ax2.bar(ks_f[1:], improvement, color="steelblue", alpha=0.8,
        edgecolor="k", linewidth=0.5)
ax2.set_yscale("log")
ax2.axvline(k_gap, color="tomato", ls="--", lw=1.5, label=f"Gap k = {k_gap}")
ax2.set_xlabel("Number of clusters k", fontsize=11)
ax2.set_ylabel("Δχ²_red  (log scale)", fontsize=11)
ax2.set_title("Marginal χ²_red improvement per added cluster", fontsize=11)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.legend(fontsize=9)

fig.suptitle("EAF region — Euler-vector clustering model selection", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig3_model_selection.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Best-k cluster map
# ═══════════════════════════════════════════════════════════════════════════════
print(f"Fig 4: k={k_gap} cluster map …")
fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)
_plot_faults(ax)

legend_handles = []
chi2_total = 0.0
for c in clusters_best:
    col  = CMAP(c.id - 1)
    pole = c.pole
    _scatter(ax, c.stations, color=col)
    _quiver(ax, c.stations, color=col)
    _pole_marker(ax, pole, color=col)
    in_ext   = EXTENT[0] <= pole.lon <= EXTENT[1] and EXTENT[2] <= pole.lat <= EXTENT[3]
    pole_str = f"  pole {pole.lat:.0f}°N {pole.lon:.0f}°E" if not in_ext else ""
    legend_handles.append(
        Line2D([0], [0], color=col, lw=0, marker="o", markersize=9,
               markeredgecolor="k", markeredgewidth=0.5,
               label=f"C{c.id}  N={c.size}{pole_str}  {pole.rate:.2f}°/Ma"))
    chi2_total += c.chi2

dof = max(2 * N - 3 * k_gap, 1)
rms = _rms(clusters_best)
_ref_arrow(ax)
ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)
ax.set_title(
    f"EAF region — VB Euler clustering  k = {k_gap}  (gap statistic)\n"
    f"RMS = {rms:.1f} mm/yr   χ²_red = {chi2_total/dof:.1f}   N = {N}",
    fontsize=11)
fig.savefig(OUT / "fig4_clusters.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Residual velocities
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 5: residuals …")
fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                         subplot_kw={"projection": ccrs.Mercator()})
for a in axes:
    _basemap(a)
    _plot_faults(a)

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
rms_res  = float(np.sqrt(np.mean(res_dve**2 + res_dvn**2)))

q = axes[1].quiver(res_lons, res_lats, res_dve, res_dvn,
                   transform=ccrs.PlateCarree(),
                   scale=60, scale_units="width", angles="uv",
                   width=0.005, headwidth=4, headlength=5,
                   color="k", alpha=0.85, zorder=4)
axes[1].quiverkey(q, X=0.88, Y=0.06, U=5,
                  label="5 mm/yr", labelpos="S",
                  fontproperties={"size": 7})

axes[0].set_title(f"Cluster assignment  (k = {k_gap})", fontsize=10)
axes[1].set_title(f"Residuals: Obs − VB Euler predicted  (RMS = {rms_res:.1f} mm/yr)",
                  fontsize=10)
fig.suptitle("EAF region — VB Euler-vector clustering residuals", fontsize=12)
fig.savefig(OUT / "fig5_residuals.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — k = 2..6 cluster grid
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 6: k=2–6 grid …")
fig, axes6 = plt.subplots(2, 3, figsize=(18, 12),
                          subplot_kw={"projection": ccrs.Mercator()})
axes6 = axes6.flatten()
max_k = _cache["meta"]["max_k"]
for ax, k in zip(axes6[:5], range(2, min(max_k, 6) + 1)):
    _basemap(ax)
    _plot_faults(ax)
    ck = _load_solution(k)
    for c in ck:
        _scatter(ax, c.stations, color=CMAP(c.id - 1), s=14)
    ax.set_title(f"k = {k}   RMS = {_rms(ck):.2f} mm/yr", fontsize=10)
axes6[5].axis("off")

fig.suptitle("EAF region — VB Euler-vector clustering  k = 2–6", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig6_k2to6.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nAll figures saved to {OUT}/")
