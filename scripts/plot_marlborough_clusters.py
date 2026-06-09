"""Plot Marlborough Fault System clustering results.

Reads pre-computed results from results/marlborough/clusters.json.
Run compute_marlborough_clusters.py first if the cache is missing.
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

from gps_cluster.domain.entities import GpsStation, Position, Velocity
from gps_cluster.domain.services.euler_math import (
    EulerVector,
    euler_vector_to_pole,
    predict_velocity,
)

ROOT  = Path(__file__).parent.parent
OUT   = ROOT / "results/marlborough"
CACHE = OUT / "clusters.json"
OUT.mkdir(parents=True, exist_ok=True)

CMAP = plt.colormaps["tab10"]

# ── load cache ─────────────────────────────────────────────────────────────────
if not CACHE.exists():
    raise FileNotFoundError(
        f"{CACHE} not found — run compute_marlborough_clusters.py first."
    )

with open(CACHE) as f:
    cache = json.load(f)

meta = cache["meta"]
N    = meta["n_stations"]
EXTENT = meta["extent"]

# Reconstruct GpsStation objects
stations = [
    GpsStation(
        name=r["name"],
        position=Position(lat=r["lat"], lon=r["lon"]),
        velocity=Velocity(ve=r["ve"], vn=r["vn"], vu=0.0,
                          se=r["se"], sn=r["sn"], su=1.0),
    )
    for r in cache["stations"]
]
station_by_name = {s.name: s for s in stations}

# Gap result namespace
_g = cache["gap"]
gap = SimpleNamespace(
    k_values      = np.array(_g["k_values"]),
    gap           = np.array(_g["gap"]),
    sk            = np.array(_g["sk"]),
    k_first_cross = _g["k_first_cross"],
    k_max_gap     = _g["k_max_gap"],
)

# F-test result namespace
_f = cache["ftest"]
ftest = SimpleNamespace(
    k_values     = np.array(_f["k_values"]),
    chi2_reduced = np.array(_f["chi2_reduced"]),
    f_statistics = np.array(_f["f_statistics"]),
    p_values     = np.array(_f["p_values"]),
    k_elbow      = _f["k_elbow"],
)

# Cluster solutions: dict[k] -> list of SimpleNamespace clusters
def _load_solution(k: int):
    sol = []
    for c in cache["solutions"][str(k)]:
        ev = EulerVector(ox=c["euler"]["ox"],
                         oy=c["euler"]["oy"],
                         oz=c["euler"]["oz"])
        ns = SimpleNamespace(
            id           = c["id"],
            size         = c["size"],
            chi2         = c["chi2"],
            euler_vector = ev,
            pole         = SimpleNamespace(**c["pole"]),
            stations     = [station_by_name[n] for n in c["stations"]
                            if n in station_by_name],
        )
        sol.append(ns)
    return sol

print(f"Loaded cache: {N} stations, solutions for k = "
      f"{list(cache['solutions'].keys())}")

# ── map helpers ────────────────────────────────────────────────────────────────
def _basemap(ax, extent=None):
    ext = extent or EXTENT
    ax.set_extent(ext, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#f5f1eb", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="gray", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--", crs=ccrs.PlateCarree())
    gl.top_labels = False; gl.right_labels = False
    gl.xlocator   = mticker.FixedLocator(range(171, 176))
    gl.ylocator   = mticker.FixedLocator(range(-44, -40))
    gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}


def _scatter(ax, sl, col, s=30):
    ax.scatter([s.position.lon for s in sl],
               [s.position.lat for s in sl],
               s=s, color=col, edgecolor="k", linewidths=0.3,
               transform=ccrs.PlateCarree(), zorder=4)


def _quiver(ax, sl, col, scale=80):
    lons = np.array([s.position.lon for s in sl])
    lats = np.array([s.position.lat for s in sl])
    ve   = np.array([s.velocity.ve  for s in sl])
    vn   = np.array([s.velocity.vn  for s in sl])
    ax.quiver(lons, lats, ve, vn,
              transform=ccrs.PlateCarree(),
              scale=scale, scale_units="width", angles="uv",
              width=0.004, headwidth=4, headlength=5,
              color=col, alpha=0.85, zorder=4)


# ── 2016 Kaikōura rupture polygons ────────────────────────────────────────────
_KAIKOURA_GJ = ROOT / "data/external/Kaikoura_Earthquake_Fault_Ruptures_2016.geojson"
_KAIKOURA_GDF = gpd.read_file(_KAIKOURA_GJ)   # already EPSG:4326


def _plot_kaikoura(ax):
    for _, row in _KAIKOURA_GDF.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            xs, ys = geom.exterior.xy
            ax.fill(list(xs), list(ys), transform=ccrs.PlateCarree(),
                    color="black", alpha=0.75, zorder=7)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                xs, ys = poly.exterior.xy
                ax.fill(list(xs), list(ys), transform=ccrs.PlateCarree(),
                        color="black", alpha=0.75, zorder=7)


# ── Litchfield fault database ──────────────────────────────────────────────────
# Litchfield et al. 2014, NZJGG 57(1):32-56, doi:10.1080/00288306.2013.854256
# Data set: https://doi.org/10.21420/W08T-TY11?x=y
_FAULTS_SHP = (ROOT /
    "data/external/Litchfield_ActiveFaultModel_Shapefile"
    "/SF4_Litchfield_MapS1_Shapefile.shp")

_MFS_GROUPS = {
    "Hope":     ("#d73027", ["Hope"]),
    "Clarence": ("#fc8d59", ["Clarence"]),
    "Awatere":  ("#4575b4", ["Awatere"]),
    "Wairau":   ("#91bfdb", ["Wairau"]),
    "Alpine":   ("#1b7837", ["Alpine"]),
}

def _load_mfs_faults():
    gdf = gpd.read_file(_FAULTS_SHP).to_crs("EPSG:4326")
    gdf = gdf.cx[170.5:175.5, -44.0:-40.5]
    def _group(name):
        for grp, (_, kws) in _MFS_GROUPS.items():
            for kw in kws:
                if kw.lower() in str(name).lower():
                    return grp
        return "other"
    gdf["group"] = gdf["FZ_Name"].apply(_group)
    return gdf

_MFS_GDF    = _load_mfs_faults()
_MFS_LEGEND: list = []


def _plot_mfs(ax):
    global _MFS_LEGEND
    _MFS_LEGEND = []
    plotted = set()
    for grp, (colour, _) in _MFS_GROUPS.items():
        subset = _MFS_GDF[_MFS_GDF["group"] == grp]
        for _, row in subset.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            xs, ys = geom.xy
            ax.plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
                    color=colour, lw=1.6, alpha=0.85, zorder=5)
        if grp not in plotted and not subset.empty:
            _MFS_LEGEND.append(Line2D([0],[0], color=colour, lw=2, label=grp))
            plotted.add(grp)
    for _, row in _MFS_GDF[_MFS_GDF["group"] == "other"].iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        xs, ys = geom.xy
        ax.plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
                color="gray", lw=0.6, alpha=0.45, zorder=4)


# ── Fig model selection ────────────────────────────────────────────────────────
print("Fig model selection …")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

ax1.errorbar(gap.k_values, gap.gap, yerr=gap.sk,
             fmt="-o", color="steelblue", lw=2, capsize=4,
             markerfacecolor="white", label="Gap(k) ± s_k")
ax1.set_xlabel("Number of clusters  k", fontsize=11)
ax1.set_ylabel("Gap statistic", fontsize=11)
ax1.set_title("Gap statistic  (Tibshirani 2001)", fontsize=11)
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.text(0.97, 0.05,
         "Monotonically rising:\ngap uninformative for\ncontinuum deformation",
         transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
         color="firebrick", style="italic",
         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
ax1.legend(fontsize=9)

ax2.semilogy(ftest.k_values, ftest.chi2_reduced, "-o", lw=2,
             color="steelblue", markerfacecolor="white", label="χ²_red(k)")
ax2.set_xlabel("Number of clusters  k", fontsize=11)
ax2.set_ylabel("Reduced χ²  (log scale)", fontsize=11)
ax2.set_title("Euler-vector fit quality  χ²_red", fontsize=11)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.text(0.97, 0.95,
         f"χ²_red at k={meta['max_k']}: {ftest.chi2_reduced[-1]:.1f}\n"
         "χ²_red = 1 not reached\n→ no adequate block model",
         transform=ax2.transAxes, ha="right", va="top", fontsize=9,
         color="firebrick", style="italic",
         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
ax2.legend(fontsize=9)

ks_p  = ftest.k_values[1:]
pvals = np.maximum(ftest.p_values, 1e-10)
ax3.semilogy(ks_p, pvals, "-o", lw=2, color="steelblue",
             markerfacecolor="white", label="F-test  p(k−1 → k)")
ax3.axhline(0.05, color="tomato", ls="--", lw=1.5, label="α = 0.05")
ax3.set_xlabel("Number of clusters  k", fontsize=11)
ax3.set_ylabel("F-test  p-value  (log scale)", fontsize=11)
ax3.set_title("Sequential F-test\nH₀: k−1 clusters sufficient", fontsize=11)
ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax3.text(0.97, 0.95,
         "p ≈ 0 for all k:\nadding clusters always\nsignificant — no natural stop",
         transform=ax3.transAxes, ha="right", va="top", fontsize=9,
         color="firebrick", style="italic",
         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
ax3.legend(fontsize=9)

fig.suptitle(
    f"Marlborough Fault System — model selection  (N = {N} stations, ITRF2008)\n"
    "All criteria indicate distributed deformation: no statistically preferred k",
    fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig5_model_selection.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Fig 1 — simplified fault map with labels ──────────────────────────────────
print("Fig 1: fault map …")
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)

# Plot faults coloured by family, thicker than cluster maps
plotted_groups = set()
for grp, (colour, _) in _MFS_GROUPS.items():
    subset = _MFS_GDF[_MFS_GDF["group"] == grp]
    for _, row in subset.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        xs, ys = geom.xy
        ax.plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
                color=colour, lw=2.2, alpha=0.9, zorder=5)
    # Label the longest segment at its midpoint — Alpine labelled separately
    if not subset.empty and grp not in plotted_groups and grp != "Alpine":
        longest = subset.loc[subset.geometry.length.idxmax()]
        geom = longest.geometry
        xs, ys = geom.xy
        mid = len(xs) // 2
        ax.text(xs[mid], ys[mid], grp, transform=ccrs.PlateCarree(),
                fontsize=9, fontweight="bold", color=colour,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, lw=0),
                zorder=9)
        plotted_groups.add(grp)

# Other faults thin grey
for _, row in _MFS_GDF[_MFS_GDF["group"] == "other"].iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    xs, ys = geom.xy
    ax.plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
            color="gray", lw=0.7, alpha=0.4, zorder=4)

# Kaikōura rupture in solid black
_plot_kaikoura(ax)
ax.text(173.15, -42.55, "2016 Kaikōura\nrupture zone",
        transform=ccrs.PlateCarree(), fontsize=8, fontweight="bold",
        color="black", ha="left", va="top", zorder=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, lw=0))

ax.legend(handles=_MFS_LEGEND, loc="lower left", fontsize=9,
          title="MFS fault families\n(Litchfield et al. 2014)", framealpha=0.95)
ax.set_title("Marlborough Fault System — active fault map", fontsize=12)

# Alpine Fault label in main frame (Springs Junction to Tophouse segment crosses domain)
ax.text(171.9, -42.6, "Alpine Fault", transform=ccrs.PlateCarree(),
        fontsize=8, fontweight="bold", color="#1b7837",
        ha="center", va="bottom", rotation=40,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, lw=0),
        zorder=10)

# ── NZ inset — bottom right ───────────────────────────────────────────────────
_nz_all = gpd.read_file(_FAULTS_SHP).to_crs("EPSG:4326")
_alpine_kws    = ["Alpine"]
_hikurangi_kws = ["Hikurangi"]
_puysegur_kws  = ["Puysegur"]

ax_in = fig.add_axes([0.62, 0.03, 0.27, 0.37], projection=ccrs.Mercator())
ax_in.set_extent([165, 179, -48, -34], crs=ccrs.PlateCarree())
ax_in.add_feature(cfeature.LAND,      facecolor="#f0ede6", zorder=0)
ax_in.add_feature(cfeature.OCEAN,     facecolor="#cde4f0", zorder=0)
ax_in.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#777", zorder=1)

for _, row in _nz_all.iterrows():
    name = str(row["FZ_Name"])
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    if any(k.lower() in name.lower() for k in _alpine_kws):
        col, lw, zord = "#1b7837", 1.8, 4
    elif any(k.lower() in name.lower() for k in _hikurangi_kws):
        col, lw, zord = "#8B0000", 1.8, 4
    elif any(k.lower() in name.lower() for k in _puysegur_kws):
        col, lw, zord = "#6a0dad", 1.4, 4
    else:
        col, lw, zord = "gray", 0.4, 2
    if geom.geom_type == "LineString":
        xs, ys = geom.xy
        ax_in.plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
                   color=col, lw=lw, alpha=0.85, zorder=zord)
    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            xs, ys = part.xy
            ax_in.plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
                       color=col, lw=lw, alpha=0.85, zorder=zord)

ext = EXTENT
ax_in.plot([ext[0], ext[1], ext[1], ext[0], ext[0]],
           [ext[2], ext[2], ext[3], ext[3], ext[2]],
           transform=ccrs.PlateCarree(), color="red", lw=1.2, zorder=6)
ax_in.text(170.5, -43.8, "Alpine\nFault", transform=ccrs.PlateCarree(),
           fontsize=5.5, color="#1b7837", fontweight="bold", ha="right",
           bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0))
ax_in.text(178.5, -39.5, "Hikurangi\nsubduction", transform=ccrs.PlateCarree(),
           fontsize=5.5, color="#8B0000", fontweight="bold", ha="right",
           bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0))
ax_in.set_title("New Zealand", fontsize=6.5, pad=2)

fig.savefig(OUT / "fig1_fault_map.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Fig 2 — GPS velocity field with faded tectonic background ─────────────────
print("Fig 2: GPS velocity field …")
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)

# Faded fault traces
for grp, (colour, _) in _MFS_GROUPS.items():
    subset = _MFS_GDF[_MFS_GDF["group"] == grp]
    for _, row in subset.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        xs, ys = geom.xy
        ax.plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
                color=colour, lw=1.4, alpha=0.35, zorder=3)
for _, row in _MFS_GDF[_MFS_GDF["group"] == "other"].iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    xs, ys = geom.xy
    ax.plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
            color="gray", lw=0.5, alpha=0.25, zorder=2)

# Faded Kaikōura rupture
for _, row in _KAIKOURA_GDF.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    if geom.geom_type == "Polygon":
        xs, ys = geom.exterior.xy
        ax.fill(list(xs), list(ys), transform=ccrs.PlateCarree(),
                color="black", alpha=0.25, zorder=4)
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            xs, ys = poly.exterior.xy
            ax.fill(list(xs), list(ys), transform=ccrs.PlateCarree(),
                    color="black", alpha=0.25, zorder=4)

# GPS vectors prominent
lons = np.array([s.position.lon for s in stations])
lats = np.array([s.position.lat for s in stations])
ve   = np.array([s.velocity.ve  for s in stations])
vn   = np.array([s.velocity.vn  for s in stations])
q = ax.quiver(lons, lats, ve, vn, transform=ccrs.PlateCarree(),
              scale=400, scale_units="width", angles="uv",
              width=0.003, headwidth=4, headlength=5,
              color="steelblue", alpha=0.9, zorder=6)
ax.quiverkey(q, X=0.12, Y=0.06, U=10, label="10 mm/yr",
             labelpos="S", fontproperties={"size": 7})
ax.set_title(f"Marlborough Fault System — GPS velocities ITRF2008  (N = {N})", fontsize=12)
fig.savefig(OUT / "fig2_velocity_field.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── boundary fault labels — hardcoded by boundary index (W→E order) ───────────
# Index 0 = westernmost boundary, 1 = next, etc.
_BOUNDARY_FAULTS: dict[int, list[str]] = {
    2: [],  # k=2 uses two-label approach: Awatere (left) and Hope (right)
    3: ["Awatere", "Hope"],
    4: ["Alpine", "Clarence", "Hope"],
    5: ["Alpine", "Clarence", "Awatere", "Hope"],
    6: ["Alpine", "Alpine", "Clarence", "Awatere", "Hope"],
    7: ["Alpine", "Alpine", "Clarence", "Clarence", "Awatere", "Hope"],
}

_sta_lons = {r["name"]: r["lon"] for r in cache["stations"]}


def _edge_centroid(cluster, lon_mid: float, half_width: float = 0.25):
    """Velocity centroid of stations within ±half_width° of the boundary longitude.
    Falls back to full cluster centroid if no stations fall in the window."""
    edge = [s for s in cluster.stations
            if abs(_sta_lons[s.name] - lon_mid) <= half_width]
    if not edge:
        edge = cluster.stations
    return (np.mean([s.velocity.ve for s in edge]),
            np.mean([s.velocity.vn for s in edge]))


# Base colour palette — index = birth order of each new cluster
_POS_COLORS = ["#e6821e", "#4575b4", "#1a9641", "#d73027",
               "#984ea3", "#8c6d31", "#e78ac3", "#666666"]


def _build_color_map() -> dict[int, dict[int, str]]:
    """Assign colours by cluster lineage: children inherit parent colour;
    only the smaller child (the 'new' cluster born at this k) gets a fresh colour."""
    color_map: dict[int, dict[int, str]] = {}
    next_new = 0   # index into _POS_COLORS for the next truly new cluster

    # k=2: assign by W→E position
    c2 = sorted(_load_solution(2),
                key=lambda c: np.mean([_sta_lons[s.name] for s in c.stations]))
    color_map[2] = {c.id: _POS_COLORS[i] for i, c in enumerate(c2)}
    next_new = len(c2)

    prev = _load_solution(2)
    for k in range(3, 9):
        curr = _load_solution(k)
        assignment: dict[int, str] = {}

        # For each current cluster find the previous cluster it overlaps most with
        best_parent: dict[int, int] = {}   # curr_id -> prev_id
        best_overlap: dict[int, int] = {}
        for c in curr:
            cn = {s.name for s in c.stations}
            bp, bo = None, 0
            for p in prev:
                ov = len(cn & {s.name for s in p.stations})
                if ov > bo:
                    bo, bp = ov, p.id
            best_parent[c.id] = bp
            best_overlap[c.id] = bo

        # Among clusters sharing the same parent, the larger one inherits the colour
        from collections import defaultdict
        children: dict[int, list] = defaultdict(list)
        for c in curr:
            children[best_parent[c.id]].append(c)

        for pid, kids in children.items():
            parent_color = color_map[k - 1].get(pid, _POS_COLORS[next_new % len(_POS_COLORS)])
            kids_sorted = sorted(kids, key=lambda c: -best_overlap[c.id])
            # Largest child keeps parent colour
            assignment[kids_sorted[0].id] = parent_color
            # Remaining children get new colours
            for extra in kids_sorted[1:]:
                assignment[extra.id] = _POS_COLORS[next_new % len(_POS_COLORS)]
                next_new += 1

        color_map[k] = assignment
        prev = curr

    return color_map


_CLUSTER_COLORS = _build_color_map()


# ── Fig 3 — velocity scatter coloured by sorted position, k=2..7 ──────────────
fig, axes = plt.subplots(2, 3, figsize=(10, 10),
                         sharex=True, sharey=True)
axes = axes.flatten()

for ax, k in zip(axes, range(2, 8)):
    clusters = _load_solution(k)
    chi2_total = sum(c.chi2 for c in clusters)
    dof = max(2 * N - 3 * k, 1)

    sorted_c = sorted(clusters,
                      key=lambda c: np.mean([_sta_lons[s.name] for s in c.stations]))
    fault_labels = _BOUNDARY_FAULTS.get(k, [])

    # Plot each cluster with its lineage-tracked colour
    for c in clusters:
        col = _CLUSTER_COLORS[k][c.id]
        ve_c = [s.velocity.ve for s in c.stations]
        vn_c = [s.velocity.vn for s in c.stations]
        ax.scatter(ve_c, vn_c, s=18, color=col, edgecolor="none", alpha=0.75,
                   zorder=3)
        ax.text(np.mean(ve_c), np.mean(vn_c), f"N={c.size}",
                fontsize=6, ha="center", va="center", color=col,
                fontweight="bold", zorder=5)

    # Boundary lines: find the actual velocity gap between adjacent clusters
    # by projecting onto the inter-centroid axis and finding the midpoint
    # between the closest stations from each cluster along that axis.
    for i in range(len(sorted_c) - 1):
        ca, cb = sorted_c[i], sorted_c[i + 1]

        # Full centroids define the separation direction
        ve_ca = np.array([s.velocity.ve for s in ca.stations])
        vn_ca = np.array([s.velocity.vn for s in ca.stations])
        ve_cb = np.array([s.velocity.ve for s in cb.stations])
        vn_cb = np.array([s.velocity.vn for s in cb.stations])

        cen_a = np.array([ve_ca.mean(), vn_ca.mean()])
        cen_b = np.array([ve_cb.mean(), vn_cb.mean()])
        direction = cen_b - cen_a
        length_dir = np.linalg.norm(direction)
        if length_dir < 1e-6:
            continue
        unit = direction / length_dir

        # Project every station onto this axis
        proj_a = np.array([s.velocity.ve for s in ca.stations]) * unit[0] \
               + np.array([s.velocity.vn for s in ca.stations]) * unit[1]
        proj_b = np.array([s.velocity.ve for s in cb.stations]) * unit[0] \
               + np.array([s.velocity.vn for s in cb.stations]) * unit[1]

        # Boundary = midpoint between the max projection of ca and min of cb
        p_max_a = proj_a.max()
        p_min_b = proj_b.min()
        p_mid   = (p_max_a + p_min_b) / 2

        # Midpoint in velocity space at the gap boundary
        ve_m = cen_a[0] + (p_mid - proj_a.mean()) * unit[0]
        vn_m = cen_a[1] + (p_mid - proj_a.mean()) * unit[1]
        dve, dvn = direction[0], direction[1]
        length = length_dir
        if length < 1e-6:
            continue
        perp_e, perp_n = -dvn / length, dve / length
        half = 4.5
        ax.plot([ve_m - half * perp_e, ve_m + half * perp_e],
                [vn_m - half * perp_n, vn_m + half * perp_n],
                color="k", lw=1.2, ls="--", zorder=6)

        fault_name = fault_labels[i] if i < len(fault_labels) else "?"
        rot = np.degrees(np.arctan2(dvn, dve))
        kw = dict(fontsize=6.5, ha="center", va="center", color="k",
                  rotation=rot,
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, lw=0))

        # k=2: two labels — "Awatere" at left end, "Hope" at right end
        if k == 2 and i == 0:
            ax.text(ve_m - half * perp_e * 1.3, vn_m - half * perp_n * 1.3,
                    "Awatere", **kw)
            ax.text(ve_m + half * perp_e * 1.3, vn_m + half * perp_n * 1.3,
                    "Hope", **kw)
        else:
            ax.text(ve_m + half * perp_e * 1.3, vn_m + half * perp_n * 1.3,
                    fault_name, **kw)

    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.axvline(0, color="gray", lw=0.4, ls="--")
    ax.set_title(f"k = {k}   χ²_red = {chi2_total/dof:.1f}", fontsize=10)
    ax.set_aspect("equal", adjustable="box")

    # Inline legend: one colour swatch per cluster, sorted W→E
    legend_handles = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=_CLUSTER_COLORS[k][c.id], markersize=7,
               label=f"C{pos+1}  N={c.size}")
        for pos, c in enumerate(sorted_c)
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=6.5, framealpha=0.85, handletextpad=0.4)

for ax in axes[3:]:
    ax.set_xlabel("Ve  (mm/yr)", fontsize=9)
for ax in axes[::3]:
    ax.set_ylabel("Vn  (mm/yr)", fontsize=9)

fig.suptitle(
    f"Marlborough Fault System — velocity space by cluster  (N = {N}, ITRF2008)\n"
    "Cluster colour tracks lineage across k panels",
    fontsize=11)
fig.tight_layout(h_pad=0.5, w_pad=0.0)
fig.subplots_adjust(wspace=0.0)
fig.savefig(OUT / "fig4_velocity_scatter.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# ── Fig 5 — combined 2×3 cluster maps k=2..7 with lineage colours ─────────────
print("Fig 5: k=2..7 cluster maps …")
K_ALL = [2, 3, 4, 5, 6, 7]
fig, axes = plt.subplots(2, 3, figsize=(22, 14),
                          subplot_kw={"projection": ccrs.Mercator()})
axes_flat = axes.flatten()

for ax, k in zip(axes_flat, K_ALL):
    clusters  = _load_solution(k)
    chi2_total = sum(c.chi2 for c in clusters)
    dof = max(2 * N - 3 * k, 1)
    _basemap(ax)
    _plot_mfs(ax)
    _plot_kaikoura(ax)
    handles = []
    for pos, c in enumerate(sorted(clusters,
                                   key=lambda c: np.mean([s.position.lon
                                                          for s in c.stations]))):
        col = _CLUSTER_COLORS[k][c.id]
        _scatter(ax, c.stations, col, s=30)
        handles.append(Line2D([0],[0], color=col, lw=0, marker="o",
            markersize=7, markeredgecolor="k", markeredgewidth=0.4,
            label=(f"C{pos+1}  N={c.size}\n"
                   f"({c.pole.lat:.0f}°N,{c.pole.lon:.0f}°E) "
                   f"{c.pole.rate:.3f}°/Ma")))
    ax.scatter(173.054, -42.737, transform=ccrs.PlateCarree(),
               marker="*", s=220, color="red", edgecolor="darkred",
               lw=0.8, zorder=8)
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, framealpha=0.92)
    ax.set_title(f"k = {k}   χ²_red = {chi2_total/dof:.1f}", fontsize=11)

# Fault legend on last panel
axes_flat[-1].legend(handles=_MFS_LEGEND, loc="upper right", fontsize=8,
                     title="Litchfield faults", framealpha=0.92, title_fontsize=8)

fig.suptitle(
    "Marlborough Fault System — Euler-vector clustering  k = 2 … 7\n"
    "Cluster colours track lineage   ★ = Mw 7.8 Kaikōura 2016",
    fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "fig3_k_all.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Appendix: predicted velocities 2×3
fig, axes = plt.subplots(2, 3, figsize=(22, 14),
                          subplot_kw={"projection": ccrs.Mercator()})
for ax, k in zip(axes.flatten(), K_ALL):
    clusters  = _load_solution(k)
    chi2_total = sum(c.chi2 for c in clusters)
    dof = max(2 * N - 3 * k, 1)
    _basemap(ax)
    _plot_mfs(ax)
    _plot_kaikoura(ax)
    for c in sorted(clusters,
                    key=lambda c: np.mean([s.position.lon for s in c.stations])):
        col = _CLUSTER_COLORS[k][c.id]
        _scatter(ax, c.stations, col, s=20)
        _quiver(ax, c.stations, col)
    ax.scatter(173.054, -42.737, transform=ccrs.PlateCarree(),
               marker="*", s=220, color="red", edgecolor="darkred",
               lw=0.8, zorder=8)
    ax.set_title(f"k = {k}  predicted velocities   χ²_red = {chi2_total/dof:.1f}",
                 fontsize=10)
fig.suptitle("Marlborough — predicted velocity vectors  k = 2 … 7  (appendix)",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "figA1_velocities.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"\nAll figures saved to {OUT}/")

# ── helpers ────────────────────────────────────────────────────────────────────
def _small_circle_lonlat(pole_lat_deg, pole_lon_deg, angular_dist_deg, n=720):
    """True small circle: all points at angular_dist_deg from the pole.
    Uses 3-D rotation around the pole vector — avoids geodesic-arc wedging."""
    R  = np.deg2rad
    # Pole unit vector in ECEF
    lat_p, lon_p = R(pole_lat_deg), R(pole_lon_deg)
    P  = np.array([np.cos(lat_p)*np.cos(lon_p),
                   np.cos(lat_p)*np.sin(lon_p),
                   np.sin(lat_p)])
    # Any vector not parallel to P — use north pole if needed
    ref = np.array([0., 0., 1.])
    if abs(np.dot(P, ref)) > 0.99:
        ref = np.array([1., 0., 0.])
    # Orthonormal basis in the plane perpendicular to P
    u = np.cross(P, ref); u /= np.linalg.norm(u)
    v = np.cross(P, u);   v /= np.linalg.norm(v)
    theta = R(angular_dist_deg)
    phis  = np.linspace(0, 2*np.pi, n, endpoint=True)
    pts   = np.cos(theta)*P[:, None] + np.sin(theta)*(np.cos(phis)*u[:, None]
                                                     + np.sin(phis)*v[:, None])
    # ECEF → lon/lat
    lons = np.degrees(np.arctan2(pts[1], pts[0]))
    lats = np.degrees(np.arcsin(np.clip(pts[2], -1, 1)))
    return lons, lats


# ── Fig 6 — Euler poles orthographic globe ─────────────────────────────────────
print("Fig 6: Euler poles globe …")

K_POLES = [3, 4, 5]
fig = plt.figure(figsize=(16, 5))

for col_idx, k in enumerate(K_POLES):
    ax = fig.add_subplot(1, 3, col_idx + 1,
                         projection=ccrs.Orthographic(
                             central_longitude=170, central_latitude=-40))
    ax.set_global()
    ax.add_feature(cfeature.LAND,      facecolor="#e8e4dc", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#cde8f5", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="gray", zorder=1)
    ax.gridlines(linewidth=0.3, color="gray", alpha=0.4, zorder=1)

    clusters   = _load_solution(k)
    chi2_total = sum(c.chi2 for c in clusters)
    dof = max(2 * N - 3 * k, 1)
    sorted_cl  = sorted(clusters,
                        key=lambda c: np.mean([s.position.lon for s in c.stations]))

    for pos, c in enumerate(sorted_cl):
        col  = _CLUSTER_COLORS[k][c.id]
        pole = c.pole

        ax.scatter(pole.lon, pole.lat, transform=ccrs.PlateCarree(),
                   s=120, color=col, edgecolor="k", lw=0.8, zorder=6, marker="*")
        ax.text(pole.lon + 2, pole.lat + 2,
                f"C{pos+1}\n{pole.lat:.0f}°N,{pole.lon:.0f}°E\n{pole.rate:.3f}°/Ma",
                transform=ccrs.PlateCarree(), fontsize=6, color=col,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, lw=0),
                zorder=7)

        # True small circle (3-D rotation, not geodesic arcs)
        mean_lat = np.mean([s.position.lat for s in c.stations])
        mean_lon = np.mean([s.position.lon for s in c.stations])
        # Angular distance from pole to cluster centroid
        dlat = np.deg2rad(mean_lat - pole.lat)
        dlon = np.deg2rad(mean_lon - pole.lon)
        a    = (np.sin(dlat/2)**2
                + np.cos(np.deg2rad(pole.lat)) * np.cos(np.deg2rad(mean_lat))
                * np.sin(dlon/2)**2)
        ang_dist = np.degrees(2 * np.arcsin(np.sqrt(a)))
        arc_lons, arc_lats = _small_circle_lonlat(pole.lat, pole.lon, ang_dist)
        ax.plot(arc_lons, arc_lats, transform=ccrs.PlateCarree(),
                color=col, lw=1.0, alpha=0.6, ls="--", zorder=4)

        ax.scatter([s.position.lon for s in c.stations],
                   [s.position.lat for s in c.stations],
                   transform=ccrs.PlateCarree(),
                   s=8, color=col, alpha=0.7, zorder=5)

    ext = EXTENT
    ax.plot([ext[0], ext[1], ext[1], ext[0], ext[0]],
            [ext[2], ext[2], ext[3], ext[3], ext[2]],
            transform=ccrs.PlateCarree(), color="gold", lw=1.8, zorder=8)
    ax.set_title(f"k = {k}   χ²_red = {chi2_total/dof:.1f}", fontsize=11)

fig.suptitle(
    "Marlborough Fault System — Euler poles (ITRF2008)\n"
    "★ = Euler pole   dashed arc = mean angular distance   gold box = study domain",
    fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig6_euler_poles.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Fig 7 — optimal k clusters + residual velocities side-by-side ─────────────
print("Fig 7: optimal clusters + residuals …")
K_OPT = 3   # k=3 recovers Kaikōura zone

clusters_opt = _load_solution(K_OPT)
sorted_opt   = sorted(clusters_opt,
                      key=lambda c: np.mean([s.position.lon for s in c.stations]))

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 8),
                                   subplot_kw={"projection": ccrs.Mercator()})

for ax in (ax_l, ax_r):
    _basemap(ax)
    _plot_mfs(ax)
    _plot_kaikoura(ax)

# Left: cluster assignments
handles = []
for pos, c in enumerate(sorted_opt):
    col = _CLUSTER_COLORS[K_OPT][c.id]
    _scatter(ax_l, c.stations, col, s=35)
    handles.append(Line2D([0],[0], color=col, lw=0, marker="o",
        markersize=8, markeredgecolor="k", markeredgewidth=0.4,
        label=f"C{pos+1}  N={c.size}\n({c.pole.lat:.0f}°N,{c.pole.lon:.0f}°E) {c.pole.rate:.3f}°/Ma"))
ax_l.scatter(173.054, -42.737, transform=ccrs.PlateCarree(),
             marker="*", s=260, color="red", edgecolor="darkred", lw=0.8, zorder=8)
ax_l.legend(handles=handles, loc="lower left", fontsize=7.5, framealpha=0.92)
ax_l.set_title(f"k = {K_OPT} cluster assignments", fontsize=12)

# Right: residual velocities
max_res = 0.0
all_res = []
for c in clusters_opt:
    for s in c.stations:
        ve_p, vn_p = predict_velocity(s, c.euler_vector)
        all_res.append((s, ve_p, vn_p, c))
        res = np.sqrt((s.velocity.ve - ve_p)**2 + (s.velocity.vn - vn_p)**2)
        if res > max_res:
            max_res = res

res_lons = np.array([r[0].position.lon for r in all_res])
res_lats = np.array([r[0].position.lat for r in all_res])
res_ve   = np.array([r[0].velocity.ve - r[1] for r in all_res])
res_vn   = np.array([r[0].velocity.vn - r[2] for r in all_res])
res_cols = [_CLUSTER_COLORS[K_OPT][r[3].id] for r in all_res]

q = ax_r.quiver(res_lons, res_lats, res_ve, res_vn,
                transform=ccrs.PlateCarree(),
                scale=40, scale_units="width", angles="uv",
                width=0.004, headwidth=4, headlength=5,
                color=res_cols, alpha=0.85, zorder=6)
ax_r.quiverkey(q, X=0.12, Y=0.06, U=5, label="5 mm/yr residual",
               labelpos="S", fontproperties={"size": 7})
ax_r.set_title(f"k = {K_OPT} residual velocities (obs − predicted)", fontsize=12)

fig.suptitle(
    "Marlborough Fault System — optimal k = 3\n"
    "★ = Mw 7.8 Kaikōura 2016 epicentre",
    fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "fig7_clusters_residuals.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ── Fig 8 — slip rates on boundary faults ─────────────────────────────────────
print("Fig 8: slip rates …")
from gps_cluster.domain.services.euler_math import fault_slip_rate

# At k=3, boundaries: C1(west)|C3(mid)=Awatere, C3(mid)|C2(east)=Hope
# Sorted by longitude: pos0=C1, pos1=C3, pos2=C2
_boundary_pairs = [
    ("Awatere", ["Awatere"], sorted_opt[0], sorted_opt[1]),
    ("Hope",    ["Hope"],    sorted_opt[1], sorted_opt[2]),
]

fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.Mercator()})
_basemap(ax)
_plot_mfs(ax)
_plot_kaikoura(ax)

# Reference scale circle
_scale_rate = 10.0  # mm/yr = reference circle size
_ref_s = 200        # scatter marker size for 10 mm/yr

legend_sr = []

for fault_name, keywords, c_west, c_east in _boundary_pairs:
    col_w = _CLUSTER_COLORS[K_OPT][c_west.id]
    col_e = _CLUSTER_COLORS[K_OPT][c_east.id]
    fault_col = _MFS_GROUPS.get(fault_name, ("#555555", []))[0]

    # Get fault segments
    fault_segs = _MFS_GDF[_MFS_GDF["group"] == fault_name]

    # Sample points along fault segments
    fault_lats, fault_lons = [], []
    for _, row in fault_segs.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        xs, ys = geom.xy
        # Sample every ~5th vertex
        for j in range(0, len(xs), max(1, len(xs)//8)):
            fault_lons.append(xs[j])
            fault_lats.append(ys[j])

    if not fault_lats:
        continue

    # Compute slip rates
    results = fault_slip_rate(
        c_west.euler_vector, c_east.euler_vector,
        fault_lats, fault_lons,
    )

    rates = np.array([r["total_mm_yr"] for r in results])
    sizes = (rates / _scale_rate) * _ref_s

    sc = ax.scatter(fault_lons, fault_lats,
                    s=sizes, c=rates, cmap="Reds",
                    vmin=0, vmax=30,
                    transform=ccrs.PlateCarree(),
                    edgecolors=fault_col, linewidths=1.2,
                    zorder=8, alpha=0.85)
    # Annotate mean rate
    ax.text(np.mean(fault_lons), np.mean(fault_lats) + 0.12,
            f"{fault_name}: {rates.mean():.1f} mm/yr",
            transform=ccrs.PlateCarree(), fontsize=8, fontweight="bold",
            color=fault_col, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, lw=0),
            zorder=9)

cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
cb.set_label("Slip rate  (mm/yr)", fontsize=9)

# Reference circle legend
for ref_r in [5, 10, 20]:
    ax.scatter([], [], s=(ref_r/_scale_rate)*_ref_s,
               c="gray", edgecolors="k", linewidths=0.8, alpha=0.7,
               label=f"{ref_r} mm/yr", transform=ccrs.PlateCarree())
ax.legend(title="Slip rate scale", loc="upper right", fontsize=8,
          framealpha=0.92, title_fontsize=8)
ax.set_title(
    f"Marlborough Fault System — fault slip rates from Euler poles  (k = {K_OPT})\n"
    "Circle size and colour = total relative rate at fault sampling points",
    fontsize=11)
fig.savefig(OUT / "fig8_slip_rates.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"\nAll figures saved to {OUT}/")
