"""EAF region — fault slip rates for k=2, 3, 4 as three subplots.

Reads results/eaf/clusters.json.  Run compute_eaf_clusters.py first.

For each k, only faults that lie on a cluster boundary produce circles.
Same-cluster faults are silently skipped (same-cluster guard).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.cm import ScalarMappable

from gps_cluster.domain.entities import GpsStation, Position, Velocity
from gps_cluster.domain.services.euler_math import (
    EulerVector,
    euler_vector_to_pole,
    fault_slip_rate,
)
from gps_cluster.domain.services.fault_analysis import (
    assign_sides,
    compute_rake,
    fault_strike_from_geom,
)

ROOT       = Path(__file__).parent.parent
CACHE      = ROOT / "results/eaf/clusters.json"
FAULT_FILE = Path("/Users/ali/Repos/StrainTool/bin/mta_emme_fault_map.shp")
OUT        = ROOT / "results/eaf"

EXTENT = [34.0, 42.0, 36.0, 39.0]
K_LIST = [2]
SAMPLE_SPACING = 25

if not CACHE.exists():
    raise FileNotFoundError(f"{CACHE} — run compute_eaf_clusters.py first.")

with open(CACHE) as f:
    cache = json.load(f)

station_by_name = {
    r["name"]: GpsStation(
        name=r["name"], position=Position(lat=r["lat"], lon=r["lon"]),
        velocity=Velocity(ve=r["ve"], vn=r["vn"], vu=0.,
                          se=r["se"], sn=r["sn"], su=1.))
    for r in cache["stations"]
}
stations = list(station_by_name.values())
N = len(stations)
print(f"EAF cache: {N} stations")

def _load_solution(k):
    sol = []
    for c in cache["solutions"][str(k)]:
        ev = EulerVector(ox=c["euler"]["ox"], oy=c["euler"]["oy"], oz=c["euler"]["oz"])
        ns = SimpleNamespace(
            id=c["id"], size=c["size"], chi2=c["chi2"],
            euler_vector=ev, pole=SimpleNamespace(**c["pole"]),
            stations=[station_by_name[n] for n in c["stations"] if n in station_by_name])
        sol.append(ns)
    return sol

# ── EMME faults clipped to EAF domain ─────────────────────────────────────────
emme = gpd.read_file(FAULT_FILE, encoding="latin-1").to_crs("EPSG:4326")
emme = emme.cx[EXTENT[0]:EXTENT[1], EXTENT[2]:EXTENT[3]]

# ── simplified fault traces (user-curated kinematic boundaries) ───────────────
SIMPLIFIED_GDF = gpd.read_file(ROOT / "data/raw/eaf_slip_rate_faults_simplified.geojson")
print(f"Simplified: {len(SIMPLIFIED_GDF)} segments — "
      f"{SIMPLIFIED_GDF['group'].value_counts().to_dict()}")

# Which groups are kinematic boundaries at each k
GROUPS_BY_K = {
    2: ["EAF", "DSF"],
    3: ["EAF", "DSF", "Surgu", "Cardak"],
    4: ["EAF", "DSF", "Surgu", "Cardak"],
}


def _build_traces(k):
    allowed = GROUPS_BY_K[k]
    return [(row["group"], row.geometry)
            for _, row in SIMPLIFIED_GDF.iterrows()
            if row["group"] in allowed]

# fault_strike_from_geom, compute_rake, assign_sides imported from
# gps_cluster.domain.services.fault_analysis
# (fault_strike is an alias kept for local readability)
fault_strike = fault_strike_from_geom

# ── colormap ──────────────────────────────────────────────────────────────────
def _make_rake_cmap(n=512):
    anchors_x    = [0.00, 0.25, 0.50, 0.75, 1.00]
    anchors_rgba = [
        (0.95, 0.25, 0.05, 1.0),
        (0.10, 0.80, 0.20, 1.0),
        (0.10, 0.30, 0.90, 1.0),
        (0.90, 0.10, 0.60, 1.0),
        (0.95, 0.25, 0.05, 1.0),
    ]
    xs   = np.linspace(0, 1, n)
    rgba = np.zeros((n, 4))
    for ch in range(4):
        vals = [c[ch] for c in anchors_rgba]
        rgba[:, ch] = np.interp(xs, anchors_x, vals)
    return mcolors.ListedColormap(rgba, name="tectonic_rake")

_RAKE_CMAP = plt.colormaps["hsv"]

def rake_to_color(rake_deg):
    return _RAKE_CMAP(((rake_deg + 180) % 360) / 360.0)

# assign_sides imported from gps_cluster.domain.services.fault_analysis

# ── compute slip rates for each k ─────────────────────────────────────────────
def compute_results(k):
    clusters     = _load_solution(k)
    fault_traces = _build_traces(k)
    results      = []
    for gname, geom in fault_traces:
        segs = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for seg in segs:
            xs, ys = list(seg.xy[0]), list(seg.xy[1])
            if len(xs) < 2: continue
            for j in range(0, len(xs), max(1, SAMPLE_SPACING)):
                lon_pt, lat_pt = xs[j], ys[j]
                strike = fault_strike(xs, ys, j)
                c_right, c_left = assign_sides(clusters, lon_pt, lat_pt, strike)
                if c_right.id == c_left.id:
                    continue
                res = fault_slip_rate(
                    c_right.euler_vector, c_left.euler_vector,
                    [lat_pt], [lon_pt], fault_strike_deg=strike)[0]
                results.append(dict(
                    lon=lon_pt, lat=lat_pt,
                    total=res["total_mm_yr"],
                    ss=res["strike_slip_mm_yr"],
                    fn=res["fault_normal_mm_yr"],
                    rake=compute_rake(res["strike_slip_mm_yr"], res["fault_normal_mm_yr"]),
                    strike=strike,
                    group=gname,
                ))
    print(f"  k={k}: {len(results)} sample points")
    return results

# ── figure: 3 subplots ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(K_LIST), figsize=(8 * len(K_LIST), 7),
                          subplot_kw={"projection": ccrs.Mercator()})

SCALE_RATE = 10.0
REF_S      = 300

for ax, k in zip(np.atleast_1d(axes), K_LIST):
    print(f"k={k} …")
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#f0ede6", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#daeef8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#666", zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="#aaa",
                   linestyle=":", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--", crs=ccrs.PlateCarree())
    gl.top_labels = False; gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(34, 43, 2))
    gl.ylocator = mticker.FixedLocator(range(36, 40))
    gl.xlabel_style = {"size": 7}; gl.ylabel_style = {"size": 7}

    # EMME faults — thin grey reference within box
    for _, row in emme.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        segs = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for seg in segs:
            xs, ys = seg.xy
            ax.plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
                    color="gray", lw=0.4, alpha=0.25, zorder=2)

    # Simplified faults — black bold, below slip rate colouring
    for _, row in SIMPLIFIED_GDF.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        segs = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for seg in segs:
            xs, ys = seg.xy
            ax.plot(list(xs), list(ys), transform=ccrs.PlateCarree(),
                    color="black", lw=1.8, alpha=0.7, zorder=5)

    # 2023 earthquake epicentres
    ax.scatter(37.55, 37.17, transform=ccrs.PlateCarree(),
               marker="*", s=200, color="red", edgecolor="darkred",
               lw=0.8, zorder=9)
    ax.scatter(37.24, 38.02, transform=ccrs.PlateCarree(),
               marker="*", s=200, color="red", edgecolor="darkred",
               lw=0.8, zorder=9)

    # Slip rates as coloured piecewise-linear fault segments (non-overlapping)
    # For each simplified fault geometry, densely resample the polyline,
    # compute rate+rake at each point, then draw each consecutive pair as a
    # coloured segment with linewidth ∝ rate.
    LW_SCALE  = 0.4    # lw per mm/yr
    LW_MIN    = 1.0
    N_RESAMP  = 200    # resample each geometry to this many points

    def _resample_geom(geom, n):
        """Return n evenly-spaced (lon, lat) points along the geometry."""
        from shapely.geometry import LineString
        if geom.geom_type == "MultiLineString":
            coords = []
            for part in geom.geoms:
                coords.extend(list(part.coords))
            geom = LineString(coords)
        total = geom.length
        dists = np.linspace(0, total, n)
        pts   = [geom.interpolate(d) for d in dists]
        return np.array([(p.x, p.y) for p in pts])

    clusters_k = _load_solution(k)
    for gname in GROUPS_BY_K[k]:
        segs_gdf = SIMPLIFIED_GDF[SIMPLIFIED_GDF["group"] == gname]
        for _, row in segs_gdf.iterrows():
            pts = _resample_geom(row.geometry, N_RESAMP)
            for i in range(len(pts) - 1):
                lon_m = (pts[i, 0] + pts[i+1, 0]) / 2
                lat_m = (pts[i, 1] + pts[i+1, 1]) / 2
                dx = pts[i+1, 0] - pts[i, 0]
                dy = pts[i+1, 1] - pts[i, 1]
                strike = float(np.degrees(np.arctan2(dx, dy)) % 360)
                if strike >= 180:
                    strike -= 180
                c_right, c_left = assign_sides(clusters_k, lon_m, lat_m, strike)
                if c_right.id == c_left.id:
                    # no boundary — draw thin grey
                    ax.plot([pts[i,0], pts[i+1,0]], [pts[i,1], pts[i+1,1]],
                            transform=ccrs.PlateCarree(),
                            color="gray", lw=0.8, alpha=0.4, zorder=5)
                    continue
                res  = fault_slip_rate(c_right.euler_vector, c_left.euler_vector,
                                       [lat_m], [lon_m], fault_strike_deg=strike)[0]
                ss   = res["strike_slip_mm_yr"]
                fn   = res["fault_normal_mm_yr"]
                tot  = res["total_mm_yr"]
                rake = compute_rake(ss, fn)
                col  = rake_to_color(rake)
                lw   = max(LW_MIN, tot * LW_SCALE)
                ax.plot([pts[i,0], pts[i+1,0]], [pts[i,1], pts[i+1,1]],
                        transform=ccrs.PlateCarree(),
                        color=col, lw=lw, solid_capstyle="butt",
                        alpha=0.92, zorder=6)

    # Cluster boundary labels (poles)
    clusters = _load_solution(k)
    for pos, c in enumerate(sorted(clusters,
                                    key=lambda c: np.mean([s.position.lon
                                                            for s in c.stations]))):
        pole = c.pole
        print(f"    C{c.id}  N={c.size}  pole=({pole.lat:.1f}N, {pole.lon:.1f}E)"
              f"  {pole.rate:.3f}°/Ma")

    ax.set_title(f"k = {k}", fontsize=12)

# Shared colorbar — explicit axes to the right of the 3rd panel
fig.tight_layout(rect=[0, 0, 0.92, 1])
cax = fig.add_axes([0.93, 0.30, 0.015, 0.40])
sm  = ScalarMappable(cmap=_RAKE_CMAP, norm=mcolors.Normalize(vmin=0, vmax=1))
sm.set_array(np.linspace(0, 1, 512))
cb  = fig.colorbar(sm, cax=cax)
cb.set_label("Rake  (°)", fontsize=9)
tick_rakes = [-180, -90, 0, 90, 180]
cb.set_ticks([((r + 180) % 360) / 360.0 for r in tick_rakes])
cb.set_ticklabels(["±180°\nDextral", "−90°\nNormal", "0°\nSinistral",
                   "+90°\nThrust", "±180°\nDextral"], fontsize=7)

# Linewidth legend
legend_ax = np.atleast_1d(axes)[-1]
for ref_r in [5, 10]:
    legend_ax.plot([], [], color="gray", lw=max(LW_MIN, ref_r * LW_SCALE),
                   label=f"{ref_r} mm/yr")
legend_ax.legend(title="Slip rate (line width)", loc="lower right",
                 fontsize=7, framealpha=0.9, title_fontsize=7)

fig.suptitle(
    "EAF region — fault slip rates from Euler-vector clustering\n"
    "k = 2, 3, 4   |   ★ = 2023 Kahramanmaraş Mw 7.8+7.7 epicentres\n"
    "Only boundaries between distinct kinematic clusters shown",
    fontsize=11, y=1.01)
fig.savefig(OUT / "fig_slip_rates_k234.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {OUT}/fig_slip_rates_k234.png")
