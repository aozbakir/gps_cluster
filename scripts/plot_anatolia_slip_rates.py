"""Fault slip rates for Anatolia — circles on fault traces coloured by rake.

Reads pre-computed clusters from results/anatolia/clusters.json.
Run compute_anatolia_clusters.py first if the cache is missing.

Rake is computed from the relative Euler velocity resolved onto the local
fault plane (strike-slip + fault-normal components):
  +180 / -180 = dextral     0 = sinistral
     +90      = thrust    -90 = normal
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
CACHE      = ROOT / "results/anatolia/clusters.json"
FAULT_FILE = Path("/Users/ali/Repos/StrainTool/bin/mta_emme_fault_map.shp")
OUT        = ROOT / "results/anatolia"
OUT.mkdir(parents=True, exist_ok=True)

EXTENT = [25.0, 45.5, 35.5, 43.0]
K_OPT  = 5

if not CACHE.exists():
    raise FileNotFoundError(f"{CACHE} not found — run compute_anatolia_clusters.py first.")

with open(CACHE) as f:
    cache = json.load(f)

# Reconstruct stations and cluster solutions
station_records = {r["name"]: r for r in cache["stations"]}
stations = [
    GpsStation(name=r["name"], position=Position(lat=r["lat"], lon=r["lon"]),
               velocity=Velocity(ve=r["ve"], vn=r["vn"], vu=0.0,
                                 se=r["se"], sn=r["sn"], su=1.0))
    for r in cache["stations"]
]
station_by_name = {s.name: s for s in stations}

def _load_solution(k):
    sol = []
    for c in cache["solutions"][str(k)]:
        ev = EulerVector(ox=c["euler"]["ox"], oy=c["euler"]["oy"], oz=c["euler"]["oz"])
        ns = SimpleNamespace(
            id=c["id"], size=c["size"], chi2=c["chi2"],
            euler_vector=ev,
            pole=SimpleNamespace(**c["pole"]),
            stations=[station_by_name[n] for n in c["stations"] if n in station_by_name],
        )
        sol.append(ns)
    return sol

clusters = _load_solution(K_OPT)
print(f"Loaded k={K_OPT}: {len(clusters)} clusters")
for c in sorted(clusters, key=lambda c: c.pole.lat):
    print(f"  C{c.id}  N={c.size}  pole=({c.pole.lat:.1f}°N, {c.pole.lon:.1f}°E) "
          f"{c.pole.rate:.3f}°/Ma")

# ── EMME background + simplified kinematic boundaries ─────────────────────────
emme_all = gpd.read_file(FAULT_FILE, encoding="latin-1").to_crs("EPSG:4326")
emme_all = emme_all.cx[EXTENT[0]:EXTENT[1], EXTENT[2]:EXTENT[3]]

SIMPLIFIED_GDF = gpd.read_file(ROOT / "data/raw/anatolia_slip_rate_faults_simplified.geojson")
# Use SOURCENAME as fault group label; fall back to index if column absent
_name_col = "SOURCENAME" if "SOURCENAME" in SIMPLIFIED_GDF.columns else (
            "group"      if "group"      in SIMPLIFIED_GDF.columns else None)
print(f"Simplified: {len(SIMPLIFIED_GDF)} segments, name col='{_name_col}'")

fault_traces = [
    (str(row[_name_col]) if _name_col else str(i), row.geometry)
    for i, (_, row) in enumerate(SIMPLIFIED_GDF.iterrows())
]

# ── rake convention ───────────────────────────────────────────────────────────
# strike_slip > 0 → dextral (right-lateral) along fault
# fault_normal > 0 → opening (normal)
# rake: arctan2(fault_normal, strike_slip), mapped to [-180, 180]
# Reference image convention:
#   0   = sinistral (left-lateral)  → strike_slip < 0
#   180 = dextral  (right-lateral)  → strike_slip > 0
#   +90 = thrust                    → fault_normal < 0 (shortening)
#   -90 = normal                    → fault_normal > 0 (opening)
# We map: rake_display = arctan2(-fault_normal, strike_slip) * 180/pi
# This gives 0=sinistral, 180=dextral, +90=thrust, -90=normal

# compute_rake, fault_strike_from_geom, assign_sides imported from
# gps_cluster.domain.services.fault_analysis


# ── Rake colormap — 4-anchor cyclic, saturated distinct colours ───────────────
# Anchors at tectonic end-members:
#   dextral (+180°) → orange-red   — warm, dominant strike-slip on NAF/EAF
#   thrust  ( +90°) → lime green   — compressional
#   sinistral(  0°) → royal blue   — sinistral strike-slip
#   normal  ( -90°) → hot pink     — extensional (W.Anatolia grabens)

import matplotlib.colors as mcolors

def _make_rake_cmap(n=512):
    anchors_x    = [0.00, 0.25, 0.50, 0.75, 1.00]
    anchors_rgba = [
        (0.95, 0.25, 0.05, 1.0),   # dextral   — orange-red
        (0.10, 0.80, 0.20, 1.0),   # thrust    — lime green
        (0.10, 0.30, 0.90, 1.0),   # sinistral — royal blue
        (0.90, 0.10, 0.60, 1.0),   # normal    — hot pink
        (0.95, 0.25, 0.05, 1.0),   # dextral   — close loop
    ]
    xs   = np.linspace(0, 1, n)
    rgba = np.zeros((n, 4))
    for ch in range(4):
        vals = [c[ch] for c in anchors_rgba]
        rgba[:, ch] = np.interp(xs, anchors_x, vals)
    return mcolors.ListedColormap(rgba, name="tectonic_rake")

_RAKE_CMAP = _make_rake_cmap()

def rake_to_color(rake_deg: float):
    norm = ((rake_deg + 180) % 360) / 360.0
    return _RAKE_CMAP(norm)


# ── sample faults and compute slip rates between adjacent clusters ─────────────
# Adjacent pairs at k=5 (sorted by centroid lat N→S, boundary defined by
# which fault separates them geographically).
# We compute for ALL fault segments: find the two clusters whose stations
# are on either side of the fault, compute relative Euler, resolve onto fault.

SAMPLE_SPACING = 40   # sample every Nth vertex — sparse, non-overlapping circles


_cluster_by_id = {c.id: c for c in clusters}


results = []   # list of dicts per sample point, keyed by fault group

print("\nComputing slip rates on major faults …")
for gname, geom in fault_traces:
    if geom is None or geom.is_empty:
        continue

    if geom.geom_type == "LineString":
        segs = [geom]
    elif geom.geom_type == "MultiLineString":
        segs = list(geom.geoms)
    else:
        continue

    for seg in segs:
        xs, ys = seg.xy
        xs, ys = list(xs), list(ys)
        if len(xs) < 2:
            continue

        for j in range(0, len(xs), max(1, SAMPLE_SPACING)):
            lon_pt, lat_pt = xs[j], ys[j]
            strike = fault_strike_from_geom(xs, ys, j)

            c_right, c_left = assign_sides(clusters, lon_pt, lat_pt, strike)

            # Skip if both sides belong to the same cluster — fault is internal
            if c_right.id == c_left.id:
                continue

            res = fault_slip_rate(
                c_right.euler_vector, c_left.euler_vector,
                [lat_pt], [lon_pt],
                fault_strike_deg=strike,
            )[0]

            ss   = res["strike_slip_mm_yr"]
            fn   = res["fault_normal_mm_yr"]
            tot  = res["total_mm_yr"]
            rake = compute_rake(ss, fn)

            results.append(dict(
                lon=lon_pt, lat=lat_pt,
                total=tot, ss=ss, fn=fn,
                rake=rake, group=gname,
                c_right=c_right.id, c_left=c_left.id,
            ))

print(f"  Total: {len(results)} sample points")

print(f"  {len(results)} fault sample points")

# ── figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 10),
                        subplot_kw={"projection": ccrs.Mercator()})
ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND,      facecolor="#f0ede6", zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor="#daeef8", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#666", zorder=1)
ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="#aaa",
               linestyle=":", zorder=1)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                  alpha=0.5, linestyle="--", crs=ccrs.PlateCarree())
gl.top_labels = False; gl.right_labels = False
gl.xlocator   = mticker.FixedLocator(range(26, 46, 2))
gl.ylocator   = mticker.FixedLocator(range(36, 44, 1))
gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}

# EMME — thin grey background
for _, row in emme_all.iterrows():
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

# Coloured piecewise slip rate segments on top
LW_SCALE = 0.4; LW_MIN = 1.0
SEG_LEN_DEG = 0.45   # ~50 km per coloured segment

def _resample_geom(geom, n):
    from shapely.geometry import LineString as _LS
    if geom.geom_type == "MultiLineString":
        coords = [c for part in geom.geoms for c in list(part.coords)]
        geom = _LS(coords)
    total = geom.length
    pts = [geom.interpolate(d) for d in np.linspace(0, total, n)]
    return np.array([(p.x, p.y) for p in pts])

for gname, geom in fault_traces:
    if geom is None or geom.is_empty: continue
    n_segs = max(2, int(round(geom.length / SEG_LEN_DEG)) + 1)
    pts = _resample_geom(geom, n_segs)
    for i in range(len(pts) - 1):
        lon_m = (pts[i,0] + pts[i+1,0]) / 2
        lat_m = (pts[i,1] + pts[i+1,1]) / 2
        dx = pts[i+1,0] - pts[i,0]; dy = pts[i+1,1] - pts[i,1]
        strike = float(np.degrees(np.arctan2(dx, dy)) % 360)
        if strike >= 180: strike -= 180
        c_right, c_left = assign_sides(clusters, lon_m, lat_m, strike)
        if c_right.id == c_left.id:
            continue
        res  = fault_slip_rate(c_right.euler_vector, c_left.euler_vector,
                               [lat_m], [lon_m], fault_strike_deg=strike)[0]
        rake = compute_rake(res["strike_slip_mm_yr"], res["fault_normal_mm_yr"])
        col  = rake_to_color(rake)
        lw   = max(LW_MIN, res["total_mm_yr"] * LW_SCALE)
        ax.plot([pts[i,0], pts[i+1,0]], [pts[i,1], pts[i+1,1]],
                transform=ccrs.PlateCarree(),
                color=col, lw=lw, solid_capstyle="butt", alpha=0.92, zorder=6)

# Colorbar
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
for ref_r in [10, 20]:
    ax.plot([], [], color="gray", lw=max(LW_MIN, ref_r * LW_SCALE),
            label=f"{ref_r} mm/yr")
ax.legend(title="Slip rate scale", loc="lower right",
          fontsize=8, framealpha=0.92, title_fontsize=8)

ax.set_title(
    f"Anatolia — Euler-vector clustering k = {K_OPT}:  fault slip rates\n"
    "Line width = total rate   Colour = rake  (HSV: 0° sinistral · ±180° dextral · +90° thrust · −90° normal)",
    fontsize=11)

fig.savefig(OUT / "fig_slip_rates.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}/fig_slip_rates.png")
