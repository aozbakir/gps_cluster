"""NAF slip rate profile — strike-slip and fault-normal vs. longitude.

Uses cached k=5 clusters (results/anatolia/clusters.json).
Cluster sides determined at each fault point via assign_sides(), consistent
with plot_anatolia_slip_rates.py.  Sign convention from euler_math:
  strike_slip > 0  → right-lateral (dextral)
  fault_normal > 0 → opening (extension)
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from gps_cluster.domain.entities import GpsStation, Position, Velocity
from gps_cluster.domain.services.euler_math import EulerVector, fault_slip_rate
from gps_cluster.domain.services.fault_analysis import assign_sides

ROOT  = Path(__file__).parent.parent
CACHE = ROOT / "results/anatolia/clusters.json"
OUT   = ROOT / "results/anatolia"

if not CACHE.exists():
    raise FileNotFoundError(f"{CACHE} — run compute_anatolia_clusters.py first")

with open(CACHE) as f:
    cache = json.load(f)

K_OPT = 5

stations = [
    GpsStation(name=r["name"], position=Position(lat=r["lat"], lon=r["lon"]),
               velocity=Velocity(ve=r["ve"], vn=r["vn"], vu=0.0,
                                 se=r["se"], sn=r["sn"], su=1.0))
    for r in cache["stations"]
]

def _load_solution(k):
    sol = []
    station_by_name = {s.name: s for s in stations}
    for c in cache["solutions"][str(k)]:
        ev = EulerVector(ox=c["euler"]["ox"], oy=c["euler"]["oy"], oz=c["euler"]["oz"])
        ns = SimpleNamespace(
            id=c["id"], size=c["size"],
            euler_vector=ev,
            pole=SimpleNamespace(**c["pole"]),
            stations=[station_by_name[n] for n in c["stations"] if n in station_by_name],
        )
        sol.append(ns)
    return sol

clusters = _load_solution(K_OPT)
_cluster_by_id = {c.id: c for c in clusters}

# assign_sides imported from gps_cluster.domain.services.fault_analysis


# ── NAF trace sampling ─────────────────────────────────────────────────────────
# Northern main strand — Marmara to Erzincan
naf_main_pts = [
    (40.42, 26.1),
    (40.68, 27.1),
    (40.81, 27.8),
    (40.86, 28.5),
    (40.75, 29.1),
    (40.72, 30.2),
    (40.77, 31.1),
    (40.80, 32.2),
    (40.58, 33.0),
    (41.11, 34.9),
    (40.69, 36.6),
    (40.26, 37.8),
    (39.53, 40.2),
]
lons_main = np.array([p[1] for p in naf_main_pts])
lats_main = np.array([p[0] for p in naf_main_pts])
lon_dense_main = np.linspace(lons_main.min(), lons_main.max(), 120)
lat_dense_main = np.interp(lon_dense_main, lons_main, lats_main)

# Southern branch
naf_south_pts = [
    (40.05, 27.5),
    (40.01, 28.5),
    (40.16, 29.0),
    (40.40, 29.8),
    (40.50, 30.4),
]
lons_south = np.array([p[1] for p in naf_south_pts])
lats_south = np.array([p[0] for p in naf_south_pts])
lon_dense_south = np.linspace(lons_south.min(), lons_south.max(), 40)
lat_dense_south = np.interp(lon_dense_south, lons_south, lats_south)


# ── compute rates using assign_sides ──────────────────────────────────────────
def profile_rates(lons, lats, strike=270):
    """Return arrays (ss, fn, tot, pair_labels) sampled via assign_sides."""
    ss, fn, tot, pairs = [], [], [], []
    for lon, lat in zip(lons, lats):
        c_right, c_left = assign_sides(clusters, lon, lat, strike)
        if c_right.id == c_left.id:
            ss.append(np.nan); fn.append(np.nan); tot.append(np.nan)
            pairs.append(None)
            continue
        res = fault_slip_rate(c_right.euler_vector, c_left.euler_vector,
                              [lat], [lon], fault_strike_deg=strike)[0]
        ss.append(res["strike_slip_mm_yr"])
        fn.append(res["fault_normal_mm_yr"])
        tot.append(res["total_mm_yr"])
        pairs.append((c_right.id, c_left.id))
    return (np.array(ss), np.array(fn), np.array(tot), pairs)


ss_main, fn_main, tot_main, pairs_main = profile_rates(lon_dense_main, lat_dense_main)
ss_south, fn_south, tot_south, pairs_south = profile_rates(lon_dense_south, lat_dense_south)

# ── published comparison values ────────────────────────────────────────────────
# Reilinger et al. 2006 — block model NAF rates (digitised from their Fig 9)
pub_lons = [27.5, 30.5, 33.0, 36.5, 39.5]
pub_ss   = [24.0, 23.5, 22.5, 20.0, 18.5]
pub_err  = [ 2.0,  2.0,  2.0,  2.0,  2.0]

# ── unique cluster pairs on main strand for legend ────────────────────────────
unique_pairs = {}
for p in pairs_main:
    if p and p not in unique_pairs:
        unique_pairs[p] = f"C{p[0]}/C{p[1]}"

# Assign a colour per pair
pair_colours = {}
palette = ["#2166ac", "#d6604d", "#4dac26", "#8856a7"]
for i, p in enumerate(unique_pairs):
    pair_colours[p] = palette[i % len(palette)]

col_south = "#888888"
col_pub   = "#333333"

# ── figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.subplots_adjust(hspace=0.08)

# Historical rupture shading
ruptures = [
    (26.0, 27.8,  "1912",       "#fee0d2"),
    (29.0, 30.8,  "1999",       "#fcbba1"),
    (30.8, 31.8,  "1999 Düzce", "#fc9272"),
    (31.6, 33.5,  "1944",       "#fee0d2"),
    (33.0, 36.0,  "1943",       "#deebf7"),
    (36.0, 38.5,  "1942+1939",  "#c6dbef"),
]

# ── Panel 1: Strike-slip rate ──────────────────────────────────────────────────
ax = axes[0]
for lon0, lon1, label, col in ruptures:
    ax.axvspan(lon0, lon1, alpha=0.18, color=col, zorder=0)
    ax.text((lon0+lon1)/2, 27.2, label, ha="center", va="bottom",
            fontsize=6.5, color="#444444")

# Plot each pair segment separately so legend is clean
for pair, colour in pair_colours.items():
    mask = np.array([p == pair for p in pairs_main], dtype=float)
    mask[mask == 0] = np.nan
    ax.plot(lon_dense_main, ss_main * mask, color=colour, lw=2.0,
            label=unique_pairs[pair])

ax.plot(lon_dense_south, ss_south, color=col_south, lw=1.8, ls="--",
        label="southern branch")
ax.errorbar(pub_lons, pub_ss, yerr=pub_err, fmt="s", color=col_pub,
            capsize=4, markersize=6, lw=1.5, zorder=5,
            label="Reilinger et al. (2006)")

ax.set_ylabel("Strike-slip rate\n(mm/yr,  +ve = right-lateral)", fontsize=10)
ax.set_ylim(10, 32)
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
ax.set_title(f"NAF slip rate profile — Euler-vector clustering k = {K_OPT}  (raw ITRF14)",
             fontsize=11)
ax.grid(axis="x", lw=0.3, alpha=0.5)

# ── Panel 2: Fault-normal rate  (+ve = opening / extension) ───────────────────
ax = axes[1]
for lon0, lon1, label, col in ruptures:
    ax.axvspan(lon0, lon1, alpha=0.18, color=col, zorder=0)

# fn > 0 = extension (opening); fn < 0 = convergence (shortening)
ax.fill_between(lon_dense_main, fn_main, 0,
                where=~np.isnan(fn_main) & (fn_main > 0),
                alpha=0.3, color="#d6604d", label="Extension (opening)")
ax.fill_between(lon_dense_main, fn_main, 0,
                where=~np.isnan(fn_main) & (fn_main < 0),
                alpha=0.3, color="#2166ac", label="Convergence (shortening)")
ax.plot(lon_dense_main, fn_main, color="#444444", lw=2.0)

ax.axhline(0, color="k", lw=1.0)
ax.set_ylabel("Fault-normal rate\n(mm/yr,  +ve = opening)", fontsize=10)
ax.set_ylim(-12, 14)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax.grid(axis="x", lw=0.3, alpha=0.5)

# Marmara = releasing/pull-apart; eastern bend ~38°E = restraining
ax.annotate("Releasing\nbend\n(pull-apart)", xy=(27.5, 5), xytext=(27.5, 10),
            fontsize=8, ha="center", color="#d6604d",
            arrowprops=dict(arrowstyle="-|>", color="#d6604d", lw=1.0))
ax.annotate("Restraining\nbend", xy=(38.5, -4), xytext=(38.5, -9.5),
            fontsize=8, ha="center", color="#2166ac",
            arrowprops=dict(arrowstyle="-|>", color="#2166ac", lw=1.0))

# ── Panel 3: Total rate ────────────────────────────────────────────────────────
ax = axes[2]
for lon0, lon1, label, col in ruptures:
    ax.axvspan(lon0, lon1, alpha=0.18, color=col, zorder=0)

for pair, colour in pair_colours.items():
    mask = np.array([p == pair for p in pairs_main], dtype=float)
    mask[mask == 0] = np.nan
    ax.plot(lon_dense_main, tot_main * mask, color=colour, lw=2.0,
            label=unique_pairs[pair])

ax.plot(lon_dense_south, tot_south, color=col_south, lw=1.8, ls="--",
        label="southern branch")
ax.errorbar(pub_lons, pub_ss, yerr=pub_err, fmt="s", color=col_pub,
            capsize=4, markersize=6, lw=1.5, zorder=5,
            label="Reilinger et al. (2006)")

ax.set_ylabel("Total relative rate\n(mm/yr)", fontsize=10)
ax.set_xlabel("Longitude (°E)", fontsize=10)
ax.set_ylim(12, 32)
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
ax.grid(axis="x", lw=0.3, alpha=0.5)

for p in naf_main_pts[::2]:
    ax.axvline(p[1], color="gray", lw=0.4, ls=":", zorder=0)

ax.set_xlim(25.5, 41.0)

fig.savefig(OUT / "fig_naf_slip_profile.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}/fig_naf_slip_profile.png")

# Print which cluster pair dominates each longitude band
print("\nCluster pairs along NAF main strand:")
prev = None
for lon, pair in zip(lon_dense_main, pairs_main):
    if pair != prev:
        print(f"  lon={lon:.1f}°E → C{pair[0]}/C{pair[1]}" if pair else f"  lon={lon:.1f}°E → same cluster")
        prev = pair
