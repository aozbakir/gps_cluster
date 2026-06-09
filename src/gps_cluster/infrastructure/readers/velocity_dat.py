"""Reader for space-delimited GPS velocity files without station names.

Expected column order (whitespace-delimited, no header):
    lon  lat  ve  vn  se  sn

Units: degrees / mm/yr.  Lines beginning with '#' or '-' are skipped.
Station names are auto-generated as S{i:04d}.
"""

from __future__ import annotations

from pathlib import Path

from gps_cluster.domain.entities import GpsStation, Position, Velocity


def read_dat_file(path: str | Path) -> list[GpsStation]:
    """Parse a headerless lon/lat/ve/vn/se/sn .dat file."""
    path = Path(path)
    stations: list[GpsStation] = []
    idx = 0

    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                ve  = float(parts[2])
                vn  = float(parts[3])
                se  = float(parts[4])
                sn  = float(parts[5])
            except ValueError:
                continue

            stations.append(
                GpsStation(
                    name=f"S{idx:04d}",
                    position=Position(lon=lon, lat=lat),
                    velocity=Velocity(ve=ve, vn=vn, vu=0.0,
                                      se=se, sn=sn, su=1.0),
                )
            )
            idx += 1

    return stations
