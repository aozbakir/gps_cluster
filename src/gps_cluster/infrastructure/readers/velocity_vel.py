"""Read headerless GLOBK-style .vel velocity files into domain entities.

Expected whitespace-delimited columns (no header row):

    name  lon  lat  ve  vn  se  sn  vu  n_obs

where ve/vn/se/sn/vu are in mm/yr.  A trailing ``*`` on the station name
(GLOBK reference-station flag) is stripped from the stored name but the
station is otherwise kept — it carries a valid ITRF velocity.

There is no ``su`` (vertical uncertainty) column; it is set to 1.0 as a
placeholder (vertical is not used in horizontal-only Euler clustering).
"""

from __future__ import annotations

from pathlib import Path

from gps_cluster.domain.entities import GpsStation, Position, Velocity


def read_vel_file(path: str | Path) -> list[GpsStation]:
    """Parse a GLOBK-style headerless .vel file and return a GpsStation list."""
    path = Path(path)
    stations: list[GpsStation] = []

    with open(path) as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            name = parts[0].rstrip("*")
            try:
                lon = float(parts[1])
                lat = float(parts[2])
                ve  = float(parts[3])
                vn  = float(parts[4])
                se  = float(parts[5])
                sn  = float(parts[6])
                vu  = float(parts[7])
            except ValueError:
                continue

            stations.append(
                GpsStation(
                    name=name,
                    position=Position(lon=lon, lat=lat),
                    velocity=Velocity(
                        ve=ve, vn=vn, vu=vu,
                        se=se, sn=sn, su=1.0,
                    ),
                )
            )

    return stations
