"""Read GPS velocity files into domain entities.

Expected column names (whitespace-delimited):
    Sta  Longitude  Latitude  Ve  Vn  Vu  Se  Sn  Su

Column aliases accepted (case-insensitive):
    lon/longitude, lat/latitude, ve/velocity_e, ...
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gps_cluster.domain.entities import GpsStation, Position, Velocity

# Maps normalized column names → canonical domain names
_COL_MAP = {
    "sta": "name",
    "station": "name",
    "name": "name",
    "lon": "lon",
    "longitude": "lon",
    "lat": "lat",
    "latitude": "lat",
    "ve": "ve",
    "vn": "vn",
    "vu": "vu",
    "se": "se",
    "sn": "sn",
    "su": "su",
    "sigve": "se",
    "sigvn": "sn",
    "sigvu": "su",
}

_REQUIRED = {"name", "lon", "lat", "ve", "vn", "vu", "se", "sn", "su"}


def read_velocity_file(path: str | Path) -> list[GpsStation]:
    """Parse a whitespace-delimited GPS velocity file and return GpsStation list."""
    df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names
    renamed = {col: _COL_MAP.get(col.lower(), col.lower()) for col in df.columns}
    df = df.rename(columns=renamed)

    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    stations: list[GpsStation] = []
    for _, row in df.iterrows():
        stations.append(
            GpsStation(
                name=str(row["name"]),
                position=Position(lon=float(row["lon"]), lat=float(row["lat"])),
                velocity=Velocity(
                    ve=float(row["ve"]),
                    vn=float(row["vn"]),
                    vu=float(row["vu"]),
                    se=float(row["se"]),
                    sn=float(row["sn"]),
                    su=float(row["su"]),
                ),
            )
        )
    return stations
