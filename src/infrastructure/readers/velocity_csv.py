"""Read GPS velocity files into domain entities.

Expected column names (whitespace-delimited):
    Sta  Longitude  Latitude  Ve  Vn  Vu  Se  Sn  Su

Column aliases accepted (case-insensitive, trailing dots stripped):
    site/sta/station/name, lon./longitude, lat./latitude, ...

Non-data lines (titles, dash separators, comment lines starting with #)
are skipped automatically.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from gps_cluster.domain.entities import GpsStation, Position, Velocity

# Maps normalized column names → canonical domain names (trailing dots stripped)
_COL_MAP = {
    "site": "name",
    "sta": "name",
    "station": "name",
    "name": "name",
    "lon": "lon",
    "lon.": "lon",
    "longitude": "lon",
    "lat": "lat",
    "lat.": "lat",
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

# Regex matching a column-header line: contains at least one known key
_HEADER_RE = re.compile(r"\b(site|sta|station|lon\.?|lat\.?|longitude|latitude)\b", re.I)
# Regex matching a separator / non-data line (all dashes, or a prose title)
_SKIP_RE = re.compile(r"^[-\s]+$|^[A-Za-z][^0-9\-]{5,}")


def _find_header_line(path: Path) -> int:
    """Return the 0-based line index of the column-header row."""
    with open(path) as fh:
        for i, line in enumerate(fh):
            if _HEADER_RE.search(line):
                return i
    raise ValueError(f"Cannot find a column-header line in {path}")


def read_velocity_file(path: str | Path) -> list[GpsStation]:
    """Parse a whitespace-delimited GPS velocity file and return GpsStation list.

    Robustly handles files with:
    - A prose title line before the column header
    - Trailing dots on column names (e.g. 'lon.', 'lat.')
    - Dash-separator lines between header and data
    - Comment lines starting with '#'
    """
    path = Path(path)
    header_row = _find_header_line(path)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        skiprows=header_row,      # skip everything above the header
        comment="#",
    )
    df.columns = [c.strip() for c in df.columns]

    # Drop separator rows (e.g. '---...---') that pandas reads as data
    df = df[~df.iloc[:, 0].astype(str).str.match(r"^-+$")]

    # Normalise column names
    renamed = {col: _COL_MAP.get(col.lower().strip("."), _COL_MAP.get(col.lower(), col.lower()))
               for col in df.columns}
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
