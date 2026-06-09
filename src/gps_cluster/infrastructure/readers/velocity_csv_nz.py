"""Reader for New Zealand GPS velocity field (Beavan et al. 2016 supplementary data).

Source: tnzg_a_1112817_sm4728.docx — Table S1
        https://doi.org/10.1080/00288306.2015.1112817

Format (space-separated, parsed to CSV):
    lon lat ve_itrf vn_itrf ve_aus vn_aus se sn site nyears ndays

Velocities in mm/yr, ITRF2008 and Australian-plate-fixed frames.
Uncertainties in mm/yr.
"""

from __future__ import annotations

import csv
from pathlib import Path

from gps_cluster.domain.entities import GpsStation, Position, Velocity


def read_nz_csv(path: str | Path, frame: str = "itrf") -> list[GpsStation]:
    """Read NZ GPS velocity CSV and return GpsStation list.

    Parameters
    ----------
    path:
        Path to the CSV file produced from tnzg_a_1112817_sm4728.docx.
    frame:
        ``"itrf"`` (default) — use ITRF2008 velocities (raw, no plate removed).
        ``"aus"``  — use Australian-plate-fixed velocities.
    """
    stations: list[GpsStation] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lon    = float(row["lon"])
            lat    = float(row["lat"])
            se     = float(row["se"])
            sn     = float(row["sn"])
            site   = str(row["site"]).strip()
            if frame == "aus":
                ve = float(row["ve_aus"])
                vn = float(row["vn_aus"])
            else:
                ve = float(row["ve_itrf"])
                vn = float(row["vn_itrf"])

            # Skip stations with unusually large uncertainties
            if se > 5.0 or sn > 5.0:
                continue
            # Skip stations with very short time series
            if float(row["nyears"]) < 2.0:
                continue

            stations.append(GpsStation(
                name=site,
                position=Position(lat=lat, lon=lon),
                velocity=Velocity(ve=ve, vn=vn, vu=0.0,
                                  se=max(se, 0.1), sn=max(sn, 0.1), su=1.0),
            ))
    return stations
