"""Preprocessing use case: clean raw GPS station lists."""

from __future__ import annotations

import numpy as np

from gps_cluster.domain.entities import GpsStation


def remove_fixed(stations: list[GpsStation]) -> list[GpsStation]:
    """Drop stations with exactly zero horizontal velocity (likely fixed reference)."""
    return [s for s in stations if not (s.velocity.ve == 0.0 and s.velocity.vn == 0.0)]


def remove_uncertain(
    stations: list[GpsStation], max_sigma: float = 0.6
) -> list[GpsStation]:
    """Drop stations whose east or north sigma exceeds max_sigma mm/yr."""
    return [s for s in stations if s.velocity.se < max_sigma and s.velocity.sn < max_sigma]


def remove_outliers(
    stations: list[GpsStation], zscore_threshold: float = 2.0
) -> list[GpsStation]:
    """Drop stations whose Ve or Vn z-score exceeds the threshold."""
    if len(stations) < 2:
        return stations
    ve = np.array([s.velocity.ve for s in stations])
    vn = np.array([s.velocity.vn for s in stations])
    # If std is zero (all identical), zscore is 0 everywhere — no outliers
    ze = np.abs((ve - ve.mean()) / ve.std()) if ve.std() > 0 else np.zeros(len(ve))
    zn = np.abs((vn - vn.mean()) / vn.std()) if vn.std() > 0 else np.zeros(len(vn))
    return [s for s, ze_i, zn_i in zip(stations, ze, zn) if ze_i <= zscore_threshold and zn_i <= zscore_threshold]


def preprocess(
    stations: list[GpsStation],
    max_sigma: float = 0.6,
    zscore_threshold: float = 2.0,
) -> list[GpsStation]:
    """Full preprocessing pipeline: remove_fixed → remove_uncertain → remove_outliers."""
    result = remove_fixed(stations)
    result = remove_uncertain(result, max_sigma=max_sigma)
    result = remove_outliers(result, zscore_threshold=zscore_threshold)
    return result
