"""Serialisation helpers: domain objects → JSON-serialisable dicts.

These functions are the single source of truth for the JSON cache schema
written by all compute scripts and read by all plot scripts.
"""

from __future__ import annotations

from gps_cluster.domain.entities import GpsStation, VelocityCluster
from gps_cluster.domain.services.euler_math import euler_vector_to_pole


def cluster_to_dict(cluster: VelocityCluster) -> dict | None:
    """Serialise a VelocityCluster to a JSON-compatible dict.

    Returns None if the cluster has no Euler vector (under-constrained).

    Schema
    ------
    {
        "id":    int,
        "size":  int,
        "chi2":  float,
        "euler": {"ox", "oy", "oz", "covariance": list[list] | null},
        "pole":  {"lat", "lon", "rate",
                  "sigma_lat", "sigma_lon", "sigma_rate"},
        "stations": [str, ...],
        "membership_weights": [float, ...] | null,
        "prob_vector": [float, ...] | null   # set externally if needed
    }
    """
    if cluster.euler_vector is None:
        return None

    pole = euler_vector_to_pole(cluster.euler_vector)
    cov  = (cluster.euler_vector.covariance.tolist()
            if cluster.euler_vector.covariance is not None else None)
    mw   = (cluster.membership_weights.tolist()
            if cluster.membership_weights is not None else None)

    return {
        "id":    cluster.id,
        "size":  cluster.size,
        "chi2":  float(cluster.chi2) if cluster.chi2 is not None else 0.0,
        "euler": {
            "ox": cluster.euler_vector.ox,
            "oy": cluster.euler_vector.oy,
            "oz": cluster.euler_vector.oz,
            "covariance": cov,
        },
        "pole": {
            "lat":        pole.lat,
            "lon":        pole.lon,
            "rate":       pole.rate,
            "sigma_lat":  pole.sigma_lat,
            "sigma_lon":  pole.sigma_lon,
            "sigma_rate": pole.sigma_rate,
        },
        "stations":          [s.name for s in cluster.stations],
        "membership_weights": mw,
        "prob_vector":        None,   # filled in by compute scripts if needed
    }


def station_to_dict(station: GpsStation) -> dict:
    """Serialise a GpsStation to a JSON-compatible dict.

    Schema
    ------
    {"name", "lat", "lon", "ve", "vn", "se", "sn"}
    """
    return {
        "name": station.name,
        "lat":  station.position.lat,
        "lon":  station.position.lon,
        "ve":   station.velocity.ve,
        "vn":   station.velocity.vn,
        "se":   station.velocity.se,
        "sn":   station.velocity.sn,
    }


def clusters_to_list(clusters: list[VelocityCluster]) -> list[dict]:
    """Serialise a list of VelocityCluster objects, skipping under-constrained ones."""
    return [d for c in clusters if (d := cluster_to_dict(c)) is not None]
