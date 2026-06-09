from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # noqa: F401 — used in type annotations via string literals


@dataclass(frozen=True)
class Position:
    lon: float  # degrees east
    lat: float  # degrees north


@dataclass(frozen=True)
class Velocity:
    ve: float  # mm/yr east
    vn: float  # mm/yr north
    vu: float  # mm/yr up
    se: float  # 1-sigma mm/yr east
    sn: float  # 1-sigma mm/yr north
    su: float  # 1-sigma mm/yr up


@dataclass(frozen=True)
class GpsStation:
    name: str
    position: Position
    velocity: Velocity


@dataclass
class EulerVector:
    """Rotation vector in ECEF Cartesian frame.

    Units: mm/yr (same as velocity), because the design matrix G is composed
    of dimensionless unit vectors, so G @ Omega has velocity units.

    Physical meaning: the velocity of a point on the unit sphere at the
    location of each axis unit vector, due to the rotation Omega x r.
    """

    ox: float
    oy: float
    oz: float
    covariance: "np.ndarray | None" = field(default=None, repr=False, compare=False)

    def to_array(self):
        import numpy as np

        return np.array([self.ox, self.oy, self.oz])


@dataclass
class EulerPole:
    """Geographic (surface) representation of an Euler vector."""

    lat: float  # degrees north
    lon: float  # degrees east
    rate: float  # deg/Myr
    sigma_lat: float = 0.0   # 1-sigma degrees
    sigma_lon: float = 0.0   # 1-sigma degrees
    sigma_rate: float = 0.0  # 1-sigma deg/Myr


@dataclass
class VelocityCluster:
    id: int
    stations: list[GpsStation] = field(default_factory=list)
    euler_vector: EulerVector | None = None
    chi2: float | None = None          # total weighted chi² (hard-assigned members)
    chi2_reduced: float | None = None  # chi² / (2N - 3)
    # EM soft-assignment: shape (N_total,) probability that each station in the
    # *full* dataset belongs to this cluster.  None for hard-assignment clusters.
    membership_weights: "np.ndarray | None" = field(
        default=None, repr=False, compare=False
    )

    @property
    def size(self) -> int:
        return len(self.stations)
