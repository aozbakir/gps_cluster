from __future__ import annotations

from dataclasses import dataclass, field


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

    def to_array(self):
        import numpy as np

        return np.array([self.ox, self.oy, self.oz])


@dataclass
class EulerPole:
    """Geographic (surface) representation of an Euler vector."""

    lat: float  # degrees north
    lon: float  # degrees east
    rate: float  # deg/Myr


@dataclass
class VelocityCluster:
    id: int
    stations: list[GpsStation] = field(default_factory=list)
    euler_vector: EulerVector | None = None

    @property
    def size(self) -> int:
        return len(self.stations)
