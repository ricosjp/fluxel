"""Rigid-pose helpers for immersed-boundary motion."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ._fluxel import _quaternion_from_axis_angle
from .enums import Axis

AxisLike = Axis | Sequence[float]


def _as_axis_vector(axis: AxisLike) -> list[float]:
    """Convert ``Axis`` or a 3-vector to a length-3 float list."""
    if isinstance(axis, Axis):
        vec = [0.0, 0.0, 0.0]
        vec[int(axis)] = 1.0
        return vec
    if len(axis) != 3:
        raise ValueError("axis must have 3 components")
    return [float(axis[0]), float(axis[1]), float(axis[2])]


def quaternion_from_axis_angle(
    axis: AxisLike,
    angle: float,
    *,
    degrees: bool = False,
) -> list[float]:
    """
    Build a unit quaternion ``[w, x, y, z]`` from an axis-angle rotation.

    Parameters
    ----------
    axis : Axis or sequence of float
        Rotation axis. Pass ``Axis.X`` / ``Axis.Y`` / ``Axis.Z``, or a
        3-vector which is normalized. A near-zero vector yields identity.
    angle : float
        Rotation angle. Radians by default; set ``degrees=True`` to pass
        degrees.
    degrees : bool, optional
        If True, ``angle`` is interpreted in degrees. Default is False.

    Returns
    -------
    list of float
        Unit quaternion ``[w, x, y, z]`` for ``update_ib`` / ``remesh``.
    """
    angle_rad = math.radians(angle) if degrees else float(angle)
    return list(_quaternion_from_axis_angle(_as_axis_vector(axis), angle_rad))
