"""Rigid-pose helpers for immersed-boundary motion."""

from __future__ import annotations

import math

from ._fluxel import _quaternion_from_axis_angle
from .enums import Axis
from .types import Float3

AxisLike = Axis | Float3


def _as_axis_vector(axis: AxisLike) -> Float3:
    """Convert ``Axis`` or a 3-vector to ``(x, y, z)``."""
    if isinstance(axis, Axis):
        components = [0.0, 0.0, 0.0]
        components[int(axis)] = 1.0
        return (components[0], components[1], components[2])
    if len(axis) != 3:
        raise ValueError("axis must have 3 components")
    return (float(axis[0]), float(axis[1]), float(axis[2]))


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
    axis : Axis or Float3
        Rotation axis. Pass ``Axis.X`` / ``Axis.Y`` / ``Axis.Z``, or a
        finite ``(x, y, z)`` which is normalized. An exactly zero vector yields
        identity; small nonzero vectors are normalized normally.
    angle : float
        Rotation angle. Radians by default; set ``degrees=True`` to pass
        degrees.
    degrees : bool, optional
        If True, ``angle`` is interpreted in degrees. Default is False.

    Returns
    -------
    list of float
        Unit quaternion ``[w, x, y, z]`` for ``update_ib`` / ``remesh``.

    Raises
    ------
    ValueError
        If the axis does not have three components, or the axis/angle is
        nonfinite. Non-numeric values may raise TypeError or ValueError.

    Notes
    -----
    Describes a rotation about the coordinate origin. Session pose updates
    interpret the quaternion as absolute, not as an incremental rotation.
    """
    angle_rad = math.radians(angle) if degrees else float(angle)
    return list(_quaternion_from_axis_angle(_as_axis_vector(axis), angle_rad))
