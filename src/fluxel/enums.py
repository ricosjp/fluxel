from enum import IntEnum

import numpy as np


class CoordinateType(IntEnum):
    Cartesian = 0
    Cylindrical = 1
    Spherical = 2


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2


class Direction(IntEnum):
    XMinus = 0
    XPlus = 1
    YMinus = 2
    YPlus = 3
    ZMinus = 4
    ZPlus = 5


def split_axis_into_directions(axis: Axis) -> tuple[Direction, Direction]:
    if axis == Axis.X:
        return Direction.XMinus, Direction.XPlus
    elif axis == Axis.Y:
        return Direction.YMinus, Direction.YPlus
    elif axis == Axis.Z:
        return Direction.ZMinus, Direction.ZPlus
    else:
        raise ValueError(f"Invalid axis: {axis}")


def get_axis_from_direction(direction: Direction) -> Axis:
    return Axis(direction.value // 2)


def get_opposite_direction(direction: Direction) -> Direction:
    return Direction(direction.value ^ 1)


def get_sign_from_direction(direction: Direction) -> int:
    return 2 * (direction.value % 2) - 1


def get_unit_vector_from_direction(direction: Direction) -> np.ndarray:
    vec = np.zeros(3, dtype=np.float64)
    axis = get_axis_from_direction(direction)
    sign = get_sign_from_direction(direction)
    vec[axis] = float(sign)
    return vec
