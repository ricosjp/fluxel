from typing import Protocol, runtime_checkable

import numpy as np

from ._fluxel import (
    ApIbmFaceData,
    ApibmSession,
    BoundingBox,
    CfdAxisProjectedMesh,
    CfdGhostCellMesh,
    FluxelManager,
    Forest,
)
from .enums import (
    Axis,
    CoordinateType,
    Direction,
    get_axis_from_direction,
    get_opposite_direction,
    get_sign_from_direction,
    get_unit_vector_from_direction,
    split_axis_into_directions,
)
from .pose import quaternion_from_axis_angle


@runtime_checkable
class ICfdMesh(Protocol):
    """Runtime-visible structural interface for CFD mesh objects."""

    @property
    def n_cells(self) -> int:
        """
        Number of background cells, including non-fluid cells.
        """
        ...

    @property
    def coordinate_type(self) -> int:
        """
        Coordinate code; generated meshes currently use 0 (Cartesian).
        """
        ...

    @property
    def cell_centers(self) -> np.ndarray:
        """
        Physical centers: float64 array of shape (n_cells, 3).
        """
        ...

    @property
    def cell_sizes(self) -> np.ndarray:
        """
        Physical side lengths [dx, dy, dz]: float64 (n_cells, 3).
        """
        ...

    @property
    def patch_name_to_id(self) -> dict[str, int]:
        """
        Surface patch names mapped to IDs; no boundary uses {"empty": 0}.
        """
        ...

    @property
    def internal_faces_owner(self) -> np.ndarray:
        """
        Owner cell indices: uintp (n_internal_faces,), local to this mesh.
        """
        ...

    @property
    def internal_faces_neighbour(self) -> np.ndarray:
        """
        Neighbour cell indices: uintp (n_internal_faces,), local to this mesh.
        """
        ...

    @property
    def internal_faces_axis(self) -> np.ndarray:
        """
        Internal-face axes: uint8 (n_internal_faces,), X=0, Y=1, Z=2.
        """
        ...

    @property
    def domain_bnd_faces_owner(self) -> np.ndarray:
        """
        Domain-boundary owner cell indices: uintp (n_domain_bnd_faces,).
        """
        ...

    @property
    def domain_bnd_faces_dir(self) -> np.ndarray:
        """
        Outward directions: uint8 (n_domain_bnd_faces,), -X,+X,-Y,+Y,-Z,+Z =
        0..5.
        """
        ...


__all__ = [
    "ApibmSession",
    "ApIbmFaceData",
    "Axis",
    "BoundingBox",
    "CfdAxisProjectedMesh",
    "CfdGhostCellMesh",
    "CoordinateType",
    "Direction",
    "FluxelManager",
    "Forest",
    "ICfdMesh",
    "get_axis_from_direction",
    "get_opposite_direction",
    "get_sign_from_direction",
    "get_unit_vector_from_direction",
    "quaternion_from_axis_angle",
    "split_axis_into_directions",
]
