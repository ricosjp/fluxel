from typing import Protocol, runtime_checkable

import numpy as np

from ._fluxel import (
    ApIbmFaceData,
    ApibmSession,
    BoundingBox,
    CfdAxisProjectedMesh,
    CfdGhostCellMesh,
    Cylindrical,
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
from .types import Bool3, Float3, Int3


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
        Coordinate code. 0 is Cartesian and 1 is cylindrical.
        """
        ...

    @property
    def cell_centers(self) -> np.ndarray:
        """
        World sample points: float64 array of shape (n_cells, 3).

        Cartesian samples are box centers. Cylindrical samples are parameter
        midpoints mapped to world, which differ from the volume centroids.
        """
        ...

    @property
    def cell_sizes(self) -> np.ndarray:
        """
        Coordinate-aligned widths: float64 (n_cells, 3).

        Cartesian widths are the edge lengths. Cylindrical widths are
        ``[Δr, r Δθ, Δz]``. Their product is not the cylindrical cell volume.
        """
        ...

    @property
    def cell_volumes(self) -> np.ndarray:
        """
        Exact cell volumes: float64 (n_cells,).
        """
        ...

    @property
    def cell_centroids(self) -> np.ndarray:
        """
        Volume centroids in world coordinates: float64 (n_cells, 3).
        """
        ...

    @property
    def cell_corners(self) -> np.ndarray:
        """
        VTK corners in world coordinates: float64 (n_cells, 8, 3).

        Cylindrical edges are straight chords through the parameter corners.
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
        Logical face axes: uint8 (n_internal_faces,). Values 0, 1, 2 are
        Cartesian X, Y, Z or cylindrical r, θ, z.
        """
        ...

    @property
    def internal_faces_area(self) -> np.ndarray:
        """
        Area vectors from owner toward neighbour: float64 (n_internal_faces, 3).
        """
        ...

    @property
    def internal_faces_min(self) -> np.ndarray:
        """
        Face minima: float64 (n_internal_faces, 3), world ``(x, y, z)`` or
        ``(r, θ, z)``.
        """
        ...

    @property
    def internal_faces_max(self) -> np.ndarray:
        """
        Face maxima: float64 (n_internal_faces, 3), same components as the
        minima.
        """
        ...

    @property
    def internal_faces_winding(self) -> np.ndarray:
        """
        Periodic wraps: int8 (n_internal_faces,). A full-turn θ seam is 1.
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
        Outward logical directions: uint8 (n_domain_bnd_faces,), codes 0..5.

        On a cylinder the pairs are ``-r,+r``, ``-θ,+θ``, ``-z,+z``.
        """
        ...

    @property
    def domain_bnd_faces_area(self) -> np.ndarray:
        """
        Outward area vectors of domain faces: float64 (n_domain_bnd_faces, 3).
        """
        ...

    @property
    def domain_bnd_faces_min(self) -> np.ndarray:
        """
        Domain-face minima: float64 (n_domain_bnd_faces, 3). Components match
        the internal-face minima.
        """
        ...

    @property
    def domain_bnd_faces_max(self) -> np.ndarray:
        """
        Domain-face maxima: float64 (n_domain_bnd_faces, 3). Components match
        the internal-face minima.
        """
        ...


__all__ = [
    "ApibmSession",
    "ApIbmFaceData",
    "Axis",
    "Bool3",
    "BoundingBox",
    "CfdAxisProjectedMesh",
    "CfdGhostCellMesh",
    "CoordinateType",
    "Cylindrical",
    "Direction",
    "Float3",
    "FluxelManager",
    "Int3",
    "Forest",
    "ICfdMesh",
    "get_axis_from_direction",
    "get_opposite_direction",
    "get_sign_from_direction",
    "get_unit_vector_from_direction",
    "quaternion_from_axis_angle",
    "split_axis_into_directions",
]
