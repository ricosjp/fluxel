from typing import Protocol, runtime_checkable

import numpy as np

from ._fluxel import *
from .enums import *
from .pose import quaternion_from_axis_angle


@runtime_checkable
class ICfdMesh(Protocol):
    """Runtime-visible structural interface for CFD mesh objects."""

    @property
    def n_cells(self) -> int: ...

    @property
    def coordinate_type(self) -> int: ...

    @property
    def cell_centers(self) -> np.ndarray: ...

    @property
    def cell_sizes(self) -> np.ndarray: ...

    @property
    def patch_name_to_id(self) -> dict[str, int]: ...

    @property
    def internal_faces_owner(self) -> np.ndarray: ...

    @property
    def internal_faces_neighbour(self) -> np.ndarray: ...

    @property
    def internal_faces_axis(self) -> np.ndarray: ...

    @property
    def domain_bnd_faces_owner(self) -> np.ndarray: ...

    @property
    def domain_bnd_faces_dir(self) -> np.ndarray: ...
