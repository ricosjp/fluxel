"""
Fluxel Python Binding (Stub File)

This module provides the Python interface for Fluxel,
an Adaptive Mesh Refinement (AMR) library based on Space-Filling Curves.
"""

import numpy as np

class BoundingBox:
    """
    Physical bounding box of the computational domain.

    Attributes
    ----------
    min : list of float
        Minimum coordinates [x, y, z] of the bounding box.
    max : list of float
        Maximum coordinates [x, y, z] of the bounding box.
    """

    min: list[float]
    max: list[float]

    def __init__(self, min: list[float], max: list[float]) -> None: ...

class CfdGhostCellMesh:
    """
    Structure of Arrays (SoA) mesh representation for CFD solvers
    using the Ghost-Cell Immersed Boundary Method (GCIBM).
    Uses zero-copy numpy arrays for maximum performance.

    Attributes
    ----------
    n_cells : int
        The number of cells in the mesh.
    coordinate_type : int
        The type of coordinate system used in the mesh.
        0: Cartesian
    cell_centers : numpy.ndarray
        Array of shape (N_cells, 3) dtype=float64
        the physical center coordinates of each background cell.
    cell_sizes : numpy.ndarray
        Array of shape (N_cells, 3) dtype=float64
        the cell side lengths [dx, dy, dz] for each background cell.
    patch_name_to_id : dict[str, int]
        A dictionary mapping patch names to their corresponding IDs.

    internal_faces_owner : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint64
        the owner cell index for internal faces.
    internal_faces_neighbour : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint64
        the neighbour cell index for internal faces.
    internal_faces_axis : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint8
        the axis of the internal faces.
        0: X, 1: Y, 2: Z
    bnd_faces_owner : numpy.ndarray
        Array of shape (N_bnd_faces,) dtype=uint64
        the owner cell index for domain boundary faces.
    bnd_faces_dir : numpy.ndarray
        Array of shape (N_bnd_faces,) dtype=uint8
        the direction of the domain boundary faces.
        0: -X, 1: +X, 2: -Y, 3: +Y, 4: -Z, 5: +Z

    [Ghost Cell IBM Specific Data]
    gc_is_fluid : numpy.ndarray
        Array of shape (N_cells,) dtype=bool
        a boolean flag indicating whether each cell is a fluid cell.

    gc_cell_ids : numpy.ndarray
        Array of shape (N_ghosts,) dtype=uint64
        the global indices of cells identified as ghost cells.
    gc_bnd_anchor_ids : numpy.ndarray
        Array of shape (N_ghosts,) dtype=uint64
        the boundary triangle IDs associated with each ghost cell.
    gc_bnd_patch_ids : numpy.ndarray
        Array of shape (N_ghosts,) dtype=uint64
        the boundary patch IDs associated with each ghost cell.
    gc_bnd_intercepts : numpy.ndarray
        Array of shape (N_ghosts, 3) dtype=float64
        the nearest boundary points.
    gc_image_points : numpy.ndarray
        Array of shape (N_ghosts, 3) dtype=float64
        the coordinates of the image points.
    gc_interp_stencil_indices : numpy.ndarray
        Array of shape (N_ghosts, 8) dtype=uint64
        the fluid cell indices used for interpolating at the Image Points.
    gc_interp_stencil_weights : numpy.ndarray
        Array of shape (N_ghosts, 8) dtype=float64
        the interpolation weights corresponding to `interp_stencil_indices`.
    """

    n_cells: int
    coordinate_type: int
    cell_centers: np.ndarray
    cell_sizes: np.ndarray
    patch_name_to_id: dict[str, int]

    internal_faces_owner: np.ndarray
    internal_faces_neighbour: np.ndarray
    internal_faces_axis: np.ndarray
    bnd_faces_owner: np.ndarray
    bnd_faces_dir: np.ndarray

    gc_is_fluid: np.ndarray
    gc_cell_ids: np.ndarray
    gc_bnd_anchor_ids: np.ndarray
    gc_bnd_patch_ids: np.ndarray
    gc_bnd_intercepts: np.ndarray
    gc_image_points: np.ndarray
    gc_interp_stencil_indices: np.ndarray
    gc_interp_stencil_weights: np.ndarray

class CfdAxisProjectedMesh:
    """
    Structure of Arrays (SoA) mesh representation for CFD solvers
    using the Axis-Projected Immersed Boundary Method (APIBM/TFIBM).
    Uses zero-copy numpy arrays for maximum performance.

    Attributes
    ----------
    n_cells: int
        The number of cells in the mesh.
    coordinate_type: int
        The type of coordinate system used in the mesh.
        0: Cartesian
    cell_centers : numpy.ndarray
        Array of shape (N_cells, 3) dtype=float64
        the physical center coordinates of each background cell.
    cell_sizes : numpy.ndarray
        Array of shape (N_cells, 3) dtype=float64
        the cell side lengths [dx, dy, dz] for each background cell.
    patch_name_to_id: dict[str, int]
        A dictionary mapping patch names to their corresponding IDs.

    internal_faces_owner : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint64
        the owner cell index for internal faces.
    internal_faces_neighbour : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint64
        the neighbour cell index for internal faces.
    internal_faces_axis : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint8
        the axis of the internal faces.
        0: X, 1: Y, 2: Z
    bnd_faces_owner : numpy.ndarray
        Array of shape (N_bnd_faces,) dtype=uint64
        the owner cell index for domain boundary faces.
    bnd_faces_dir : numpy.ndarray
        Array of shape (N_bnd_faces,) dtype=uint8
        the direction of the domain boundary faces.
        0: -X, 1: +X, 2: -Y, 3: +Y, 4: -Z, 5: +Z

    [Axis-Projected IBM Specific Data]
    ap_is_immersed_face : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=bool
        a boolean flag indicating
        whether each internal face intersects the immersed boundary.
    ap_dist_owner_to_bnd : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=float64
        the distance from the owner cell center
        to the boundary intersection point along each axis.
    ap_dist_neighbour_to_bnd : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=float64
        the distance from the neighbour cell center
        to the boundary intersection point along each axis.
    ap_owner_far_cell_id : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint64
        the far-cell index on the owner side
        used for APIBM reconstruction along each axis.
    ap_neighbour_far_cell_id : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint64
        the far-cell index on the neighbour side
        used for APIBM reconstruction along each axis.
    ap_owner_weights : numpy.ndarray
        Array of shape (Nx_faces_with_bnd, 3) dtype=float64
        APIBM reconstruction weights for the owner side on each axis.
    ap_neighbour_weights : numpy.ndarray
        Array of shape (N_internal_faces, 3) dtype=float64
        APIBM reconstruction weights for the neighbour side on each axis.
    ap_owner_bnd_anchor_id : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint64
        the boundary anchor id for owner side of each axis.
    ap_owner_bnd_patch_id : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint64
        the boundary patch id for owner side of each axis.
    ap_neighbour_bnd_anchor_id : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint64
        the boundary anchor id for neighbour side of each axis.
    ap_neighbour_bnd_patch_id : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint64
        the boundary patch id for neighbour side of each axis.
    """

    n_cells: int
    coordinate_type: int
    cell_centers: np.ndarray
    cell_sizes: np.ndarray
    patch_name_to_id: dict[str, int]

    internal_faces_owner: np.ndarray
    internal_faces_neighbour: np.ndarray
    internal_faces_axis: np.ndarray
    bnd_faces_owner: np.ndarray
    bnd_faces_dir: np.ndarray

    ap_is_immersed_face: np.ndarray
    ap_dist_owner_to_bnd: np.ndarray
    ap_dist_neighbour_to_bnd: np.ndarray
    ap_owner_far_cell_id: np.ndarray
    ap_neighbour_far_cell_id: np.ndarray
    ap_owner_weights: np.ndarray
    ap_neighbour_weights: np.ndarray
    ap_owner_bnd_anchor_id: np.ndarray
    ap_owner_bnd_patch_id: np.ndarray
    ap_neighbour_bnd_anchor_id: np.ndarray
    ap_neighbour_bnd_patch_id: np.ndarray

class FluxelManager:
    """
    High-level API manager for automated AMR mesh generation.
    """
    def __init__(
        self,
        bbox: BoundingBox,
        base_res: list[int],
        n_leaf_refinement: int = 3,
    ) -> None:
        """
        Initialize the FluxelManager.

        Parameters
        ----------
        bbox : BoundingBox
            The physical bounds of the overall domain.
        base_res : list of int
            The initial number of root blocks (trees) in [X, Y, Z] directions.
        n_leaf_refinement : int (default: 3)
            The number of times that
            the uniform refinement is applied to the final mesh.
            n_leaf_refinement=3 will generate 8x8x8 micro-cells per octree leaf.
        """
        ...

    def build_ghost_cell_mesh(
        self,
        stl_path: str,
        target_level: int,
        fluid_seed_point: list[float],
    ) -> CfdGhostCellMesh:
        """
        Builds a CFD mesh
        tailored for the Ghost-Cell Immersed Boundary Method (GCIBM).

        This method reads an STL file, refines the octree cells near the surface
        up to the `target_level`, performs inside/outside determination,
        and calculates GCIBM data such as image points or extrapolation weights.

        Parameters
        ----------
        stl_path : str
            Path to the input STL geometry file.
        target_level : int
            The maximum octree refinement level applied around the STL surface.
        fluid_seed_point : list of float
            The point in the physical domain to seed the fluid region.

        Returns
        -------
        CfdGhostCellMesh
            The generated SoA mesh ready for CFD solvers.

        Raises
        ------
        ValueError
            If the specified `fluid_seed_point` lies on the boundary or
            outside the computational domain.
        """
        ...

    def build_axis_projected_mesh(
        self, stl_path: str, target_level: int
    ) -> CfdAxisProjectedMesh:
        """
        Builds a CFD mesh
        tailored for the Axis-Projected Immersed Boundary Method (APIBM).

        This method reads an STL file, refines the octree cells near the surface
        up to the `target_level`, performs axis raytracing, and calculates
        APIBM data such as distance to STL mesh along each axis
        or extrapolation weights.

        Parameters
        ----------
        stl_path : str
            Path to the input STL geometry file.
        target_level : int
            The maximum octree refinement level applied around the STL surface.

        Returns
        -------
        CfdAxisProjectedMesh
            The generated SoA mesh ready for CFD solvers.
        """
        ...

class Forest:
    """
    Low-level API for manual AMR control.
    """
    def __init__(
        self,
        bbox: BoundingBox,
        base_res: list[int],
    ) -> None:
        """
        Initialize the Forest.

        Parameters
        ----------
        bbox : BoundingBox
            The physical bounds of the overall domain.
        base_res : list of int
            The initial number of root blocks (trees) in [X, Y, Z] directions.
        """
        ...

    def num_cells(self) -> int:
        """
        Returns the current total number of cells.

        Returns
        -------
        int
            Number of cells.
        """
        ...

    def refine_by_flags(self, flags: list[bool]) -> None:
        """
        Refines cells based on a boolean mask.

        Parameters
        ----------
        flags : list[bool]
            A boolean list of length `num_cells()`. True indicates refinement.
        """
        ...

    def coarsen_by_flags(self, flags: list[bool]) -> None:
        """
        Coarsens sibling cells back to their parent based on a boolean mask.

        Parameters
        ----------
        flags : list[bool]
            A boolean list of length `num_cells()`.
        """
        ...

    def refine_by_bbox(
        self, min: list[float], max: list[float], target_level: int
    ) -> None:
        """
        Refines all cells that intersect with the specified bounding box
        up to the given target level.

        Parameters
        ----------
        min : list[float]
            The [x, y, z] coordinates of the minimum corner of the box.
        max : list[float]
            The [x, y, z] coordinates of the maximum corner of the box.
        target_level : int
            The desired refinement level inside the box.
        """
        ...

    def enforce_2_to_1_balance(self) -> None:
        """
        Enforces the 2:1 balancing constraint across the entire mesh,
        ensuring adjacent cells differ by at most one refinement level.
        """
        ...

    def uniform_refinement(self, n_times: int) -> None:
        """
        Applies uniform refinement to every cell in the mesh, repeated `n_times`
        times.

        Parameters
        ----------
        n_times : int
            The number of times the uniform refinement is applied.
        """
        ...
