"""
Fluxel Python Binding (Stub File)

This module provides the Python interface for Fluxel,
an Adaptive Mesh Refinement (AMR) library based on Space-Filling Curves.
"""

from typing import Protocol, runtime_checkable

import numpy as np

RefinementRegion = tuple[list[float], list[float], int]

def _quaternion_from_axis_angle(
    axis: list[float], angle: float
) -> list[float]: ...

class BoundingBox:
    """
    Physical axis-aligned domain bounds, in the surface's length units.

    Parameters
    ----------
    min, max : list of float
        Three finite coordinates with min < max on every axis and finite
        extents.

    Raises
    ------
    ValueError
        If bounds are nonfinite, reversed, or have zero/overflowing extents.

    Notes
    -----
    min and max are read-only properties returning coordinate copies.
    Point location includes the minimum faces and excludes the maximum faces.
    """

    min: list[float]
    max: list[float]

    def __init__(self, min: list[float], max: list[float]) -> None:
        """
        Validate finite, strictly ordered 3D bounds; see BoundingBox.
        """
        ...

@runtime_checkable
class ICfdMesh(Protocol):
    """
    Interface for CFD meshes.
    """

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

class CfdGhostCellMesh(ICfdMesh):
    """
    Structure of Arrays (SoA) mesh representation for CFD solvers
    using the Ghost-Cell Immersed Boundary Method (GCIBM).
    Arrays use NumPy storage with explicit ownership.
    Default snapshots are writable.

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
        Array of shape (N_internal_faces,) dtype=uintp
        the owner cell index for internal faces.
    internal_faces_neighbour : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uintp
        the neighbour cell index for internal faces.
    internal_faces_axis : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint8
        the axis of the internal faces.
        0: X, 1: Y, 2: Z
    domain_bnd_faces_owner : numpy.ndarray
        Array of shape (N_domain_bnd_faces,) dtype=uintp
        the owner cell index for domain boundary faces.
    domain_bnd_faces_dir : numpy.ndarray
        Array of shape (N_domain_bnd_faces,) dtype=uint8
        the direction of the domain boundary faces.
        0: -X, 1: +X, 2: -Y, 3: +Y, 4: -Z, 5: +Z

    [Ghost Cell IBM Specific Data]
    gc_is_fluid : numpy.ndarray
        Array of shape (N_cells,) dtype=bool
        a boolean flag indicating whether each cell is a fluid cell.

    gc_cell_ids : numpy.ndarray
        Array of shape (N_ghosts,) dtype=uintp
        the global indices of cells identified as ghost cells.
    gc_bnd_anchor_ids : numpy.ndarray
        Array of shape (N_ghosts,) dtype=uintp
        the boundary triangle IDs associated with each ghost cell.
    gc_bnd_patch_ids : numpy.ndarray
        Array of shape (N_ghosts,) dtype=uintp
        the boundary patch IDs associated with each ghost cell.
    gc_bnd_intercepts : numpy.ndarray
        Array of shape (N_ghosts, 3) dtype=float64
        the nearest boundary points.
    gc_image_points : numpy.ndarray
        Array of shape (N_ghosts, 3) dtype=float64
        the coordinates of the image points.
    gc_interp_stencil_indices : numpy.ndarray
        Array of shape (N_ghosts, 8) dtype=uintp
        the fluid cell indices used for interpolating at the Image Points.
    gc_interp_stencil_weights : numpy.ndarray
        Array of shape (N_ghosts, 8) dtype=float64
        the interpolation weights corresponding to `interp_stencil_indices`.

    Notes
    -----
    All gc_* rows except gc_is_fluid follow gc_cell_ids order. Unused stencil
    columns have zero weight and must be ignored. Linear least-squares weights
    can be negative. Failed linear interpolation falls back to inverse-distance
    weights; no candidates produce a self-reference with weight 1, not a valid
    fluid stencil. Counts for these cases are available only in the Rust report.
    Cell indices must not be reused across separate builds or remeshing.
    """

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
    @property
    def gc_is_fluid(self) -> np.ndarray:
        """
        Fluid classification: bool (n_cells,), True means fluid.
        """
        ...
    @property
    def gc_cell_ids(self) -> np.ndarray:
        """
        Ghost-cell indices into the background: uintp (n_ghosts,).
        """
        ...
    @property
    def gc_bnd_anchor_ids(self) -> np.ndarray:
        """
        Nearest surface triangle IDs: uintp (n_ghosts,).
        """
        ...
    @property
    def gc_bnd_patch_ids(self) -> np.ndarray:
        """
        Patch IDs at the nearest boundary points: uintp (n_ghosts,).
        """
        ...
    @property
    def gc_bnd_intercepts(self) -> np.ndarray:
        """
        Nearest boundary points in physical units: float64 (n_ghosts, 3).
        """
        ...
    @property
    def gc_image_points(self) -> np.ndarray:
        """
        Mirrored physical points, 2 * intercept - center: float64 (n_ghosts, 3).
        """
        ...
    @property
    def gc_interp_stencil_indices(self) -> np.ndarray:
        """
        Background interpolation indices: uintp (n_ghosts, 8); ignore
        zero-weight slots.
        """
        ...
    @property
    def gc_interp_stencil_weights(self) -> np.ndarray:
        """
        Interpolation weights: float64 (n_ghosts, 8); see class notes for
        fallback rows.
        """
        ...

class ApIbmFaceData:
    """
    Compressed axis-projected immersed-boundary payload for internal faces.

    `is_immersed_face` has length ``N_internal_faces``. All other arrays are
    compressed to ``N_immersed = count(is_immersed_face)`` and store data only
    for immersed faces, in the same order as ``True`` entries in
    `is_immersed_face`.

    Distances have shape ``(N_immersed,)`` and dtype ``float64``. They are
    physical distances from each cell center to its first boundary hit along
    the face axis, including zero for a boundary through the center.
    ``owner_near_boundary`` and ``neighbour_near_boundary`` are boolean arrays
    of the same shape. Each flags a candidate for a cell-center Dirichlet
    constraint when ``d / delta_x <= delta_x / L``, where ``delta_x`` is that
    side's cell width along the face axis and ``L`` is the largest extent of
    the computational domain. These flags do not alter the distances or
    apply boundary conditions; the solver must handle the constraints.
    """

    @property
    def is_immersed_face(self) -> np.ndarray:
        """
        Per-internal-face bool mask; both axis rays must hit a boundary.
        """
        ...
    @property
    def dist_owner_to_bnd(self) -> np.ndarray:
        """
        Owner-side physical first-hit distances: float64 (n_immersed,), possibly
        zero.
        """
        ...
    @property
    def dist_neighbour_to_bnd(self) -> np.ndarray:
        """
        Neighbour-side physical first-hit distances: float64 (n_immersed,),
        possibly zero.
        """
        ...
    @property
    def owner_near_boundary(self) -> np.ndarray:
        """
        Owner constraint candidates: bool (n_immersed,), using d / dx <= dx / L.
        """
        ...
    @property
    def neighbour_near_boundary(self) -> np.ndarray:
        """
        Neighbour constraint candidates: bool (n_immersed,), using d / dx <= dx
        / L.
        """
        ...
    @property
    def owner_bnd_anchor_id(self) -> np.ndarray:
        """
        Owner-side first-hit surface triangle IDs: uintp (n_immersed,).
        """
        ...
    @property
    def owner_bnd_patch_id(self) -> np.ndarray:
        """
        Owner-side first-hit patch IDs: uintp (n_immersed,).
        """
        ...
    @property
    def neighbour_bnd_anchor_id(self) -> np.ndarray:
        """
        Neighbour-side first-hit surface triangle IDs: uintp (n_immersed,).
        """
        ...
    @property
    def neighbour_bnd_patch_id(self) -> np.ndarray:
        """
        Neighbour-side first-hit patch IDs: uintp (n_immersed,).
        """
        ...

class CfdAxisProjectedMesh(ICfdMesh):
    """
    Structure of Arrays (SoA) mesh representation for CFD solvers
    using the Axis-Projected Immersed Boundary Method (APIBM/TFIBM).
    Arrays use NumPy storage with explicit ownership.
    Default snapshots are writable.

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
        Array of shape (N_internal_faces,) dtype=uintp
        the owner cell index for internal faces.
    internal_faces_neighbour : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uintp
        the neighbour cell index for internal faces.
    internal_faces_axis : numpy.ndarray
        Array of shape (N_internal_faces,) dtype=uint8
        the axis of the internal faces.
        0: X, 1: Y, 2: Z
    domain_bnd_faces_owner : numpy.ndarray
        Array of shape (N_domain_bnd_faces,) dtype=uintp
        the owner cell index for domain boundary faces.
    domain_bnd_faces_dir : numpy.ndarray
        Array of shape (N_domain_bnd_faces,) dtype=uint8
        the direction of the domain boundary faces.
        0: -X, 1: +X, 2: -Y, 3: +Y, 4: -Z, 5: +Z
    ap : ApIbmFaceData
        Axis-projected immersed-boundary face payload.

    Notes
    -----
    Index arrays refer to this background mesh. Indices stay stable for
    fixed-grid
    session updates but must not be reused after remeshing. Session.snapshot()
    and updates with copy=False publish read-only arrays; ordinary builds and
    session.mesh return independent writable copies. Editing copies does not
    alter Rust state. Surface patch IDs do not label domain boundary faces.
    """

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
    @property
    def ap(self) -> ApIbmFaceData:
        """
        Compressed immersed-face payload; see ApIbmFaceData for its indexing
        contract.
        """
        ...

class FluxelManager:
    """
    High-level API manager for automated AMR mesh generation.
    """
    def __init__(
        self,
        bbox: BoundingBox,
        base_res: list[int],
        n_leaf_refinement: int = 3,
        *,
        max_cells: int | None = None,
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
        max_cells : int | None (default: None)
            Positive upper bound on cells during generation, including balancing
            and final uniform refinement. Exceeding it raises ValueError.
            Also applies to sessions and subsequent remeshing. None adds no
            user-specified limit. This is not a target cell count or memory cap.

        Raises
        ------
        ValueError
            If bounds or root resolution are invalid, the root count exceeds
            max_cells, or n_leaf_refinement exceeds 32. Each resolution
            component
            must be positive and the product must not exceed 2**26.
        TypeError, OverflowError
            If an argument cannot be converted to the required native type.

        Notes
        -----
        No surface or cell mesh is created until a build/session method is
        called.
        Lengths use the same units as the surface coordinates.
        """
        ...

    def build_ghost_cell_mesh(
        self,
        mesh_path: str | None,
        target_level: int,
        fluid_seed_point: list[float],
        refinement_regions: list[RefinementRegion] | None = None,
    ) -> CfdGhostCellMesh:
        """
        Builds a CFD mesh
        tailored for the Ghost-Cell Immersed Boundary Method (GCIBM).

        This method reads an STL / OBJ file, refines the octree cells near the
        surface
        up to the `target_level`, performs inside/outside determination,
        and calculates GCIBM data such as image points or extrapolation weights.

        Parameters
        ----------
        mesh_path : str | None
            Path to the input STL / OBJ geometry file.
            Specifying None will generate a mesh without immersed boundary.
        target_level : int
            Surface refinement target before final uniform leaf refinement.
        fluid_seed_point : list of float
            The point in the physical domain to seed the fluid region.
        refinement_regions : list of tuple[list[float], list[float], int] | None
            Optional region refinement requests as ``(min, max, level)``.
            Cells whose AABB intersects a region are refined up to ``level``
            before 2:1 balancing and final uniform leaf refinement.

        Returns
        -------
        CfdGhostCellMesh
            The generated SoA mesh ready for CFD solvers.

        Raises
        ------
        ValueError
            If the seed is nonfinite, outside the half-open domain [min, max),
            or its containing cell intersects the boundary, even when the seed
            itself is not on the surface. Also raised for invalid
            levels/regions,
            unreadable or invalid surface files, and max_cells violations.
        RuntimeError
            If a geometry query or internal mesh validation fails.
        TypeError, OverflowError
            If an argument cannot be converted to the required native type.

        Notes
        -----
        target_level and region levels are targets before final uniform
        refinement,
        not upper bounds on final cell levels. Each must be in [0, 32], and the
        largest requested level plus n_leaf_refinement must not exceed 32.
        Regions use finite, strictly ordered corners; only positive-volume
        overlap
        counts. None and [] both mean no region requests for a new build.
        max_cells also applies to balancing and final uniform refinement.
        The surface starts at its file coordinates with no rotation or
        translation.

        The result owns independent writable NumPy arrays. Cell/face indices are
        local to this build; changing an array does not alter the manager. This
        one-shot call does not retain a session. Rust build diagnostics are not
        returned or emitted as warnings by this Python method.

        With mesh_path=None, a valid seed is still required; all cells are fluid
        and ghost arrays are empty. With a surface, seed-connected
        non-intersecting
        cells and intersecting centers on the same side as the seed are fluid.
        Ghost cells use weighted linear least squares, with inverse-distance
        fallback. If no fluid candidate exists, a row references the ghost cell
        itself with weight 1; this is not a valid fluid interpolation. The Rust
        unresolved-stencil count is currently not exposed in Python.
        """
        ...

    def build_axis_projected_mesh(
        self,
        mesh_path: str | None,
        target_level: int,
        refinement_regions: list[RefinementRegion] | None = None,
    ) -> CfdAxisProjectedMesh:
        """
        Builds a CFD mesh
        tailored for the Axis-Projected Immersed Boundary Method (APIBM).

        This method reads an STL / OBJ file, refines the octree cells near the
        surface
        up to the `target_level`, performs axis raytracing, and calculates
        APIBM data such as physical distances to the immersed boundary along
        each axis and cell-center Dirichlet-constraint candidates. These are
        available in the returned mesh's ``ap`` payload.

        Parameters
        ----------
        mesh_path : str | None
            Path to the input STL / OBJ geometry file.
            Specifying None will generate a mesh without immersed boundary.
        target_level : int
            Surface refinement target before final uniform leaf refinement.
        refinement_regions : list of tuple[list[float], list[float], int] | None
            Optional region refinement requests as ``(min, max, level)``.
            Cells whose AABB intersects a region are refined up to ``level``
            before 2:1 balancing and final uniform leaf refinement.

        Returns
        -------
        CfdAxisProjectedMesh
            The generated SoA mesh ready for CFD solvers.

        Notes
        -----
        target_level and region levels are targets before final uniform
        refinement,
        not upper bounds on final cell levels. Each must be in [0, 32], and the
        largest requested level plus n_leaf_refinement must not exceed 32.
        Regions use finite, strictly ordered corners; only positive-volume
        overlap
        counts. None and [] both mean no region requests for a new build.
        max_cells also applies to balancing and final uniform refinement.
        The surface starts at its file coordinates with no rotation or
        translation.

        Raises
        ------
        ValueError
            If levels, regions, file format, file contents, or surface geometry
            are
            invalid, the file cannot be read, or max_cells would be exceeded.
        RuntimeError
            If a geometry query or internal mesh validation fails.
        TypeError, OverflowError
            If an argument cannot be converted to the required native type.

        The result owns independent writable NumPy arrays. Cell/face indices are
        local to this build; changing an array does not alter the manager. This
        one-shot call does not retain a session. Rust build diagnostics are not
        returned or emitted as warnings by this Python method.

        Both owner and neighbour axis rays must hit for a face to be immersed.
        This method does not classify fluid/solid cells or apply boundary
        conditions.
        With mesh_path=None, all immersed-face flags are false and the
        compressed
        APIBM arrays are empty.
        """
        ...

    def create_axis_projected_session(
        self,
        mesh_path: str | None,
        target_level: int,
        refinement_regions: list[RefinementRegion] | None = None,
    ) -> ApibmSession:
        """
        Creates a stateful APIBM session that retains the background forest
        for rigid moving-boundary updates via ``update_ib`` / ``remesh``.

        Parameters
        ----------
        mesh_path : str | None
            Path to the input STL / OBJ geometry file.
            Specifying None will generate a mesh without immersed boundary.
        target_level : int
            Surface refinement target before final uniform leaf refinement.
        refinement_regions : list of tuple[list[float], list[float], int] | None
            Optional region refinement requests as ``(min, max, level)``.
            Cells whose AABB intersects a region are refined up to ``level``
            before 2:1 balancing and final uniform leaf refinement.

        Returns
        -------
        ApibmSession
            Session holding the forest, IBM geometry, and current CFD mesh.

        Notes
        -----
        target_level and region levels are targets before final uniform
        refinement,
        not upper bounds on final cell levels. Each must be in [0, 32], and the
        largest requested level plus n_leaf_refinement must not exceed 32.
        Regions use finite, strictly ordered corners; only positive-volume
        overlap
        counts. None and [] both mean no region requests for a new build.
        max_cells also applies to balancing and final uniform refinement.
        The surface starts at its file coordinates with no rotation or
        translation.

        Raises
        ------
        ValueError
            If levels, regions, file format, file contents, or surface geometry
            are
            invalid, the file cannot be read, or max_cells would be exceeded.
        RuntimeError
            If a geometry query or internal mesh validation fails.
        TypeError, OverflowError
            If an argument cannot be converted to the required native type.

        The session retains the background grid and surface for rigid motion.
        Use session.mesh for writable copies or session.snapshot() for read-only
        shared publication. The manager's max_cells applies to every remesh.
        """
        ...

class ApibmSession:
    """
    Stateful APIBM session for rigid immersed-boundary motion.

    Keep this object alive across timesteps. Use ``update_ib`` when the
    background mesh can stay fixed, and ``remesh`` when AMR should follow
    the boundary.

    Attributes
    ----------
    mesh : CfdAxisProjectedMesh
        Current CFD mesh snapshot (a new Python object on each access).
    translation : list of float
        Current rigid translation ``[tx, ty, tz]`` applied to the IB mesh.
    rotation_quaternion : list of float
        Current rigid rotation as a unit quaternion ``[w, x, y, z]``.
    """

    @property
    def mesh(self) -> CfdAxisProjectedMesh:
        """
        Return a fresh snapshot with independent writable arrays; changes do not
        affect the session.
        """
        ...
    def snapshot(self) -> CfdAxisProjectedMesh:
        """
        Return a read-only snapshot, sharing cached immutable array storage.

        Old snapshots remain valid after updates and session destruction.
        Initial publication copies to immutable storage.
        Repeated reads reuse that storage.

        Returns
        -------
        CfdAxisProjectedMesh
            Read-only arrays backed by immutable Python bytes, with fresh
            ndarray
            headers and a fresh patch dictionary. This is not end-to-end
            zero-copy.
            Array metadata changes do not change subsequent snapshots.
        """
        ...
    @property
    def translation(self) -> list[float]:
        """
        Current absolute translation [tx, ty, tz] in surface length units.
        """
        ...
    @property
    def rotation_quaternion(self) -> list[float]:
        """
        Current normalized absolute rotation [w, x, y, z].
        """
        ...
    def update_ib(
        self,
        translation: list[float] | None = None,
        rotation_quaternion: list[float] | None = None,
        warn_outside_refinement: bool = True,
        *,
        copy: bool = True,
    ) -> CfdAxisProjectedMesh:
        """
        Recomputes IB face data only (topology fixed).

        Pose arguments are absolute. Quaternion order is ``[w, x, y, z]``.
        Omitted components keep the current pose values.
        Use ``quaternion_from_axis_angle`` to build a quaternion from an
        axis and angle.

        Parameters
        ----------
        translation : list of float or None
            Absolute translation ``[tx, ty, tz]``. ``None`` keeps the current
            translation.
        rotation_quaternion : list of float or None
            Absolute unit quaternion ``[w, x, y, z]``. ``None`` keeps the
            current rotation. Build from an axis and angle with
            ``quaternion_from_axis_angle``.
        warn_outside_refinement : bool, default True
            If True, warn when the IB intersects cells below ``target_level``.
            When warnings raise exceptions, the current session stays unchanged.
        copy : bool, default True
            Return independent writable arrays. False returns read-only arrays
            sharing cached background storage. Existing snapshots stay valid.

        Returns
        -------
        CfdAxisProjectedMesh
            Updated mesh snapshot with rebuilt immersed-boundary payload.

        Raises
        ------
        ValueError
            If a pose is nonfinite, or requested refinement is invalid or
            exceeds
            max_cells (remesh only).
        RuntimeError
            If geometry computation fails, or the session is already borrowed by
            a concurrent or reentrant operation.
        UserWarning
            Emitted when requested and the boundary crosses cells below
            target_level;
            raised as an exception when warnings are configured as errors.
        TypeError, OverflowError
            If an argument cannot be converted to the required native type.

        Notes
        -----
        Pose values are absolute, not increments. Rotation is about the original
        coordinate origin, followed by translation, in the surface's length
        units.
        Finite nonzero quaternions are normalized. For Python compatibility, a
        finite quaternion with squared norm <= float64 epsilon becomes identity.
        Omitted pose components retain their current values.
        All fallible computation, warnings, and Python publication complete
        before
        the state is committed; failure leaves the current mesh, pose, and
        settings
        unchanged. Previously returned snapshots remain valid.

        Cell and face indices stay fixed. copy=False reuses the cached
        background
        storage but publishes newly computed boundary data.
        """
        ...

    def remesh(
        self,
        target_level: int | None = None,
        refinement_regions: list[RefinementRegion] | None = None,
        translation: list[float] | None = None,
        rotation_quaternion: list[float] | None = None,
        warn_outside_refinement: bool = True,
        *,
        copy: bool = True,
    ) -> CfdAxisProjectedMesh:
        """
        Rebuilds the AMR background mesh and IB payload for the current pose.

        Parameters
        ----------
        target_level : int or None
            Surface refinement target before final uniform leaf refinement.
            ``None`` keeps the current target level.
        refinement_regions : list of tuple[list[float], list[float], int] | None
            Optional region refinement requests as ``(min, max, level)``.
            ``None`` keeps the current region list.
        translation : list of float or None
            Absolute translation ``[tx, ty, tz]``. ``None`` keeps the current
            translation.
        rotation_quaternion : list of float or None
            Absolute unit quaternion ``[w, x, y, z]``. ``None`` keeps the
            current rotation. Build from an axis and angle with
            ``quaternion_from_axis_angle``.
        warn_outside_refinement : bool, default True
            If True, warn when the IB intersects cells below ``target_level``.
            When warnings raise exceptions, the current session stays unchanged.
        copy : bool, default True
            Return independent writable arrays. False returns read-only arrays
            sharing cached background storage. Existing snapshots stay valid.

        Returns
        -------
        CfdAxisProjectedMesh
            Newly built mesh snapshot after AMR and IB reconstruction.

        Raises
        ------
        ValueError
            If a pose is nonfinite, or requested refinement is invalid or
            exceeds
            max_cells (remesh only).
        RuntimeError
            If geometry computation fails, or the session is already borrowed by
            a concurrent or reentrant operation.
        UserWarning
            Emitted when requested and the boundary crosses cells below
            target_level;
            raised as an exception when warnings are configured as errors.
        TypeError, OverflowError
            If an argument cannot be converted to the required native type.

        Notes
        -----
        Pose values are absolute, not increments. Rotation is about the original
        coordinate origin, followed by translation, in the surface's length
        units.
        Finite nonzero quaternions are normalized. For Python compatibility, a
        finite quaternion with squared norm <= float64 epsilon becomes identity.
        Omitted pose components retain their current values.
        All fallible computation, warnings, and Python publication complete
        before
        the state is committed; failure leaves the current mesh, pose, and
        settings
        unchanged. Previously returned snapshots remain valid.

        refinement_regions=None retains existing regions; [] clears them.
        The manager's max_cells and n_leaf_refinement remain in effect.
        Cell and face indices may change even if the cell count does not.
        Rebuild
        solver caches and transfer solution fields separately; this API does not
        transfer them. copy=False publishes the new background as read-only
        data.
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

        Raises
        ------
        ValueError
            If bounds are invalid or resolution is nonpositive or exceeds 2**26
            roots.

        Notes
        -----
        Creates populated level-0 cells, unlike the empty low-level Rust Forest.
        This Python API has no max_cells argument and does not inherit a
        manager's
        limit. Refinement changes cell ordering; flags refer to the current
        order.
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

        Notes
        -----
        Each flagged leaf is replaced by eight children; level-32 leaves are
        skipped.
        Does not automatically restore 2:1 balance. Mutates this forest in
        place.

        Raises
        ------
        ValueError
            If the flag count differs from num_cells() or cell-count arithmetic
            overflows. Validation failure leaves the forest unchanged.
        """
        ...

    def coarsen_by_flags(self, flags: list[bool]) -> None:
        """
        Coarsens sibling cells back to their parent based on a boolean mask.

        Parameters
        ----------
        flags : list[bool]
            A boolean list of length `num_cells()`.

        Notes
        -----
        Only complete groups of eight flagged siblings are replaced by their
        parent.
        Other flagged cells are left unchanged. Does not automatically restore
        2:1 balance. Mutates this forest in place and changes cell indices.

        Raises
        ------
        ValueError
            If the flag count differs from num_cells(); the forest is unchanged.
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

        Notes
        -----
        Only positive-volume overlap counts; touching a region face is
        insufficient.
        Coordinates are in the domain's physical units. Does not coarsen finer
        cells
        or automatically restore balance. Mutates in place after successful
        work.

        Raises
        ------
        ValueError
            If corners are nonfinite or not strictly ordered, target_level is
            outside
            [0, 32], or cell-count arithmetic overflows. Failure leaves the
            forest
            unchanged.
        """
        ...

    def enforce_2_to_1_balance(self) -> None:
        """
        Enforces the 2:1 balancing constraint across the entire mesh,
        ensuring adjacent cells differ by at most one refinement level.

        Notes
        -----
        Checks face, edge and corner neighbours (26 probes); only refines coarse
        cells. Mutates in place and may change cell indices.

        Raises
        ------
        ValueError
            If cell-count arithmetic overflows. Failure leaves the forest
            unchanged.
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

        Notes
        -----
        Zero is a no-op; every additional level multiplies cell count by eight.
        Preserves an existing 2:1 balance but does not fix an unbalanced forest.
        Mutates in place and changes cell indices.

        Raises
        ------
        ValueError
            If the resulting level exceeds 32 or cell-count arithmetic
            overflows.
            Validation failure leaves the forest unchanged.
        """
        ...
