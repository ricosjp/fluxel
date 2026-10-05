"""
Fluxel Python Binding (Stub File)

This module provides the Python interface for Fluxel,
an Adaptive Mesh Refinement (AMR) library based on Space-Filling Curves.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from fluxel.types import Bool3, Float3, Int3

RefinementRegion = tuple[Float3, Float3, int]

def _quaternion_from_axis_angle(axis: Float3, angle: float) -> list[float]: ...

class BoundingBox:
    """
    Physical axis-aligned domain bounds, in the surface's length units.

    Parameters
    ----------
    min, max : Float3
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

    min: Float3
    max: Float3

    def __init__(self, min: Float3, max: Float3) -> None:
        """
        Validate finite, strictly ordered 3D bounds; see BoundingBox.
        """
        ...

class Cylindrical:
    """
    Annular sector or full turn about an axis parallel to world Z.

    Parameters
    ----------
    origin : Float3
        World point where ``r = 0`` and ``z = 0``.
    r_min, r_max : float
        Inner and outer radii. ``r_min`` must be positive.
    theta_start, theta_extent : float
        Start angle and angular width in radians. One turn selects a
        periodic full turn.
    z_min, z_max : float
        Axial limits in the same length units as the radii.

    Raises
    ------
    ValueError
        If a value is nonfinite, the radius includes the axis, the angle
        is outside ``(0, 2π]``, or the axial interval is empty.

    Notes
    -----
    Properties are read-only copies. ``r`` and ``z`` include the lower
    face and exclude the upper face. ``θ`` is
    ``[theta_start, theta_start + extent)``. ``r = 0`` is not supported.
    """

    origin: Float3
    r_min: float
    r_max: float
    theta_start: float
    theta_extent: float
    z_min: float
    z_max: float

    def __init__(
        self,
        origin: Float3,
        r_min: float,
        r_max: float,
        theta_start: float,
        theta_extent: float,
        z_min: float,
        z_max: float,
    ) -> None:
        """
        Validate an annular sector or full turn; see Cylindrical.
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
        Coordinate code: 0 is Cartesian and 1 is cylindrical.
        """
        ...
    @property
    def cell_centers(self) -> np.ndarray:
        """
        World sample points: float64 (n_cells, 3).

        Cartesian samples are box centers. Cylindrical samples are
        parameter midpoints mapped to world and differ from volume centroids.
        """
        ...
    @property
    def cell_sizes(self) -> np.ndarray:
        """
        Coordinate-aligned widths: float64 (n_cells, 3).

        Cartesian widths are edge lengths. Cylindrical widths are
        ``[Δr, r Δθ, Δz]``. Their product is not the cell volume.
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

        Cylindrical edges are straight chords through the parameter
        corners.
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
    def domain_bnd_faces_owner(self) -> np.ndarray:
        """
        Domain-boundary owner cell indices: uintp (n_domain_bnd_faces,).
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
        Domain-face minima: float64 (n_domain_bnd_faces, 3). Components
        match the internal-face minima.
        """
        ...
    @property
    def domain_bnd_faces_max(self) -> np.ndarray:
        """
        Domain-face maxima: float64 (n_domain_bnd_faces, 3). Components
        match the internal-face minima.
        """
        ...

class CfdGhostCellMesh(ICfdMesh):
    """
    SoA mesh for the ghost-cell immersed-boundary method.
    Arrays use NumPy storage. One-shot builds are writable.

    ``coordinate_type`` is 0 for Cartesian and 1 for cylindrical.
    Logical axes 0, 1, 2 are X, Y, Z or ``r``, ``θ``, ``z``.
    Ghost rows, except ``gc_is_fluid``, follow ``gc_cell_ids``.

    Notes
    -----
    Unused stencil columns have zero weight and must be ignored. Linear
    least-squares weights can be negative. Failed linear interpolation
    falls back to inverse-distance weights. No candidates produce a
    self-reference with weight 1, which is not a valid fluid stencil.
    Those counts stay in the Rust report. Cell indices must not be reused
    across builds or remeshing.
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
        Coordinate code: 0 is Cartesian and 1 is cylindrical.
        """
        ...
    @property
    def cell_centers(self) -> np.ndarray:
        """
        World sample points: float64 (n_cells, 3).

        Cartesian samples are box centers. Cylindrical samples are
        parameter midpoints mapped to world and differ from volume centroids.
        """
        ...
    @property
    def cell_sizes(self) -> np.ndarray:
        """
        Coordinate-aligned widths: float64 (n_cells, 3).

        Cartesian widths are edge lengths. Cylindrical widths are
        ``[Δr, r Δθ, Δz]``. Their product is not the cell volume.
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

        Cylindrical edges are straight chords through the parameter
        corners.
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
    def domain_bnd_faces_owner(self) -> np.ndarray:
        """
        Domain-boundary owner cell indices: uintp (n_domain_bnd_faces,).
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
        Domain-face minima: float64 (n_domain_bnd_faces, 3). Components
        match the internal-face minima.
        """
        ...
    @property
    def domain_bnd_faces_max(self) -> np.ndarray:
        """
        Domain-face maxima: float64 (n_domain_bnd_faces, 3). Components
        match the internal-face minima.
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
    Compressed axis-projected payload for internal faces.

    ``is_immersed_face`` has one entry per internal face. Every other array
    follows the ``True`` entries, in that order.

    A distance is the path length from that side's sample to its first hit,
    including zero when the boundary passes through the sample. Cartesian
    paths follow the world axis. Cylindrical ``r`` and ``z`` follow ``e_r``
    and ``e_z``. Cylindrical ``θ`` is the arc length ``r|Δθ|``.

    ``owner_near_boundary`` and ``neighbour_near_boundary`` mark a
    cell-center Dirichlet candidate when ``d / width <= width / L``.
    ``width`` is that side's cell width on the face axis and ``L`` is the
    largest domain extent. The flags do not change the distances or apply
    a boundary condition.

    Points, tangents, and normals have shape ``(N_immersed, 3)``. A tangent
    is the unit search direction at the hit. A normal follows triangle
    winding and is shared by both sides of that triangle.
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
        Owner-side path length to the first hit: float64 (n_immersed,),
        possibly zero.
        """
        ...
    @property
    def dist_neighbour_to_bnd(self) -> np.ndarray:
        """
        Neighbour-side path length to the first hit: float64 (n_immersed,),
        possibly zero.
        """
        ...
    @property
    def owner_near_boundary(self) -> np.ndarray:
        """
        Owner Dirichlet candidates: bool (n_immersed,). See the class notes.
        """
        ...
    @property
    def neighbour_near_boundary(self) -> np.ndarray:
        """
        Neighbour Dirichlet candidates: bool (n_immersed,). See the class
        notes.
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
    @property
    def owner_bnd_point(self) -> np.ndarray:
        """
        Owner-side world intersections: float64 (n_immersed, 3).
        """
        ...
    @property
    def owner_bnd_tangent(self) -> np.ndarray:
        """
        Owner-side unit search direction at the hit: float64 (n_immersed, 3).
        """
        ...
    @property
    def owner_bnd_normal(self) -> np.ndarray:
        """
        Owner-side unit triangle normal: float64 (n_immersed, 3).
        """
        ...
    @property
    def neighbour_bnd_point(self) -> np.ndarray:
        """
        Neighbour-side world intersections: float64 (n_immersed, 3).
        """
        ...
    @property
    def neighbour_bnd_tangent(self) -> np.ndarray:
        """
        Neighbour-side unit search direction at the hit: float64
        (n_immersed, 3).
        """
        ...
    @property
    def neighbour_bnd_normal(self) -> np.ndarray:
        """
        Neighbour-side unit triangle normal: float64 (n_immersed, 3).
        """
        ...

class CfdAxisProjectedMesh(ICfdMesh):
    """
    SoA mesh for the axis-projected immersed-boundary method.
    Arrays use NumPy storage. Ordinary builds and ``session.mesh`` are
    writable.

    ``coordinate_type`` is 0 for Cartesian and 1 for cylindrical.
    Logical axes 0, 1, 2 are X, Y, Z or ``r``, ``θ``, ``z``.

    Notes
    -----
    Index arrays refer to this background mesh. Indices stay stable for
    fixed-grid session updates but must not be reused after remeshing.
    ``snapshot()`` and updates with ``copy=False`` publish read-only
    arrays. Editing copies does not alter Rust state. Surface patch IDs
    do not label domain boundary faces.
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
        Coordinate code: 0 is Cartesian and 1 is cylindrical.
        """
        ...
    @property
    def cell_centers(self) -> np.ndarray:
        """
        World sample points: float64 (n_cells, 3).

        Cartesian samples are box centers. Cylindrical samples are
        parameter midpoints mapped to world and differ from volume centroids.
        """
        ...
    @property
    def cell_sizes(self) -> np.ndarray:
        """
        Coordinate-aligned widths: float64 (n_cells, 3).

        Cartesian widths are edge lengths. Cylindrical widths are
        ``[Δr, r Δθ, Δz]``. Their product is not the cell volume.
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

        Cylindrical edges are straight chords through the parameter
        corners.
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
    def domain_bnd_faces_owner(self) -> np.ndarray:
        """
        Domain-boundary owner cell indices: uintp (n_domain_bnd_faces,).
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
        Domain-face minima: float64 (n_domain_bnd_faces, 3). Components
        match the internal-face minima.
        """
        ...
    @property
    def domain_bnd_faces_max(self) -> np.ndarray:
        """
        Domain-face maxima: float64 (n_domain_bnd_faces, 3). Components
        match the internal-face minima.
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
        domain: BoundingBox | Cylindrical,
        base_res: Int3,
        n_leaf_refinement: int = 3,
        *,
        max_cells: int | None = None,
        periodic: Bool3 | None = None,
    ) -> None:
        """
        Initialize the FluxelManager.

        Parameters
        ----------
        domain : BoundingBox or Cylindrical
            Physical domain. A box uses world ``(x, y, z)``. A cylinder
            uses ``(r, θ, z)``.
        base_res : Int3
            Root counts along the domain axes.
        n_leaf_refinement : int (default: 3)
            The number of times that
            the uniform refinement is applied to the final mesh.
            n_leaf_refinement=3 will generate 8x8x8 micro-cells per octree leaf.
        max_cells : int | None (default: None)
            Positive upper bound on cells during generation, including balancing
            and final uniform refinement. Exceeding it raises ValueError.
            Also applies to sessions and subsequent remeshing. None adds no
            user-specified limit. This is not a target cell count or memory cap.
        periodic : Bool3 or None (default: None)
            Axes that wrap across the root grid. Cartesian order is
            ``(x, y, z)``. Cylindrical order is ``(r, θ, z)``. ``None``
            wraps none on a box and wraps only θ on a full-turn cylinder.

        Raises
        ------
        ValueError
            If the domain, root resolution, or refinement limits are invalid.
        TypeError, OverflowError
            If an argument cannot be converted to the required native type.

        Notes
        -----
        No surface or cell mesh is created until a build/session method is
        called. Lengths use the same units as the surface coordinates. A
        cylinder uses parameter refinement ``(r, θ, z)``. A full turn may
        wrap θ with ``max[1] < min[1]``. World-box refinement on a cylinder
        is rejected.
        """
        ...

    def build_ghost_cell_mesh(
        self,
        mesh_path: str | None,
        target_level: int,
        fluid_seed_point: Float3,
        refinement_regions: list[RefinementRegion] | None = None,
    ) -> CfdGhostCellMesh:
        """
        Builds a CFD mesh
        tailored for the Ghost-Cell Immersed Boundary Method (GCIBM).

        This method reads an STL / OBJ file, refines cells near the surface
        up to `target_level`, classifies inside and outside, and stores
        image points and extrapolation weights.

        Parameters
        ----------
        mesh_path : str | None
            Path to the input STL / OBJ geometry file.
            Specifying None will generate a mesh without immersed boundary.
        target_level : int
            Surface refinement target before final uniform leaf refinement.
        fluid_seed_point : Float3
            The point in the physical domain to seed the fluid region.
        refinement_regions : list of tuple[Float3, Float3, int] | None
            ``(min, max, level)`` intervals refined before balancing.
            Cartesian intervals are world ``(x, y, z)``. Cylindrical
            intervals are ``(r, θ, z)``; a full turn may wrap ``θ`` with
            ``max[1] < min[1]``.

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
        Regions need finite corners and positive overlap. On a cylinder,
        ``r`` and ``z`` stay ordered and ``θ`` may wrap. None and []
        request no regions.
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

        This method reads an STL / OBJ file, refines cells near the surface
        up to `target_level`, and stores the axis-projected hit paths on
        ``ap``.

        Parameters
        ----------
        mesh_path : str | None
            Path to the input STL / OBJ geometry file.
            Specifying None will generate a mesh without immersed boundary.
        target_level : int
            Surface refinement target before final uniform leaf refinement.
        refinement_regions : list of tuple[Float3, Float3, int] | None
            ``(min, max, level)`` intervals refined before balancing.
            Cartesian intervals are world ``(x, y, z)``. Cylindrical
            intervals are ``(r, θ, z)``; a full turn may wrap ``θ`` with
            ``max[1] < min[1]``.

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
        Regions need finite corners and positive overlap. On a cylinder,
        ``r`` and ``z`` stay ordered and ``θ`` may wrap. None and []
        request no regions.
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
        refinement_regions : list of tuple[Float3, Float3, int] | None
            ``(min, max, level)`` intervals refined before balancing.
            Cartesian intervals are world ``(x, y, z)``. Cylindrical
            intervals are ``(r, θ, z)``; a full turn may wrap ``θ`` with
            ``max[1] < min[1]``.

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
        Regions need finite corners and positive overlap. On a cylinder,
        ``r`` and ``z`` stay ordered and ``θ`` may wrap. None and []
        request no regions.
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
    translation : Float3
        Current rigid translation ``(tx, ty, tz)`` applied to the IB mesh.
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
    def translation(self) -> Float3:
        """
        Current absolute translation (tx, ty, tz) in surface length units.
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
        translation: Float3 | None = None,
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
        translation : Float3 or None
            Absolute translation ``(tx, ty, tz)``. ``None`` keeps the current
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
        translation: Float3 | None = None,
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
        refinement_regions : list of tuple[Float3, Float3, int] | None
            ``(min, max, level)`` intervals. ``None`` keeps the current list.
            Cartesian intervals are world ``(x, y, z)``. Cylindrical
            intervals are ``(r, θ, z)``; a full turn may wrap ``θ`` with
            ``max[1] < min[1]``.
        translation : Float3 or None
            Absolute translation ``(tx, ty, tz)``. ``None`` keeps the current
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
        base_res: Int3,
        *,
        periodic: Bool3 | None = None,
    ) -> None:
        """
        Initialize the Forest.

        Parameters
        ----------
        bbox : BoundingBox
            The physical bounds of the overall domain.
        base_res : Int3
            The initial number of root blocks (trees) in [X, Y, Z] directions.
        periodic : Bool3 or None (default: None)
            Axes that wrap across the root grid as ``(x, y, z)``. ``None``
            wraps none.

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
        self, min: Float3, max: Float3, target_level: int
    ) -> None:
        """
        Refines all cells that intersect with the specified bounding box
        up to the given target level.

        Parameters
        ----------
        min : Float3
            Minimum corner ``(x, y, z)``.
        max : Float3
            Maximum corner ``(x, y, z)``.
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
