//! Thin Python entry points into the Rust engine.
use crate::{
    apibm_session::ApibmSession,
    arguments::{self, RegionInput},
    cfd_mesh::{BoundingBox, CfdAxisProjectedMesh, CfdGhostCellMesh, Cylindrical},
    errors::to_python,
};
use fluxel_engine::{self as engine, BuildLimits, MeshBuildConfig};
use pyo3::prelude::*;
use std::path::Path;
/// High-level API manager for automated AMR mesh generation.
///
/// Initialize the FluxelManager.
///
/// Parameters
/// ----------
/// domain : BoundingBox or Cylindrical
///     Physical domain. A box uses world ``(x, y, z)``. A cylinder uses
///     ``(r, θ, z)``.
/// base_res : Int3
///     Root counts along the domain axes.
/// n_leaf_refinement : int (default: 3)
///     The number of times that
///     the uniform refinement is applied to the final mesh.
///     n_leaf_refinement=3 will generate 8x8x8 micro-cells per octree leaf.
/// max_cells : int | None (default: None)
///     Positive upper bound on cells during generation, including balancing
///     and final uniform refinement. Exceeding it raises ValueError.
///     Also applies to sessions and subsequent remeshing. None adds no
///     user-specified limit. This is not a target cell count or memory cap.
/// periodic : Bool3 or None (default: None)
///     Axes that wrap across the root grid. Cartesian order is ``(x, y, z)``.
///     Cylindrical order is ``(r, θ, z)``. ``None`` wraps none on a box and
///     wraps only θ on a full-turn cylinder.
///
/// Raises
/// ------
/// ValueError
///     If bounds or root resolution are invalid, the root count exceeds
///     max_cells, or n_leaf_refinement exceeds 32. Each resolution component
///     must be positive and the product must not exceed 2**26.
/// TypeError, OverflowError
///     If an argument cannot be converted to the required native type.
///
/// Notes
/// -----
/// No surface or cell mesh is created until a build/session method is called.
/// Lengths use the same units as the surface coordinates. A cylinder uses
/// parameter refinement ``(r, θ, z)``. A full turn may wrap θ with
/// ``max[1] < min[1]``. World-box refinement on a cylinder is rejected.
#[pyclass]
pub struct FluxelManager {
    config: MeshBuildConfig,
}
#[pymethods]
impl FluxelManager {
    #[new]
    #[pyo3(signature = (
        domain, base_res, n_leaf_refinement = 3, *, max_cells = None, periodic = None
    ))]
    /// Initialize the FluxelManager.
    ///
    /// Parameters
    /// ----------
    /// domain : BoundingBox or Cylindrical
    ///     Physical domain. A box uses world ``(x, y, z)``. A cylinder uses
    ///     ``(r, θ, z)``.
    /// base_res : Int3
    ///     Root counts along the domain axes.
    /// n_leaf_refinement : int (default: 3)
    ///     The number of times that
    ///     the uniform refinement is applied to the final mesh.
    ///     n_leaf_refinement=3 will generate 8x8x8 micro-cells per octree leaf.
    /// max_cells : int | None (default: None)
    ///     Positive upper bound on cells during generation, including balancing
    ///     and final uniform refinement. Exceeding it raises ValueError.
    ///     Also applies to sessions and subsequent remeshing. None adds no
    ///     user-specified limit. This is not a target cell count or memory cap.
    /// periodic : Bool3 or None (default: None)
    ///     Axes that wrap across the root grid. Cartesian order is ``(x, y, z)``.
    ///     Cylindrical order is ``(r, θ, z)``. ``None`` wraps none on a box and
    ///     wraps only θ on a full-turn cylinder.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the domain, root resolution, or refinement limits are invalid.
    /// TypeError, OverflowError
    ///     If an argument cannot be converted to the required native type.
    ///
    /// Notes
    /// -----
    /// No surface or cell mesh is created until a build/session method is called.
    /// Lengths use the same units as the surface coordinates. A cylinder uses
    /// parameter refinement ``(r, θ, z)``. A full turn may wrap θ with
    /// ``max[1] < min[1]``. World-box refinement on a cylinder is rejected.
    pub fn new(
        domain: &Bound<'_, PyAny>,
        base_res: [u32; 3],
        n_leaf_refinement: u8,
        max_cells: Option<usize>,
        periodic: Option<[bool; 3]>,
    ) -> PyResult<Self> {
        let limits = BuildLimits {
            max_cells: max_cells.unwrap_or(usize::MAX),
        };
        let config = if let Ok(bbox) = domain.extract::<PyRef<BoundingBox>>() {
            let bounds = fluxel_geometry::BoundingBox::new(bbox.min, bbox.max)
                .map_err(|error| to_python(error.into()))?;
            MeshBuildConfig::new(
                bounds,
                base_res,
                n_leaf_refinement,
                limits,
                periodic.unwrap_or([false; 3]),
            )
        } else if let Ok(cylinder) = domain.extract::<PyRef<Cylindrical>>() {
            let geometry = cylinder.geometry();
            let axes = periodic.unwrap_or([false, geometry.is_full_turn(), false]);
            MeshBuildConfig::cylindrical(
                geometry.origin(),
                geometry.r_min(),
                geometry.r_max(),
                geometry.theta_start(),
                geometry.theta_extent(),
                geometry.z_min(),
                geometry.z_max(),
                base_res,
                n_leaf_refinement,
                limits,
                axes,
            )
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "domain must be a BoundingBox or a Cylindrical",
            ));
        };
        Ok(Self {
            config: config.map_err(to_python)?,
        })
    }
    #[pyo3(signature = (mesh_path, target_level, fluid_seed_point, refinement_regions = None))]
    /// Builds a CFD mesh
    /// tailored for the Ghost-Cell Immersed Boundary Method (GCIBM).
    ///
    /// This method reads an STL / OBJ file, refines cells near the surface up to
    /// `target_level`, classifies inside and outside, and stores image points
    /// and extrapolation weights.
    ///
    /// Parameters
    /// ----------
    /// mesh_path : str | None
    ///     Path to the input STL / OBJ geometry file.
    ///     Specifying None will generate a mesh without immersed boundary.
    /// target_level : int
    ///     Surface refinement target before final uniform leaf refinement.
    /// fluid_seed_point : Float3
    ///     The point in the physical domain to seed the fluid region.
    /// refinement_regions : list of tuple[Float3, Float3, int] | None
    ///     ``(min, max, level)`` intervals refined before balancing.
    ///     Cartesian intervals are world ``(x, y, z)``. Cylindrical intervals
    ///     are ``(r, θ, z)``; a full turn may wrap ``θ`` with ``max[1] < min[1]``.
    ///
    /// Returns
    /// -------
    /// CfdGhostCellMesh
    ///     The generated SoA mesh ready for CFD solvers.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the seed is nonfinite, outside the half-open domain [min, max),
    ///     or its containing cell intersects the boundary, even when the seed
    ///     itself is not on the surface. Also raised for invalid levels/regions,
    ///     unreadable or invalid surface files, and max_cells violations.
    /// RuntimeError
    ///     If a geometry query or internal mesh validation fails.
    /// TypeError, OverflowError
    ///     If an argument cannot be converted to the required native type.
    ///
    /// Notes
    /// -----
    /// target_level and region levels are targets before final uniform refinement,
    /// not upper bounds on final cell levels. Each must be in [0, 32], and the
    /// largest requested level plus n_leaf_refinement must not exceed 32.
    /// Regions need finite corners and positive overlap. On a cylinder, ``r`` and
    /// ``z`` stay ordered and ``θ`` may wrap. None and [] request no regions.
    /// max_cells also applies to balancing and final uniform refinement.
    /// The surface starts at its file coordinates with no rotation or translation.
    ///
    /// The result owns independent writable NumPy arrays. Cell/face indices are
    /// local to this build; changing an array does not alter the manager. This
    /// one-shot call does not retain a session. Rust build diagnostics are not
    /// returned or emitted as warnings by this Python method.
    ///
    /// With mesh_path=None, a valid seed is still required; all cells are fluid
    /// and ghost arrays are empty. With a surface, seed-connected non-intersecting
    /// cells and intersecting centers on the same side as the seed are fluid.
    /// Ghost cells use weighted linear least squares, with inverse-distance
    /// fallback. If no fluid candidate exists, a row references the ghost cell
    /// itself with weight 1; this is not a valid fluid interpolation. The Rust
    /// unresolved-stencil count is currently not exposed in Python.
    fn build_ghost_cell_mesh(
        &self,
        py: Python<'_>,
        mesh_path: Option<&str>,
        target_level: u8,
        fluid_seed_point: [f64; 3],
        refinement_regions: Option<Vec<RegionInput>>,
    ) -> PyResult<CfdGhostCellMesh> {
        let plan = arguments::plan(
            target_level,
            refinement_regions,
            self.config.is_cylindrical(),
        )?;
        let output = py
            .detach(|| {
                let boundary = engine::io::load_boundary(mesh_path.map(Path::new))?;
                engine::build_gcibm(&self.config, boundary, plan, fluid_seed_point)
            })
            .map_err(to_python)?;
        CfdGhostCellMesh::copied(py, &output.mesh)
    }
    #[pyo3(signature = (mesh_path, target_level, refinement_regions = None))]
    /// Builds a CFD mesh
    /// tailored for the Axis-Projected Immersed Boundary Method (APIBM).
    ///
    /// This method reads an STL / OBJ file, refines cells near the surface up to
    /// `target_level`, and stores the axis-projected hit paths on ``ap``.
    ///
    /// Parameters
    /// ----------
    /// mesh_path : str | None
    ///     Path to the input STL / OBJ geometry file.
    ///     Specifying None will generate a mesh without immersed boundary.
    /// target_level : int
    ///     Surface refinement target before final uniform leaf refinement.
    /// refinement_regions : list of tuple[Float3, Float3, int] | None
    ///     ``(min, max, level)`` intervals refined before balancing.
    ///     Cartesian intervals are world ``(x, y, z)``. Cylindrical intervals
    ///     are ``(r, θ, z)``; a full turn may wrap ``θ`` with ``max[1] < min[1]``.
    ///
    /// Returns
    /// -------
    /// CfdAxisProjectedMesh
    ///     The generated SoA mesh ready for CFD solvers.
    ///
    /// Notes
    /// -----
    /// target_level and region levels are targets before final uniform refinement,
    /// not upper bounds on final cell levels. Each must be in [0, 32], and the
    /// largest requested level plus n_leaf_refinement must not exceed 32.
    /// Regions need finite corners and positive overlap. On a cylinder, ``r`` and
    /// ``z`` stay ordered and ``θ`` may wrap. None and [] request no regions.
    /// max_cells also applies to balancing and final uniform refinement.
    /// The surface starts at its file coordinates with no rotation or translation.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If levels, regions, file format, file contents, or surface geometry are
    ///     invalid, the file cannot be read, or max_cells would be exceeded.
    /// RuntimeError
    ///     If a geometry query or internal mesh validation fails.
    /// TypeError, OverflowError
    ///     If an argument cannot be converted to the required native type.
    ///
    /// The result owns independent writable NumPy arrays. Cell/face indices are
    /// local to this build; changing an array does not alter the manager. This
    /// one-shot call does not retain a session. Rust build diagnostics are not
    /// returned or emitted as warnings by this Python method.
    ///
    /// Both owner and neighbour axis rays must hit for a face to be immersed.
    /// This method does not classify fluid/solid cells or apply boundary conditions.
    /// With mesh_path=None, all immersed-face flags are false and the compressed
    /// APIBM arrays are empty.
    fn build_axis_projected_mesh(
        &self,
        py: Python<'_>,
        mesh_path: Option<&str>,
        target_level: u8,
        refinement_regions: Option<Vec<RegionInput>>,
    ) -> PyResult<CfdAxisProjectedMesh> {
        let plan = arguments::plan(
            target_level,
            refinement_regions,
            self.config.is_cylindrical(),
        )?;
        let output = py
            .detach(|| {
                let boundary = engine::io::load_boundary(mesh_path.map(Path::new))?;
                engine::build_apibm(&self.config, boundary, plan)
            })
            .map_err(to_python)?;
        CfdAxisProjectedMesh::copied(py, &output.mesh)
    }
    #[pyo3(signature = (mesh_path, target_level, refinement_regions = None))]
    /// Creates a stateful APIBM session that retains the background forest
    /// for rigid moving-boundary updates via ``update_ib`` / ``remesh``.
    ///
    /// Parameters
    /// ----------
    /// mesh_path : str | None
    ///     Path to the input STL / OBJ geometry file.
    ///     Specifying None will generate a mesh without immersed boundary.
    /// target_level : int
    ///     Surface refinement target before final uniform leaf refinement.
    /// refinement_regions : list of tuple[Float3, Float3, int] | None
    ///     ``(min, max, level)`` intervals refined before balancing.
    ///     Cartesian intervals are world ``(x, y, z)``. Cylindrical intervals
    ///     are ``(r, θ, z)``; a full turn may wrap ``θ`` with ``max[1] < min[1]``.
    ///
    /// Returns
    /// -------
    /// ApibmSession
    ///     Session holding the forest, IBM geometry, and current CFD mesh.
    ///
    /// Notes
    /// -----
    /// target_level and region levels are targets before final uniform refinement,
    /// not upper bounds on final cell levels. Each must be in [0, 32], and the
    /// largest requested level plus n_leaf_refinement must not exceed 32.
    /// Regions need finite corners and positive overlap. On a cylinder, ``r`` and
    /// ``z`` stay ordered and ``θ`` may wrap. None and [] request no regions.
    /// max_cells also applies to balancing and final uniform refinement.
    /// The surface starts at its file coordinates with no rotation or translation.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If levels, regions, file format, file contents, or surface geometry are
    ///     invalid, the file cannot be read, or max_cells would be exceeded.
    /// RuntimeError
    ///     If a geometry query or internal mesh validation fails.
    /// TypeError, OverflowError
    ///     If an argument cannot be converted to the required native type.
    ///
    /// The session retains the background grid and surface for rigid motion.
    /// Use session.mesh for writable copies or session.snapshot() for read-only
    /// shared publication. The manager's max_cells applies to every remesh.
    fn create_axis_projected_session(
        &self,
        py: Python<'_>,
        mesh_path: Option<&str>,
        target_level: u8,
        refinement_regions: Option<Vec<RegionInput>>,
    ) -> PyResult<ApibmSession> {
        let plan = arguments::plan(
            target_level,
            refinement_regions,
            self.config.is_cylindrical(),
        )?;
        let inner = py
            .detach(|| {
                let boundary = engine::io::load_boundary(mesh_path.map(Path::new))?;
                engine::ApibmSession::new(self.config.clone(), boundary, plan)
            })
            .map_err(to_python)?;
        Ok(ApibmSession::new(inner))
    }
}
