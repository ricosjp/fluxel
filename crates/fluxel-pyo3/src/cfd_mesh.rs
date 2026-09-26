//! Existing Python mesh classes, backed by a single common background converter.
use crate::conversion::{array1, array2, view};
use fluxel_engine::{ApibmMesh, GcibmMesh};
use fluxel_ibm::{ApIbmData, BoundaryRevision, PatchTable};
use fluxel_mesh::{BackgroundMesh, MeshId};
use numpy::{PyArray1, PyArray2};
use pyo3::{prelude::*, types::PyDict};

/// Physical axis-aligned domain bounds, in the surface's length units.
///
/// Parameters
/// ----------
/// min, max : list of float
///     Three finite coordinates with min < max on every axis and finite extents.
///
/// Raises
/// ------
/// ValueError
///     If bounds are nonfinite, reversed, or have zero/overflowing extents.
///
/// Notes
/// -----
/// min and max are read-only properties returning coordinate copies.
/// Point location includes the minimum faces and excludes the maximum faces.
#[pyclass(get_all)]
pub struct BoundingBox {
    /// Minimum physical corner [x, y, z], included by point location.
    pub min: [f64; 3],
    /// Maximum physical corner [x, y, z], excluded by point location.
    pub max: [f64; 3],
}
#[pymethods]
impl BoundingBox {
    #[new]
    /// Validate finite, strictly ordered 3D bounds; see BoundingBox.
    pub fn new(min: [f64; 3], max: [f64; 3]) -> PyResult<Self> {
        fluxel_geometry::BoundingBox::new(min, max)
            .map_err(|e| crate::errors::to_python(e.into()))?;
        Ok(Self { min, max })
    }
}
/// Create an independent Python patch-name dictionary, preserving the no-boundary empty=0 convention.
fn patches(py: Python<'_>, table: &PatchTable) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    if table.names().is_empty() {
        dict.set_item("empty", 0)?;
    }
    for (index, name) in table.names().iter().enumerate() {
        dict.set_item(name, index)?;
    }
    Ok(dict.unbind())
}
pub(crate) struct BackgroundArrays {
    pub cell_centers: Py<PyArray2<f64>>,
    pub cell_sizes: Py<PyArray2<f64>>,
    pub internal_faces_owner: Py<PyArray1<usize>>,
    pub internal_faces_neighbour: Py<PyArray1<usize>>,
    pub internal_faces_axis: Py<PyArray1<u8>>,
    pub domain_bnd_faces_owner: Py<PyArray1<usize>>,
    pub domain_bnd_faces_dir: Py<PyArray1<u8>>,
}
impl BackgroundArrays {
    /// Copy background arrays; readonly selects immutable byte-backed publication.
    fn new(py: Python<'_>, mesh: &BackgroundMesh, readonly: bool) -> PyResult<Self> {
        let cells = mesh.geometry();
        let faces = mesh.topology();
        Ok(Self {
            cell_centers: array2(py, cells.centers(), readonly)?,
            cell_sizes: array2(py, cells.sizes(), readonly)?,
            internal_faces_owner: array1(
                py,
                &faces
                    .internal_owner()
                    .iter()
                    .map(|i| i.index())
                    .collect::<Vec<_>>(),
                readonly,
            )?,
            internal_faces_neighbour: array1(
                py,
                &faces
                    .internal_neighbour()
                    .iter()
                    .map(|i| i.index())
                    .collect::<Vec<_>>(),
                readonly,
            )?,
            internal_faces_axis: array1(
                py,
                &faces
                    .internal_axis()
                    .iter()
                    .map(|a| *a as u8)
                    .collect::<Vec<_>>(),
                readonly,
            )?,
            domain_bnd_faces_owner: array1(
                py,
                &faces
                    .boundary_owner()
                    .iter()
                    .map(|i| i.index())
                    .collect::<Vec<_>>(),
                readonly,
            )?,
            domain_bnd_faces_dir: array1(
                py,
                &faces
                    .boundary_direction()
                    .iter()
                    .map(|d| *d as u8)
                    .collect::<Vec<_>>(),
                readonly,
            )?,
        })
    }
}
/// Compressed axis-projected immersed-boundary payload for internal faces.
///
/// `is_immersed_face` has length ``N_internal_faces``. All other arrays are
/// compressed to ``N_immersed = count(is_immersed_face)`` and store data only
/// for immersed faces, in the same order as ``True`` entries in
/// `is_immersed_face`.
///
/// Distances have shape ``(N_immersed,)`` and dtype ``float64``. They are
/// physical distances from each cell center to its first boundary hit along
/// the face axis, including zero for a boundary through the center.
/// ``owner_near_boundary`` and ``neighbour_near_boundary`` are boolean arrays
/// of the same shape. Each flags a candidate for a cell-center Dirichlet
/// constraint when ``d / delta_x <= delta_x / L``, where ``delta_x`` is that
/// side's cell width along the face axis and ``L`` is the largest extent of
/// the computational domain. These flags do not alter the distances or
/// apply boundary conditions; the solver must handle the constraints.
#[pyclass(get_all)]
pub struct ApIbmFaceData {
    /// Per-internal-face bool mask; both axis rays must hit a boundary.
    pub is_immersed_face: Py<PyArray1<bool>>,
    /// Owner-side physical first-hit distances: float64 (n_immersed,), possibly zero.
    pub dist_owner_to_bnd: Py<PyArray1<f64>>,
    /// Neighbour-side physical first-hit distances: float64 (n_immersed,), possibly zero.
    pub dist_neighbour_to_bnd: Py<PyArray1<f64>>,
    /// Owner constraint candidates: bool (n_immersed,), using d / dx <= dx / L.
    pub owner_near_boundary: Py<PyArray1<bool>>,
    /// Neighbour constraint candidates: bool (n_immersed,), using d / dx <= dx / L.
    pub neighbour_near_boundary: Py<PyArray1<bool>>,
    /// Owner-side first-hit surface triangle IDs: uintp (n_immersed,).
    pub owner_bnd_anchor_id: Py<PyArray1<usize>>,
    /// Owner-side first-hit patch IDs: uintp (n_immersed,).
    pub owner_bnd_patch_id: Py<PyArray1<usize>>,
    /// Neighbour-side first-hit surface triangle IDs: uintp (n_immersed,).
    pub neighbour_bnd_anchor_id: Py<PyArray1<usize>>,
    /// Neighbour-side first-hit patch IDs: uintp (n_immersed,).
    pub neighbour_bnd_patch_id: Py<PyArray1<usize>>,
}
impl ApIbmFaceData {
    /// Copy the compressed APIBM arrays under the requested ownership policy.
    fn new(py: Python<'_>, data: &ApIbmData, readonly: bool) -> PyResult<Self> {
        Ok(Self {
            is_immersed_face: array1(py, data.is_immersed_face(), readonly)?,
            dist_owner_to_bnd: array1(py, data.dist_owner_to_bnd(), readonly)?,
            dist_neighbour_to_bnd: array1(py, data.dist_neighbour_to_bnd(), readonly)?,
            owner_near_boundary: array1(py, data.owner_near_boundary(), readonly)?,
            neighbour_near_boundary: array1(py, data.neighbour_near_boundary(), readonly)?,
            owner_bnd_anchor_id: array1(py, data.owner_bnd_anchor_id(), readonly)?,
            owner_bnd_patch_id: array1(py, data.owner_bnd_patch_id(), readonly)?,
            neighbour_bnd_anchor_id: array1(py, data.neighbour_bnd_anchor_id(), readonly)?,
            neighbour_bnd_patch_id: array1(py, data.neighbour_bnd_patch_id(), readonly)?,
        })
    }
    /// Create fresh ndarray headers sharing payload storage, isolating metadata changes.
    fn view(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            is_immersed_face: view(py, &self.is_immersed_face)?,
            dist_owner_to_bnd: view(py, &self.dist_owner_to_bnd)?,
            dist_neighbour_to_bnd: view(py, &self.dist_neighbour_to_bnd)?,
            owner_near_boundary: view(py, &self.owner_near_boundary)?,
            neighbour_near_boundary: view(py, &self.neighbour_near_boundary)?,
            owner_bnd_anchor_id: view(py, &self.owner_bnd_anchor_id)?,
            owner_bnd_patch_id: view(py, &self.owner_bnd_patch_id)?,
            neighbour_bnd_anchor_id: view(py, &self.neighbour_bnd_anchor_id)?,
            neighbour_bnd_patch_id: view(py, &self.neighbour_bnd_patch_id)?,
        })
    }
}
/// Structure of Arrays (SoA) mesh representation for CFD solvers
/// using the Axis-Projected Immersed Boundary Method (APIBM/TFIBM).
/// Arrays use NumPy storage with explicit ownership.
/// Default snapshots are writable.
///
/// Attributes
/// ----------
/// n_cells: int
///     The number of cells in the mesh.
/// coordinate_type: int
///     The type of coordinate system used in the mesh.
///     0: Cartesian
/// cell_centers : numpy.ndarray
///     Array of shape (N_cells, 3) dtype=float64
///     the physical center coordinates of each background cell.
/// cell_sizes : numpy.ndarray
///     Array of shape (N_cells, 3) dtype=float64
///     the cell side lengths [dx, dy, dz] for each background cell.
/// patch_name_to_id: dict[str, int]
///     A dictionary mapping patch names to their corresponding IDs.
///
/// internal_faces_owner : numpy.ndarray
///     Array of shape (N_internal_faces,) dtype=uintp
///     the owner cell index for internal faces.
/// internal_faces_neighbour : numpy.ndarray
///     Array of shape (N_internal_faces,) dtype=uintp
///     the neighbour cell index for internal faces.
/// internal_faces_axis : numpy.ndarray
///     Array of shape (N_internal_faces,) dtype=uint8
///     the axis of the internal faces.
///     0: X, 1: Y, 2: Z
/// domain_bnd_faces_owner : numpy.ndarray
///     Array of shape (N_domain_bnd_faces,) dtype=uintp
///     the owner cell index for domain boundary faces.
/// domain_bnd_faces_dir : numpy.ndarray
///     Array of shape (N_domain_bnd_faces,) dtype=uint8
///     the direction of the domain boundary faces.
///     0: -X, 1: +X, 2: -Y, 3: +Y, 4: -Z, 5: +Z
/// ap : ApIbmFaceData
///     Axis-projected immersed-boundary face payload.
///
/// Notes
/// -----
/// Index arrays refer to this background mesh. Indices stay stable for fixed-grid
/// session updates but must not be reused after remeshing. Session.snapshot()
/// and updates with copy=False publish read-only arrays; ordinary builds and
/// session.mesh return independent writable copies. Editing copies does not
/// alter Rust state. Surface patch IDs do not label domain boundary faces.
#[pyclass(get_all)]
pub struct CfdAxisProjectedMesh {
    /// Number of background cells, including non-fluid cells.
    pub n_cells: usize,
    /// Coordinate code; generated meshes currently use 0 (Cartesian).
    pub coordinate_type: u8,
    /// Surface patch names mapped to IDs; no boundary uses {"empty": 0}.
    pub patch_name_to_id: Py<PyDict>,
    /// Physical centers: float64 array of shape (n_cells, 3).
    pub cell_centers: Py<PyArray2<f64>>,
    /// Physical side lengths [dx, dy, dz]: float64 (n_cells, 3).
    pub cell_sizes: Py<PyArray2<f64>>,
    /// Owner cell indices: uintp (n_internal_faces,), local to this mesh.
    pub internal_faces_owner: Py<PyArray1<usize>>,
    /// Neighbour cell indices: uintp (n_internal_faces,), local to this mesh.
    pub internal_faces_neighbour: Py<PyArray1<usize>>,
    /// Internal-face axes: uint8 (n_internal_faces,), X=0, Y=1, Z=2.
    pub internal_faces_axis: Py<PyArray1<u8>>,
    /// Domain-boundary owner cell indices: uintp (n_domain_bnd_faces,).
    pub domain_bnd_faces_owner: Py<PyArray1<usize>>,
    /// Outward directions: uint8 (n_domain_bnd_faces,), -X,+X,-Y,+Y,-Z,+Z = 0..5.
    pub domain_bnd_faces_dir: Py<PyArray1<u8>>,
    /// Compressed immersed-face payload; see ApIbmFaceData for its indexing contract.
    pub ap: Py<ApIbmFaceData>,
}
impl CfdAxisProjectedMesh {
    /// Assemble a mesh with fresh background views and patch dictionary; Python allocation may fail.
    fn assemble(
        py: Python<'_>,
        mesh: &ApibmMesh,
        base: &BackgroundArrays,
        ap: ApIbmFaceData,
    ) -> PyResult<Self> {
        Ok(Self {
            n_cells: mesh.background().n_cells(),
            coordinate_type: mesh.background().coordinate_type() as u8,
            patch_name_to_id: patches(py, mesh.patches())?,
            cell_centers: view(py, &base.cell_centers)?,
            cell_sizes: view(py, &base.cell_sizes)?,
            internal_faces_owner: view(py, &base.internal_faces_owner)?,
            internal_faces_neighbour: view(py, &base.internal_faces_neighbour)?,
            internal_faces_axis: view(py, &base.internal_faces_axis)?,
            domain_bnd_faces_owner: view(py, &base.domain_bnd_faces_owner)?,
            domain_bnd_faces_dir: view(py, &base.domain_bnd_faces_dir)?,
            ap: Py::new(py, ap)?,
        })
    }
    /// Publish fresh writable background and payload storage independent of Rust state.
    pub(crate) fn copied(py: Python<'_>, mesh: &ApibmMesh) -> PyResult<Self> {
        let base = BackgroundArrays::new(py, mesh.background(), false)?;
        Self::assemble(
            py,
            mesh,
            &base,
            ApIbmFaceData::new(py, mesh.payload(), false)?,
        )
    }
}
/// Only one background and one payload revision are retained by the adapter.
#[derive(Default)]
pub(crate) struct SnapshotCache {
    background: Option<(MeshId, BackgroundArrays)>,
    payload: Option<(BoundaryRevision, ApIbmFaceData)>,
}
impl SnapshotCache {
    /// Publish a read-only snapshot using one cached background and one payload revision.
    /// Initial publication copies into immutable bytes; cache hits reuse storage with
    /// fresh headers and dictionaries. Old Python snapshots own their buffers and can
    /// outlive cache replacement or session destruction. Conversion/allocation errors
    /// propagate; this method may update the cache but never commits session state.
    pub fn export(&mut self, py: Python<'_>, mesh: &ApibmMesh) -> PyResult<CfdAxisProjectedMesh> {
        if self.background.as_ref().map(|(id, _)| *id) != Some(mesh.background().id()) {
            self.background = Some((
                mesh.background().id(),
                BackgroundArrays::new(py, mesh.background(), true)?,
            ));
        }
        if self.payload.as_ref().map(|(id, _)| *id) != Some(mesh.boundary_revision()) {
            self.payload = Some((
                mesh.boundary_revision(),
                ApIbmFaceData::new(py, mesh.payload(), true)?,
            ));
        }
        let base = &self.background.as_ref().expect("cache just populated").1;
        let payload = self
            .payload
            .as_ref()
            .expect("cache just populated")
            .1
            .view(py)?;
        CfdAxisProjectedMesh::assemble(py, mesh, base, payload)
    }
}
/// Structure of Arrays (SoA) mesh representation for CFD solvers
/// using the Ghost-Cell Immersed Boundary Method (GCIBM).
/// Arrays use NumPy storage with explicit ownership.
/// Default snapshots are writable.
///
/// Attributes
/// ----------
/// n_cells : int
///     The number of cells in the mesh.
/// coordinate_type : int
///     The type of coordinate system used in the mesh.
///     0: Cartesian
/// cell_centers : numpy.ndarray
///     Array of shape (N_cells, 3) dtype=float64
///     the physical center coordinates of each background cell.
/// cell_sizes : numpy.ndarray
///     Array of shape (N_cells, 3) dtype=float64
///     the cell side lengths [dx, dy, dz] for each background cell.
/// patch_name_to_id : dict[str, int]
///     A dictionary mapping patch names to their corresponding IDs.
///
/// internal_faces_owner : numpy.ndarray
///     Array of shape (N_internal_faces,) dtype=uintp
///     the owner cell index for internal faces.
/// internal_faces_neighbour : numpy.ndarray
///     Array of shape (N_internal_faces,) dtype=uintp
///     the neighbour cell index for internal faces.
/// internal_faces_axis : numpy.ndarray
///     Array of shape (N_internal_faces,) dtype=uint8
///     the axis of the internal faces.
///     0: X, 1: Y, 2: Z
/// domain_bnd_faces_owner : numpy.ndarray
///     Array of shape (N_domain_bnd_faces,) dtype=uintp
///     the owner cell index for domain boundary faces.
/// domain_bnd_faces_dir : numpy.ndarray
///     Array of shape (N_domain_bnd_faces,) dtype=uint8
///     the direction of the domain boundary faces.
///     0: -X, 1: +X, 2: -Y, 3: +Y, 4: -Z, 5: +Z
///
/// [Ghost Cell IBM Specific Data]
/// gc_is_fluid : numpy.ndarray
///     Array of shape (N_cells,) dtype=bool
///     a boolean flag indicating whether each cell is a fluid cell.
///
/// gc_cell_ids : numpy.ndarray
///     Array of shape (N_ghosts,) dtype=uintp
///     the global indices of cells identified as ghost cells.
/// gc_bnd_anchor_ids : numpy.ndarray
///     Array of shape (N_ghosts,) dtype=uintp
///     the boundary triangle IDs associated with each ghost cell.
/// gc_bnd_patch_ids : numpy.ndarray
///     Array of shape (N_ghosts,) dtype=uintp
///     the boundary patch IDs associated with each ghost cell.
/// gc_bnd_intercepts : numpy.ndarray
///     Array of shape (N_ghosts, 3) dtype=float64
///     the nearest boundary points.
/// gc_image_points : numpy.ndarray
///     Array of shape (N_ghosts, 3) dtype=float64
///     the coordinates of the image points.
/// gc_interp_stencil_indices : numpy.ndarray
///     Array of shape (N_ghosts, 8) dtype=uintp
///     the fluid cell indices used for interpolating at the Image Points.
/// gc_interp_stencil_weights : numpy.ndarray
///     Array of shape (N_ghosts, 8) dtype=float64
///     the interpolation weights corresponding to `interp_stencil_indices`.
///
/// Notes
/// -----
/// All gc_* rows except gc_is_fluid follow gc_cell_ids order. Unused stencil
/// columns have zero weight and must be ignored. Linear least-squares weights
/// can be negative. Failed linear interpolation falls back to inverse-distance
/// weights; no candidates produce a self-reference with weight 1, not a valid
/// fluid stencil. Counts for these cases are available only in the Rust report.
/// Cell indices must not be reused across separate builds or remeshing.
#[pyclass(get_all)]
pub struct CfdGhostCellMesh {
    /// Number of background cells, including non-fluid cells.
    pub n_cells: usize,
    /// Coordinate code; generated meshes currently use 0 (Cartesian).
    pub coordinate_type: u8,
    /// Surface patch names mapped to IDs; no boundary uses {"empty": 0}.
    pub patch_name_to_id: Py<PyDict>,
    /// Physical centers: float64 array of shape (n_cells, 3).
    pub cell_centers: Py<PyArray2<f64>>,
    /// Physical side lengths [dx, dy, dz]: float64 (n_cells, 3).
    pub cell_sizes: Py<PyArray2<f64>>,
    /// Owner cell indices: uintp (n_internal_faces,), local to this mesh.
    pub internal_faces_owner: Py<PyArray1<usize>>,
    /// Neighbour cell indices: uintp (n_internal_faces,), local to this mesh.
    pub internal_faces_neighbour: Py<PyArray1<usize>>,
    /// Internal-face axes: uint8 (n_internal_faces,), X=0, Y=1, Z=2.
    pub internal_faces_axis: Py<PyArray1<u8>>,
    /// Domain-boundary owner cell indices: uintp (n_domain_bnd_faces,).
    pub domain_bnd_faces_owner: Py<PyArray1<usize>>,
    /// Outward directions: uint8 (n_domain_bnd_faces,), -X,+X,-Y,+Y,-Z,+Z = 0..5.
    pub domain_bnd_faces_dir: Py<PyArray1<u8>>,
    /// Fluid classification: bool (n_cells,), True means fluid.
    pub gc_is_fluid: Py<PyArray1<bool>>,
    /// Ghost-cell indices into the background: uintp (n_ghosts,).
    pub gc_cell_ids: Py<PyArray1<usize>>,
    /// Nearest surface triangle IDs: uintp (n_ghosts,).
    pub gc_bnd_anchor_ids: Py<PyArray1<usize>>,
    /// Patch IDs at the nearest boundary points: uintp (n_ghosts,).
    pub gc_bnd_patch_ids: Py<PyArray1<usize>>,
    /// Nearest boundary points in physical units: float64 (n_ghosts, 3).
    pub gc_bnd_intercepts: Py<PyArray2<f64>>,
    /// Mirrored physical points, 2 * intercept - center: float64 (n_ghosts, 3).
    pub gc_image_points: Py<PyArray2<f64>>,
    /// Background interpolation indices: uintp (n_ghosts, 8); ignore zero-weight slots.
    pub gc_interp_stencil_indices: Py<PyArray2<usize>>,
    /// Interpolation weights: float64 (n_ghosts, 8); see class notes for fallback rows.
    pub gc_interp_stencil_weights: Py<PyArray2<f64>>,
}
impl CfdGhostCellMesh {
    /// Copy background, classification and ghost rows into independent writable NumPy arrays.
    pub(crate) fn copied(py: Python<'_>, mesh: &GcibmMesh) -> PyResult<Self> {
        let base = BackgroundArrays::new(py, mesh.background(), false)?;
        let data = mesh.payload();
        Ok(Self {
            n_cells: mesh.background().n_cells(),
            coordinate_type: mesh.background().coordinate_type() as u8,
            patch_name_to_id: patches(py, mesh.patches())?,
            cell_centers: base.cell_centers,
            cell_sizes: base.cell_sizes,
            internal_faces_owner: base.internal_faces_owner,
            internal_faces_neighbour: base.internal_faces_neighbour,
            internal_faces_axis: base.internal_faces_axis,
            domain_bnd_faces_owner: base.domain_bnd_faces_owner,
            domain_bnd_faces_dir: base.domain_bnd_faces_dir,
            gc_is_fluid: array1(py, data.gc_is_fluid(), false)?,
            gc_cell_ids: array1(py, data.gc_cell_ids(), false)?,
            gc_bnd_anchor_ids: array1(py, data.gc_bnd_anchor_ids(), false)?,
            gc_bnd_patch_ids: array1(py, data.gc_bnd_patch_ids(), false)?,
            gc_bnd_intercepts: array2(py, data.gc_bnd_intercepts(), false)?,
            gc_image_points: array2(py, data.gc_image_points(), false)?,
            gc_interp_stencil_indices: array2(py, data.gc_interp_stencil_indices(), false)?,
            gc_interp_stencil_weights: array2(py, data.gc_interp_stencil_weights(), false)?,
        })
    }
}
