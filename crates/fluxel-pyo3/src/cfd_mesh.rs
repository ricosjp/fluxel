//! Existing Python mesh classes, backed by a single common background converter.
use crate::conversion::{array1, array2, array3, view};
use fluxel_engine::{ApibmMesh, GcibmMesh};
use fluxel_geometry::{CylindricalGeometry, ParameterBox};
use fluxel_ibm::{ApIbmData, BoundaryRevision, PatchTable};
use fluxel_mesh::{BackgroundMesh, MeshId};
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::{prelude::*, types::PyDict};

/// Physical axis-aligned domain bounds, in the surface's length units.
///
/// Parameters
/// ----------
/// min, max : Float3
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
#[pyclass]
pub struct BoundingBox {
    pub min: [f64; 3],
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
    #[getter]
    /// Minimum corner ``(x, y, z)``, included by point location.
    fn min(&self) -> (f64, f64, f64) {
        (self.min[0], self.min[1], self.min[2])
    }
    #[getter]
    /// Maximum corner ``(x, y, z)``, excluded by point location.
    fn max(&self) -> (f64, f64, f64) {
        (self.max[0], self.max[1], self.max[2])
    }
}
/// Annular sector or full turn about an axis parallel to world Z.
///
/// Parameters
/// ----------
/// origin : Float3
///     World point where ``r = 0`` and ``z = 0``.
/// r_min, r_max : float
///     Inner and outer radii. ``r_min`` must be positive.
/// theta_start, theta_extent : float
///     Start angle and angular width in radians. One turn selects a periodic
///     full turn.
/// z_min, z_max : float
///     Axial limits in the same length units as the radii.
///
/// Raises
/// ------
/// ValueError
///     If a value is nonfinite, the radius includes the axis, the angle is
///     outside ``(0, 2π]``, or the axial interval is empty.
///
/// Notes
/// -----
/// Properties are read-only copies. ``r`` and ``z`` include the lower face and
/// exclude the upper face. ``θ`` is ``[theta_start, theta_start + extent)``.
/// ``r = 0`` is not supported.
#[pyclass]
pub struct Cylindrical {
    geometry: CylindricalGeometry,
}
impl Cylindrical {
    pub(crate) fn geometry(&self) -> CylindricalGeometry {
        self.geometry
    }
}
#[pymethods]
impl Cylindrical {
    #[new]
    /// Validate an annular sector or full turn; see Cylindrical.
    pub fn new(
        origin: [f64; 3],
        r_min: f64,
        r_max: f64,
        theta_start: f64,
        theta_extent: f64,
        z_min: f64,
        z_max: f64,
    ) -> PyResult<Self> {
        let geometry = CylindricalGeometry::from_extent(
            origin,
            r_min,
            r_max,
            theta_start,
            theta_extent,
            z_min,
            z_max,
        )
        .map_err(|error| crate::errors::to_python(error.into()))?;
        Ok(Self { geometry })
    }
    #[getter]
    /// World origin ``(x, y, z)`` of the axis.
    fn origin(&self) -> (f64, f64, f64) {
        let [x, y, z] = self.geometry.origin();
        (x, y, z)
    }
    #[getter]
    /// Inclusive inner radius.
    fn r_min(&self) -> f64 {
        self.geometry.r_min()
    }
    #[getter]
    /// Exclusive outer radius.
    fn r_max(&self) -> f64 {
        self.geometry.r_max()
    }
    #[getter]
    /// Inclusive start angle in radians.
    fn theta_start(&self) -> f64 {
        self.geometry.theta_start()
    }
    #[getter]
    /// Angular width in radians. A full turn is ``2π`` and is periodic.
    fn theta_extent(&self) -> f64 {
        self.geometry.theta_extent()
    }
    #[getter]
    /// Inclusive lower axial bound, relative to the origin.
    fn z_min(&self) -> f64 {
        self.geometry.z_min()
    }
    #[getter]
    /// Exclusive upper axial bound, relative to the origin.
    fn z_max(&self) -> f64 {
        self.geometry.z_max()
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
    pub cell_volumes: Py<PyArray1<f64>>,
    pub cell_centroids: Py<PyArray2<f64>>,
    pub cell_corners: Py<PyArray3<f64>>,
    pub internal_faces_owner: Py<PyArray1<usize>>,
    pub internal_faces_neighbour: Py<PyArray1<usize>>,
    pub internal_faces_axis: Py<PyArray1<u8>>,
    pub internal_faces_area: Py<PyArray2<f64>>,
    pub internal_faces_min: Py<PyArray2<f64>>,
    pub internal_faces_max: Py<PyArray2<f64>>,
    pub internal_faces_winding: Py<PyArray1<i8>>,
    pub domain_bnd_faces_owner: Py<PyArray1<usize>>,
    pub domain_bnd_faces_dir: Py<PyArray1<u8>>,
    pub domain_bnd_faces_area: Py<PyArray2<f64>>,
    pub domain_bnd_faces_min: Py<PyArray2<f64>>,
    pub domain_bnd_faces_max: Py<PyArray2<f64>>,
}
impl BackgroundArrays {
    /// Copy background arrays; readonly selects immutable byte-backed publication.
    fn new(py: Python<'_>, mesh: &BackgroundMesh, readonly: bool) -> PyResult<Self> {
        let cells = mesh.geometry();
        let faces = mesh.topology();
        let (internal_min, internal_max) = box_corners(faces.internal_bounds());
        let (boundary_min, boundary_max) = box_corners(faces.boundary_bounds());
        Ok(Self {
            cell_centers: array2(py, cells.centers(), readonly)?,
            cell_sizes: array2(py, cells.sizes(), readonly)?,
            cell_volumes: array1(py, cells.volumes(), readonly)?,
            cell_centroids: array2(py, cells.centroids(), readonly)?,
            cell_corners: array3(py, cells.corners(), readonly)?,
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
            internal_faces_area: array2(py, faces.internal_area(), readonly)?,
            internal_faces_min: array2(py, &internal_min, readonly)?,
            internal_faces_max: array2(py, &internal_max, readonly)?,
            internal_faces_winding: array1(py, faces.internal_winding(), readonly)?,
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
            domain_bnd_faces_area: array2(py, faces.boundary_area(), readonly)?,
            domain_bnd_faces_min: array2(py, &boundary_min, readonly)?,
            domain_bnd_faces_max: array2(py, &boundary_max, readonly)?,
        })
    }
}
fn box_corners(boxes: &[ParameterBox]) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    (
        boxes.iter().map(|bounds| bounds.min).collect(),
        boxes.iter().map(|bounds| bounds.max).collect(),
    )
}
/// Compressed axis-projected payload for internal faces.
///
/// ``is_immersed_face`` has one entry per internal face. Every other array
/// follows the ``True`` entries, in that order.
///
/// A distance is the path length from that side's sample to its first hit,
/// including zero when the boundary passes through the sample. Cartesian paths
/// follow the world axis. Cylindrical ``r`` and ``z`` follow ``e_r`` and
/// ``e_z``. Cylindrical ``θ`` is the arc length ``r|Δθ|``.
///
/// ``owner_near_boundary`` and ``neighbour_near_boundary`` mark a cell-center
/// Dirichlet candidate when ``d / width <= width / L``. ``width`` is that
/// side's cell width on the face axis and ``L`` is the largest domain extent.
/// The flags do not change the distances or apply a boundary condition.
///
/// Points, tangents, and normals have shape ``(N_immersed, 3)``. A tangent is
/// the unit search direction at the hit. A normal follows triangle winding and
/// is shared by both sides of that triangle.
#[pyclass(get_all)]
pub struct ApIbmFaceData {
    /// Per-internal-face bool mask; both axis rays must hit a boundary.
    pub is_immersed_face: Py<PyArray1<bool>>,
    /// Owner-side path length to the first hit: float64 (n_immersed,), possibly zero.
    pub dist_owner_to_bnd: Py<PyArray1<f64>>,
    /// Neighbour-side path length to the first hit: float64 (n_immersed,), possibly zero.
    pub dist_neighbour_to_bnd: Py<PyArray1<f64>>,
    /// Owner Dirichlet candidates: bool (n_immersed,). See the class notes.
    pub owner_near_boundary: Py<PyArray1<bool>>,
    /// Neighbour Dirichlet candidates: bool (n_immersed,). See the class notes.
    pub neighbour_near_boundary: Py<PyArray1<bool>>,
    /// Owner-side first-hit surface triangle IDs: uintp (n_immersed,).
    pub owner_bnd_anchor_id: Py<PyArray1<usize>>,
    /// Owner-side first-hit patch IDs: uintp (n_immersed,).
    pub owner_bnd_patch_id: Py<PyArray1<usize>>,
    /// Neighbour-side first-hit surface triangle IDs: uintp (n_immersed,).
    pub neighbour_bnd_anchor_id: Py<PyArray1<usize>>,
    /// Neighbour-side first-hit patch IDs: uintp (n_immersed,).
    pub neighbour_bnd_patch_id: Py<PyArray1<usize>>,
    /// Owner-side world intersections: float64 (n_immersed, 3).
    pub owner_bnd_point: Py<PyArray2<f64>>,
    /// Owner-side unit search direction at the hit: float64 (n_immersed, 3).
    pub owner_bnd_tangent: Py<PyArray2<f64>>,
    /// Owner-side unit triangle normal: float64 (n_immersed, 3).
    pub owner_bnd_normal: Py<PyArray2<f64>>,
    /// Neighbour-side world intersections: float64 (n_immersed, 3).
    pub neighbour_bnd_point: Py<PyArray2<f64>>,
    /// Neighbour-side unit search direction at the hit: float64 (n_immersed, 3).
    pub neighbour_bnd_tangent: Py<PyArray2<f64>>,
    /// Neighbour-side unit triangle normal: float64 (n_immersed, 3).
    pub neighbour_bnd_normal: Py<PyArray2<f64>>,
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
            owner_bnd_point: array2(py, data.owner_bnd_point(), readonly)?,
            owner_bnd_tangent: array2(py, data.owner_bnd_tangent(), readonly)?,
            owner_bnd_normal: array2(py, data.owner_bnd_normal(), readonly)?,
            neighbour_bnd_point: array2(py, data.neighbour_bnd_point(), readonly)?,
            neighbour_bnd_tangent: array2(py, data.neighbour_bnd_tangent(), readonly)?,
            neighbour_bnd_normal: array2(py, data.neighbour_bnd_normal(), readonly)?,
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
            owner_bnd_point: view(py, &self.owner_bnd_point)?,
            owner_bnd_tangent: view(py, &self.owner_bnd_tangent)?,
            owner_bnd_normal: view(py, &self.owner_bnd_normal)?,
            neighbour_bnd_point: view(py, &self.neighbour_bnd_point)?,
            neighbour_bnd_tangent: view(py, &self.neighbour_bnd_tangent)?,
            neighbour_bnd_normal: view(py, &self.neighbour_bnd_normal)?,
        })
    }
}
/// SoA mesh for the axis-projected immersed-boundary method.
/// Arrays use NumPy storage. Ordinary builds and ``session.mesh`` are writable.
///
/// ``coordinate_type`` is 0 for Cartesian and 1 for cylindrical.
/// Logical axes 0, 1, 2 are X, Y, Z or ``r``, ``θ``, ``z``.
///
/// Notes
/// -----
/// Index arrays refer to this background mesh. Indices stay stable for fixed-grid
/// session updates but must not be reused after remeshing. ``snapshot()`` and
/// updates with ``copy=False`` publish read-only arrays. Editing copies does not
/// alter Rust state. Surface patch IDs do not label domain boundary faces.
#[pyclass(get_all)]
pub struct CfdAxisProjectedMesh {
    /// Number of background cells, including non-fluid cells.
    pub n_cells: usize,
    /// Coordinate code: 0 is Cartesian and 1 is cylindrical.
    pub coordinate_type: u8,
    /// Surface patch names mapped to IDs; no boundary uses {"empty": 0}.
    pub patch_name_to_id: Py<PyDict>,
    /// World sample points: float64 (n_cells, 3).
    ///
    /// Cartesian samples are box centers. Cylindrical samples are parameter
    /// midpoints mapped to world and differ from the volume centroids.
    pub cell_centers: Py<PyArray2<f64>>,
    /// Coordinate-aligned widths: float64 (n_cells, 3).
    ///
    /// Cartesian widths are the edge lengths. Cylindrical widths are
    /// `[Δr, r Δθ, Δz]`. Their product is not the cell volume.
    pub cell_sizes: Py<PyArray2<f64>>,
    /// Exact cell volumes: float64 (n_cells,).
    pub cell_volumes: Py<PyArray1<f64>>,
    /// Volume centroids in world coordinates: float64 (n_cells, 3).
    pub cell_centroids: Py<PyArray2<f64>>,
    /// VTK corners in world coordinates: float64 (n_cells, 8, 3).
    ///
    /// Cylindrical edges are straight chords through the parameter corners.
    pub cell_corners: Py<PyArray3<f64>>,
    /// Owner cell indices: uintp (n_internal_faces,), local to this mesh.
    pub internal_faces_owner: Py<PyArray1<usize>>,
    /// Neighbour cell indices: uintp (n_internal_faces,), local to this mesh.
    pub internal_faces_neighbour: Py<PyArray1<usize>>,
    /// Logical face axes: uint8 (n_internal_faces,). Values 0, 1, 2 are
    /// Cartesian X, Y, Z or cylindrical r, θ, z.
    pub internal_faces_axis: Py<PyArray1<u8>>,
    /// Area vectors from owner toward neighbour: float64 (n_internal_faces, 3).
    pub internal_faces_area: Py<PyArray2<f64>>,
    /// Face minima: float64 (n_internal_faces, 3), world `(x, y, z)` or `(r, θ, z)`.
    pub internal_faces_min: Py<PyArray2<f64>>,
    /// Face maxima: float64 (n_internal_faces, 3), same components as the minima.
    pub internal_faces_max: Py<PyArray2<f64>>,
    /// Periodic wraps of each internal face: int8 (n_internal_faces,). A full-turn seam is 1.
    pub internal_faces_winding: Py<PyArray1<i8>>,
    /// Domain-boundary owner cell indices: uintp (n_domain_bnd_faces,).
    pub domain_bnd_faces_owner: Py<PyArray1<usize>>,
    /// Outward logical directions: uint8 (n_domain_bnd_faces,), codes 0..5.
    /// On a cylinder the pairs are `-r,+r`, `-θ,+θ`, `-z,+z`.
    pub domain_bnd_faces_dir: Py<PyArray1<u8>>,
    /// Outward area vectors of domain faces: float64 (n_domain_bnd_faces, 3).
    pub domain_bnd_faces_area: Py<PyArray2<f64>>,
    /// Domain-face minima: float64 (n_domain_bnd_faces, 3).
    /// Components match the internal-face minima.
    pub domain_bnd_faces_min: Py<PyArray2<f64>>,
    /// Domain-face maxima: float64 (n_domain_bnd_faces, 3).
    /// Components match the internal-face minima.
    pub domain_bnd_faces_max: Py<PyArray2<f64>>,
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
            cell_volumes: view(py, &base.cell_volumes)?,
            cell_centroids: view(py, &base.cell_centroids)?,
            cell_corners: view(py, &base.cell_corners)?,
            internal_faces_owner: view(py, &base.internal_faces_owner)?,
            internal_faces_neighbour: view(py, &base.internal_faces_neighbour)?,
            internal_faces_axis: view(py, &base.internal_faces_axis)?,
            internal_faces_area: view(py, &base.internal_faces_area)?,
            internal_faces_min: view(py, &base.internal_faces_min)?,
            internal_faces_max: view(py, &base.internal_faces_max)?,
            internal_faces_winding: view(py, &base.internal_faces_winding)?,
            domain_bnd_faces_owner: view(py, &base.domain_bnd_faces_owner)?,
            domain_bnd_faces_dir: view(py, &base.domain_bnd_faces_dir)?,
            domain_bnd_faces_area: view(py, &base.domain_bnd_faces_area)?,
            domain_bnd_faces_min: view(py, &base.domain_bnd_faces_min)?,
            domain_bnd_faces_max: view(py, &base.domain_bnd_faces_max)?,
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
/// SoA mesh for the ghost-cell immersed-boundary method.
/// Arrays use NumPy storage. One-shot builds are writable.
///
/// ``coordinate_type`` is 0 for Cartesian and 1 for cylindrical.
/// Logical axes 0, 1, 2 are X, Y, Z or ``r``, ``θ``, ``z``.
/// Ghost rows, except ``gc_is_fluid``, follow ``gc_cell_ids``.
///
/// Notes
/// -----
/// Unused stencil columns have zero weight and must be ignored. Linear
/// least-squares weights can be negative. Failed linear interpolation falls
/// back to inverse-distance weights. No candidates produce a self-reference
/// with weight 1, which is not a valid fluid stencil. Those counts stay in the
/// Rust report. Cell indices must not be reused across builds or remeshing.
#[pyclass(get_all)]
pub struct CfdGhostCellMesh {
    /// Number of background cells, including non-fluid cells.
    pub n_cells: usize,
    /// Coordinate code: 0 is Cartesian and 1 is cylindrical.
    pub coordinate_type: u8,
    /// Surface patch names mapped to IDs; no boundary uses {"empty": 0}.
    pub patch_name_to_id: Py<PyDict>,
    /// World sample points: float64 (n_cells, 3).
    ///
    /// Cartesian samples are box centers. Cylindrical samples are parameter
    /// midpoints mapped to world and differ from the volume centroids.
    pub cell_centers: Py<PyArray2<f64>>,
    /// Coordinate-aligned widths: float64 (n_cells, 3).
    ///
    /// Cartesian widths are the edge lengths. Cylindrical widths are
    /// `[Δr, r Δθ, Δz]`. Their product is not the cell volume.
    pub cell_sizes: Py<PyArray2<f64>>,
    /// Exact cell volumes: float64 (n_cells,).
    pub cell_volumes: Py<PyArray1<f64>>,
    /// Volume centroids in world coordinates: float64 (n_cells, 3).
    pub cell_centroids: Py<PyArray2<f64>>,
    /// VTK corners in world coordinates: float64 (n_cells, 8, 3).
    ///
    /// Cylindrical edges are straight chords through the parameter corners.
    pub cell_corners: Py<PyArray3<f64>>,
    /// Owner cell indices: uintp (n_internal_faces,), local to this mesh.
    pub internal_faces_owner: Py<PyArray1<usize>>,
    /// Neighbour cell indices: uintp (n_internal_faces,), local to this mesh.
    pub internal_faces_neighbour: Py<PyArray1<usize>>,
    /// Logical face axes: uint8 (n_internal_faces,). Values 0, 1, 2 are
    /// Cartesian X, Y, Z or cylindrical r, θ, z.
    pub internal_faces_axis: Py<PyArray1<u8>>,
    /// Area vectors from owner toward neighbour: float64 (n_internal_faces, 3).
    pub internal_faces_area: Py<PyArray2<f64>>,
    /// Face minima: float64 (n_internal_faces, 3), world `(x, y, z)` or `(r, θ, z)`.
    pub internal_faces_min: Py<PyArray2<f64>>,
    /// Face maxima: float64 (n_internal_faces, 3), same components as the minima.
    pub internal_faces_max: Py<PyArray2<f64>>,
    /// Periodic wraps of each internal face: int8 (n_internal_faces,). A full-turn seam is 1.
    pub internal_faces_winding: Py<PyArray1<i8>>,
    /// Domain-boundary owner cell indices: uintp (n_domain_bnd_faces,).
    pub domain_bnd_faces_owner: Py<PyArray1<usize>>,
    /// Outward logical directions: uint8 (n_domain_bnd_faces,), codes 0..5.
    /// On a cylinder the pairs are `-r,+r`, `-θ,+θ`, `-z,+z`.
    pub domain_bnd_faces_dir: Py<PyArray1<u8>>,
    /// Outward area vectors of domain faces: float64 (n_domain_bnd_faces, 3).
    pub domain_bnd_faces_area: Py<PyArray2<f64>>,
    /// Domain-face minima: float64 (n_domain_bnd_faces, 3).
    /// Components match the internal-face minima.
    pub domain_bnd_faces_min: Py<PyArray2<f64>>,
    /// Domain-face maxima: float64 (n_domain_bnd_faces, 3).
    /// Components match the internal-face minima.
    pub domain_bnd_faces_max: Py<PyArray2<f64>>,
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
            cell_volumes: base.cell_volumes,
            cell_centroids: base.cell_centroids,
            cell_corners: base.cell_corners,
            internal_faces_owner: base.internal_faces_owner,
            internal_faces_neighbour: base.internal_faces_neighbour,
            internal_faces_axis: base.internal_faces_axis,
            internal_faces_area: base.internal_faces_area,
            internal_faces_min: base.internal_faces_min,
            internal_faces_max: base.internal_faces_max,
            internal_faces_winding: base.internal_faces_winding,
            domain_bnd_faces_owner: base.domain_bnd_faces_owner,
            domain_bnd_faces_dir: base.domain_bnd_faces_dir,
            domain_bnd_faces_area: base.domain_bnd_faces_area,
            domain_bnd_faces_min: base.domain_bnd_faces_min,
            domain_bnd_faces_max: base.domain_bnd_faces_max,
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
