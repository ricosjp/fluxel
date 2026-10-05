//! Python publication and warnings around an atomic Rust session.
use crate::{
    arguments::{self, RegionInput},
    cfd_mesh::{CfdAxisProjectedMesh, SnapshotCache},
    errors::to_python,
};
use fluxel_engine::{self as engine, PreparedUpdate, RemeshRequest};
use pyo3::{exceptions::PyUserWarning, prelude::*};
/// Stateful APIBM session for rigid immersed-boundary motion.
///
/// Keep this object alive across timesteps. Use ``update_ib`` when the
/// background mesh can stay fixed, and ``remesh`` when AMR should follow
/// the boundary.
///
/// Attributes
/// ----------
/// mesh : CfdAxisProjectedMesh
///     Current CFD mesh snapshot (a new Python object on each access).
/// translation : Float3
///     Current rigid translation ``(tx, ty, tz)`` applied to the IB mesh.
/// rotation_quaternion : list of float
///     Current rigid rotation as a unit quaternion ``[w, x, y, z]``.
#[pyclass]
pub struct ApibmSession {
    inner: engine::ApibmSession,
    cache: SnapshotCache,
}
impl ApibmSession {
    /// Wrap an initialized Rust session with an empty publication cache.
    pub(crate) fn new(inner: engine::ApibmSession) -> Self {
        Self {
            inner,
            cache: SnapshotCache::default(),
        }
    }
    /// Warn, allocate Python arrays/object, then commit the prepared Rust candidate.
    /// Any pre-commit exception leaves session state unchanged; the conversion cache
    /// may have advanced but is keyed by snapshot provenance. Exclusive PyO3 borrowing
    /// rejects reentrant/concurrent session access during the operation.
    fn publish(
        &mut self,
        py: Python<'_>,
        update: PreparedUpdate,
        warn: bool,
        copy: bool,
    ) -> PyResult<Py<CfdAxisProjectedMesh>> {
        if warn && update.report().under_refined_cells > 0 {
            py.import("warnings")?.call_method1("warn", (
                "Immersed boundary intersects cells below target_level; consider remesh() to restore refinement near the boundary.",
                py.get_type::<PyUserWarning>(),
            ))?;
        }
        // PyO3's exclusive receiver borrow rejects reentrant or concurrent mutation.
        // All fallible publication precedes commit, including warnings-as-errors.
        let result = if copy {
            CfdAxisProjectedMesh::copied(py, update.mesh())?
        } else {
            self.cache.export(py, update.mesh())?
        };
        let result = Py::new(py, result)?;
        self.inner.commit(update).map_err(to_python)?;
        Ok(result)
    }
}
#[pymethods]
impl ApibmSession {
    #[getter]
    /// Return a fresh snapshot with independent writable arrays; changes do not affect the session.
    fn mesh(&self, py: Python<'_>) -> PyResult<CfdAxisProjectedMesh> {
        CfdAxisProjectedMesh::copied(py, self.inner.mesh())
    }
    /// Return a read-only snapshot, sharing cached immutable array storage.
    ///
    /// Old snapshots remain valid after updates and session destruction.
    /// Initial publication copies to immutable storage.
    /// Repeated reads reuse that storage.
    ///
    /// Returns
    /// -------
    /// CfdAxisProjectedMesh
    ///     Read-only arrays backed by immutable Python bytes, with fresh ndarray
    ///     headers and a fresh patch dictionary. This is not end-to-end zero-copy.
    ///     Array metadata changes do not change subsequent snapshots.
    fn snapshot(&mut self, py: Python<'_>) -> PyResult<CfdAxisProjectedMesh> {
        self.cache.export(py, self.inner.mesh())
    }
    #[getter]
    /// Current absolute translation (tx, ty, tz) in surface length units.
    fn translation(&self) -> (f64, f64, f64) {
        let [tx, ty, tz] = self.inner.pose().translation();
        (tx, ty, tz)
    }
    #[getter]
    /// Current normalized absolute rotation [w, x, y, z].
    fn rotation_quaternion(&self) -> [f64; 4] {
        self.inner.pose().quaternion()
    }
    #[pyo3(signature = (translation = None, rotation_quaternion = None, warn_outside_refinement = true, *, copy = true))]
    /// Recomputes IB face data only (topology fixed).
    ///
    /// Pose arguments are absolute. Quaternion order is ``[w, x, y, z]``.
    /// Omitted components keep the current pose values.
    /// Use ``quaternion_from_axis_angle`` to build a quaternion from an
    /// axis and angle.
    ///
    /// Parameters
    /// ----------
    /// translation : Float3 or None
    ///     Absolute translation ``(tx, ty, tz)``. ``None`` keeps the current
    ///     translation.
    /// rotation_quaternion : list of float or None
    ///     Absolute unit quaternion ``[w, x, y, z]``. ``None`` keeps the
    ///     current rotation. Build from an axis and angle with
    ///     ``quaternion_from_axis_angle``.
    /// warn_outside_refinement : bool, default True
    ///     If True, warn when the IB intersects cells below ``target_level``.
    ///     When warnings raise exceptions, the current session stays unchanged.
    /// copy : bool, default True
    ///     Return independent writable arrays. False returns read-only arrays
    ///     sharing cached background storage. Existing snapshots stay valid.
    ///
    /// Returns
    /// -------
    /// CfdAxisProjectedMesh
    ///     Updated mesh snapshot with rebuilt immersed-boundary payload.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a pose is nonfinite, or requested refinement is invalid or exceeds
    ///     max_cells (remesh only).
    /// RuntimeError
    ///     If geometry computation fails, or the session is already borrowed by
    ///     a concurrent or reentrant operation.
    /// UserWarning
    ///     Emitted when requested and the boundary crosses cells below target_level;
    ///     raised as an exception when warnings are configured as errors.
    /// TypeError, OverflowError
    ///     If an argument cannot be converted to the required native type.
    ///
    /// Notes
    /// -----
    /// Pose values are absolute, not increments. Rotation is about the original
    /// coordinate origin, followed by translation, in the surface's length units.
    /// Finite nonzero quaternions are normalized. For Python compatibility, a
    /// finite quaternion with squared norm <= float64 epsilon becomes identity.
    /// Omitted pose components retain their current values.
    /// All fallible computation, warnings, and Python publication complete before
    /// the state is committed; failure leaves the current mesh, pose, and settings
    /// unchanged. Previously returned snapshots remain valid.
    ///
    /// Cell and face indices stay fixed. copy=False reuses the cached background
    /// storage but publishes newly computed boundary data.
    fn update_ib(
        &mut self,
        py: Python<'_>,
        translation: Option<[f64; 3]>,
        rotation_quaternion: Option<[f64; 4]>,
        warn_outside_refinement: bool,
        copy: bool,
    ) -> PyResult<Py<CfdAxisProjectedMesh>> {
        let pose = arguments::pose(translation, rotation_quaternion);
        let update = py
            .detach(|| self.inner.prepare_update(pose))
            .map_err(to_python)?;
        self.publish(py, update, warn_outside_refinement, copy)
    }
    #[pyo3(signature = (target_level = None, refinement_regions = None, translation = None, rotation_quaternion = None, warn_outside_refinement = true, *, copy = true))]
    #[allow(clippy::too_many_arguments)]
    /// Rebuilds the AMR background mesh and IB payload for the current pose.
    ///
    /// Parameters
    /// ----------
    /// target_level : int or None
    ///     Surface refinement target before final uniform leaf refinement.
    ///     ``None`` keeps the current target level.
    /// refinement_regions : list of tuple[Float3, Float3, int] | None
    ///     ``(min, max, level)`` intervals. ``None`` keeps the current list.
    ///     Cartesian intervals are world ``(x, y, z)``. Cylindrical intervals
    ///     are ``(r, θ, z)``; a full turn may wrap ``θ`` with ``max[1] < min[1]``.
    /// translation : Float3 or None
    ///     Absolute translation ``(tx, ty, tz)``. ``None`` keeps the current
    ///     translation.
    /// rotation_quaternion : list of float or None
    ///     Absolute unit quaternion ``[w, x, y, z]``. ``None`` keeps the
    ///     current rotation. Build from an axis and angle with
    ///     ``quaternion_from_axis_angle``.
    /// warn_outside_refinement : bool, default True
    ///     If True, warn when the IB intersects cells below ``target_level``.
    ///     When warnings raise exceptions, the current session stays unchanged.
    /// copy : bool, default True
    ///     Return independent writable arrays. False returns read-only arrays
    ///     sharing cached background storage. Existing snapshots stay valid.
    ///
    /// Returns
    /// -------
    /// CfdAxisProjectedMesh
    ///     Newly built mesh snapshot after AMR and IB reconstruction.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a pose is nonfinite, or requested refinement is invalid or exceeds
    ///     max_cells (remesh only).
    /// RuntimeError
    ///     If geometry computation fails, or the session is already borrowed by
    ///     a concurrent or reentrant operation.
    /// UserWarning
    ///     Emitted when requested and the boundary crosses cells below target_level;
    ///     raised as an exception when warnings are configured as errors.
    /// TypeError, OverflowError
    ///     If an argument cannot be converted to the required native type.
    ///
    /// Notes
    /// -----
    /// Pose values are absolute, not increments. Rotation is about the original
    /// coordinate origin, followed by translation, in the surface's length units.
    /// Finite nonzero quaternions are normalized. For Python compatibility, a
    /// finite quaternion with squared norm <= float64 epsilon becomes identity.
    /// Omitted pose components retain their current values.
    /// All fallible computation, warnings, and Python publication complete before
    /// the state is committed; failure leaves the current mesh, pose, and settings
    /// unchanged. Previously returned snapshots remain valid.
    ///
    /// refinement_regions=None retains existing regions; [] clears them.
    /// The manager's max_cells and n_leaf_refinement remain in effect.
    /// Cell and face indices may change even if the cell count does not. Rebuild
    /// solver caches and transfer solution fields separately; this API does not
    /// transfer them. copy=False publishes the new background as read-only data.
    fn remesh(
        &mut self,
        py: Python<'_>,
        target_level: Option<u8>,
        refinement_regions: Option<Vec<RegionInput>>,
        translation: Option<[f64; 3]>,
        rotation_quaternion: Option<[f64; 4]>,
        warn_outside_refinement: bool,
        copy: bool,
    ) -> PyResult<Py<CfdAxisProjectedMesh>> {
        let pose = arguments::pose(translation, rotation_quaternion);
        let request = if self.inner.is_cylindrical() {
            RemeshRequest {
                target_level,
                parameter_regions: refinement_regions
                    .map(arguments::parameter_regions)
                    .transpose()?,
                pose,
                ..Default::default()
            }
        } else {
            RemeshRequest {
                target_level,
                regions: refinement_regions.map(arguments::regions).transpose()?,
                pose,
                ..Default::default()
            }
        };
        let update = py
            .detach(|| self.inner.prepare_remesh(request))
            .map_err(to_python)?;
        self.publish(py, update, warn_outside_refinement, copy)
    }
}
