//! Manual Python AMR delegates to the same engine policies as automatic generation.
use crate::{arguments, cfd_mesh::BoundingBox, errors::to_python};
use fluxel_engine::{BuildLimits, ManualGrid};
use pyo3::prelude::*;
/// Low-level API for manual AMR control.
///
/// Initialize the Forest.
///
/// Parameters
/// ----------
/// bbox : BoundingBox
///     The physical bounds of the overall domain.
/// base_res : list of int
///     The initial number of root blocks (trees) in [X, Y, Z] directions.
///
/// Raises
/// ------
/// ValueError
///     If bounds are invalid or resolution is nonpositive or exceeds 2**26 roots.
///
/// Notes
/// -----
/// Creates populated level-0 cells, unlike the empty low-level Rust Forest.
/// This Python API has no max_cells argument and does not inherit a manager's
/// limit. Refinement changes cell ordering; flags refer to the current order.
#[pyclass]
pub struct Forest {
    inner: ManualGrid,
}
#[pymethods]
impl Forest {
    #[new]
    /// Initialize the Forest.
    ///
    /// Parameters
    /// ----------
    /// bbox : BoundingBox
    ///     The physical bounds of the overall domain.
    /// base_res : list of int
    ///     The initial number of root blocks (trees) in [X, Y, Z] directions.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If bounds are invalid or resolution is nonpositive or exceeds 2**26 roots.
    ///
    /// Notes
    /// -----
    /// Creates populated level-0 cells, unlike the empty low-level Rust Forest.
    /// This Python API has no max_cells argument and does not inherit a manager's
    /// limit. Refinement changes cell ordering; flags refer to the current order.
    fn new(bbox: &BoundingBox, base_res: [u32; 3]) -> PyResult<Self> {
        let bbox = fluxel_geometry::BoundingBox::new(bbox.min, bbox.max)
            .map_err(|e| to_python(e.into()))?;
        Ok(Self {
            inner: ManualGrid::new(bbox, base_res, BuildLimits::default()).map_err(to_python)?,
        })
    }
    /// Returns the current total number of cells.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of cells.
    fn num_cells(&self) -> usize {
        self.inner.num_cells()
    }
    /// Refines cells based on a boolean mask.
    ///
    /// Parameters
    /// ----------
    /// flags : list[bool]
    ///     A boolean list of length `num_cells()`. True indicates refinement.
    ///
    /// Notes
    /// -----
    /// Each flagged leaf is replaced by eight children; level-32 leaves are skipped.
    /// Does not automatically restore 2:1 balance. Mutates this forest in place.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the flag count differs from num_cells() or cell-count arithmetic
    ///     overflows. Validation failure leaves the forest unchanged.
    fn refine_by_flags(&mut self, py: Python<'_>, flags: Vec<bool>) -> PyResult<()> {
        py.detach(|| self.inner.refine_by_flags(&flags))
            .map_err(to_python)
    }
    /// Coarsens sibling cells back to their parent based on a boolean mask.
    ///
    /// Parameters
    /// ----------
    /// flags : list[bool]
    ///     A boolean list of length `num_cells()`.
    ///
    /// Notes
    /// -----
    /// Only complete groups of eight flagged siblings are replaced by their parent.
    /// Other flagged cells are left unchanged. Does not automatically restore
    /// 2:1 balance. Mutates this forest in place and changes cell indices.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the flag count differs from num_cells(); the forest is unchanged.
    fn coarsen_by_flags(&mut self, py: Python<'_>, flags: Vec<bool>) -> PyResult<()> {
        py.detach(|| self.inner.coarsen_by_flags(&flags))
            .map_err(to_python)
    }
    /// Refines all cells that intersect with the specified bounding box
    /// up to the given target level.
    ///
    /// Parameters
    /// ----------
    /// min : list[float]
    ///     The [x, y, z] coordinates of the minimum corner of the box.
    /// max : list[float]
    ///     The [x, y, z] coordinates of the maximum corner of the box.
    /// target_level : int
    ///     The desired refinement level inside the box.
    ///
    /// Notes
    /// -----
    /// Only positive-volume overlap counts; touching a region face is insufficient.
    /// Coordinates are in the domain's physical units. Does not coarsen finer cells
    /// or automatically restore balance. Mutates in place after successful work.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If corners are nonfinite or not strictly ordered, target_level is outside
    ///     [0, 32], or cell-count arithmetic overflows. Failure leaves the forest
    ///     unchanged.
    fn refine_by_bbox(
        &mut self,
        py: Python<'_>,
        min: [f64; 3],
        max: [f64; 3],
        target_level: u8,
    ) -> PyResult<()> {
        let region = arguments::regions(vec![(min, max, target_level)])?.remove(0);
        py.detach(|| self.inner.refine_region(region))
            .map_err(to_python)
    }
    /// Enforces the 2:1 balancing constraint across the entire mesh,
    /// ensuring adjacent cells differ by at most one refinement level.
    ///
    /// Notes
    /// -----
    /// Checks face, edge and corner neighbours (26 probes); only refines coarse
    /// cells. Mutates in place and may change cell indices.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If cell-count arithmetic overflows. Failure leaves the forest unchanged.
    fn enforce_2_to_1_balance(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.enforce_2_to_1_balance())
            .map_err(to_python)
    }
    /// Applies uniform refinement to every cell in the mesh, repeated `n_times`
    /// times.
    ///
    /// Parameters
    /// ----------
    /// n_times : int
    ///     The number of times the uniform refinement is applied.
    ///
    /// Notes
    /// -----
    /// Zero is a no-op; every additional level multiplies cell count by eight.
    /// Preserves an existing 2:1 balance but does not fix an unbalanced forest.
    /// Mutates in place and changes cell indices.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the resulting level exceeds 32 or cell-count arithmetic overflows.
    ///     Validation failure leaves the forest unchanged.
    fn uniform_refinement(&mut self, py: Python<'_>, n_times: u8) -> PyResult<()> {
        py.detach(|| self.inner.uniform_refinement(n_times))
            .map_err(to_python)
    }
}
