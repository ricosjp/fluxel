//! Python `Forest` wrapper for manual AMR control.

use crate::cfd_mesh::BoundingBox;
use fluxel_core::Forest as CoreForest;
use fluxel_geometry::{BoundingBox as CoreBoundingBox, Geometry};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Use case 2: `Forest` wrapper for manual AMR control.
#[pyclass]
pub struct Forest {
    inner: CoreForest,
    geom: Geometry,
}

#[pymethods]
impl Forest {
    #[new]
    pub fn new(bbox: &BoundingBox, base_res: [u32; 3]) -> Self {
        let core_bbox = CoreBoundingBox::new(bbox.min, bbox.max);
        let geom = Geometry::new(core_bbox, base_res);
        let mut inner = CoreForest::new(base_res);
        inner.populate_root_cells();

        Self { inner, geom }
    }

    pub fn num_cells(&self) -> usize {
        self.inner.num_cells()
    }

    /// Batch-refines cells where the corresponding flag is `true`.
    pub fn refine_by_flags(&mut self, flags: Vec<bool>) -> PyResult<()> {
        if flags.len() != self.inner.num_cells() {
            return Err(PyValueError::new_err("Flags length must match num_cells"));
        }
        self.inner.refine_by_flags(&flags);
        Ok(())
    }

    /// Batch-coarsens sibling cells where the corresponding flag is `true`.
    pub fn coarsen_by_flags(&mut self, flags: Vec<bool>) -> PyResult<()> {
        if flags.len() != self.inner.num_cells() {
            return Err(PyValueError::new_err("Flags length must match num_cells"));
        }
        self.inner.coarsen_by_flags(&flags);
        Ok(())
    }

    /// Refines cells whose AABB overlaps `[min, max]` until `target_level` (refinementRegions-style).
    pub fn refine_by_bbox(
        &mut self,
        min: [f64; 3],
        max: [f64; 3],
        target_level: u8,
    ) -> PyResult<()> {
        for _ in 0..target_level {
            let mut flags = vec![false; self.inner.num_cells()];
            let mut should_refine = false;
            (0..self.inner.num_cells()).for_each(|i| {
                let key = self.inner.keys()[i];
                if key.level() < target_level {
                    let logical = key.to_logical();
                    let (center, size) = self.geom.cell_bounds(&logical);

                    let half_x = size[0] / 2.0;
                    let half_y = size[1] / 2.0;
                    let half_z = size[2] / 2.0;

                    let c_min = [center[0] - half_x, center[1] - half_y, center[2] - half_z];
                    let c_max = [center[0] + half_x, center[1] + half_y, center[2] + half_z];

                    let overlap_x = c_min[0] < max[0] && c_max[0] > min[0];
                    let overlap_y = c_min[1] < max[1] && c_max[1] > min[1];
                    let overlap_z = c_min[2] < max[2] && c_max[2] > min[2];

                    if overlap_x && overlap_y && overlap_z {
                        flags[i] = true;
                        should_refine = true;
                    }
                }
            });

            if !should_refine {
                break;
            }
            self.inner.refine_by_flags(&flags);
        }
        Ok(())
    }

    /// Enforces the 2:1 neighbour-balance constraint.
    pub fn enforce_2_to_1_balance(&mut self) {
        self.inner.enforce_2_to_1_balance();
    }

    /// Uniformly refines all cells by the specified number of times.
    pub fn uniform_refinement(&mut self, n_times: u8) {
        self.inner.uniform_refinement(n_times);
    }
}
