//! Manual AMR uses the same region policy as automatic builds.
use crate::{BuildError, BuildLimits, MeshBuildConfig, RefinementRegion};
use fluxel_core::Forest;
use fluxel_geometry::{BoundingBox, Geometry};
#[derive(Debug)]
/// Mutable AMR forest with physical geometry and a fixed cell limit.
/// Cell positions in Morton order change after refinement/coarsening. Operations
/// do not enforce balance unless explicitly requested; geometry remains fixed.
pub struct ManualGrid {
    forest: Forest,
    geometry: Geometry,
}
impl ManualGrid {
    /// Validate the domain and resolution and populate level-0 root cells.
    /// Errors match [`MeshBuildConfig::new`], including a root count above `limits`.
    pub fn new(
        bbox: BoundingBox,
        base_res: [u32; 3],
        limits: BuildLimits,
    ) -> Result<Self, BuildError> {
        MeshBuildConfig::new(bbox, base_res, 0, limits)?;
        let mut forest = Forest::with_cell_limit(base_res, limits.max_cells)?;
        forest.populate_root_cells();
        Ok(Self {
            forest,
            geometry: Geometry::new(bbox, base_res)?,
        })
    }
    /// Current leaf-cell count.
    pub fn num_cells(&self) -> usize {
        self.forest.num_cells()
    }
    /// Split flagged current-order leaves into eight, skipping leaves at MAX_LEVEL.
    /// Does not rebalance. Invalid flag length or cell-limit failure returns an error
    /// without modifying the forest.
    pub fn refine_by_flags(&mut self, flags: &[bool]) -> Result<(), BuildError> {
        Ok(self.forest.refine_by_flags(flags)?)
    }
    /// Merge complete flagged sibling groups in current cell order; ignore others.
    /// Does not rebalance. Invalid flag length errors without modifying the forest.
    pub fn coarsen_by_flags(&mut self, flags: &[bool]) -> Result<(), BuildError> {
        Ok(self.forest.coarsen_by_flags(flags)?)
    }
    /// Refine positive-volume region overlaps to its target without coarsening or balancing.
    /// Works on a clone and installs it only on success; cell-limit failures leave
    /// the current forest unchanged.
    pub fn refine_region(&mut self, region: RefinementRegion) -> Result<(), BuildError> {
        let mut candidate = self.forest.clone();
        crate::refinement::refine_regions(&mut candidate, &self.geometry, &[region])?;
        self.forest = candidate;
        Ok(())
    }
    /// Refine coarse 26-neighbours until their level difference is at most one.
    /// Unlike the low-level core operation, failure leaves this grid unchanged:
    /// balancing is performed on a candidate forest before replacement.
    pub fn enforce_2_to_1_balance(&mut self) -> Result<(), BuildError> {
        let mut candidate = self.forest.clone();
        candidate.enforce_2_to_1_balance()?;
        self.forest = candidate;
        Ok(())
    }
    /// Split every leaf `n` times; zero is a no-op. Existing balance is preserved.
    /// Prevalidates final levels and cell count; validation errors leave state unchanged.
    pub fn uniform_refinement(&mut self, n: u8) -> Result<(), BuildError> {
        Ok(self.forest.uniform_refinement(n)?)
    }
}
