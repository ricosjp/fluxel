//! Intersection is independent of the GCIBM fluid/solid classification.
use crate::{BoundaryRevision, BoundaryState, IbmError};
use fluxel_core::Forest;
use fluxel_geometry::Geometry;
use fluxel_mesh::{GridContext, MeshId};
use rayon::prelude::*;
#[derive(Debug)]
/// Per-cell intersection flags tied to a particular grid and boundary revision.
pub struct IntersectionMask {
    mesh: MeshId,
    boundary: BoundaryRevision,
    values: Vec<bool>,
}
impl IntersectionMask {
    /// True for cells whose physical boxes intersect the boundary, in forest order.
    pub fn values(&self) -> &[bool] {
        &self.values
    }
    /// Reject a different grid or boundary revision with IbmError::StaleClassification.
    pub fn validate(&self, grid: &GridContext, boundary: &BoundaryState) -> Result<(), IbmError> {
        if self.mesh != grid.background().id() || self.boundary != boundary.revision() {
            return Err(IbmError::StaleClassification);
        }
        Ok(())
    }
}
/// Used during refinement, before final topology exists.
/// Return one surface-intersection flag per forest leaf, in current forest order.
/// `geometry` must describe the forest's root resolution. No final topology or
/// balance is required. Boundary::None yields all false. Query failures return
/// IbmError without mutating the inputs; this mask has no provenance metadata.
pub fn intersect_cells(
    forest: &Forest,
    geometry: &Geometry,
    boundary: &BoundaryState,
) -> Result<Vec<bool>, IbmError> {
    let Some(surface) = boundary.surface() else {
        return Ok(vec![false; forest.num_cells()]);
    };
    forest
        .keys()
        .par_iter()
        .map(|key| {
            let (center, size) = geometry.cell_bounds(&key.to_logical());
            surface.intersects(center, size, &boundary.pose())
        })
        .collect()
}
/// Compute intersection flags on a validated grid and stamp their provenance.
/// Returns boundary query failures from [`intersect_cells`]. A pose change requires
/// a new mask even when the background grid is unchanged.
pub fn classify_intersections(
    grid: &GridContext,
    boundary: &BoundaryState,
) -> Result<IntersectionMask, IbmError> {
    Ok(IntersectionMask {
        mesh: grid.background().id(),
        boundary: boundary.revision(),
        values: intersect_cells(grid.forest(), grid.geometry(), boundary)?,
    })
}
