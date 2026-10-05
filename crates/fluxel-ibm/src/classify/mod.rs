//! Intersection is independent of the GCIBM fluid/solid classification.
//!
//! [`cartesian`] tests cell boxes. [`cylindrical`] tests annular wedges.
//! [`classify_intersections`] chooses one from the grid's spatial domain.

mod cartesian;
mod cylindrical;

use crate::{BoundaryRevision, BoundaryState, IbmError};
use fluxel_core::Forest;
use fluxel_geometry::SpatialDomain;
use fluxel_mesh::{GridContext, MeshId};

pub use cartesian::intersect_cells;

/// Intersection flags before a grid context exists, using the same tests as classification.
pub fn intersect_domain(
    forest: &Forest,
    domain: &SpatialDomain,
    boundary: &BoundaryState,
) -> Result<Vec<bool>, IbmError> {
    match domain {
        SpatialDomain::Cartesian(geometry) => {
            cartesian::intersect_cells(forest, geometry, boundary)
        }
        SpatialDomain::Cylindrical(geometry) => {
            cylindrical::intersect_cells(forest, geometry, boundary)
        }
    }
}

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

/// Compute intersection flags on a validated grid and stamp their provenance.
/// Returns boundary query failures from [`intersect_cells`]. A pose change requires
/// a new mask even when the background grid is unchanged.
pub fn classify_intersections(
    grid: &GridContext,
    boundary: &BoundaryState,
) -> Result<IntersectionMask, IbmError> {
    let values = match grid.domain() {
        SpatialDomain::Cartesian(geometry) => {
            cartesian::intersect_cells(grid.forest(), geometry, boundary)?
        }
        SpatialDomain::Cylindrical(geometry) => {
            cylindrical::intersect_cells(grid.forest(), geometry, boundary)?
        }
    };
    Ok(IntersectionMask {
        mesh: grid.background().id(),
        boundary: boundary.revision(),
        values,
    })
}
