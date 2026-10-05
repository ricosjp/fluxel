//! Cartesian cell intersection: the cell box against the posed surface.
//!
//! The cylindrical counterpart is [`super::cylindrical`].

use crate::{BoundaryState, IbmError};
use fluxel_core::Forest;
use fluxel_geometry::Geometry;
use rayon::prelude::*;

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
