//! Cylindrical cell intersection: an axis-aligned candidate filter, then the exact wedge.
//!
//! The Cartesian counterpart is [`super::cartesian`].

use crate::{BoundaryState, IbmError};
use fluxel_core::Forest;
use fluxel_geometry::CylindricalGeometry;
use rayon::prelude::*;

pub(super) fn intersect_cells(
    forest: &Forest,
    geometry: &CylindricalGeometry,
    boundary: &BoundaryState,
) -> Result<Vec<bool>, IbmError> {
    let Some(surface) = boundary.surface() else {
        return Ok(vec![false; forest.num_cells()]);
    };
    let triangles = surface.world_triangles(&boundary.pose());
    forest
        .keys()
        .par_iter()
        .map(|key| {
            let interval = geometry
                .cell_interval(&key.to_logical(), forest.base_resolution())
                .map_err(|error| IbmError::Query(error.to_string()))?;
            // Reject cells whose world box misses the surface before the exact wedge test.
            let (center, size) = geometry.world_aabb(interval);
            if !surface.intersects(center, size, &boundary.pose())? {
                return Ok(false);
            }
            // A point or edge left by clipping still counts when its radius overlaps.
            Ok(triangles
                .iter()
                .any(|triangle| geometry.intersects_triangle(interval, *triangle)))
        })
        .collect()
}
