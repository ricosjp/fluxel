use crate::{BoundaryRevision, BoundaryState, IbmError, IntersectionMask};
use fluxel_core::Direction;
use fluxel_mesh::{GridContext, MeshId};
use rayon::prelude::*;
use std::collections::VecDeque;

/// Final fluid/solid result; it cannot be confused with intersection status.
pub struct FluidClassification {
    mesh: MeshId,
    boundary: BoundaryRevision,
    values: Vec<bool>,
}
impl FluidClassification {
    /// Final per-cell fluid flags: true is fluid, false is non-fluid.
    pub fn values(&self) -> &[bool] {
        &self.values
    }
    /// Background identity for these flags.
    pub fn mesh_id(&self) -> MeshId {
        self.mesh
    }
    /// Boundary revision used for these flags.
    pub fn boundary_revision(&self) -> BoundaryRevision {
        self.boundary
    }
}
/// Flood non-intersecting cells from a fluid seed, then classify intersecting centers.
/// An intersecting center is fluid if its parity-based inside/outside result agrees
/// with the seed. Unreached non-intersecting cells remain non-fluid; no second flood
/// is performed after resolving intersecting cells. Boundary::None yields all true.
///
/// # Errors
/// The mask must match both grid and boundary. The finite seed must lie in the
/// half-open domain and its containing cell must not intersect the surface, even
/// if the point itself is off-surface. Violations return StaleClassification,
/// SeedOutside or SeedOnBoundary respectively. Inputs remain unchanged.
pub fn classify_fluid(
    grid: &GridContext,
    boundary: &BoundaryState,
    mask: &IntersectionMask,
    seed: [f64; 3],
) -> Result<FluidClassification, IbmError> {
    mask.validate(grid, boundary)?;
    let forest = grid.forest();
    let seed_id = grid
        .domain()
        .locate(forest, seed)
        .ok_or(IbmError::SeedOutside)?;
    if mask.values()[seed_id] {
        return Err(IbmError::SeedOnBoundary);
    }
    if boundary.surface().is_none() {
        return Ok(FluidClassification {
            mesh: grid.background().id(),
            boundary: boundary.revision(),
            values: vec![true; forest.num_cells()],
        });
    }
    let mut visited = mask.values().to_vec();
    let mut values = vec![false; forest.num_cells()];
    let mut queue = VecDeque::from([seed_id]);
    visited[seed_id] = true;
    values[seed_id] = true;
    while let Some(cell) = queue.pop_front() {
        for direction in Direction::ALL {
            for neighbour in forest.face_neighbour_global_ids(cell, direction) {
                if !visited[neighbour] {
                    visited[neighbour] = true;
                    values[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
    }
    // Preserve the existing sequence: flood non-intersecting cells, then resolve
    // intersecting centers by parity relative to the user-selected fluid seed.
    if let Some(surface) = boundary.surface() {
        let seed_inside = surface.contains(seed, &boundary.pose());
        values.par_iter_mut().enumerate().for_each(|(id, value)| {
            if mask.values()[id] {
                *value = surface
                    .contains(grid.background().geometry().centers()[id], &boundary.pose())
                    == seed_inside;
            }
        });
    }
    Ok(FluidClassification {
        mesh: grid.background().id(),
        boundary: boundary.revision(),
        values,
    })
}
