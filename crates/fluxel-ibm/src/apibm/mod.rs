//! Compressed APIBM payload, independent of topology construction and sessions.
mod payload;
mod ray;
use crate::{BoundaryState, IbmError, IntersectionMask};
use fluxel_mesh::GridContext;
pub use payload::ApIbmData;
use rayon::prelude::*;
/// Compute compressed first-hit data for internal faces of an immutable grid.
/// The mask must match this grid ID and boundary revision. Only faces adjacent to
/// an intersecting cell are tested; both independent axis rays must hit. Distances
/// use world units and can be zero. The output preserves topology face order and
/// performs no fluid/solid classification or boundary-condition enforcement.
///
/// # Errors
/// Returns `IbmError::StaleClassification` for a mismatched mask, or a query error
/// if a backend query fails. Boundary::None produces an all-false face mask and
/// empty compressed arrays. Inputs remain unchanged.
pub fn compute_apibm(
    grid: &GridContext,
    boundary: &BoundaryState,
    mask: &IntersectionMask,
) -> Result<ApIbmData, IbmError> {
    mask.validate(grid, boundary)?;
    let topology = grid.background().topology();
    let hits: Result<Vec<_>, IbmError> = (0..topology.n_internal_faces())
        .into_par_iter()
        .map(|i| {
            let owner = topology.internal_owner()[i].index();
            let neighbour = topology.internal_neighbour()[i].index();
            if !mask.values()[owner] && !mask.values()[neighbour] {
                return Ok(None);
            }
            let axis = topology.internal_axis()[i];
            let Some(owner_hit) = ray::side(grid, boundary, owner, neighbour, axis, 1.0)? else {
                return Ok(None);
            };
            let Some(neighbour_hit) = ray::side(grid, boundary, neighbour, owner, axis, -1.0)?
            else {
                return Ok(None);
            };
            Ok(Some((owner_hit, neighbour_hit)))
        })
        .collect();
    let mut result = ApIbmData::default();
    result.is_immersed_face.reserve(topology.n_internal_faces());
    for hit in hits? {
        result.is_immersed_face.push(hit.is_some());
        if let Some((owner, neighbour)) = hit {
            result.dist_owner_to_bnd.push(owner.distance);
            result.dist_neighbour_to_bnd.push(neighbour.distance);
            result.owner_near_boundary.push(owner.near);
            result.neighbour_near_boundary.push(neighbour.near);
            result.owner_bnd_anchor_id.push(owner.anchor);
            result.neighbour_bnd_anchor_id.push(neighbour.anchor);
            result.owner_bnd_patch_id.push(owner.patch);
            result.neighbour_bnd_patch_id.push(neighbour.patch);
        }
    }
    Ok(result)
}
