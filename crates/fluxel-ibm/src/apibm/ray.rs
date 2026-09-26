use crate::{BoundaryState, IbmError};
use fluxel_core::Axis;
use fluxel_mesh::GridContext;
#[derive(Debug)]
pub(super) struct SideHit {
    pub distance: f64,
    pub near: bool,
    pub anchor: usize,
    pub patch: usize,
}
/// Cast from one cell center along a face axis toward the other cell's axis coordinate.
/// The caller supplies a positive-axis owner or negative-axis neighbour orientation;
/// the transverse center coordinates need not coincide at coarse/fine interfaces.
/// Pads the ray for roundoff, then clamps the corrected first-hit distance to the
/// center separation. A hit at the center is valid. Near-boundary status uses this
/// side's width and the largest domain extent. Returns None without a surface or
/// hit; propagates backend query errors.
pub(super) fn side(
    grid: &GridContext,
    boundary: &BoundaryState,
    local: usize,
    other: usize,
    axis: Axis,
    sign: f64,
) -> Result<Option<SideHit>, IbmError> {
    let Some(surface) = boundary.surface() else {
        return Ok(None);
    };
    let cells = grid.background().geometry();
    let center = cells.centers()[local];
    let target = cells.centers()[other];
    let ax = axis.as_index();
    let max_dist = (target[ax] - center[ax]).abs();
    let padding = 64.0 * f64::EPSILON * center.iter().map(|x| x.abs()).fold(max_dist, f64::max);
    let mut direction = [0.0; 3];
    direction[ax] = sign;
    let mut origin = center;
    origin[ax] -= sign * padding;
    let Some(hit) = surface.ray(
        origin,
        direction,
        max_dist + 2.0 * padding,
        &boundary.pose(),
    )?
    else {
        return Ok(None);
    };
    let distance = (hit.distance - padding).clamp(0.0, max_dist);
    let width = cells.sizes()[local][ax].abs();
    let bbox = grid.geometry().bounding_box();
    let length = (0..3)
        .map(|a| bbox.max()[a] - bbox.min()[a])
        .fold(0.0, f64::max);
    Ok(Some(SideHit {
        distance,
        near: distance / width <= width / length,
        anchor: hit.anchor,
        patch: hit.patch,
    }))
}
