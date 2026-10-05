//! Straight world-axis rays from a Cartesian cell sample.
//!
//! The cylindrical counterpart is [`super::cylindrical`].

use super::ray::{finish, triangle_normal, HitGeometry, SideHit};
use crate::{BoundaryState, IbmError};
use fluxel_core::Axis;
use fluxel_mesh::GridContext;

pub(super) fn side(
    grid: &GridContext,
    boundary: &BoundaryState,
    local: usize,
    other: usize,
    axis: Axis,
    sign: f64,
    triangles: &[[[f64; 3]; 3]],
) -> Result<Option<SideHit>, IbmError> {
    let Some(surface) = boundary.surface() else {
        return Ok(None);
    };
    let cells = grid.background().geometry();
    let center = cells.centers()[local];
    let target = cells.centers()[other];
    let ax = axis.as_index();
    let max_dist = (target[ax] - center[ax]).abs();
    // Start behind the sample so a surface through the center is still reported.
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
    // Path length is measured from the sample. The point keeps the unpadded hit.
    let distance = (hit.distance - padding).clamp(0.0, max_dist);
    let point = [
        origin[0] + direction[0] * hit.distance,
        origin[1] + direction[1] * hit.distance,
        origin[2] + direction[2] * hit.distance,
    ];
    let normal = triangles
        .get(hit.anchor)
        .copied()
        .map(triangle_normal)
        .ok_or_else(|| IbmError::Query("ray hit did not match a surface triangle".into()))?;
    finish(
        grid,
        local,
        axis,
        distance,
        hit.anchor,
        hit.patch,
        HitGeometry {
            point,
            tangent: direction,
            normal,
        },
    )
}
