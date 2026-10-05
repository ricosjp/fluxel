//! Dispatch a face cast to the Cartesian or cylindrical implementation.

use super::{cartesian, cylindrical};
use crate::{BoundaryState, IbmError};
use fluxel_core::Axis;
use fluxel_geometry::SpatialDomain;
use fluxel_mesh::GridContext;

#[derive(Debug)]
pub(super) struct SideHit {
    pub distance: f64,
    pub near: bool,
    pub anchor: usize,
    pub patch: usize,
    /// World intersection. This is not reconstructed from the clamped path length.
    pub point: [f64; 3],
    /// Unit tangent of the search path at the hit, in world components.
    pub tangent: [f64; 3],
    /// Unit triangle normal from vertex winding, in world components.
    pub normal: [f64; 3],
}

pub(super) struct HitGeometry {
    pub point: [f64; 3],
    pub tangent: [f64; 3],
    pub normal: [f64; 3],
}

/// Cast from one cell sample toward the other along that face's basis.
///
/// Cartesian casts are straight world-axis rays. Cylindrical `r` and `z` casts
/// follow `e_r` and `e_z`. A cylindrical `θ` cast follows the arc at the caster's
/// own radius and height. Across a periodic seam the forward angle is measured
/// with `rem_euclid`.
pub(super) fn side(
    grid: &GridContext,
    boundary: &BoundaryState,
    local: usize,
    other: usize,
    axis: Axis,
    sign: f64,
    triangles: &[[[f64; 3]; 3]],
) -> Result<Option<SideHit>, IbmError> {
    if boundary.surface().is_none() {
        return Ok(None);
    }
    match grid.domain() {
        SpatialDomain::Cartesian(_) => {
            cartesian::side(grid, boundary, local, other, axis, sign, triangles)
        }
        SpatialDomain::Cylindrical(_) => {
            cylindrical::side(grid, boundary, local, other, axis, sign, triangles)
        }
    }
}

pub(super) fn finish(
    grid: &GridContext,
    local: usize,
    axis: Axis,
    distance: f64,
    anchor: usize,
    patch: usize,
    geometry: HitGeometry,
) -> Result<Option<SideHit>, IbmError> {
    let width = grid.background().geometry().sizes()[local][axis.as_index()].abs();
    let length = grid.domain().domain_length();
    Ok(Some(SideHit {
        distance,
        near: distance / width <= width / length,
        anchor,
        patch,
        point: geometry.point,
        tangent: geometry.tangent,
        normal: geometry.normal,
    }))
}

/// Unit normal `AB × AC` of one world triangle. A degenerate triangle returns zero.
pub(super) fn triangle_normal(triangle: [[f64; 3]; 3]) -> [f64; 3] {
    let ab = [
        triangle[1][0] - triangle[0][0],
        triangle[1][1] - triangle[0][1],
        triangle[1][2] - triangle[0][2],
    ];
    let ac = [
        triangle[2][0] - triangle[0][0],
        triangle[2][1] - triangle[0][1],
        triangle[2][2] - triangle[0][2],
    ];
    let normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let length = normal[0].hypot(normal[1]).hypot(normal[2]);
    if length == 0.0 {
        [0.0; 3]
    } else {
        [normal[0] / length, normal[1] / length, normal[2] / length]
    }
}
