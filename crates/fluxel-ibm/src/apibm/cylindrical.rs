//! Cylindrical face casts: radial and axial segments, and a constant-radius arc in angle.
//!
//! The Cartesian counterpart is [`super::cartesian`]. Angular hits use [`super::arc`].

use super::arc;
use super::ray::{finish, triangle_normal, HitGeometry, SideHit};
use crate::{BoundaryState, IbmError};
use fluxel_core::Axis;
use fluxel_geometry::CylindricalGeometry;
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
    let geometry = grid
        .domain()
        .cylindrical()
        .ok_or_else(|| IbmError::Query("cylindrical cast requires a cylindrical domain".into()))?;
    let Some(surface) = boundary.surface() else {
        return Ok(None);
    };
    let local_param = parameters(geometry, grid, local)?;
    let other_param = parameters(geometry, grid, other)?;
    let sample = geometry.world_from_param(local_param[0], local_param[1], local_param[2]);
    // θ travels on the caster's own circle. The short forward angle crosses the seam.
    if axis == Axis::Y {
        let (minus_theta, plus_theta) = if sign >= 0.0 {
            (local_param[1], other_param[1])
        } else {
            (other_param[1], local_param[1])
        };
        let travel = (plus_theta - minus_theta).rem_euclid(std::f64::consts::TAU);
        let center = [geometry.origin()[0], geometry.origin()[1], sample[2]];
        let Some(hit) = arc::first_hit(
            center,
            local_param[0],
            local_param[1],
            travel,
            sign,
            triangles,
        ) else {
            return Ok(None);
        };
        let distance = hit.distance.clamp(0.0, local_param[0] * travel);
        let patch = surface_patch(surface, hit.anchor)?;
        return finish(
            grid,
            local,
            axis,
            distance,
            hit.anchor,
            patch,
            HitGeometry {
                point: hit.point,
                tangent: hit.tangent,
                normal: hit.normal,
            },
        );
    }
    // r follows e_r and z follows e_z. The ray starts slightly behind the sample.
    let max_dist = if axis == Axis::X {
        (other_param[0] - local_param[0]).abs()
    } else {
        (other_param[2] - local_param[2]).abs()
    };
    let frame = geometry.local_frame(local_param[1]);
    let basis = if axis == Axis::X { frame[0] } else { frame[2] };
    let direction = [basis[0] * sign, basis[1] * sign, basis[2] * sign];
    let padding = 64.0
        * f64::EPSILON
        * sample
            .iter()
            .map(|value| value.abs())
            .fold(max_dist, f64::max);
    let origin = [
        sample[0] - direction[0] * padding,
        sample[1] - direction[1] * padding,
        sample[2] - direction[2] * padding,
    ];
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

fn parameters(
    geometry: &CylindricalGeometry,
    grid: &GridContext,
    cell: usize,
) -> Result<[f64; 3], IbmError> {
    let interval = geometry
        .cell_interval(
            &grid.forest().keys()[cell].to_logical(),
            grid.forest().base_resolution(),
        )
        .map_err(|error| IbmError::Query(error.to_string()))?;
    Ok([
        0.5 * (interval.min[0] + interval.max[0]),
        0.5 * (interval.min[1] + interval.max[1]),
        0.5 * (interval.min[2] + interval.max[2]),
    ])
}

fn surface_patch(surface: &crate::BoundarySurface, anchor: usize) -> Result<usize, IbmError> {
    surface
        .patch_of(anchor)
        .ok_or_else(|| IbmError::Query("arc hit did not match a surface triangle".into()))
}
