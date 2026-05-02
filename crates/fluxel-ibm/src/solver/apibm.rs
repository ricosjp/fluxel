//! Axis-projected IBM (APIBM): ray–boundary intersection and 1D reconstruction weights per face.
//!
//! Each interior face between an owner and neighbour cell is resolved along one coordinate axis.
//! `resolve_owner_to_neighbour` and `resolve_neighbour_to_owner` cast opposite rays along that
//! axis so that two distinct boundary sheets between the cells (if any) each get a distance and
//! stencil. [`resolve_apibm_face`] merges both one-sided results into [`ApibmIntersection`].

use crate::mesh::IBMMesh;
use fluxel_core::{Axis, Direction, Forest};
use fluxel_geometry::Geometry;
use parry3d_f64::math::{Pose, Vector};
use parry3d_f64::query::{Ray, RayCast};

trait ToVector {
    fn to_vector(&self) -> Vector;
}

impl ToVector for Direction {
    fn to_vector(&self) -> Vector {
        match self {
            Self::XMinus => Vector::new(-1.0, 0.0, 0.0),
            Self::XPlus => Vector::new(1.0, 0.0, 0.0),
            Self::YMinus => Vector::new(0.0, -1.0, 0.0),
            Self::YPlus => Vector::new(0.0, 1.0, 0.0),
            Self::ZMinus => Vector::new(0.0, 0.0, -1.0),
            Self::ZPlus => Vector::new(0.0, 0.0, 1.0),
        }
    }
}

/// Full APIBM data for one owner–neighbour face: both ray hits and stencils on the shared axis.
pub struct ApibmIntersection {
    /// Distance along `axis` from the owner cell center to the boundary hit on the owner→neighbour ray.
    pub dist_owner_to_bnd: f64,
    /// Distance along `axis` from the neighbour cell center to the boundary hit on the neighbour→owner ray.
    pub dist_neighbour_to_bnd: f64,
    /// Owner-side weights `(w_b, w_o)` for boundary, owner center.
    pub owner_weights: [f64; 2],
    /// Neighbour-side weights `(w_b, w_n)` for boundary, neighbour center.
    pub neighbour_weights: [f64; 2],
    /// IBM surface triangle id for the hit resolved from the owner side.
    pub owner_bnd_anchor_id: usize,
    /// IBM surface patch id for the hit resolved from the owner side.
    pub owner_bnd_patch_id: usize,
    /// IBM surface triangle id for the hit resolved from the neighbour side.
    pub neighbour_bnd_anchor_id: usize,
    /// IBM surface patch id for the hit resolved from the neighbour side.
    pub neighbour_bnd_patch_id: usize,
}

/// Resolves APIBM on a face by combining one-sided ray casts from owner and neighbour.
///
/// Runs the owner→neighbour and neighbour→owner ray tests in this module and returns [`None`] if
/// either ray misses the immersed boundary within the segment between the two cell centers.
///
/// # Arguments
///
/// * `axis` — Coordinate axis for this face: `0` = X, `1` = Y, `2` = Z. Rays are cast along
///   increasing coordinate from the owner and decreasing from the neighbour; the pair should match
///   the forest connectivity used for this face (e.g. `x_plus` neighbours along +X).
pub fn resolve_apibm_face(
    forest: &Forest,
    geom: &Geometry,
    mesh: &IBMMesh,
    owner_global_id: usize,
    neighbour_global_id: usize,
    axis: Axis,
) -> Option<ApibmIntersection> {
    let owner_side = resolve_owner_to_neighbour(
        forest,
        geom,
        mesh,
        owner_global_id,
        neighbour_global_id,
        axis,
    )?;
    let neighbour_side = resolve_neighbour_to_owner(
        forest,
        geom,
        mesh,
        owner_global_id,
        neighbour_global_id,
        axis,
    )?;

    Some(ApibmIntersection {
        dist_owner_to_bnd: owner_side.dist_to_bnd,
        dist_neighbour_to_bnd: neighbour_side.dist_to_bnd,
        owner_weights: owner_side.weights,
        neighbour_weights: neighbour_side.weights,
        owner_bnd_anchor_id: owner_side.bnd_anchor_id,
        owner_bnd_patch_id: owner_side.bnd_patch_id,
        neighbour_bnd_anchor_id: neighbour_side.bnd_anchor_id,
        neighbour_bnd_patch_id: neighbour_side.bnd_patch_id,
    })
}

/// Computes APIBM reconstruction weights on one axis.
///
/// - `d`: The distance to the boundary.
/// - `delta_x`: The size of the cell in the direction of the axis.
fn calc_axis_weights(d: f64, delta_x: f64) -> (f64, [f64; 2]) {
    let theta = d / delta_x;
    if theta < delta_x {
        return (1.0, [1.0, 0.0]);
    }

    let w_b = 1.0 / theta;
    let w_i = (theta - 1.0) / theta;
    (theta, [w_b, w_i])
}

/// Casts a ray from the owner cell center toward the neighbour along `axis` and builds a one-sided APIBM stencil.
///
/// The ray direction is **+X / +Y / +Z** for `axis` 0 / 1 / 2 respectively, up to the distance between
/// cell centers. The first boundary intersection yields `dist_to_bnd`, triangle `bnd_anchor_id`, and
/// axis weights extrapolating toward the owner’s outward face (opposite to the ray).
///
/// # Arguments
///
/// * `axis` — `0` = X, `1` = Y, `2` = Z.
pub fn resolve_owner_to_neighbour(
    forest: &Forest,
    geom: &Geometry,
    mesh: &IBMMesh,
    owner_global_id: usize,
    neighbour_global_id: usize,
    axis: Axis,
) -> Option<ApibmOneSideIntersection> {
    let o_logical = forest.keys()[owner_global_id].to_logical();
    let n_logical = forest.keys()[neighbour_global_id].to_logical();

    let (c_o, s_o) = geom.cell_bounds(&o_logical);
    let (c_n, _) = geom.cell_bounds(&n_logical);

    let ax = axis.as_index();
    let (_, fwd_dir) = axis.split_into_directions();
    let dir_vec = fwd_dir.to_vector();

    let center_o = Vector::new(c_o[0], c_o[1], c_o[2]);
    let max_dist = (c_n[ax] - c_o[ax]).abs();

    let ray = Ray::new(center_o, dir_vec);
    let num_triangles = mesh.bvh.indices().len();

    if let Some(intersection) =
        mesh.bvh
            .cast_ray_and_get_normal(&Pose::identity(), &ray, max_dist, false)
    {
        let toi = intersection.time_of_impact;
        let bnd_anchor_id = intersection.feature.unwrap_face() as usize % num_triangles;
        let bnd_patch_id = mesh.anchor_to_patch_id[bnd_anchor_id];

        let d_bi = toi.abs();
        let delta_x = s_o[ax].abs();
        let (theta, weights) = calc_axis_weights(d_bi, delta_x);

        let dist_to_bnd = theta * delta_x;

        Some(ApibmOneSideIntersection {
            dist_to_bnd,
            weights,
            bnd_anchor_id,
            bnd_patch_id,
        })
    } else {
        None
    }
}

/// Casts a ray from the neighbour cell center toward the owner along `axis` and builds a one-sided APIBM stencil.
///
/// The ray direction is **−X / −Y / −Z** for `axis` 0 / 1 / 2. This is the counterpart of
/// [`resolve_owner_to_neighbour`] for the same face; weights extrapolate toward the neighbour’s
/// outward face (opposite to this ray).
///
/// # Arguments
///
/// * `axis` — `0` = X, `1` = Y, `2` = Z.
pub fn resolve_neighbour_to_owner(
    forest: &Forest,
    geom: &Geometry,
    mesh: &IBMMesh,
    owner_global_id: usize,
    neighbour_global_id: usize,
    axis: Axis,
) -> Option<ApibmOneSideIntersection> {
    let o_logical = forest.keys()[owner_global_id].to_logical();
    let n_logical = forest.keys()[neighbour_global_id].to_logical();

    let (c_o, _) = geom.cell_bounds(&o_logical);
    let (c_n, s_n) = geom.cell_bounds(&n_logical);

    let ax = axis.as_index();
    let (rev_dir, _) = axis.split_into_directions();
    let dir_vec = rev_dir.to_vector();

    let center_n = Vector::new(c_n[0], c_n[1], c_n[2]);
    let max_dist = (c_o[ax] - c_n[ax]).abs();

    let ray = Ray::new(center_n, dir_vec);
    let num_triangles = mesh.bvh.indices().len();

    if let Some(intersection) =
        mesh.bvh
            .cast_ray_and_get_normal(&Pose::identity(), &ray, max_dist, false)
    {
        let toi = intersection.time_of_impact;
        let bnd_anchor_id = intersection.feature.unwrap_face() as usize % num_triangles;
        let bnd_patch_id = mesh.anchor_to_patch_id[bnd_anchor_id];

        let d_bi = toi.abs();
        let delta_x = s_n[ax].abs();
        let (theta, weights) = calc_axis_weights(d_bi, delta_x);

        let dist_to_bnd = theta * delta_x;

        Some(ApibmOneSideIntersection {
            dist_to_bnd,
            weights,
            bnd_anchor_id,
            bnd_patch_id,
        })
    } else {
        None
    }
}

/// Result of a single ray cast from one cell toward its opposite neighbour on the chosen axis.
pub struct ApibmOneSideIntersection {
    /// Ray length from the cast origin (owner or neighbour center) to the first boundary hit.
    pub dist_to_bnd: f64,
    /// Weights `(w_b, w_center)` for boundary, local cell center.
    pub weights: [f64; 2],
    /// Hit IBM mesh triangle index (modulo triangle count).
    pub bnd_anchor_id: usize,
    /// IBM surface patch id for the hit.
    pub bnd_patch_id: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::IBMMesh;
    use fluxel_core::Forest;
    use fluxel_geometry::{BoundingBox, Geometry};

    fn setup_vertical_plane_between_cells() -> (Forest, Geometry, IBMMesh) {
        let mut forest = Forest::new([2, 1, 1]);
        forest.populate_root_cells();

        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let geom = Geometry::new(bbox, [2, 1, 1]);

        let verts = [[0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.0, 1.0]];
        let indices = [[0u32, 1, 2]];
        let mesh = IBMMesh::from_vertices_indices_and_patches(
            &verts,
            &indices,
            vec!["wall".into()],
            vec![0],
        );
        (forest, geom, mesh)
    }

    #[test]
    fn resolve_apibm_face_hits_plane_between_two_cells() {
        let (forest, geom, mesh) = setup_vertical_plane_between_cells();

        let merged = resolve_apibm_face(&forest, &geom, &mesh, 0, 1, Axis::X);
        assert!(merged.is_some());

        let o = resolve_owner_to_neighbour(&forest, &geom, &mesh, 0, 1, Axis::X);
        let n = resolve_neighbour_to_owner(&forest, &geom, &mesh, 0, 1, Axis::X);
        assert!(o.is_some());
        assert!(n.is_some());
    }
}
