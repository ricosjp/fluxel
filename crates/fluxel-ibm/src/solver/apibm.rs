//! Axis-projected IBM (APIBM): ray–boundary distances and boundary-cell candidates per face.
//!
//! Each interior face between an owner and neighbour cell is resolved along one coordinate axis.
//! `resolve_owner_to_neighbour` and `resolve_neighbour_to_owner` cast opposite rays along that
//! axis so that two distinct boundary sheets between the cells (if any) each get a distance and
//! boundary-cell flag. [`resolve_apibm_face`] merges both one-sided results into [`ApibmIntersection`].

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

/// Full APIBM data for one owner–neighbour face: both ray hits and boundary-cell flags on the shared axis.
pub struct ApibmIntersection {
    /// Distance along `axis` from the owner cell center to the boundary hit on the owner→neighbour ray.
    pub dist_owner_to_bnd: f64,
    /// Distance along `axis` from the neighbour cell center to the boundary hit on the neighbour→owner ray.
    pub dist_neighbour_to_bnd: f64,
    /// Owner-center Dirichlet-constraint candidate: `d / delta_x <= delta_x / L`,
    /// using the owner cell width and largest domain extent `L`; distance is unchanged.
    pub owner_near_boundary: bool,
    /// Neighbour-center Dirichlet-constraint candidate: `d / delta_x <= delta_x / L`,
    /// using the neighbour cell width and largest domain extent `L`; distance is unchanged.
    pub neighbour_near_boundary: bool,
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
/// Segment endpoints are included with roundoff padding, so a boundary through a
/// cell center can produce a zero distance and a near-boundary flag.
///
/// # Arguments
///
/// * `axis` — Coordinate axis for this face: `0` = X, `1` = Y, `2` = Z. Rays are cast along
///   increasing coordinate from the owner and decreasing from the neighbour; the pair should match
///   the forest connectivity used for this face (e.g. `x_plus` neighbours along +X).
/// * `pose` — Rigid transform applied to the immersed-boundary mesh.
pub fn resolve_apibm_face(
    forest: &Forest,
    geom: &Geometry,
    mesh: &IBMMesh,
    owner_global_id: usize,
    neighbour_global_id: usize,
    axis: Axis,
    pose: &Pose,
) -> Option<ApibmIntersection> {
    let owner_side = resolve_owner_to_neighbour(
        forest,
        geom,
        mesh,
        owner_global_id,
        neighbour_global_id,
        axis,
        pose,
    )?;
    let neighbour_side = resolve_neighbour_to_owner(
        forest,
        geom,
        mesh,
        owner_global_id,
        neighbour_global_id,
        axis,
        pose,
    )?;

    Some(ApibmIntersection {
        dist_owner_to_bnd: owner_side.dist_to_bnd,
        dist_neighbour_to_bnd: neighbour_side.dist_to_bnd,
        owner_near_boundary: owner_side.near_boundary,
        neighbour_near_boundary: neighbour_side.near_boundary,
        owner_bnd_anchor_id: owner_side.bnd_anchor_id,
        owner_bnd_patch_id: owner_side.bnd_patch_id,
        neighbour_bnd_anchor_id: neighbour_side.bnd_anchor_id,
        neighbour_bnd_patch_id: neighbour_side.bnd_patch_id,
    })
}

/// Gibou et al. (2002), p. 8: theta <= h in normalized coordinates.
/// The reference length is the largest extent of the computational domain.
/// This marks a candidate for a Dirichlet constraint on the cell UNKNOWN;
/// it does not replace the physical distance or define a ghost value.
fn is_near_boundary(d: f64, delta_x: f64, geom: &Geometry) -> bool {
    let length = (0..3)
        .map(|a| geom.global_bbox.max[a] - geom.global_bbox.min[a])
        .fold(0.0_f64, f64::max);
    d / delta_x <= delta_x / length
}

/// Casts a ray from the owner cell center toward the neighbour along `axis` and builds a one-sided APIBM boundary-cell flag.
///
/// The ray direction is **+X / +Y / +Z** for `axis` 0 / 1 / 2 respectively, up to the distance between
/// cell centers. The first boundary intersection yields `dist_to_bnd`, triangle `bnd_anchor_id`, and
/// whether the owner center is a candidate for a Dirichlet constraint.
///
/// # Arguments
///
/// * `axis` — `0` = X, `1` = Y, `2` = Z.
/// * `pose` — Rigid transform applied to the immersed-boundary mesh.
pub fn resolve_owner_to_neighbour(
    forest: &Forest,
    geom: &Geometry,
    mesh: &IBMMesh,
    owner_global_id: usize,
    neighbour_global_id: usize,
    axis: Axis,
    pose: &Pose,
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

    // Include segment endpoints: a surface through a center must still
    // produce a boundary-cell constraint. Padding only covers roundoff.
    let padding = 64.0 * f64::EPSILON * c_o.iter().map(|x| x.abs()).fold(max_dist, f64::max);
    let ray = Ray::new(center_o - dir_vec * padding, dir_vec);
    let num_triangles = mesh.bvh.indices().len();

    if let Some(intersection) =
        mesh.bvh
            .cast_ray_and_get_normal(pose, &ray, max_dist + 2.0 * padding, false)
    {
        let toi = intersection.time_of_impact;
        let bnd_anchor_id = intersection.feature.unwrap_face() as usize % num_triangles;
        let bnd_patch_id = mesh.anchor_to_patch_id[bnd_anchor_id];

        let d_bi = (toi - padding).clamp(0.0, max_dist);
        let delta_x = s_o[ax].abs();
        let near_boundary = is_near_boundary(d_bi, delta_x, geom);
        let dist_to_bnd = d_bi;

        Some(ApibmOneSideIntersection {
            dist_to_bnd,
            near_boundary,
            bnd_anchor_id,
            bnd_patch_id,
        })
    } else {
        None
    }
}

/// Casts a ray from the neighbour cell center toward the owner along `axis` and builds a one-sided APIBM boundary-cell flag.
///
/// The ray direction is **−X / −Y / −Z** for `axis` 0 / 1 / 2. This is the counterpart of
/// [`resolve_owner_to_neighbour`] for the same face; the near-boundary flag refers to the neighbour center.
///
/// # Arguments
///
/// * `axis` — `0` = X, `1` = Y, `2` = Z.
/// * `pose` — Rigid transform applied to the immersed-boundary mesh.
pub fn resolve_neighbour_to_owner(
    forest: &Forest,
    geom: &Geometry,
    mesh: &IBMMesh,
    owner_global_id: usize,
    neighbour_global_id: usize,
    axis: Axis,
    pose: &Pose,
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

    // Include segment endpoints: a surface through a center must still
    // produce a boundary-cell constraint. Padding only covers roundoff.
    let padding = 64.0 * f64::EPSILON * c_n.iter().map(|x| x.abs()).fold(max_dist, f64::max);
    let ray = Ray::new(center_n - dir_vec * padding, dir_vec);
    let num_triangles = mesh.bvh.indices().len();

    if let Some(intersection) =
        mesh.bvh
            .cast_ray_and_get_normal(pose, &ray, max_dist + 2.0 * padding, false)
    {
        let toi = intersection.time_of_impact;
        let bnd_anchor_id = intersection.feature.unwrap_face() as usize % num_triangles;
        let bnd_patch_id = mesh.anchor_to_patch_id[bnd_anchor_id];

        let d_bi = (toi - padding).clamp(0.0, max_dist);
        let delta_x = s_n[ax].abs();
        let near_boundary = is_near_boundary(d_bi, delta_x, geom);
        let dist_to_bnd = d_bi;

        Some(ApibmOneSideIntersection {
            dist_to_bnd,
            near_boundary,
            bnd_anchor_id,
            bnd_patch_id,
        })
    } else {
        None
    }
}

/// Result of a single ray cast from one cell toward its opposite neighbour on the chosen axis.
pub struct ApibmOneSideIntersection {
    /// Physical distance from the owner or neighbour center to the first boundary hit.
    /// Ray-origin padding is removed and the result is clamped to the center-to-center
    /// segment, so a boundary through the local center has distance zero.
    pub dist_to_bnd: f64,
    /// Candidate for a Dirichlet constraint at the local cell center.
    pub near_boundary: bool,
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

    #[test]
    fn boundary_snapping_is_invariant_under_length_units() {
        for scale in [0.001, 1.0, 1000.0] {
            let geom = Geometry::new(BoundingBox::new([0.0; 3], [scale, scale, scale]), [4, 4, 4]);
            let dx = 0.25 * scale;
            assert!(is_near_boundary(0.0, dx, &geom));
            assert!(is_near_boundary(0.0625 * scale, dx, &geom));
            assert!(!is_near_boundary(0.063 * scale, dx, &geom));
        }
    }

    #[test]
    fn near_boundary_keeps_the_actual_ray_distance() {
        let (forest, geom, mesh) = setup_vertical_plane_between_cells();
        let pose = Pose::translation(-0.249, 0.0, 0.0);
        let hit = resolve_owner_to_neighbour(&forest, &geom, &mesh, 0, 1, Axis::X, &pose).unwrap();
        assert!(hit.near_boundary);
        assert!((hit.dist_to_bnd - 0.001).abs() < 1e-14);
    }

    #[test]
    fn surface_at_cell_center_is_not_lost_by_raycast() {
        let (forest, geom, mesh) = setup_vertical_plane_between_cells();
        let pose = Pose::translation(-0.25, 0.0, 0.0);
        let hit = resolve_apibm_face(&forest, &geom, &mesh, 0, 1, Axis::X, &pose).unwrap();
        assert!(hit.owner_near_boundary);
        assert!(hit.dist_owner_to_bnd.abs() < 1e-14);
        assert!((hit.dist_neighbour_to_bnd - 0.5).abs() < 1e-14);
    }

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
        let identity = Pose::identity();

        let merged = resolve_apibm_face(&forest, &geom, &mesh, 0, 1, Axis::X, &identity);
        assert!(merged.is_some());

        let o = resolve_owner_to_neighbour(&forest, &geom, &mesh, 0, 1, Axis::X, &identity);
        let n = resolve_neighbour_to_owner(&forest, &geom, &mesh, 0, 1, Axis::X, &identity);
        assert!(o.is_some());
        assert!(n.is_some());
    }

    #[test]
    fn resolve_apibm_face_respects_translation_pose() {
        let (forest, geom, mesh) = setup_vertical_plane_between_cells();
        // Move the plane far away so the face segment no longer hits.
        let pose = Pose::translation(10.0, 0.0, 0.0);
        let merged = resolve_apibm_face(&forest, &geom, &mesh, 0, 1, Axis::X, &pose);
        assert!(merged.is_none());
    }
}
