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
    /// Global cell index used as the far point in the owner-side three-point stencil along the axis.
    pub owner_far_cell_id: usize,
    /// Global cell index used as the far stencil point on the neighbour side.
    pub neighbour_far_cell_id: usize,
    /// Owner-side weights `(w_b, w_o, w_far)` for boundary, owner center, and far cell.
    pub owner_weights: [f64; 3],
    /// Neighbour-side weights `(w_b, w_n, w_far)` for boundary, neighbour center, and far cell.
    pub neighbour_weights: [f64; 3],
    /// IBM surface triangle id for the hit resolved from the owner side.
    pub owner_bnd_anchor_id: usize,
    /// IBM surface triangle id for the hit resolved from the neighbour side.
    pub neighbour_bnd_anchor_id: usize,
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
        owner_far_cell_id: owner_side.far_cell_id,
        neighbour_far_cell_id: neighbour_side.far_cell_id,
        owner_weights: owner_side.weights,
        neighbour_weights: neighbour_side.weights,
        owner_bnd_anchor_id: owner_side.bnd_anchor_id,
        neighbour_bnd_anchor_id: neighbour_side.bnd_anchor_id,
    })
}

/// Computes APIBM reconstruction weights on one axis.
///
/// - `x_tgt`: Target evaluation point (extrapolation location).
/// - `x_b`: Boundary point.
/// - `x_i`: First fluid-cell center.
/// - `x_far`: Second fluid-cell center (far cell).
/// - `use_far`: Whether the far-cell center should be used.
fn calc_axis_weights(x_tgt: f64, x_b: f64, x_i: f64, _x_far: f64, use_far: bool) -> [f64; 3] {
    let d_bi = (x_b - x_i).abs();
    let dx = (x_tgt - x_i).abs();
    let eta = d_bi / dx;

    if use_far && eta < 0.5 {
        let w_b = 2.0;
        let w_i = -2.0 * eta;
        let w_far = 2.0 * eta - 1.0;
        return [w_b, w_i, w_far];
    }

    if eta < 1e-4 {
        return [1.0, 0.0, 0.0];
    }

    let w_b = 1.0 / eta;
    let w_i = (eta - 1.0) / eta;
    let w_far = 0.0;
    [w_b, w_i, w_far]
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
    let (rev_dir, fwd_dir) = axis.split_into_directions();
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
        let dist_to_bnd = toi;

        let owner_far_cells = forest.face_neighbour_global_ids(owner_global_id, rev_dir);

        let rev_dir_vec = rev_dir.to_vector();
        let ray_rev = Ray::new(center_o, rev_dir_vec);
        let far_valid = !owner_far_cells.is_empty()
            && !mesh
                .bvh
                .intersects_ray(&Pose::identity(), &ray_rev, s_o[ax] * 1.5);

        let far_cell_id = if far_valid {
            owner_far_cells[0]
        } else {
            owner_global_id
        };

        let x_o = c_o[ax];
        let x_b = x_o + toi;
        let x_o_far = x_o - s_o[ax];
        let x_o_tgt = x_o + s_o[ax];

        let weights = calc_axis_weights(x_o_tgt, x_b, x_o, x_o_far, far_valid);

        Some(ApibmOneSideIntersection {
            dist_to_bnd,
            far_cell_id,
            weights,
            bnd_anchor_id,
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
    let (rev_dir, fwd_dir) = axis.split_into_directions();
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
        let dist_to_bnd = toi;

        let neighbour_far_cells = forest.face_neighbour_global_ids(neighbour_global_id, fwd_dir);

        let fwd_dir_vec = fwd_dir.to_vector();
        let ray_fwd = Ray::new(center_n, fwd_dir_vec);
        let far_valid = !neighbour_far_cells.is_empty()
            && !mesh
                .bvh
                .intersects_ray(&Pose::identity(), &ray_fwd, s_n[ax] * 1.5);

        let far_cell_id = if far_valid {
            neighbour_far_cells[0]
        } else {
            neighbour_global_id
        };

        let x_n = c_n[ax];
        let x_b = x_n - toi;
        let x_n_far = x_n + s_n[ax];
        let x_n_tgt = x_n - s_n[ax];

        let weights = calc_axis_weights(x_n_tgt, x_b, x_n, x_n_far, far_valid);

        Some(ApibmOneSideIntersection {
            dist_to_bnd,
            far_cell_id,
            weights,
            bnd_anchor_id,
        })
    } else {
        None
    }
}

/// Result of a single ray cast from one cell toward its opposite neighbour on the chosen axis.
pub struct ApibmOneSideIntersection {
    /// Ray length from the cast origin (owner or neighbour center) to the first boundary hit.
    pub dist_to_bnd: f64,
    /// Global index of the far fluid cell for the 3-point stencil, or the cast cell if unused.
    pub far_cell_id: usize,
    /// Weights `(w_b, w_center, w_far)` for boundary, local cell center, and far cell.
    pub weights: [f64; 3],
    /// Hit IBM mesh triangle index (modulo triangle count).
    pub bnd_anchor_id: usize,
}
