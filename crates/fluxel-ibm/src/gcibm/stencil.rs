//! Weighted linear least squares with the existing inverse-distance fallback.
use crate::{BoundarySurface, IbmError};
use fluxel_core::{Direction, Forest};
use fluxel_geometry::{get_global_id_from_phys, Geometry, RigidPose};
use nalgebra::{Matrix4, Vector3 as Vector, Vector4};
#[inline]
/// Locate a physical sample only if it falls in a cell marked fluid; otherwise return None.
fn fluid_cell_from_phys(
    forest: &Forest,
    geom: &Geometry,
    gc_is_fluid: &[bool],
    px: f64,
    py: f64,
    pz: f64,
) -> Option<usize> {
    get_global_id_from_phys(forest, geom, [px, py, pz]).filter(|&gid| gc_is_fluid[gid])
}

/// One computational ghost cell: boundary data and a weighted linear interpolation row.
pub(super) struct GhostStencilRow {
    pub fallback: bool,
    pub unresolved: bool,
    pub cell_id: usize,
    pub bnd_anchor_id: usize,
    pub bnd_patch_id: usize,
    pub bnd_intercept: [f64; 3],
    pub image_point: [f64; 3],
    pub stencil_indices: [usize; 8],
    pub stencil_weights: [f64; 8],
}

/// Build one row for a non-fluid cell with a face-adjacent fluid neighbour.
/// Returns None for other cells. Inputs must share the forest ordering and physical
/// mapping; the classification length and global_id must be valid (unchecked).
///
/// Reflects the center through its nearest surface point. Samples the eight offsets
/// at +/- 0.1 cell width around that image point, deduplicates fluid cells, and fits
/// a weighted linear polynomial. Singular fits, fewer than four candidates, or a
/// near-zero weight sum use IDW. Zero candidates produce a self-index/weight-one
/// row marked unresolved. Unused columns have zero weight; linear weights may be
/// negative. Geometry query errors propagate without changing the inputs.
pub(super) fn try_build_ghost_stencil_row(
    global_id: usize,
    forest: &Forest,
    geom: &Geometry,
    mesh: &BoundarySurface,
    pose: &RigidPose,
    gc_is_fluid: &[bool],
) -> Result<Option<GhostStencilRow>, IbmError> {
    let logical = forest.keys()[global_id].to_logical();

    if gc_is_fluid[global_id] {
        return Ok(None);
    }

    let touches_fluid = Direction::ALL.iter().any(|&dir| {
        forest
            .face_neighbour_global_ids(global_id, dir)
            .iter()
            .any(|&nbr_id| gc_is_fluid[nbr_id])
    });

    if !touches_fluid {
        return Ok(None);
    }

    let (center_arr, size_arr) = geom.cell_bounds(&logical);
    let center = Vector::new(center_arr[0], center_arr[1], center_arr[2]);

    let hit = mesh.closest(center_arr, pose)?;
    let closest_point = Vector::new(hit.point[0], hit.point[1], hit.point[2]);
    let anchor_id = hit.anchor;
    let patch_id = hit.patch;

    let image_point = Vector::new(
        2.0 * closest_point.x - center.x,
        2.0 * closest_point.y - center.y,
        2.0 * closest_point.z - center.z,
    );

    let mut stencil_indices = [0usize; 8];
    let mut stencil_weights = [0.0f64; 8];

    let eps_x = size_arr[0] * 0.1;
    let eps_y = size_arr[1] * 0.1;
    let eps_z = size_arr[2] * 0.1;

    let offsets = [
        [-eps_x, -eps_y, -eps_z],
        [eps_x, -eps_y, -eps_z],
        [-eps_x, eps_y, -eps_z],
        [eps_x, eps_y, -eps_z],
        [-eps_x, -eps_y, eps_z],
        [eps_x, -eps_y, eps_z],
        [-eps_x, eps_y, eps_z],
        [eps_x, eps_y, eps_z],
    ];

    let mut unique_fluid_cells = Vec::new();
    for offset in &offsets {
        if let Some(idx) = fluid_cell_from_phys(
            forest,
            geom,
            gc_is_fluid,
            image_point.x + offset[0],
            image_point.y + offset[1],
            image_point.z + offset[2],
        ) {
            if !unique_fluid_cells.contains(&idx) {
                unique_fluid_cells.push(idx);
            }
        }
    }

    let num_points = unique_fluid_cells.len().min(8);

    let mut m_mat = Matrix4::zeros();
    let mut a_row = [Vector4::zeros(); 8];
    let mut idw_w = [0.0; 8];

    let l_ref = size_arr[0].max(1e-12);

    for (k, &fluid_global_id) in unique_fluid_cells.iter().take(num_points).enumerate() {
        let (c_arr, _) = geom.cell_bounds(&forest.keys()[fluid_global_id].to_logical());

        let dx = (c_arr[0] - image_point.x) / l_ref;
        let dy = (c_arr[1] - image_point.y) / l_ref;
        let dz = (c_arr[2] - image_point.z) / l_ref;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        let w = if dist < 1e-9 { 1e9 } else { 1.0 / dist };
        idw_w[k] = w;

        let a_k = Vector4::new(1.0, dx, dy, dz);
        a_row[k] = a_k;

        m_mat += a_k * a_k.transpose() * w;
    }

    let mut use_idw_fallback = true;

    if num_points >= 4 {
        if let Some(inv_m) = m_mat.try_inverse() {
            use_idw_fallback = false;
            let mut sum_w = 0.0;

            let row_0 = Vector4::new(inv_m[(0, 0)], inv_m[(0, 1)], inv_m[(0, 2)], inv_m[(0, 3)]);

            (0..num_points).for_each(|k| {
                let final_w = row_0.dot(&a_row[k]);

                let weight = idw_w[k] * final_w;
                stencil_weights[k] = weight;
                stencil_indices[k] = unique_fluid_cells[k];
                sum_w += weight;
            });

            if sum_w.abs() > 1e-12 {
                stencil_weights
                    .iter_mut()
                    .take(num_points)
                    .for_each(|w| *w /= sum_w);
            } else {
                use_idw_fallback = true;
            }
        }
    }

    if use_idw_fallback {
        let mut sum_w = 0.0;
        (0..num_points).for_each(|k| {
            stencil_weights[k] = idw_w[k];
            stencil_indices[k] = unique_fluid_cells[k];
            sum_w += idw_w[k];
        });
        if sum_w > 0.0 {
            stencil_weights
                .iter_mut()
                .take(num_points)
                .for_each(|w| *w /= sum_w);
        } else {
            stencil_indices[0] = global_id;
            stencil_weights[0] = 1.0;
        }
    }

    Ok(Some(GhostStencilRow {
        cell_id: global_id,
        bnd_anchor_id: anchor_id,
        bnd_patch_id: patch_id,
        bnd_intercept: [closest_point.x, closest_point.y, closest_point.z],
        image_point: [image_point.x, image_point.y, image_point.z],
        stencil_indices,
        stencil_weights,
        fallback: use_idw_fallback,
        unresolved: num_points == 0,
    }))
}
